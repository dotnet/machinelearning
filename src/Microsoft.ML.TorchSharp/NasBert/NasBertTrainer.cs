// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Tokenizers;
using Microsoft.ML.TorchSharp.NasBert.Models;
using Microsoft.ML.Transforms;
using TorchSharp;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Microsoft.ML.TorchSharp.Utils;
using Microsoft.ML.TorchSharp.NasBert.Optimizers;
using Microsoft.ML.TorchSharp.Extensions;
using System.IO;
using System.CodeDom;
using System.Runtime.CompilerServices;
using TorchSharp.Modules;
using System.Diagnostics;

namespace Microsoft.ML.TorchSharp.NasBert
{

    public class NasBertTrainer
    {
        public class NasBertOptions : TorchSharpBaseTrainer.Options
        {
            /// <summary>
            /// The first sentence column.
            /// </summary>
            public string Sentence1ColumnName = "Sentence";

            /// <summary>
            /// The second sentence column.
            /// </summary>
            public string Sentence2ColumnName = default;

            /// <summary>
            /// Whether to freeze encoder parameters.
            /// </summary>
            public bool FreezeEncoder = false;

            /// <summary>
            /// Whether to freeze transfer module parameters.
            /// </summary>
            public bool FreezeTransfer = false;

            /// <summary>
            /// Whether to train layer norm parameters.
            /// </summary>
            public bool LayerNormTraining = false;

            /// <summary>
            /// Whether to apply layer normalization before each encoder block.
            /// </summary>
            public bool EncoderNormalizeBefore = true;

            /// <summary>
            /// Dropout rate for general situations. Should be within [0, 1).
            /// </summary>
            public double Dropout = .1;

            /// <summary>
            /// Dropout rate for attention weights. Should be within [0, 1).
            /// </summary>
            public double AttentionDropout = .1;

            /// <summary>
            /// Dropout rate after activation functions in FFN layers. Should be within [0, 1).
            /// </summary>
            public double ActivationDropout = 0;

            /// <summary>
            /// Whether to use dynamic dropout.
            /// </summary>
            public bool DynamicDropout = false;

            /// <summary>
            /// Dropout rate in the masked language model pooler layers. Should be within [0, 1).
            /// </summary>
            public double PoolerDropout = 0;

            /// <summary>
            /// Betas for Adam optimizer.
            /// </summary>
            public IReadOnlyList<double> AdamBetas = new List<double> { .9, .999 };

            /// <summary>
            /// Epsilon for Adam optimizer.
            /// </summary>
            public double AdamEps = 1e-8;

            /// <summary>
            /// The clipping threshold of gradients. Should be within [0, +Inf). 0 means not to clip norm.
            /// </summary>
            public double ClipNorm = 5.0;

            /// <summary>
            /// Proportion of warmup steps for polynomial decay scheduler.
            /// </summary>
            public double WarmupRatio = .06;

            /// <summary>
            /// Learning rate for the first N epochs; all epochs >N using LR_N.
            /// Note: this may be interpreted differently depending on the scheduler.
            /// </summary>
            public List<double> LearningRate = new List<double> { 1e-4 };

            /// <summary>
            /// Task type, which is related to the model head.
            /// </summary>
            public BertTaskType TaskType = BertTaskType.None;

            /// <summary>
            /// The index numbers of model architecture. Fixed by the TorchSharp model.
            /// </summary>
            internal IReadOnlyList<int> Arches = new int[] { 9, 11, 7, 0, 0, 0, 11, 11, 7, 0, 0, 0, 9, 7, 11, 0, 0, 0, 10, 7, 9, 0, 0, 0 };

            /// <summary>
            /// Maximum length of a sample. Set by the TorchSharp model.
            /// </summary>
            internal int MaxSequenceLength = 512;

            /// <summary>
            /// Number of the embedding dimensions. Should be positive. Set by the TorchSharp model.
            /// </summary>
            internal int EmbeddingDim = 64;

            /// <summary>
            /// Number of encoder layers. Set by the TorchSharp model.
            /// </summary>
            internal int EncoderLayers = 24;

            /// <summary>
            ///  Number of the output dimensions of encoder. Should be positive. Set by the TorchSharp model. 3 * EmbeddingDim
            /// </summary>
            internal int EncoderOutputDim = 192;

            /// <summary>
            /// The activation function to use for general situations. Set by the TorchSharp model.
            /// </summary>
            internal string ActivationFunction = "gelu";

            /// <summary>
            /// The activation function to use for pooler layers. Set by the TorchSharp model.
            /// </summary>
            internal string PoolerActivationFunction = "tanh";

            /// <summary>
            /// Reduction of criterion loss function. Set by the TorchSharp model.
            /// </summary>
            internal torch.nn.Reduction Reduction = Reduction.Sum;

            internal BertModelType ModelType = BertModelType.NasBert;
        }
    }

    public abstract class NasBertTrainer<TLabelCol, TTargetsCol> : TorchSharpBaseTrainer<TLabelCol, TTargetsCol>
    {

        internal readonly NasBertTrainer.NasBertOptions BertOptions;

        internal NasBertTrainer(IHostEnvironment env, Options options) : base(env, options)
        {
            BertOptions = options as NasBertTrainer.NasBertOptions;
            Contracts.AssertValue(BertOptions.Sentence1ColumnName);
            Contracts.Assert(BertOptions.TaskType != BertTaskType.None, "BertTaskType must be specified");
        }

        private protected abstract class NasBertTrainerBase : TrainerBase
        {
            public Tokenizer Tokenizer;
            public new BaseOptimizer Optimizer;
            public new NasBertTrainer<TLabelCol, TTargetsCol> Parent => base.Parent as NasBertTrainer<TLabelCol, TTargetsCol>;
            public new NasBertModel Model;
            private protected ValueGetter<ReadOnlyMemory<char>> Sentence1Getter;
            private protected ValueGetter<ReadOnlyMemory<char>> Sentence2Getter;

            public NasBertTrainerBase(TorchSharpBaseTrainer<TLabelCol, TTargetsCol> parent, IChannel ch, IDataView input, string modelUrl) : base(parent, ch, input, modelUrl)
            {
                // Get the parameters that need optimization and set up the optimizer
                var parameters = Model.parameters().Where(p => p.requires_grad);
                Optimizer = BaseOptimizer.GetOptimizer(Parent.BertOptions, parameters);
                base.Optimizer = Optimizer.Optimizer;
                LearningRateScheduler = torch.optim.lr_scheduler.OneCycleLR(
                   Optimizer.Optimizer,
                   max_lr: Parent.BertOptions.LearningRate[0],
                   total_steps: ((TrainingRowCount / Parent.Option.BatchSize) + 1) * Parent.Option.MaxEpoch,
                   pct_start: Parent.BertOptions.WarmupRatio,
                   anneal_strategy: torch.optim.lr_scheduler.impl.OneCycleLR.AnnealStrategy.Linear,
                   div_factor: 1.0 / Parent.Option.StartLearningRateRatio,
                   final_div_factor: Parent.Option.StartLearningRateRatio / Parent.Option.FinalLearningRateRatio);
            }

            private protected override Module CreateModule(IChannel ch, IDataView input)
            {
                Tokenizer = TokenizerExtensions.GetInstance(ch);
                EnglishRobertaTokenizer tokenizerModel = Tokenizer.RobertaModel();

                NasBertModel model;
                if (Parent.BertOptions.TaskType == BertTaskType.NamedEntityRecognition)
                    model = new NerModel(Parent.BertOptions, tokenizerModel.PadIndex, tokenizerModel.SymbolsCount, Parent.Option.NumberOfClasses);
                else
                    model = new ModelForPrediction(Parent.BertOptions, tokenizerModel.PadIndex, tokenizerModel.SymbolsCount, Parent.Option.NumberOfClasses);
                model.GetEncoder().load(GetModelPath(ModelUrl));
                Model = model;
                return model;
            }

            private protected override DataViewRowCursor GetRowCursor(IDataView input)
            {
                if (Parent.BertOptions.Sentence2ColumnName != default)
                    return input.GetRowCursor(input.Schema[Parent.BertOptions.Sentence1ColumnName], input.Schema[Parent.BertOptions.Sentence2ColumnName], input.Schema[Parent.Option.LabelColumnName]);
                else
                    return input.GetRowCursor(input.Schema[Parent.BertOptions.Sentence1ColumnName], input.Schema[Parent.Option.LabelColumnName]);
            }

            private protected override void InitializeDataGetters(IDataView input, DataViewRowCursor cursor)
            {
                Sentence1Getter = cursor.GetGetter<ReadOnlyMemory<char>>(input.Schema[Parent.BertOptions.Sentence1ColumnName]);
                Sentence2Getter = Parent.BertOptions.Sentence2ColumnName != default ? cursor.GetGetter<ReadOnlyMemory<char>>(input.Schema[Parent.BertOptions.Sentence2ColumnName]) : default;
            }

            private protected override void RunModelAndUpdateValidationStats(ref Tensor inputTensor, ref Tensor targetsTensor, ref int numCorrect)
            {
                var logits = Model.forward(inputTensor);
                var predictions = GetPredictions(logits);
                var targetss = GetTargets(targetsTensor);
                numCorrect = GetNumCorrect(predictions, targetss);
            }

            private protected override torch.Tensor PrepareBatchTensor(ref List<Tensor> inputTensors, Device device)
            {
                return DataUtils.CollateTokens(inputTensors, Tokenizer.RobertaModel().PadIndex, device: Device);
            }

            private protected override torch.Tensor PrepareRowTensor(ref TLabelCol target)
            {
                ReadOnlyMemory<char> sentence1 = default;
                Sentence1Getter(ref sentence1);
                Tensor t;
                if (Sentence2Getter == default)
                {
                    t = torch.tensor((new[] { 0 /* InitToken */ }).Concat(Tokenizer.EncodeToConverted(sentence1.ToString())).ToList(), device: Device);
                }
                else
                {

                    ReadOnlyMemory<char> sentence2 = default;
                    Sentence2Getter(ref sentence2);

                    t = torch.tensor((new[] { 0 /* InitToken */ }).Concat(Tokenizer.EncodeToConverted(sentence1.ToString()))
                        .Concat(new[] { 2 /* SeparatorToken */ }).Concat(Tokenizer.EncodeToConverted(sentence2.ToString())).ToList(), device: Device);
                }

                if (t.NumberOfElements > 512)
                    t = t.slice(0, 0, 512, 1);

                return t;
            }

            private protected override void RunModelAndBackPropagate(ref List<Tensor> inputTensors, ref Tensor targetsTensor)
            {
                Tensor logits = default;
                if (Parent.BertOptions.TaskType == BertTaskType.NamedEntityRecognition)
                {
                    int[,] lengthArray = new int[inputTensors.Count, 1];
                    for (int i = 0; i < inputTensors.Count; i++)
                    {
                        lengthArray[i, 0] = (int)inputTensors[i].shape[0];
                    }
                    Tensor lengths = torch.tensor(lengthArray, device: Device);

                    var inputTensor = PrepareBatchTensor(ref inputTensors, device: Device);
                    var tokenMask = torch.arange(512).expand(lengths.numel(), 512).to(lengths.device) < lengths;

                    logits = Model.forward(inputTensor, tokenMask: tokenMask);

                }
                else
                {
                    var inputTensor = PrepareBatchTensor(ref inputTensors, device: Device);

                    logits = Model.forward(inputTensor);
                }

                torch.Tensor loss;
                if (Parent.BertOptions.TaskType == BertTaskType.TextClassification)
                    loss = torch.nn.CrossEntropyLoss(reduction: Parent.BertOptions.Reduction).forward(logits, targetsTensor);
                else if (Parent.BertOptions.TaskType == BertTaskType.NamedEntityRecognition)
                {
                    targetsTensor = targetsTensor.@long().view(-1);
                    logits = logits.view(-1, logits.size(-1));
                    loss = torch.nn.CrossEntropyLoss(reduction: Parent.BertOptions.Reduction).forward(logits, targetsTensor);
                }
                else
                {
                    loss = torch.nn.MSELoss(reduction: Parent.BertOptions.Reduction).forward(logits.squeeze(), targetsTensor);
                }

                loss.backward();
            }

            private protected override void OptimizeStep()
            {
                Optimizer.Step();
                LearningRateScheduler.step();
            }
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);

            var outColumns = inputSchema.ToDictionary(x => x.Name);

            if (BertOptions.TaskType == BertTaskType.TextClassification)
            {

                var metadata = new List<SchemaShape.Column>();
                metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector,
                    TextDataViewType.Instance, false));

                // Get label column for score column annotations. Already verified it exists.
                inputSchema.TryFindColumn(Option.LabelColumnName, out var labelCol);

                outColumns[Option.PredictionColumnName] = new SchemaShape.Column(Option.PredictionColumnName, SchemaShape.Column.VectorKind.Scalar,
                        NumberDataViewType.UInt32, true, new SchemaShape(metadata.ToArray()));

                outColumns[Option.ScoreColumnName] = new SchemaShape.Column(Option.ScoreColumnName, SchemaShape.Column.VectorKind.Vector,
                    NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.AnnotationsForMulticlassScoreColumn(labelCol)));
            }
            else if (BertOptions.TaskType == BertTaskType.NamedEntityRecognition)
            {
                var metadata = new List<SchemaShape.Column>();
                metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector,
                    TextDataViewType.Instance, false));

                // Get label column for score column annotations. Already verified it exists.
                inputSchema.TryFindColumn(Option.LabelColumnName, out var labelCol);

                outColumns[Option.PredictionColumnName] = new SchemaShape.Column(Option.PredictionColumnName, SchemaShape.Column.VectorKind.VariableVector,
                        NumberDataViewType.UInt32, true, new SchemaShape(metadata.ToArray()));
            }
            else
            {
                outColumns[Option.ScoreColumnName] = new SchemaShape.Column(Option.ScoreColumnName, SchemaShape.Column.VectorKind.Scalar,
                    NumberDataViewType.Single, false);
            }

            return new SchemaShape(outColumns.Values);
        }

        private protected override void CheckInputSchema(SchemaShape inputSchema)
        {
            // Verify that all required input columns are present, and are of the same type.
            if (!inputSchema.TryFindColumn(BertOptions.Sentence1ColumnName, out var sentenceCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence", BertOptions.Sentence1ColumnName);
            if (sentenceCol.ItemType != TextDataViewType.Instance)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence", BertOptions.Sentence1ColumnName,
                    TextDataViewType.Instance.ToString(), sentenceCol.GetTypeString());

            if (!inputSchema.TryFindColumn(Option.LabelColumnName, out var labelCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", Option.LabelColumnName);

            if (BertOptions.TaskType == BertTaskType.TextClassification)
            {
                if (labelCol.ItemType != NumberDataViewType.UInt32)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", Option.LabelColumnName,
                        NumberDataViewType.UInt32.ToString(), labelCol.GetTypeString());


                if (BertOptions.Sentence2ColumnName != default)
                {
                    if (!inputSchema.TryFindColumn(BertOptions.Sentence2ColumnName, out var sentenceCol2))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", BertOptions.Sentence2ColumnName);
                    if (sentenceCol2.ItemType != TextDataViewType.Instance)
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", BertOptions.Sentence2ColumnName,
                            TextDataViewType.Instance.ToString(), sentenceCol2.GetTypeString());
                }
            }
            else if (BertOptions.TaskType == BertTaskType.NamedEntityRecognition)
            {
                if (labelCol.ItemType != NumberDataViewType.UInt32)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", Option.LabelColumnName,
                        NumberDataViewType.UInt32.ToString(), labelCol.GetTypeString());
            }
            else
            {
                if (labelCol.ItemType != NumberDataViewType.Single)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", Option.LabelColumnName,
                        NumberDataViewType.Single.ToString(), labelCol.GetTypeString());

                if (!inputSchema.TryFindColumn(BertOptions.Sentence2ColumnName, out var sentenceCol2))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", BertOptions.Sentence2ColumnName);
                if (sentenceCol2.ItemType != TextDataViewType.Instance)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", BertOptions.Sentence2ColumnName,
                        TextDataViewType.Instance.ToString(), sentenceCol2.GetTypeString());
            }
        }
    }

    public abstract class NasBertTransformer<TLabelCol, TTargetsCol> : TorchSharpBaseTransformer<TLabelCol, TTargetsCol>
    {
        internal readonly NasBertTrainer.NasBertOptions BertOptions;


        public readonly SchemaShape.Column SentenceColumn;
        public readonly SchemaShape.Column SentenceColumn2;

        internal NasBertTransformer(IHostEnvironment env, NasBertTrainer.NasBertOptions options, NasBertModel model, DataViewSchema.DetachedColumn labelColumn)
           : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(NasBertTransformer<TLabelCol, TTargetsCol>)), options, model, labelColumn)
        {
            BertOptions = options;
            SentenceColumn = new SchemaShape.Column(options.Sentence1ColumnName, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance, false);
            SentenceColumn2 = options.Sentence2ColumnName == default ? default : new SchemaShape.Column(options.Sentence2ColumnName, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance, false);
        }

        private protected override SchemaShape GetOutputSchemaCore(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);

            var outColumns = inputSchema.ToDictionary(x => x.Name);
            if (BertOptions.TaskType == BertTaskType.TextClassification)
            {
                var labelAnnotationsColumn = new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, LabelColumn.Annotations.Schema[AnnotationUtils.Kinds.SlotNames].Type, false);
                var predLabelMetadata = new SchemaShape(new SchemaShape.Column[] { labelAnnotationsColumn }
                    .Concat(AnnotationUtils.GetTrainerOutputAnnotation()));

                outColumns[Options.PredictionColumnName] = new SchemaShape.Column(Options.PredictionColumnName, SchemaShape.Column.VectorKind.Scalar,
                    NumberDataViewType.UInt32, true, predLabelMetadata);

                outColumns[ScoreColumnName] = new SchemaShape.Column(ScoreColumnName, SchemaShape.Column.VectorKind.Vector,
                       NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.AnnotationsForMulticlassScoreColumn(labelAnnotationsColumn)));
            }
            else
            {
                outColumns[ScoreColumnName] = new SchemaShape.Column(ScoreColumnName, SchemaShape.Column.VectorKind.Scalar,
                       NumberDataViewType.Single, false);
            }

            return new SchemaShape(outColumns.Values);
        }

        private protected override void CheckInputSchema(SchemaShape inputSchema)
        {
            // Verify that all required input columns are present, and are of the same type.
            if (!inputSchema.TryFindColumn(SentenceColumn.Name, out var sentenceCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence", SentenceColumn.Name);
            if (!SentenceColumn.IsCompatibleWith(sentenceCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence", SentenceColumn.Name,
                    SentenceColumn.GetTypeString(), sentenceCol.GetTypeString());

            if (BertOptions.Sentence2ColumnName != default || BertOptions.TaskType == BertTaskType.SentenceRegression)
            {
                if (!inputSchema.TryFindColumn(SentenceColumn2.Name, out var sentenceCol2))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", SentenceColumn2.Name);
                if (!SentenceColumn2.IsCompatibleWith(sentenceCol2))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", SentenceColumn2.Name,
                        SentenceColumn2.GetTypeString(), sentenceCol2.GetTypeString());
            }
        }

        private protected abstract override void SaveModel(ModelSaveContext ctx);


        private protected new void SaveBaseModel(ModelSaveContext ctx, VersionInfo versionInfo)
        {
            base.SaveBaseModel(ctx, versionInfo);

            // *** Binary format ***
            // int: id of sentence 1 column name
            // int: id of sentence 2 column name

            ctx.SaveNonEmptyString(BertOptions.Sentence1ColumnName);
            ctx.SaveStringOrNull(BertOptions.Sentence2ColumnName);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => GetRowMapper(this, schema);

        private protected abstract class NasBertMapper : TorchSharpBaseMapper
        {
            private protected new NasBertTransformer<TLabelCol, TTargetsCol> Parent => base.Parent as NasBertTransformer<TLabelCol, TTargetsCol>;

            private static readonly FuncInstanceMethodInfo1<NasBertMapper, DataViewSchema.DetachedColumn, Delegate> _makeLabelAnnotationGetter
                = FuncInstanceMethodInfo1<NasBertMapper, DataViewSchema.DetachedColumn, Delegate>.Create(target => target.GetLabelAnnotations<int>);
            internal static readonly int[] InitTokenArray = new[] { 0 /* InitToken */ };
            internal static readonly int[] SeperatorTokenArray = new[] { 2 /* SeperatorToken */ };

            public NasBertMapper(TorchSharpBaseTransformer<TLabelCol, TTargetsCol> parent, DataViewSchema inputSchema) :
                base(parent, inputSchema)
            {
            }

            private protected override void AddInputColumnIndices(DataViewSchema inputSchema)
            {
                if (inputSchema.TryGetColumnIndex(Parent.BertOptions.Sentence1ColumnName, out var col))
                    InputColIndices.Add(col);

                if (Parent.BertOptions.Sentence2ColumnName != default || Parent.BertOptions.TaskType == BertTaskType.SentenceRegression)
                    if (inputSchema.TryGetColumnIndex(Parent.BertOptions.Sentence2ColumnName, out col))
                        InputColIndices.Add(col);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                if (Parent.BertOptions.TaskType == BertTaskType.TextClassification)
                {
                    var info = new DataViewSchema.DetachedColumn[2];
                    var keyType = Parent.LabelColumn.Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues)?.Type as VectorDataViewType;
                    var getter = Microsoft.ML.Internal.Utilities.Utils.MarshalInvoke(_makeLabelAnnotationGetter, this, keyType.ItemType.RawType, Parent.LabelColumn);


                    var meta = new DataViewSchema.Annotations.Builder();
                    meta.Add(AnnotationUtils.Kinds.ScoreColumnKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreColumnKind.MulticlassClassification.AsMemory(); });
                    meta.Add(AnnotationUtils.Kinds.ScoreColumnSetId, AnnotationUtils.ScoreColumnSetIdType, GetScoreColumnSetId(InputSchema));
                    meta.Add(AnnotationUtils.Kinds.ScoreValueKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreValueKind.Score.AsMemory(); });
                    meta.Add(AnnotationUtils.Kinds.TrainingLabelValues, keyType, getter);
                    meta.Add(AnnotationUtils.Kinds.SlotNames, keyType, getter);

                    var labelBuilder = new DataViewSchema.Annotations.Builder();
                    labelBuilder.Add(AnnotationUtils.Kinds.KeyValues, keyType, getter);

                    info[0] = new DataViewSchema.DetachedColumn(Parent.Options.PredictionColumnName, new KeyDataViewType(typeof(uint), Parent.Options.NumberOfClasses), labelBuilder.ToAnnotations());

                    info[1] = new DataViewSchema.DetachedColumn(Parent.Options.ScoreColumnName, new VectorDataViewType(NumberDataViewType.Single, Parent.Options.NumberOfClasses), meta.ToAnnotations());
                    return info;
                }
                else if (Parent.BertOptions.TaskType == BertTaskType.NamedEntityRecognition)
                {
                    var info = new DataViewSchema.DetachedColumn[1];
                    var keyType = Parent.LabelColumn.Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues)?.Type as VectorDataViewType;
                    var getter = Microsoft.ML.Internal.Utilities.Utils.MarshalInvoke(_makeLabelAnnotationGetter, this, keyType.ItemType.RawType, Parent.LabelColumn);

                    var labelBuilder = new DataViewSchema.Annotations.Builder();
                    labelBuilder.Add(AnnotationUtils.Kinds.KeyValues, keyType, getter);

                    info[0] = new DataViewSchema.DetachedColumn(Parent.Options.PredictionColumnName, new VectorDataViewType(new KeyDataViewType(typeof(uint), Parent.Options.NumberOfClasses - 1)), labelBuilder.ToAnnotations());

                    return info;
                }
                else
                {
                    var info = new DataViewSchema.DetachedColumn[1];

                    info[0] = new DataViewSchema.DetachedColumn(Parent.Options.ScoreColumnName, NumberDataViewType.Single);
                    return info;
                }
            }

            private Delegate GetLabelAnnotations<T>(DataViewSchema.DetachedColumn labelCol)
            {
                return labelCol.Annotations.GetGetter<VBuffer<T>>(labelCol.Annotations.Schema[AnnotationUtils.Kinds.KeyValues]);
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
                => throw new NotImplementedException("This should never be called!");

            private protected class BertTensorCacher : TorchSharpBaseMapper.TensorCacher<Tensor>
            {
                public override void DisposeCore()
                {
                    Result?.Dispose();
                }
            }

            private protected override TensorCacher GetTensorCacher()
            {
                return new BertTensorCacher();
            }

            private IList<int> PrepInputTokens(ref ReadOnlyMemory<char> sentence1, ref ReadOnlyMemory<char> sentence2, ref ValueGetter<ReadOnlyMemory<char>> getSentence1, ref ValueGetter<ReadOnlyMemory<char>> getSentence2, Tokenizer tokenizer)
            {
                getSentence1(ref sentence1);
                if (getSentence2 == default)
                {
                    List<int> newList = new List<int>(tokenizer.EncodeToConverted(sentence1.ToString()));
                    // 0 Is the init token and must be at the beginning.
                    newList.Insert(0, 0);
                    return newList;
                }
                else
                {
                    getSentence2(ref sentence2);
                    return InitTokenArray.Concat(tokenizer.EncodeToConverted(sentence1.ToString()))
                                              .Concat(SeperatorTokenArray).Concat(tokenizer.EncodeToConverted(sentence2.ToString())).ToList();
                }
            }

            private Tensor PrepAndRunModel(IList<int> tokens)
            {
                using (torch.no_grad())
                {
                    var inputTensor = torch.tensor(tokens, device: Parent.Device);
                    if (inputTensor.NumberOfElements > 512)
                        inputTensor = inputTensor.slice(0, 0, 512, 1);
                    inputTensor = inputTensor.reshape(1, inputTensor.NumberOfElements);
                    return (Parent.Model as NasBertModel).forward(inputTensor);
                }
            }

            private protected void UpdateCacheIfNeeded(long position, TensorCacher outputCache, ref ReadOnlyMemory<char> sentence1, ref ReadOnlyMemory<char> sentence2, ref ValueGetter<ReadOnlyMemory<char>> getSentence1, ref ValueGetter<ReadOnlyMemory<char>> getSentence2, Tokenizer tokenizer)
            {
                var cache = outputCache as BertTensorCacher;
                if (outputCache.Position != position)
                {
                    cache.Result?.Dispose();
                    cache.Result = PrepAndRunModel(PrepInputTokens(ref sentence1, ref sentence2, ref getSentence1, ref getSentence2, tokenizer));
                    cache.Result.MoveToOuterDisposeScope();
                    cache.Position = position;
                }
            }

            private protected override void SaveModel(ModelSaveContext ctx) => Parent.SaveModel(ctx);
        }
    }
}
