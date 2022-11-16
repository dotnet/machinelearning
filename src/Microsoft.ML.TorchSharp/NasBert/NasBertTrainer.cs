// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Tokenizers;
using Microsoft.ML.TorchSharp.NasBert.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using TorchSharp;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.TensorExtensionMethods;
using Microsoft.ML.TorchSharp.Utils;
using Microsoft.ML.TorchSharp.NasBert.Optimizers;
using Microsoft.ML;
using Microsoft.ML.TorchSharp.NasBert;
using Microsoft.ML.TorchSharp.Extensions;
using System.IO;
using System.CodeDom;
using System.Runtime.CompilerServices;
using Microsoft.ML.Data.IO;

namespace Microsoft.ML.TorchSharp.NasBert
{
    public abstract class NasBertTrainer : IEstimator<NasBertTransformer>
    {
        public abstract NasBertTransformer Fit(IDataView input);
        public abstract SchemaShape GetOutputSchema(SchemaShape inputSchema);

        internal sealed class Options : TransformInputBase
        {
            /// <summary>
            /// The label column name.
            /// </summary>
            public string LabelColumnName = DefaultColumnNames.Label;

            /// <summary>
            /// The Score column name.
            /// </summary>
            public string ScoreColumnName = DefaultColumnNames.Score;

            /// <summary>
            /// The Prediction column name.
            /// </summary>
            public string PredictionColumnName = DefaultColumnNames.PredictedLabel;

            /// <summary>
            /// The first sentence column.
            /// </summary>
            public string Sentence1ColumnName = "Sentence";

            /// <summary>
            /// The second sentence column.
            /// </summary>
            public string Sentence2ColumnName = default;

            /// <summary>
            /// Number of samples to use for mini-batch training.
            /// </summary>
            public int BatchSize = 32;

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
            /// The start learning rate for polynomial decay scheduler.
            /// </summary>
            public double StartLearningRateRatio = .1;

            /// <summary>
            /// The final learning rate for polynomial decay scheduler.
            /// </summary>
            public double FinalLearningRateRatio = .1;

            /// <summary>
            /// Betas for Adam optimizer.
            /// </summary>
            public IReadOnlyList<double> AdamBetas = new List<double> { .9, .999 };

            /// <summary>
            /// Epsilon for Adam optimizer.
            /// </summary>
            public double AdamEps = 1e-8;

            /// <summary>
            /// Coefficiency of weight decay. Should be within [0, +Inf).
            /// </summary>
            public double WeightDecay = 0;

            /// <summary>
            /// The clipping threshold of gradients. Should be within [0, +Inf). 0 means not to clip norm.
            /// </summary>
            public double ClipNorm = 25;

            /// <summary>
            /// Proportion of warmup steps for polynomial decay scheduler.
            /// </summary>
            public double WarmupRatio = .06;

            /// <summary>
            /// Stop training when reaching this number of epochs.
            /// </summary>
            public int MaxEpoch = 100;

            /// <summary>
            /// The validation set used while training to improve model quality.
            /// </summary>
            public IDataView ValidationSet = null;

            /// <summary>
            /// Number of classes for the data.
            /// </summary>
            internal int NumberOfClasses;

            /// <summary>
            /// Learning rate for the first N epochs; all epochs >N using LR_N.
            /// Note: this may be interpreted differently depending on the scheduler.
            /// </summary>
            internal List<double> LearningRate = new List<double> { 1e-4 };

            /// <summary>
            /// The index numbers of model architecture. Fixed by the TorchSharp model.
            /// </summary>
            internal IReadOnlyList<int> Arches = new int[] { 9, 11, 7, 0, 0, 0, 11, 11, 7, 0, 0, 0, 9, 7, 11, 0, 0, 0, 10, 7, 9, 0, 0, 0 };

            /// <summary>
            /// Task type, which is related to the model head.
            /// </summary>
            internal BertTaskType TaskType = BertTaskType.TextClassification;

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
        }
    }

    public abstract class NasBertTrainer<TLabelCol, TTargetsCol> : NasBertTrainer
    {
        private protected readonly IHost Host;
        internal readonly Options Option;
        private const string ModelUrl = "models/NasBert2000000.tsm";

        internal NasBertTrainer(IHostEnvironment env, Options options)
        {
            Host = Contracts.CheckRef(env, nameof(env)).Register(nameof(NasBertTrainer));
            Contracts.Assert(options.BatchSize > 0);
            Contracts.Assert(options.MaxEpoch > 0);
            Contracts.AssertValue(options.Sentence1ColumnName);
            Contracts.AssertValue(options.LabelColumnName);
            Contracts.AssertValue(options.PredictionColumnName);
            Contracts.AssertValue(options.ScoreColumnName);
            Option = options;
        }

        public override NasBertTransformer Fit(IDataView input)
        {
            CheckInputSchema(SchemaShape.Create(input.Schema));

            NasBertTransformer<TLabelCol, TTargetsCol> transformer = default;

            using (var ch = Host.Start("TrainModel"))
            using (var pch = Host.StartProgressChannel("Training model"))
            {
                var header = new ProgressHeader(new[] { "Accuracy" }, null);
                var trainer = CreateTrainer(this, ch, input);
                pch.SetHeader(header, e => e.SetMetric(0, trainer.Accuracy));
                for (int i = 0; i < Option.MaxEpoch; i++)
                {
                    ch.Trace($"Starting epoch {i}");
                    trainer.Train(input);
                    ch.Trace($"Finished epoch {i}");
                    if (Option.ValidationSet != null)
                        trainer.Validate(pch, ch, i);
                }
                var labelCol = input.Schema.GetColumnOrNull(Option.LabelColumnName);

                transformer = CreateTransformer(Host, Option, trainer.Model, new DataViewSchema.DetachedColumn(labelCol.Value));

                transformer.GetOutputSchema(input.Schema);
            }
            return transformer;
        }

        private protected abstract NasBertTransformer<TLabelCol, TTargetsCol> CreateTransformer(IHost host, NasBertTrainer<TLabelCol, TTargetsCol>.Options options, NasBertModel model, DataViewSchema.DetachedColumn labelColumn);

        private protected abstract TrainerBase CreateTrainer(NasBertTrainer<TLabelCol, TTargetsCol> parent, IChannel ch, IDataView input);

        private protected abstract class TrainerBase
        {
            public Tokenizer Tokenizer;
            public NasBertModel Model;
            public torch.Device Device;
            public BaseOptimizer Optimizer;
            public optim.lr_scheduler.LRScheduler LearningRateScheduler;
            protected readonly NasBertTrainer<TLabelCol, TTargetsCol> Parent;
            public int Updates;
            public float Accuracy;

            public TrainerBase(NasBertTrainer<TLabelCol, TTargetsCol> parent, IChannel ch, IDataView input)
            {
                Parent = parent;
                Updates = 0;
                Accuracy = 0;

                // Get the tokenizer
                Tokenizer = TokenizerExtensions.GetInstance(ch);
                EnglishRoberta tokenizerModel = Tokenizer.RobertaModel();

                // Get row count and figure out num of unique labels
                var rowCount = GetRowCountAndSetLabelCount(input);

                // Initialize the model and load pre-trained weights
                Model = new NasBertModel(Parent.Option, tokenizerModel.PadIndex, tokenizerModel.SymbolsCount, Parent.Option.NumberOfClasses);
                Model.GetEncoder().load(GetModelPath());

                // Figure out if we are running on GPU or CPU
                Device = TorchUtils.InitializeDevice(Parent.Host);

                // Move to GPU if we are running there
                if (Device == CUDA)
                    Model.cuda();

                // Get the parameters that need optimization and set up the optimizer
                var parameters = Model.parameters().Where(p => p.requires_grad);
                Optimizer = BaseOptimizer.GetOptimizer(Parent.Option, parameters);
                LearningRateScheduler = torch.optim.lr_scheduler.OneCycleLR(
                   Optimizer.Optimizer,
                   max_lr: Parent.Option.LearningRate[0],
                   total_steps: ((rowCount / Parent.Option.BatchSize) + 1) * Parent.Option.MaxEpoch,
                   pct_start: Parent.Option.WarmupRatio,
                   anneal_strategy: torch.optim.lr_scheduler.impl.OneCycleLR.AnnealStrategy.Linear,
                   div_factor: 1.0 / Parent.Option.StartLearningRateRatio,
                   final_div_factor: 1.0 / Parent.Option.FinalLearningRateRatio);
            }

            private protected abstract int GetRowCountAndSetLabelCount(IDataView input);

            private string GetModelPath()
            {
                var destDir = Path.Combine(((IHostEnvironmentInternal)Parent.Host).TempFilePath, "mlnet");
                var destFileName = ModelUrl.Split('/').Last();

                Directory.CreateDirectory(destDir);

                string relativeFilePath = Path.Combine(destDir, destFileName);

                int timeout = 10 * 60 * 1000;
                using (var ch = (Parent.Host as IHostEnvironment).Start("Ensuring model file is present."))
                {
                    var ensureModel = ResourceManagerUtils.Instance.EnsureResourceAsync(Parent.Host, ch, ModelUrl, destFileName, destDir, timeout);
                    ensureModel.Wait();
                    var errorResult = ResourceManagerUtils.GetErrorMessage(out var errorMessage, ensureModel.Result);
                    if (errorResult != null)
                    {
                        var directory = Path.GetDirectoryName(errorResult.FileName);
                        var name = Path.GetFileName(errorResult.FileName);
                        throw ch.Except($"{errorMessage}\nmodel file could not be downloaded!");
                    }
                }

                return relativeFilePath;
            }

            public void Validate(IProgressChannel pch, IChannel ch, int epoch)
            {
                var validationSet = Parent.Option.ValidationSet;
                Model.eval();

                DataViewRowCursor cursor = default;
                if (Parent.Option.Sentence2ColumnName != default)
                    cursor = validationSet.GetRowCursor(validationSet.Schema[Parent.Option.Sentence1ColumnName], validationSet.Schema[Parent.Option.Sentence2ColumnName], validationSet.Schema[Parent.Option.LabelColumnName]);
                else
                    cursor = validationSet.GetRowCursor(validationSet.Schema[Parent.Option.Sentence1ColumnName], validationSet.Schema[Parent.Option.LabelColumnName]);

                var sentence1Getter = cursor.GetGetter<ReadOnlyMemory<char>>(validationSet.Schema[Parent.Option.Sentence1ColumnName]);
                var sentence2Getter = Parent.Option.Sentence2ColumnName != default ? cursor.GetGetter<ReadOnlyMemory<char>>(validationSet.Schema[Parent.Option.Sentence2ColumnName]) : default;
                var labelGetter = cursor.GetGetter<TLabelCol>(validationSet.Schema[Parent.Option.LabelColumnName]);

                // Pre-allocate the memory so it's only done once (though this step needs to be optimized)
                List<Tensor> inputTensors = new List<Tensor>(Parent.Option.BatchSize);
                List<TTargetsCol> targets = new List<TTargetsCol>(Parent.Option.BatchSize);
                int numCorrect = 0;
                int numRows = 0;

                var cursorValid = true;
                while (cursorValid)
                {
                    cursorValid = ValidateStep(cursor, sentence1Getter, sentence2Getter, labelGetter, ref inputTensors, ref targets, ref numCorrect, ref numRows);
                }
                Accuracy = numCorrect / (float)numRows;
                pch.Checkpoint(Accuracy);
                ch.Info($"Accuracy for epoch {epoch}: {Accuracy}");

                Model.train();
            }

            private bool ValidateStep(DataViewRowCursor cursor,
                ValueGetter<ReadOnlyMemory<char>> sentence1Getter,
                ValueGetter<ReadOnlyMemory<char>> sentence2Getter,
                ValueGetter<TLabelCol> labelGetter,
                ref List<Tensor> inputTensors,
                ref List<TTargetsCol> targets,
                ref int numCorrect,
                ref int numRows)
            {
                // Make sure list is clear before use
                inputTensors.Clear();
                targets.Clear();
                using var disposeScope = torch.NewDisposeScope();
                var cursorValid = true;
                for (int i = 0; i < Parent.Option.BatchSize && cursorValid; i++)
                {
                    cursorValid = cursor.MoveNext();
                    if (cursorValid)
                    {
                        inputTensors.Add(PrepareData(sentence1Getter, sentence2Getter));
                        TLabelCol target = default;
                        labelGetter(ref target);
                        targets.Add(AddToTargets(target));
                    }
                    else
                    {
                        inputTensors.TrimExcess();
                        targets.TrimExcess();
                        if (inputTensors.Count() == 0)
                            return cursorValid;
                    }
                }

                using (torch.no_grad())
                {
                    var inputTensor = DataUtils.CollateTokens(inputTensors, Tokenizer.RobertaModel().PadIndex, device: Device);
                    var targetsTensor = CreateTargetsTensor(ref targets, device: Device);
                    var logits = Model.forward(inputTensor);
                    var predictions = GetPredictions(logits);
                    var targetss = GetTargets(targetsTensor);
                    numCorrect = GetNumCorrect(predictions, targetss);
                    numRows = inputTensors.Count;
                }

                return cursorValid;
            }

            public void Train(IDataView input)
            {
                // Get the cursor and the correct columns based on the inputs
                DataViewRowCursor cursor = default;
                if (Parent.Option.Sentence2ColumnName != default)
                    cursor = input.GetRowCursor(input.Schema[Parent.Option.Sentence1ColumnName], input.Schema[Parent.Option.Sentence2ColumnName], input.Schema[Parent.Option.LabelColumnName]);
                else
                    cursor = input.GetRowCursor(input.Schema[Parent.Option.Sentence1ColumnName], input.Schema[Parent.Option.LabelColumnName]);

                var sentence1Getter = cursor.GetGetter<ReadOnlyMemory<char>>(input.Schema[Parent.Option.Sentence1ColumnName]);
                var sentence2Getter = Parent.Option.Sentence2ColumnName != default ? cursor.GetGetter<ReadOnlyMemory<char>>(input.Schema[Parent.Option.Sentence2ColumnName]) : default;
                var labelGetter = cursor.GetGetter<TLabelCol>(input.Schema[Parent.Option.LabelColumnName]);

                // Pre-allocate the memory so it's only done once (though this step needs to be optimized)
                List<Tensor> inputTensors = new List<Tensor>(Parent.Option.BatchSize);
                List<TTargetsCol> targets = new List<TTargetsCol>(Parent.Option.BatchSize);

                var cursorValid = true;
                while (cursorValid)
                {
                    cursorValid = TrainStep(cursor, sentence1Getter, sentence2Getter, labelGetter, ref inputTensors, ref targets);
                }
            }

            private bool TrainStep(DataViewRowCursor cursor,
            ValueGetter<ReadOnlyMemory<char>> sentence1Getter,
            ValueGetter<ReadOnlyMemory<char>> sentence2Getter,
            ValueGetter<TLabelCol> labelGetter,
            ref List<Tensor> inputTensors,
            ref List<TTargetsCol> targets)
            {
                // Make sure list is clear before use
                inputTensors.Clear();
                targets.Clear();
                using var disposeScope = torch.NewDisposeScope();
                var cursorValid = true;
                for (int i = 0; i < Parent.Option.BatchSize && cursorValid; i++)
                {
                    cursorValid = cursor.MoveNext();
                    if (cursorValid)
                    {
                        inputTensors.Add(PrepareData(sentence1Getter, sentence2Getter));
                        TLabelCol target = default;
                        labelGetter(ref target);
                        targets.Add(AddToTargets(target));
                    }
                    else
                    {
                        inputTensors.TrimExcess();
                        targets.TrimExcess();
                        if (inputTensors.Count() == 0)
                            return cursorValid;
                    }
                }

                Updates++;

                torch.random.manual_seed(1 + Updates);
                torch.cuda.manual_seed(1 + Updates);
                Model.train();
                Optimizer.zero_grad();

                var inputTensor = DataUtils.CollateTokens(inputTensors, Tokenizer.RobertaModel().PadIndex, device: Device);
                var targetsTensor = CreateTargetsTensor(ref targets, device: Device);
                var logits = Model.forward(inputTensor);

                torch.Tensor loss;
                if (Parent.Option.TaskType == BertTaskType.TextClassification)
                    loss = torch.nn.CrossEntropyLoss(reduction: Parent.Option.Reduction).forward(logits, targetsTensor);
                else
                {
                    loss = torch.nn.MSELoss(reduction: Parent.Option.Reduction).forward(logits, targetsTensor);
                    logits = logits.squeeze();
                }

                loss.backward();
                OptimizeStep();

                return cursorValid;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private protected abstract TTargetsCol AddToTargets(TLabelCol target);

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private protected abstract Tensor CreateTargetsTensor(ref List<TTargetsCol> targets, Device device);

            private protected abstract torch.Tensor GetPredictions(torch.Tensor logits);

            private protected abstract torch.Tensor GetTargets(torch.Tensor labels);

            private protected abstract int GetNumCorrect(torch.Tensor predictions, torch.Tensor targets);

            private void OptimizeStep()
            {
                Optimizer.Step();
                LearningRateScheduler.step();
            }

            protected torch.Tensor PrepareData(ValueGetter<ReadOnlyMemory<char>> sentence1Getter, ValueGetter<ReadOnlyMemory<char>> sentence2Getter)
            {
                ReadOnlyMemory<char> sentence1 = default;
                sentence1Getter(ref sentence1);
                Tensor t;
                if (sentence2Getter == default)
                {
                    t = torch.tensor((new[] { 0 /* InitToken */ }).Concat(Tokenizer.EncodeToConverted(sentence1.ToString())).ToList(), device: Device);
                }
                else
                {

                    ReadOnlyMemory<char> sentence2 = default;
                    sentence2Getter(ref sentence2);

                    t = torch.tensor((new[] { 0 /* InitToken */ }).Concat(Tokenizer.EncodeToConverted(sentence1.ToString()))
                        .Concat(new[] { 2 /* SeparatorToken */ }).Concat(Tokenizer.EncodeToConverted(sentence2.ToString())).ToList(), device: Device);
                }

                if (t.NumberOfElements > 512)
                    t = t.slice(0, 0, 512, 1);

                return t;
            }
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);

            var outColumns = inputSchema.ToDictionary(x => x.Name);

            if (Option.TaskType == BertTaskType.TextClassification)
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
            else
            {
                outColumns[Option.ScoreColumnName] = new SchemaShape.Column(Option.ScoreColumnName, SchemaShape.Column.VectorKind.Scalar,
                    NumberDataViewType.Single, false);
            }

            return new SchemaShape(outColumns.Values);
        }

        private void CheckInputSchema(SchemaShape inputSchema)
        {
            // Verify that all required input columns are present, and are of the same type.
            if (!inputSchema.TryFindColumn(Option.Sentence1ColumnName, out var sentenceCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence", Option.Sentence1ColumnName);
            if (sentenceCol.ItemType != TextDataViewType.Instance)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence", Option.Sentence1ColumnName,
                    TextDataViewType.Instance.ToString(), sentenceCol.GetTypeString());

            if (!inputSchema.TryFindColumn(Option.LabelColumnName, out var labelCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", Option.LabelColumnName);

            if (Option.TaskType == BertTaskType.TextClassification)
            {
                if (labelCol.ItemType != NumberDataViewType.UInt32)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", Option.LabelColumnName,
                        NumberDataViewType.UInt32.ToString(), labelCol.GetTypeString());


                if (Option.Sentence2ColumnName != default)
                {
                    if (!inputSchema.TryFindColumn(Option.Sentence2ColumnName, out var sentenceCol2))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", Option.Sentence2ColumnName);
                    if (sentenceCol2.ItemType != TextDataViewType.Instance)
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", Option.Sentence2ColumnName,
                            TextDataViewType.Instance.ToString(), sentenceCol2.GetTypeString());
                }
            }
            else
            {
                if (labelCol.ItemType != NumberDataViewType.Single)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", Option.LabelColumnName,
                        NumberDataViewType.Single.ToString(), labelCol.GetTypeString());

                if (!inputSchema.TryFindColumn(Option.Sentence2ColumnName, out var sentenceCol2))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", Option.Sentence2ColumnName);
                if (sentenceCol2.ItemType != TextDataViewType.Instance)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", Option.Sentence2ColumnName,
                        TextDataViewType.Instance.ToString(), sentenceCol2.GetTypeString());
            }
        }
    }

    public abstract class NasBertTransformer : RowToRowTransformerBase
    {
        private protected NasBertTransformer(IHost host) : base(host)
        {
        }
    }

    public abstract class NasBertTransformer<TLabelCol, TTargetsCol> : NasBertTransformer
    {
        private protected readonly Device Device;
        private protected readonly NasBertModel Model;
        internal readonly NasBertTrainer.Options Options;

        private protected readonly string ScoreColumnName;

        public readonly SchemaShape.Column SentenceColumn;
        public readonly SchemaShape.Column SentenceColumn2;
        public readonly DataViewSchema.DetachedColumn LabelColumn;

        internal NasBertTransformer(IHostEnvironment env, NasBertTrainer.Options options, NasBertModel model, DataViewSchema.DetachedColumn labelColumn)
           : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(NasBertTransformer)))
        {
            Device = TorchUtils.InitializeDevice(env);

            Options = options;
            LabelColumn = labelColumn;
            SentenceColumn = new SchemaShape.Column(Options.Sentence1ColumnName, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance, false);
            SentenceColumn2 = Options.Sentence2ColumnName == default ? default : new SchemaShape.Column(Options.Sentence2ColumnName, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance, false);
            ScoreColumnName = Options.ScoreColumnName;

            Model = model;

            if (Device == CUDA)
                Model.cuda();
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);

            var outColumns = inputSchema.ToDictionary(x => x.Name);
            if (Options.TaskType == BertTaskType.TextClassification)
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

        private void CheckInputSchema(SchemaShape inputSchema)
        {
            // Verify that all required input columns are present, and are of the same type.
            if (!inputSchema.TryFindColumn(SentenceColumn.Name, out var sentenceCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence", SentenceColumn.Name);
            if (!SentenceColumn.IsCompatibleWith(sentenceCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence", SentenceColumn.Name,
                    SentenceColumn.GetTypeString(), sentenceCol.GetTypeString());

            if (Options.Sentence2ColumnName != default || Options.TaskType == BertTaskType.SentenceRegression)
            {
                if (!inputSchema.TryFindColumn(SentenceColumn2.Name, out var sentenceCol2))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", SentenceColumn2.Name);
                if (!SentenceColumn2.IsCompatibleWith(sentenceCol2))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", SentenceColumn2.Name,
                        SentenceColumn2.GetTypeString(), sentenceCol2.GetTypeString());
            }
        }


        private protected abstract override void SaveModel(ModelSaveContext ctx);


        private protected void SaveBaseModel(ModelSaveContext ctx, VersionInfo versionInfo)
        {
            Host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(versionInfo);

            // *** Binary format ***
            // int: id of label column name
            // int: id of the score column name
            // int: id of output column name
            // int: id of sentence 1 column name
            // int: id of sentence 2 column name
            // int: number of classes

            ctx.SaveNonEmptyString(Options.LabelColumnName);
            ctx.SaveNonEmptyString(Options.ScoreColumnName);
            ctx.SaveNonEmptyString(Options.PredictionColumnName);
            ctx.SaveNonEmptyString(Options.Sentence1ColumnName);
            ctx.SaveStringOrNull(Options.Sentence2ColumnName);
            ctx.Writer.Write(Options.NumberOfClasses);

            ctx.SaveBinaryStream("TSModel", w =>
            {
                Model.save(w);
            });
        }

        private protected abstract IRowMapper GetRowMapper(NasBertTransformer<TLabelCol, TTargetsCol> parent, DataViewSchema schema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => GetRowMapper(this, schema);

        private protected abstract class NasBertMapper : MapperBase
        {
            private protected readonly NasBertTransformer<TLabelCol, TTargetsCol> Parent;
            private protected readonly HashSet<int> InputColIndices;

            private static readonly FuncInstanceMethodInfo1<NasBertMapper, DataViewSchema.DetachedColumn, Delegate> _makeLabelAnnotationGetter
                = FuncInstanceMethodInfo1<NasBertMapper, DataViewSchema.DetachedColumn, Delegate>.Create(target => target.GetLabelAnnotations<int>);


            public NasBertMapper(NasBertTransformer<TLabelCol, TTargetsCol> parent, DataViewSchema inputSchema) :
                base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(NasBertMapper)), inputSchema, parent)
            {
                Parent = parent;
                InputColIndices = new HashSet<int>();
                if (inputSchema.TryGetColumnIndex(parent.Options.Sentence1ColumnName, out var col))
                    InputColIndices.Add(col);

                if (parent.Options.Sentence2ColumnName != default || Parent.Options.TaskType == BertTaskType.SentenceRegression)
                    if (inputSchema.TryGetColumnIndex(parent.Options.Sentence2ColumnName, out col))
                        InputColIndices.Add(col);

                torch.random.manual_seed(1);
                torch.cuda.manual_seed(1);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                if (Parent.Options.TaskType == BertTaskType.TextClassification)
                {
                    var info = new DataViewSchema.DetachedColumn[2];
                    var keyType = Parent.LabelColumn.Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues)?.Type as VectorDataViewType;
                    var getter = Microsoft.ML.Internal.Utilities.Utils.MarshalInvoke(_makeLabelAnnotationGetter, this, keyType.ItemType.RawType, Parent.LabelColumn);


                    var meta = new DataViewSchema.Annotations.Builder();
                    meta.Add(AnnotationUtils.Kinds.ScoreColumnKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreColumnKind.MulticlassClassification.AsMemory(); });
                    meta.Add(AnnotationUtils.Kinds.ScoreColumnSetId, new KeyDataViewType(typeof(uint), Parent.Options.NumberOfClasses), GetScoreColumnSetId(InputSchema));
                    meta.Add(AnnotationUtils.Kinds.ScoreValueKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreValueKind.Score.AsMemory(); });
                    meta.Add(AnnotationUtils.Kinds.TrainingLabelValues, keyType, getter);
                    meta.Add(AnnotationUtils.Kinds.SlotNames, keyType, getter);

                    var labelBuilder = new DataViewSchema.Annotations.Builder();
                    labelBuilder.Add(AnnotationUtils.Kinds.KeyValues, keyType, getter);

                    info[0] = new DataViewSchema.DetachedColumn(Parent.Options.PredictionColumnName, new KeyDataViewType(typeof(uint), Parent.Options.NumberOfClasses), labelBuilder.ToAnnotations());

                    info[1] = new DataViewSchema.DetachedColumn(Parent.Options.ScoreColumnName, new VectorDataViewType(NumberDataViewType.Single, Parent.Options.NumberOfClasses), meta.ToAnnotations());
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

            private ValueGetter<uint> GetScoreColumnSetId(DataViewSchema schema)
            {
                int c;
                var max = schema.GetMaxAnnotationKind(out c, AnnotationUtils.Kinds.ScoreColumnSetId);
                uint id = checked(max + 1);
                return
                    (ref uint dst) => dst = id;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
                => throw new NotImplementedException("This should never be called!");

            private protected abstract Delegate CreateGetter(DataViewRow input, int iinfo, TensorCacher outputCacher);

            public override Delegate[] CreateGetters(DataViewRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Contracts.Assert(input.Schema == base.InputSchema);

                TensorCacher outputCacher = new TensorCacher();
                var ch = Host.Start("Make Getters");
                Parent.Model.eval();

                int n = OutputColumns.Value.Length;
                var result = new Delegate[n];
                for (int i = 0; i < n; i++)
                {
                    if (!activeOutput(i))
                        continue;
                    result[i] = CreateGetter(input, i, outputCacher);
                }
                disposer = () =>
                {
                    outputCacher.Dispose();
                };
                return result;
            }

            private IList<int> PrepInputTokens(ref ReadOnlyMemory<char> sentence1, ref ReadOnlyMemory<char> sentence2, ref ValueGetter<ReadOnlyMemory<char>> getSentence1, ref ValueGetter<ReadOnlyMemory<char>> getSentence2, Tokenizer tokenizer)
            {
                getSentence1(ref sentence1);
                if (getSentence2 == default)
                {
                    return new[] { 0 /* InitToken */ }.Concat(tokenizer.EncodeToConverted(sentence1.ToString())).ToList();
                }
                else
                {
                    getSentence2(ref sentence2);
                    return new[] { 0 /* InitToken */ }.Concat(tokenizer.EncodeToConverted(sentence1.ToString()))
                                              .Concat(new[] { 2 /* SeperatorToken */ }).Concat(tokenizer.EncodeToConverted(sentence2.ToString())).ToList();
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
                    return Parent.Model.forward(inputTensor);
                }
            }

            private protected class TensorCacher : IDisposable
            {
                public long Position;
                public Tensor Result;

                public TensorCacher()
                {
                    Position = -1;
                    Result = default;
                }

                private bool _isDisposed;

                public void Dispose()
                {
                    if (_isDisposed)
                        return;

                    Result?.Dispose();
                    _isDisposed = true;
                }
            }

            private protected void UpdateCacheIfNeeded(long position, TensorCacher outputCache, ref ReadOnlyMemory<char> sentence1, ref ReadOnlyMemory<char> sentence2, ref ValueGetter<ReadOnlyMemory<char>> getSentence1, ref ValueGetter<ReadOnlyMemory<char>> getSentence2, Tokenizer tokenizer)
            {
                if (outputCache.Position != position)
                {
                    outputCache.Result?.Dispose();
                    outputCache.Result = PrepAndRunModel(PrepInputTokens(ref sentence1, ref sentence2, ref getSentence1, ref getSentence2, tokenizer));
                    outputCache.Result.MoveToOuterDisposeScope();
                    outputCache.Position = position;
                }
            }

            private protected override void SaveModel(ModelSaveContext ctx) => Parent.SaveModel(ctx);
        }
    }
}
