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
using Microsoft.ML.TorchSharp.NasBert.Models;
using Microsoft.ML.TorchSharp.NasBert.Preprocessing;
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

[assembly: LoadableClass(typeof(TextClassificationTransformer), null, typeof(SignatureLoadModel),
    TextClassificationTransformer.UserName, TextClassificationTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(TextClassificationTransformer), null, typeof(SignatureLoadRowMapper),
    TextClassificationTransformer.UserName, TextClassificationTransformer.LoaderSignature)]

namespace Microsoft.ML.TorchSharp.NasBert
{
    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a Deep Neural Network(DNN) to classify text.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer, use [TextClassification](xref:Microsoft.ML.TorchSharpCatalog.TextClassification(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,Int32,System.String,System.String,System.String,System.String,Int32,Int32,Int32,Microsoft.ML.TorchSharp.NasBert.BertArchitecture,Microsoft.ML.IDataView)).
    ///
    /// ### Input and Output Columns
    /// The input label column data must be [key](xref:Microsoft.ML.Data.KeyDataViewType) type and the sentence columns must be of type<xref:Microsoft.ML.Data.TextDataViewType>.
    ///
    /// This trainer outputs the following columns:
    ///
    /// | Output Column Name | Column Type | Description|
    /// | -- | -- | -- |
    /// | `PredictedLabel` | [key](xref:Microsoft.ML.Data.KeyDataViewType) type | The predicted label's index. If its value is i, the actual label would be the i-th category in the key-valued input label type. |
    ///
    /// ### Trainer Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Multiclass classification |
    /// | Is normalization required? | No |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | Microsoft.ML.TorchSharp and libtorch-cpu or libtorch-cuda-11.3 or any of the OS specific variants. |
    /// | Exportable to ONNX | No |
    ///    ///
    /// ### Training Algorithm Details
    /// Trains a Deep Neural Network(DNN) by leveraging an existing pre-trained NAS-BERT roBERTa model for the purpose of classifying text.
    /// ]]>
    /// </format>
    /// </remarks>
    public sealed class TextClassificationTrainer : IEstimator<TextClassificationTransformer>
    {
        private readonly IHost _host;
        private readonly Options _options;
        private TextClassificationTransformer _transformer;
        private const string ModelUrl = "models/NasBert2000000.tsm";

        internal sealed class Options : TransformInputBase
        {
            /// <summary>
            /// The label column name.
            /// </summary>
            public string LabelColumnName = DefaultColumnNames.Label;

            /// <summary>
            /// The label column name.
            /// </summary>
            public string OutputColumnName = DefaultColumnNames.PredictedLabel;

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
            /// Number of classes for the data.
            /// </summary>
            public int NumberOfClasses = 2;

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
            /// Stop training when reaching this number of updates.
            /// </summary>
            public int MaxUpdate = 2147483647;

            /// <summary>
            /// Stop training when reaching this number of epochs.
            /// </summary>
            public int MaxEpoch = 100;

            /// <summary>
            /// The validation set used while training to improve model quality.
            /// </summary>
            public IDataView ValidationSet = null;

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

        internal TextClassificationTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string outputColumnName = DefaultColumnNames.PredictedLabel,
            string sentence1ColumnName = "Sentence1",
            string sentence2ColumnName = default,
            int numberOfClasses = 2,
            int batchSize = 32,
            int maxEpochs = 10,
            int maxUpdates = 2147483647,
            IDataView validationSet = null,
            BertArchitecture architecture = BertArchitecture.Roberta) :
            this(env, new Options
            {
                OutputColumnName = outputColumnName,
                Sentence1ColumnName = sentence1ColumnName,
                Sentence2ColumnName = sentence2ColumnName,
                LabelColumnName = labelColumnName,
                BatchSize = batchSize,
                NumberOfClasses = numberOfClasses,
                MaxEpoch = maxEpochs,
                MaxUpdate = maxUpdates,
                ValidationSet = validationSet,
            })
        {
        }

        internal TextClassificationTrainer(IHostEnvironment env, Options options)
        {
            _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(TextClassificationTrainer));
            _options = options;
        }

        public TextClassificationTransformer Fit(IDataView input)
        {
            using (var ch = _host.Start("TrainModel"))
            using (var pch = _host.StartProgressChannel("Training model"))
            {
                var header = new ProgressHeader(new[] { "Accuracy" }, null);
                var trainer = new Trainer(this, ch);
                pch.SetHeader(header, e => e.SetMetric(0, trainer.Accuracy));
                for (int i = 0; i < _options.MaxEpoch && trainer.Updates < _options.MaxUpdate; i++)
                {
                    ch.Trace($"Starting epoch {i}");
                    trainer.Train(input);
                    ch.Trace($"Finished epoch {i}");
                    if (_options.ValidationSet != null)
                        trainer.Validate(pch, ch, i);
                }
                _transformer = new TextClassificationTransformer(_host, _options, trainer.Model, trainer.Tokenizer.Vocabulary);

                _transformer.GetOutputSchema(input.Schema);
            }
            return _transformer;
        }

        private class Trainer
        {
            public BpeTokenizer Tokenizer;
            public TextClassificationModel Model;
            public torch.Device Device;
            public BaseOptimizer Optimizer;
            public optim.lr_scheduler.LRScheduler LearningRateScheduler;
            private readonly TextClassificationTrainer _parent;
            public int Updates;
            public float Accuracy;

            public Trainer(TextClassificationTrainer parent, IChannel ch)
            {
                _parent = parent;
                Updates = 0;
                Accuracy = 0;
                // Get the tokenizer
                Tokenizer = BpeTokenizer.GetInstance(ch);

                // Initialize the vocab
                var vocabulary = Tokenizer.Vocabulary;
                vocabulary.AddMaskSymbol();

                // Initialize the model and load pre-trained weights
                Model = new TextClassificationModel(_parent._options, vocabulary, _parent._options.NumberOfClasses);
                Model.GetEncoder().load(GetModelPath());
                Model.train();

                // Figure out if we are running on GPU or CPU
                Device = ((IHostEnvironmentInternal)_parent._host).GpuDeviceId != null && cuda.is_available() ? CUDA : CPU;

                // Move to GPU if we are running there
                if (Device == CUDA)
                    Model.cuda();

                // Get the paramters that need optimization and set up the optimizer
                var parameters = Model.parameters().Where(p => p.requires_grad);
                Optimizer = BaseOptimizer.GetOptimizer(_parent._options, parameters);
                LearningRateScheduler = torch.optim.lr_scheduler.OneCycleLR(
                   Optimizer.Optimizer,
                   max_lr: _parent._options.LearningRate[0],
                   total_steps: _parent._options.MaxUpdate,
                   pct_start: _parent._options.WarmupRatio,
                   anneal_strategy: torch.optim.lr_scheduler.impl.OneCycleLR.AnnealStrategy.Linear,
                   div_factor: 1.0 / _parent._options.StartLearningRateRatio,
                   final_div_factor: 1.0 / _parent._options.FinalLearningRateRatio);
            }

            private string GetModelPath()
            {
                var destDir = Path.Combine(((IHostEnvironmentInternal)_parent._host).TempFilePath, "mlnet");
                var destFileName = ModelUrl.Split('/').Last();

                Directory.CreateDirectory(destDir);

                string relativeFilePath = Path.Combine(destDir, destFileName);

                int timeout = 10 * 60 * 1000;
                using (var ch = (_parent._host as IHostEnvironment).Start("Ensuring model file is present."))
                {
                    var ensureModel = ResourceManagerUtils.Instance.EnsureResourceAsync(_parent._host, ch, ModelUrl, destFileName, destDir, timeout);
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
                var validationSet = _parent._options.ValidationSet;
                Model.eval();

                DataViewRowCursor cursor = default;
                if (_parent._options.Sentence2ColumnName != default)
                    cursor = validationSet.GetRowCursor(validationSet.Schema[_parent._options.Sentence1ColumnName], validationSet.Schema[_parent._options.Sentence2ColumnName], validationSet.Schema[_parent._options.LabelColumnName]);
                else
                    cursor = validationSet.GetRowCursor(validationSet.Schema[_parent._options.Sentence1ColumnName], validationSet.Schema[_parent._options.LabelColumnName]);

                var sentence1Getter = cursor.GetGetter<ReadOnlyMemory<char>>(validationSet.Schema[_parent._options.Sentence1ColumnName]);
                var sentence2Getter = _parent._options.Sentence2ColumnName != default ? cursor.GetGetter<ReadOnlyMemory<char>>(validationSet.Schema[_parent._options.Sentence2ColumnName]) : default;
                var labelGetter = cursor.GetGetter<uint>(validationSet.Schema[_parent._options.LabelColumnName]);

                // Pre-allocate the memory so it's only done once (though this step needs to be optimized)
                List<Tensor> inputTensors = new List<Tensor>(_parent._options.BatchSize);
                List<long> targets = new List<long>(_parent._options.BatchSize);
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
                ValueGetter<uint> labelGetter,
                ref List<Tensor> inputTensors,
                ref List<long> targets,
                ref int numCorrect,
                ref int numRows)
            {
                // Make sure list is clear before use
                inputTensors.Clear();
                targets.Clear();
                using var disposeScope = torch.NewDisposeScope();
                var cursorValid = true;
                for (int i = 0; i < _parent._options.BatchSize && cursorValid; i++)
                {
                    cursorValid = cursor.MoveNext();
                    if (cursorValid)
                    {
                        inputTensors.Add(PrepareData(sentence1Getter, sentence2Getter));
                        uint target = default;
                        labelGetter(ref target);
                        // keys are 1 based but the model is 0 based
                        targets.Add(target - 1);
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
                    var inputTensor = DataUtils.CollateTokens(inputTensors, Tokenizer.Vocabulary.PadIndex, device: Device);
                    var targetsTensor = tensor(targets, device: Device);
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
                // Set the torch random seed to match ML.NET if one was provided
                //if (((IHostEnvironmentInternal)_host).Seed.HasValue)
                torch.random.manual_seed(1);
                torch.cuda.manual_seed(1);

                // Get the cursor and the correct columns based on the inputs
                DataViewRowCursor cursor = default;
                if (_parent._options.Sentence2ColumnName != default)
                    cursor = input.GetRowCursor(input.Schema[_parent._options.Sentence1ColumnName], input.Schema[_parent._options.Sentence2ColumnName], input.Schema[_parent._options.LabelColumnName]);
                else
                    cursor = input.GetRowCursor(input.Schema[_parent._options.Sentence1ColumnName], input.Schema[_parent._options.LabelColumnName]);

                var sentence1Getter = cursor.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent._options.Sentence1ColumnName]);
                var sentence2Getter = _parent._options.Sentence2ColumnName != default ? cursor.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent._options.Sentence2ColumnName]) : default;
                var labelGetter = cursor.GetGetter<UInt32>(input.Schema[_parent._options.LabelColumnName]);

                // Pre-allocate the memory so it's only done once (though this step needs to be optimized)
                List<Tensor> inputTensors = new List<Tensor>(_parent._options.BatchSize);
                List<long> targets = new List<long>(_parent._options.BatchSize);

                var cursorValid = true;
                while (cursorValid)
                {
                    cursorValid = TrainStep(cursor, sentence1Getter, sentence2Getter, labelGetter, ref inputTensors, ref targets);
                }
            }

            private bool TrainStep(DataViewRowCursor cursor,
            ValueGetter<ReadOnlyMemory<char>> sentence1Getter,
            ValueGetter<ReadOnlyMemory<char>> sentence2Getter,
            ValueGetter<UInt32> labelGetter,
            ref List<Tensor> inputTensors,
            ref List<long> targets)
            {
                // Make sure list is clear before use
                inputTensors.Clear();
                targets.Clear();
                using var disposeScope = torch.NewDisposeScope();
                var cursorValid = true;
                for (int i = 0; i < _parent._options.BatchSize && cursorValid; i++)
                {
                    cursorValid = cursor.MoveNext();
                    if (cursorValid)
                    {
                        inputTensors.Add(PrepareData(sentence1Getter, sentence2Getter));
                        UInt32 target = default;
                        labelGetter(ref target);
                        // keys are 1 based but the model is 0 based
                        targets.Add(target - 1);
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

                Optimizer.zero_grad();

                var inputTensor = DataUtils.CollateTokens(inputTensors, Tokenizer.Vocabulary.PadIndex, device: Device);
                var targetsTensor = tensor(targets, device: Device);
                var logits = Model.forward(inputTensor);
                var lossFunction = torch.nn.functional.cross_entropy_loss(reduction: _parent._options.Reduction);
                var loss = lossFunction(logits, targetsTensor);
                loss.backward();
                OptimizeStep();

                return cursorValid;
            }

            private static torch.Tensor GetPredictions(torch.Tensor logits)
            {
                logits = logits ?? throw new ArgumentNullException(nameof(logits));
                var (_, indexes) = logits.max(-1, false);
                return indexes;
            }

            private static torch.Tensor GetTargets(torch.Tensor labels)
            {
                return labels.view(-1);
            }

            private static int GetNumCorrect(torch.Tensor predictions, torch.Tensor targets)
            {
                predictions = predictions ?? throw new ArgumentNullException(nameof(predictions));
                return (int)predictions.eq(targets).sum().ToInt64();
            }

            private void OptimizeStep()
            {
                // the gradients will accumulate until [TrainingStates.OptimizeStep] steps,
                // and then we update the parameters. In this way, we can enlarge the actual batch size
                Optimizer.Step();
                LearningRateScheduler.step();
            }

            private torch.Tensor PrepareData(ValueGetter<ReadOnlyMemory<char>> sentence1Getter, ValueGetter<ReadOnlyMemory<char>> sentence2Getter)
            {
                ReadOnlyMemory<char> sentence1 = default;
                sentence1Getter(ref sentence1);
                Tensor t;
                if (sentence2Getter == default)
                {
                    t = torch.tensor((new[] { BpeTokenizer.InitToken }).Concat(Tokenizer.EncodeToConverted(sentence1.ToString())).ToList(), device: Device);
                }
                else
                {

                    ReadOnlyMemory<char> sentence2 = default;
                    sentence2Getter(ref sentence2);

                    t = torch.tensor((new[] { BpeTokenizer.InitToken }).Concat(Tokenizer.EncodeToConverted(sentence1.ToString()))
                        .Concat(new[] { BpeTokenizer.SeperatorToken }).Concat(Tokenizer.EncodeToConverted(sentence2.ToString())).ToList(), device: Device);
                }

                if (t.NumberOfElements > 512)
                    t = t.slice(0, 0, 512, 1);

                return t;
            }
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);

            var outColumns = inputSchema.ToDictionary(x => x.Name);
            var metadata = new List<SchemaShape.Column>();
            metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector,
                TextDataViewType.Instance, false));

            outColumns[_options.OutputColumnName] = new SchemaShape.Column(_options.OutputColumnName, SchemaShape.Column.VectorKind.Scalar,
                    NumberDataViewType.UInt32, true, new SchemaShape(metadata.ToArray()));

            return new SchemaShape(outColumns.Values);
        }

        private void CheckInputSchema(SchemaShape inputSchema)
        {
            // Verify that all required input columns are present, and are of the same type.
            if (!inputSchema.TryFindColumn(_options.Sentence1ColumnName, out var sentenceCol))
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "sentence", _options.Sentence1ColumnName);
            if (sentenceCol.ItemType != TextDataViewType.Instance)
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "sentence", _options.Sentence1ColumnName,
                    TextDataViewType.Instance.ToString(), sentenceCol.GetTypeString());

            if (!inputSchema.TryFindColumn(_options.LabelColumnName, out var labelCol))
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "label", _options.LabelColumnName);
            if (labelCol.ItemType != NumberDataViewType.UInt32)
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "label", _options.LabelColumnName,
                    NumberDataViewType.UInt32.ToString(), labelCol.GetTypeString());

            if (_options.Sentence2ColumnName != default)
            {
                if (!inputSchema.TryFindColumn(_options.Sentence2ColumnName, out var sentenceCol2))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", _options.Sentence2ColumnName);
                if (sentenceCol2.ItemType != TextDataViewType.Instance)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", _options.Sentence2ColumnName,
                        TextDataViewType.Instance.ToString(), sentenceCol2.GetTypeString());
            }
        }
    }

    public sealed class TextClassificationTransformer : RowToRowTransformerBase
    {
        internal const string LoadName = "NASBERTTrainer";
        internal const string UserName = "NASBERT Trainer";
        internal const string ShortName = "NASBERT";
        internal const string Summary = "NLP with NAS-BERT";

        private readonly Device _device;
        private readonly TextClassificationModel _model;
        private readonly Vocabulary _vocabulary;
        private readonly TextClassificationTrainer.Options _options;

        private readonly string _predictedLabelColumnName;

        public readonly SchemaShape.Column SentenceColumn;
        public readonly SchemaShape.Column SentenceColumn2;
        public readonly SchemaShape.Column LabelColumn;

        internal const string LoaderSignature = "NASBERT";

        internal TextClassificationTransformer(IHostEnvironment env, TextClassificationTrainer.Options options, TextClassificationModel model, Vocabulary vocabulary)
           : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TextClassificationTransformer)))
        {
            _device = ((IHostEnvironmentInternal)env).GpuDeviceId != null && cuda.is_available() ? CUDA : CPU;
            Contracts.Assert(_device == CPU || (((IHostEnvironmentInternal)env).FallbackToCpu != false || _device != CUDA), "Fallback to CPU is false but no GPU detected");

            _options = options;
            LabelColumn = new SchemaShape.Column(_options.LabelColumnName, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.UInt32, true);
            SentenceColumn = new SchemaShape.Column(_options.Sentence1ColumnName, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance, false);
            SentenceColumn2 = _options.Sentence2ColumnName == default ? default : new SchemaShape.Column(_options.Sentence2ColumnName, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance, false);
            _predictedLabelColumnName = _options.OutputColumnName;

            _vocabulary = vocabulary;
            _model = model;

            if (_device == CUDA)
                _model.cuda();

            //if (((IHostEnvironmentInternal)env).Seed.HasValue)
            torch.random.manual_seed(1);
            torch.cuda.manual_seed(1);

        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        // Factory method for SignatureLoadModel.
        private static TextClassificationTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: id of label column name
            // int: id of output column name
            // int: id of sentence 1 column name
            // int: id of sentence 2 column name
            // int: number of classes
            var options = new TextClassificationTrainer.Options()
            {
                LabelColumnName = ctx.LoadString(),
                OutputColumnName = ctx.LoadString(),
                Sentence1ColumnName = ctx.LoadString(),
                Sentence2ColumnName = ctx.LoadStringOrNull(),
                NumberOfClasses = ctx.Reader.ReadInt32(),
            };

            var ch = env.Start("Load Model");
            var tokenizer = BpeTokenizer.GetInstance(ch);
            var vocabulary = tokenizer.Vocabulary;
            vocabulary.AddMaskSymbol();

            var model = new TextClassificationModel(options, vocabulary, options.NumberOfClasses);
            if (!ctx.TryLoadBinaryStream("TSModel", r => model.load(r)))
                throw env.ExceptDecode();

            return new TextClassificationTransformer(env, options, model, vocabulary);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);
            inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol);
            var predLabelMetadata = new SchemaShape(labelCol.Annotations.Where(x => x.Name == AnnotationUtils.Kinds.KeyValues)
                .Concat(AnnotationUtils.GetTrainerOutputAnnotation()));

            var outColumns = inputSchema.ToDictionary(x => x.Name);
            outColumns[_predictedLabelColumnName] = new SchemaShape.Column(_predictedLabelColumnName, SchemaShape.Column.VectorKind.Scalar,
                    NumberDataViewType.UInt32, true, predLabelMetadata);

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

            if (_options.Sentence2ColumnName != default)
            {
                if (!inputSchema.TryFindColumn(SentenceColumn2.Name, out var sentenceCol2))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", SentenceColumn2.Name);
                if (!SentenceColumn2.IsCompatibleWith(sentenceCol2))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", SentenceColumn2.Name,
                        SentenceColumn2.GetTypeString(), sentenceCol2.GetTypeString());
            }

            if (!inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", LabelColumn.Name);
            if (!LabelColumn.IsCompatibleWith(labelCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", LabelColumn.Name,
                    LabelColumn.GetTypeString(), labelCol.GetTypeString());
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NAS-BERT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TextClassificationTransformer).Assembly.FullName);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: id of label column name
            // int: id of output column name
            // int: id of sentence 1 column name
            // int: id of sentence 2 column name
            // int: number of classes

            ctx.SaveNonEmptyString(_options.LabelColumnName);
            ctx.SaveNonEmptyString(_options.OutputColumnName);
            ctx.SaveNonEmptyString(_options.Sentence1ColumnName);
            ctx.SaveStringOrNull(_options.Sentence2ColumnName);
            ctx.Writer.Write(_options.NumberOfClasses);

            ctx.SaveBinaryStream("TSModel", w =>
            {
                _model.save(w);
            });
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : MapperBase
        {
            private readonly TextClassificationTransformer _parent;
            private readonly HashSet<int> _inputColIndices;
            private readonly DataViewSchema.Column _labelCol;

            public Mapper(TextClassificationTransformer parent, DataViewSchema inputSchema) :
                base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
                _inputColIndices = new HashSet<int>();
                int col = 0;
                if (inputSchema.TryGetColumnIndex(parent._options.Sentence1ColumnName, out col))
                    _inputColIndices.Add(col);

                if (parent._options.Sentence2ColumnName != default)
                    if (inputSchema.TryGetColumnIndex(parent._options.Sentence2ColumnName, out col))
                        _inputColIndices.Add(col);

                _labelCol = inputSchema[_parent._options.LabelColumnName];
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var info = new DataViewSchema.DetachedColumn[1];
                //var meta = new DataViewSchema.Annotations.Builder();
                //foreach (var kvp in _labelCol.Annotations)
                //    meta.Add(kvp.Value.Kind, kvp.Value.AnnotationType, kvp.Value.GetGetterDelegate());

                info[0] = new DataViewSchema.DetachedColumn(_parent._options.OutputColumnName, new KeyDataViewType(typeof(uint), _parent._options.NumberOfClasses), _labelCol.Annotations);
                return info;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                var ch = Host.Start("Make Getter");
                disposer = null;

                Host.Assert(_inputColIndices.All(i => input.IsColumnActive(input.Schema[i])));

                ValueGetter<ReadOnlyMemory<char>> getSentence1 = default;
                ValueGetter<ReadOnlyMemory<char>> getSentence2 = default;

                BpeTokenizer tokenizer = BpeTokenizer.GetInstance(ch);

                getSentence1 = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent.SentenceColumn.Name]);
                if (_parent.SentenceColumn2.IsValid)
                    getSentence2 = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent.SentenceColumn2.Name]);

                ReadOnlyMemory<char> sentence1 = default;
                ReadOnlyMemory<char> sentence2 = default;

                ValueGetter<UInt32> classification = (ref UInt32 dst) =>
                {
                    using var disposeScope = torch.NewDisposeScope();
                    _parent._model.eval();

                    getSentence1(ref sentence1);
                    Tensor inputTensor = default;
                    if (getSentence2 == default)
                    {
                        using (torch.no_grad())
                        {
                            inputTensor = torch.tensor(tokenizer.EncodeToConverted(sentence1.ToString()), device: _parent._device);
                            if (inputTensor.NumberOfElements > 512)
                                inputTensor = inputTensor.slice(0, 0, 512, 1);
                            inputTensor = inputTensor.reshape(1, inputTensor.NumberOfElements);
                            var result = _parent._model.forward(inputTensor);
                            dst = (UInt32)result.argmax(-1).cpu().item<long>() + 1;
                        }
                        return;
                    }
                    getSentence2(ref sentence2);

                    inputTensor = torch.tensor((new[] { BpeTokenizer.InitToken }).Concat(tokenizer.EncodeToConverted(sentence1.ToString()))
                        .Concat(new[] { BpeTokenizer.SeperatorToken }).Concat(tokenizer.EncodeToConverted(sentence2.ToString())).ToList(), device: _parent._device);
                    if (inputTensor.NumberOfElements > 512)
                        inputTensor = inputTensor.slice(0, 0, 512, 1);
                    inputTensor = inputTensor.reshape(1, inputTensor.NumberOfElements);

                    using (torch.no_grad())
                    {
                        var result2 = _parent._model.forward(inputTensor);
                        dst = (UInt32)result2.argmax(-1).cpu().item<long>() + 1;
                    }
                };

                return classification;
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => activeOutput(0) && _inputColIndices.Any(i => i == col);
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);
        }
    }
}
