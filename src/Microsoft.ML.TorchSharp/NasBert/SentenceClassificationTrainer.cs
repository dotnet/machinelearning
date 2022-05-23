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

[assembly: LoadableClass(typeof(SentenceClassificationTransformer), null, typeof(SignatureLoadModel),
    SentenceClassificationTransformer.UserName, SentenceClassificationTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(SentenceClassificationTransformer), null, typeof(SignatureLoadRowMapper),
    SentenceClassificationTransformer.UserName, SentenceClassificationTransformer.LoaderSignature)]

namespace Microsoft.ML.TorchSharp.NasBert
{
    public sealed class SentenceClassificationTrainer : IEstimator<SentenceClassificationTransformer>
    {
        private readonly IHost _host;
        private readonly Options _options;
        private SentenceClassificationTransformer _transformer;
        private const string ModelUrl = "models/NasBert2000000.tsm";

        /// <summary>
        /// Sentence classification model.
        /// </summary>
        public enum Architecture
        {
            Roberta
        };

        internal sealed class Options : TransformInputBase
        {
            /// <summary>
            /// The label column name.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "The label column name", ShortName = "label", SortOrder = 1)]
            public string LabelColumnName = DefaultColumnNames.Label;

            /// <summary>
            /// The label column name.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "The name of the output", ShortName = "output", SortOrder = 1)]
            public string OutputColumnName = DefaultColumnNames.PredictedLabel;

            /// <summary>
            /// The first sentence column.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "The first sentence column name", ShortName = "sent1", SortOrder = 1)]
            public string Sentence1ColumnName = "Sentence";

            /// <summary>
            /// The second sentence column. O
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The second sentence column name", ShortName = "sent2", SortOrder = 2)]
            public string Sentence2ColumnName = default;

            /// <summary>
            /// Number of samples to use for mini-batch training.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of samples to use for mini-batch training.", SortOrder = 9)]
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
            internal TaskType TaskType = TaskType.SentenceClassification;

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

        internal SentenceClassificationTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string outputColumnName = DefaultColumnNames.PredictedLabel,
            string sentence1ColumnName = "Sentence1",
            string sentence2ColumnName = default,
            int numberOfClasses = 2,
            int batchSize = 32,
            int maxEpochs = 10,
            int maxUpdates = 2147483647,
            SentenceClassificationTrainer.Architecture architecture = Architecture.Roberta) :
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
            })
        {
        }

        internal SentenceClassificationTrainer(IHostEnvironment env, Options options)
        {
            _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(SentenceClassificationTrainer));
            _options = options;
        }

        public SentenceClassificationTransformer Fit(IDataView input)
        {
            using (var ch2 = _host.Start("TrainModel"))
            using (var pch = _host.StartProgressChannel("Training model"))
            {
                var trainer = new Trainer(this);
                for (int i = 0; i < _options.MaxEpoch && trainer.Updates < _options.MaxUpdate; i++)
                {
                    ch2.Info($"Starting epoch {i}");
                    trainer.Train(input, i);
                }
                _transformer = new SentenceClassificationTransformer(_host, _options, trainer.Model, trainer.Tokenizer.Vocabulary);

                _transformer.GetOutputSchema(input.Schema);
            }
            return _transformer;
        }

        private class Trainer
        {
            public BpeTokenizer Tokenizer;
            public SentenceClassificationModel Model;
            public torch.Device Device;
            public BaseOptimizer Optimizer;
            public optim.lr_scheduler.LRScheduler LearningRateScheduler;
            private readonly SentenceClassificationTrainer _parent;
            public int Updates;

            public Trainer(SentenceClassificationTrainer parent)
            {
                _parent = parent;
                Updates = 0;

                // Get the tokenizer
                Tokenizer = BpeTokenizer.GetInstance();

                // Initialize the vocab
                var vocabulary = Tokenizer.Vocabulary;
                vocabulary.AddMaskSymbol();

                // Initialize the model and load pre-trained weights
                Model = new SentenceClassificationModel(_parent._options, vocabulary, _parent._options.NumberOfClasses);
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

            public void Train(IDataView input, int epoch)
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
                var labelGetter = cursor.GetGetter<long>(input.Schema[_parent._options.LabelColumnName]);

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
            ValueGetter<long> labelGetter,
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
                        long target = default;
                        labelGetter(ref target);
                        targets.Add(target);
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

                var predictions = GetPredictions(logits);
                var predCpu = predictions.cpu();
                var targetss = GetTargets(targetsTensor);
                var targetssCpu = targetss.cpu();
                var numCorrect = GetNumCorrect(predictions, targetss);
                var acc = numCorrect / (double)inputTensors.Count;
                OptimizeStep();
                return cursorValid;
            }

            private torch.Tensor GetPredictions(torch.Tensor logits)
            {
                logits = logits ?? throw new ArgumentNullException(nameof(logits));
                var (_, indexes) = logits.max(-1, false);
                return indexes;
            }

            private torch.Tensor GetTargets(torch.Tensor labels)
            {
                return labels.view(-1);
            }
            private int GetNumCorrect(torch.Tensor predictions, torch.Tensor targets)
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
                if (sentence2Getter == default)
                    return torch.tensor((new[] { BpeTokenizer.InitToken }).Concat(Tokenizer.EncodeToConverted(sentence1.ToString())).ToList(), device: Device);

                ReadOnlyMemory<char> sentence2 = default;
                sentence2Getter(ref sentence2);

                return torch.tensor((new[] { BpeTokenizer.InitToken }).Concat(Tokenizer.EncodeToConverted(sentence1.ToString()))
                    .Concat(new[] { BpeTokenizer.SeperatorToken }).Concat(Tokenizer.EncodeToConverted(sentence2.ToString())).ToList(), device: Device);
            }
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);

            var outColumns = inputSchema.ToDictionary(x => x.Name);
            outColumns[_options.OutputColumnName] = new SchemaShape.Column(_options.OutputColumnName, SchemaShape.Column.VectorKind.Scalar,
                    NumberDataViewType.Double, false, null);

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
            if (labelCol.ItemType != NumberDataViewType.Int64)
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "label", _options.LabelColumnName,
                    NumberDataViewType.Int64.ToString(), labelCol.GetTypeString());

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

    public sealed class SentenceClassificationTransformer : RowToRowTransformerBase
    {
        internal const string LoadName = "NASBERTTrainer";
        internal const string UserName = "NASBERT Trainer";
        internal const string ShortName = "NASBERT";
        internal const string Summary = "NLP with NAS-BERT";

        private readonly Device _device;
        private readonly SentenceClassificationModel _model;
        private readonly Vocabulary _vocabulary;
        private readonly SentenceClassificationTrainer.Options _options;

        private readonly string _predictedLabelColumnName;

        public readonly SchemaShape.Column SentenceColumn;
        public readonly SchemaShape.Column SentenceColumn2;
        public readonly SchemaShape.Column LabelColumn;

        internal const string LoaderSignature = "NASBERT";

        internal SentenceClassificationTransformer(IHostEnvironment env, SentenceClassificationTrainer.Options options, SentenceClassificationModel model, Vocabulary vocabulary)
           : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(SentenceClassificationTransformer)))
        {
            Contracts.Assert(((IHostEnvironmentInternal)env).FallbackToCpu != false || _device != CPU, "Fallback to CPU is false but no GPU detected");

            _options = options;
            LabelColumn = new SchemaShape.Column(_options.LabelColumnName, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Int64, false);
            SentenceColumn = new SchemaShape.Column(_options.Sentence1ColumnName, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance, false);
            SentenceColumn2 = _options.Sentence2ColumnName == default ? default : new SchemaShape.Column(_options.Sentence2ColumnName, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance, false);
            _predictedLabelColumnName = _options.OutputColumnName;

            _vocabulary = vocabulary;
            _model = model;

            _device = ((IHostEnvironmentInternal)env).GpuDeviceId != null && cuda.is_available() ? CUDA : CPU;

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
        private static SentenceClassificationTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: id of label column name
            // int: id of output column name
            // int: id of sentence 1 column name
            // int: id of sentence 2 column name
            // int: batch size
            // int: number of classes
            // bool: freeze encoder
            // bool: freeze transfer
            // bool: layer norm training
            // bool: encoder normalize before
            // double: dropout
            // double: attention dropout
            // double: activation dropout
            // bool: dynamic dropout
            // double: pooler dropout
            // stream: state dictionary
            var options = new SentenceClassificationTrainer.Options()
            {
                LabelColumnName = ctx.LoadString(),
                OutputColumnName = ctx.LoadString(),
                Sentence1ColumnName = ctx.LoadString(),
                Sentence2ColumnName = ctx.LoadStringOrNull(),
                BatchSize = ctx.Reader.ReadInt32(),
                NumberOfClasses = ctx.Reader.ReadInt32(),
                FreezeEncoder = ctx.Reader.ReadBoolean(),
                FreezeTransfer = ctx.Reader.ReadBoolean(),
                LayerNormTraining = ctx.Reader.ReadBoolean(),
                EncoderNormalizeBefore = ctx.Reader.ReadBoolean(),
                Dropout = ctx.Reader.ReadDouble(),
                AttentionDropout = ctx.Reader.ReadDouble(),
                ActivationDropout = ctx.Reader.ReadDouble(),
                DynamicDropout = ctx.Reader.ReadBoolean(),
                PoolerDropout = ctx.Reader.ReadDouble()
            };

            var tokenizer = BpeTokenizer.GetInstance();
            var vocabulary = tokenizer.Vocabulary;
            vocabulary.AddMaskSymbol();

            var model = new SentenceClassificationModel(options, vocabulary, options.NumberOfClasses);
            if (!ctx.TryLoadBinaryStream("TSModel", r => model.load(r)))
                throw env.ExceptDecode();

            return new SentenceClassificationTransformer(env, options, model, vocabulary);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);

            var outColumns = inputSchema.ToDictionary(x => x.Name);
            outColumns[_predictedLabelColumnName] = new SchemaShape.Column(_predictedLabelColumnName, SchemaShape.Column.VectorKind.Scalar,
                    NumberDataViewType.Double, false, null);

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
                loaderAssemblyName: typeof(SentenceClassificationTransformer).Assembly.FullName);
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
            // int: batch size
            // int: number of classes
            // bool: freeze encoder
            // bool: freeze transfer
            // bool: layer norm training
            // bool: encoder normalize before
            // double: dropout
            // double: attention dropout
            // double: activation dropout
            // bool: dynamic dropout
            // double: pooler dropout
            // stream: state dictionary

            ctx.SaveNonEmptyString(_options.LabelColumnName);
            ctx.SaveNonEmptyString(_options.OutputColumnName);
            ctx.SaveNonEmptyString(_options.Sentence1ColumnName);
            ctx.SaveStringOrNull(_options.Sentence2ColumnName);
            ctx.Writer.Write(_options.BatchSize);
            ctx.Writer.Write(_options.NumberOfClasses);
            ctx.Writer.Write(_options.FreezeEncoder);
            ctx.Writer.Write(_options.FreezeTransfer);
            ctx.Writer.Write(_options.LayerNormTraining);
            ctx.Writer.Write(_options.EncoderNormalizeBefore);
            ctx.Writer.Write(_options.Dropout);
            ctx.Writer.Write(_options.AttentionDropout);
            ctx.Writer.Write(_options.ActivationDropout);
            ctx.Writer.Write(_options.DynamicDropout);
            ctx.Writer.Write(_options.PoolerDropout);

            ctx.SaveBinaryStream("TSModel", w =>
            {
                _model.save(w);
            });
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : MapperBase
        {
            private readonly SentenceClassificationTransformer _parent;
            private readonly HashSet<int> _inputColIndices;

            public Mapper(SentenceClassificationTransformer parent, DataViewSchema inputSchema) :
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
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var info = new DataViewSchema.DetachedColumn[1];
                info[0] = new DataViewSchema.DetachedColumn(_parent._options.OutputColumnName, NumberDataViewType.Double, null);
                return info;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                disposer = null;

                Host.Assert(_inputColIndices.All(i => input.IsColumnActive(input.Schema[i])));

                ValueGetter<ReadOnlyMemory<char>> getSentence1 = default;
                ValueGetter<ReadOnlyMemory<char>> getSentence2 = default;

                BpeTokenizer tokenizer = BpeTokenizer.GetInstance();

                getSentence1 = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent.SentenceColumn.Name]);
                if (_parent.SentenceColumn2.IsValid)
                    getSentence2 = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent.SentenceColumn2.Name]);

                ReadOnlyMemory<char> sentence1 = default;
                ReadOnlyMemory<char> sentence2 = default;

                ValueGetter<double> classification = (ref double dst) =>
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
                            inputTensor = inputTensor.reshape(1, inputTensor.NumberOfElements);
                            var result = _parent._model.forward(inputTensor);
                            dst = (double)result.argmax(-1).cpu().item<long>();
                        }
                        return;
                    }
                    getSentence2(ref sentence2);

                    inputTensor = torch.tensor((new[] { BpeTokenizer.InitToken }).Concat(tokenizer.EncodeToConverted(sentence1.ToString()))
                        .Concat(new[] { BpeTokenizer.SeperatorToken }).Concat(tokenizer.EncodeToConverted(sentence2.ToString())).ToList(), device: _parent._device);
                    inputTensor = inputTensor.reshape(1, inputTensor.NumberOfElements);

                    using (torch.no_grad())
                    {
                        var result2 = _parent._model.forward(inputTensor);
                        dst = (double)result2.argmax(-1).cpu().item<long>();
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
