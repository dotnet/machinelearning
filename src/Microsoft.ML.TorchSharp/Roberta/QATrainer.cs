// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using TorchSharp;
using System.Runtime.CompilerServices;

using static TorchSharp.torch;
using static TorchSharp.TensorExtensionMethods;
using Microsoft.ML.TorchSharp.Utils;
using Microsoft.ML;
using System.IO;
using static Microsoft.ML.TorchSharp.NasBert.NasBertTrainer;
using static Microsoft.ML.Data.AnnotationUtils;
using Microsoft.ML.TorchSharp.NasBert.Optimizers;
using Microsoft.ML.TorchSharp.Roberta;
using Microsoft.ML.TorchSharp.NasBert;
using Microsoft.ML.Tokenizers;
using Microsoft.ML.TorchSharp.Extensions;

[assembly: LoadableClass(typeof(QATransformer), null, typeof(SignatureLoadModel),
    QATransformer.UserName, QATransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(QATransformer), null, typeof(SignatureLoadRowMapper),
    QATransformer.UserName, QATransformer.LoaderSignature)]

namespace Microsoft.ML.TorchSharp.Roberta
{
    public class QATrainer : IEstimator<QATransformer>
    {
        public sealed class Options : NasBertOptions
        {
            /// <summary>
            /// Context Column Name
            /// </summary>
            public string ContextColumnName = DefaultColumnNames.Context;

            /// <summary>
            /// Question Column Name
            /// </summary>
            public string QuestionColumnName = DefaultColumnNames.Question;

            /// <summary>
            /// Answer Column Name for the training data
            /// </summary>
            public string TrainingAnswerColumnName = DefaultColumnNames.TrainingAnswer;

            /// <summary>
            /// Answer Column Name for the predicted answers
            /// </summary>
            public string PredictedAnswerColumnName = DefaultColumnNames.Answer;

            /// <summary>
            /// Answer Index Start Column Name
            /// </summary>
            public string AnswerIndexStartColumnName = DefaultColumnNames.AnswerIndex;

            /// <summary>
            /// Number of top predicted answers in question answering task.
            /// </summary>
            public int TopKAnswers = DefaultColumnNames.TopKAnswers;

            /// <summary>
            /// How often to log the loss.
            /// </summary>
            public int LogEveryNStep = 50;

            public Options()
            {
                EncoderOutputDim = 768;
                EmbeddingDim = 768;
                PoolerDropout = 0;
                ModelType = BertModelType.Roberta;
                TaskType = BertTaskType.QuestionAnswering;
                LearningRate = new List<double>() { .000001 };
                WeightDecay = 0.01;
            }
        }

        internal static class DefaultColumnNames
        {
            public const string Context = "Context";
            public const string Question = "Question";
            public const string Answer = "Answer";
            public const string TrainingAnswer = "TrainingAnswer";
            public const string AnswerIndex = "AnswerStart";
            public const int TopKAnswers = 3;
            public const string Score = "Score";
        }

        private protected readonly IHost Host;
        internal readonly Options Option;
        private const string ModelUrl = "models/pretrained_Roberta_encoder.tsm";

        internal QATrainer(IHostEnvironment env, Options options)
        {
            Host = Contracts.CheckRef(env, nameof(env)).Register(nameof(QATrainer));
            Contracts.Assert(options.MaxEpoch > 0);
            Contracts.AssertValue(options.ContextColumnName);
            Contracts.AssertValue(options.QuestionColumnName);
            Contracts.AssertValue(options.TrainingAnswerColumnName);
            Contracts.AssertValue(options.AnswerIndexStartColumnName);
            Contracts.AssertValue(options.ScoreColumnName);
            Contracts.AssertValue(options.PredictedAnswerColumnName);

            Option = options;
        }

        internal QATrainer(IHostEnvironment env,
            string contextColumnName = DefaultColumnNames.Context,
            string questionColumnName = DefaultColumnNames.Question,
            string trainingAnswerColumnName = DefaultColumnNames.TrainingAnswer,
            string answerIndexColumnName = DefaultColumnNames.AnswerIndex,
            string predictedAnswerColumnName = DefaultColumnNames.Answer,
            string scoreColumnName = DefaultColumnNames.Score,
            int topk = 3,
            int batchSize = 4,
            int maxEpochs = 10,
            IDataView validationSet = null,
            BertArchitecture architecture = BertArchitecture.Roberta) :
            this(env, new Options
            {
                ContextColumnName = contextColumnName,
                QuestionColumnName = questionColumnName,
                TrainingAnswerColumnName = trainingAnswerColumnName,
                AnswerIndexStartColumnName = answerIndexColumnName,
                PredictedAnswerColumnName = predictedAnswerColumnName,
                ScoreColumnName = scoreColumnName,
                TopKAnswers = topk,
                BatchSize = batchSize,
                MaxEpoch = maxEpochs,
                ValidationSet = validationSet
            })
        {
        }

        public QATransformer Fit(IDataView input)
        {
            CheckInputSchema(SchemaShape.Create(input.Schema));

            QATransformer transformer = default;

            using (var ch = Host.Start("TrainModel"))
            using (var pch = Host.StartProgressChannel("Training model"))
            {
                var header = new ProgressHeader(new[] { "Loss" }, new[] { "Total Rows" });

                var trainer = new Trainer(this, ch, input);
                pch.SetHeader(header,
                    e =>
                    {
                        e.SetProgress(0, trainer.Updates, trainer.RowCount);
                        e.SetMetric(0, trainer.LossValue);
                    });

                for (int i = 0; i < Option.MaxEpoch; i++)
                {
                    ch.Trace($"Starting epoch {i}");
                    Host.CheckAlive();
                    trainer.Train(Host, input, pch);
                    ch.Trace($"Finished epoch {i}");
                }

                trainer.Optimizer.Optimizer.Dispose();

                transformer = new QATransformer(Host, Option, trainer.Model);
                transformer.GetOutputSchema(input.Schema);
            }
            return transformer;
        }

        internal class Trainer
        {
            public RobertaModelForQA Model;
            public torch.Device Device;
            public BaseOptimizer Optimizer;
            public optim.lr_scheduler.LRScheduler LearningRateScheduler;
            protected readonly QATrainer Parent;
            public int Updates;
            public float LossValue;
            public readonly int RowCount;
            private readonly IChannel _channel;
            public Tokenizer Tokenizer;

            public Trainer(QATrainer parent, IChannel ch, IDataView input)
            {
                Parent = parent;
                Updates = 0;
                LossValue = 0;
                _channel = ch;

                // Get row count
                RowCount = GetRowCount(input);
                Device = TorchUtils.InitializeDevice(Parent.Host);

                // Initialize the model and load pre-trained weights
                Model = new RobertaModelForQA(Parent.Option);

                Model.GetEncoder().load(GetModelPath());

                // Figure out if we are running on GPU or CPU
                Device = TorchUtils.InitializeDevice(Parent.Host);

                // Move to GPU if we are running there
                if (Device.type == DeviceType.CUDA)
                    Model.cuda();

                Tokenizer = TokenizerExtensions.GetInstance(ch);

                // Get the parameters that need optimization and set up the optimizer
                var parameters = Model.parameters().Where(p => p.requires_grad);
                Optimizer = BaseOptimizer.GetOptimizer(Parent.Option, parameters);
                LearningRateScheduler = torch.optim.lr_scheduler.OneCycleLR(
                   Optimizer.Optimizer,
                   max_lr: Parent.Option.LearningRate[0],
                   total_steps: ((RowCount / Parent.Option.BatchSize) + 1) * Parent.Option.MaxEpoch,
                   pct_start: Parent.Option.WarmupRatio,
                   anneal_strategy: torch.optim.lr_scheduler.impl.OneCycleLR.AnnealStrategy.Linear,
                   div_factor: 1.0 / Parent.Option.StartLearningRateRatio,
                   final_div_factor: Parent.Option.StartLearningRateRatio / Parent.Option.FinalLearningRateRatio);
            }

            private protected int GetRowCount(IDataView input)
            {
                var labelCol = input.GetColumn<int>(Parent.Option.AnswerIndexStartColumnName);
                var rowCount = 0;

                foreach (var label in labelCol)
                {
                    rowCount++;
                }

                return rowCount;
            }

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

            public void Train(IHost host, IDataView input, IProgressChannel pch)
            {
                // Get the cursor and the correct columns based on the inputs
                DataViewRowCursor cursor = input.GetRowCursor(input.Schema[Parent.Option.ContextColumnName], input.Schema[Parent.Option.QuestionColumnName], input.Schema[Parent.Option.TrainingAnswerColumnName], input.Schema[Parent.Option.AnswerIndexStartColumnName]);

                var contextGetter = cursor.GetGetter<ReadOnlyMemory<char>>(input.Schema[Parent.Option.ContextColumnName]);
                var questionGetter = cursor.GetGetter<ReadOnlyMemory<char>>(input.Schema[Parent.Option.QuestionColumnName]);
                var answerGetter = cursor.GetGetter<ReadOnlyMemory<char>>(input.Schema[Parent.Option.TrainingAnswerColumnName]);
                var answerIndexGetter = cursor.GetGetter<int>(input.Schema[Parent.Option.AnswerIndexStartColumnName]);

                var cursorValid = true;
                Updates = 0;

                List<Tensor> inputTensors = new List<Tensor>(Parent.Option.BatchSize);
                List<Tensor> targetTensors = new List<Tensor>(Parent.Option.BatchSize);

                while (cursorValid)
                {

                    if (host is IHostEnvironmentInternal hostInternal)
                    {
                        torch.random.manual_seed(hostInternal.Seed + Updates ?? 1);
                        torch.cuda.manual_seed(hostInternal.Seed + Updates ?? 1);
                    }
                    else
                    {
                        torch.random.manual_seed(1);
                        torch.cuda.manual_seed(1);
                    }
                    cursorValid = TrainStep(host, cursor, contextGetter, questionGetter, answerGetter, answerIndexGetter, ref inputTensors, ref targetTensors, pch);
                }
            }

            private bool TrainStep(IHost host,
                DataViewRowCursor cursor,
                ValueGetter<ReadOnlyMemory<char>> contextGetter,
                ValueGetter<ReadOnlyMemory<char>> questionGetter,
                ValueGetter<ReadOnlyMemory<char>> answerGetter,
                ValueGetter<int> answerIndexGetter,
                ref List<Tensor> inputTensors,
                ref List<Tensor> targetTensors,
                IProgressChannel pch)
            {
                // Make sure list is clear before use
                inputTensors.Clear();
                targetTensors.Clear();

                using var disposeScope = torch.NewDisposeScope();
                var cursorValid = true;
                Tensor srcTensor = default;
                Tensor targetTensor = default;

                host.CheckAlive();

                for (int i = 0; i < Parent.Option.BatchSize && cursorValid; i++)
                {
                    host.CheckAlive();
                    cursorValid = cursor.MoveNext();
                    if (cursorValid)
                    {
                        (srcTensor, targetTensor, bool valid) = PrepareData(contextGetter, questionGetter, answerGetter, answerIndexGetter);
                        if (valid)
                        {
                            inputTensors.Add(srcTensor);
                            targetTensors.Add(targetTensor);
                        }
                        else
                            i--;
                    }
                    else
                    {
                        inputTensors.TrimExcess();
                        targetTensors.TrimExcess();
                        if (inputTensors.Count() == 0)
                            return cursorValid;
                    }
                }

                Updates++;
                host.CheckAlive();
                Model.train();
                Optimizer.zero_grad();

                srcTensor = PrepareBatchTensor(ref inputTensors, device: Device, Tokenizer.RobertaModel().PadIndex);
                targetTensor = PrepareBatchTensor(ref targetTensors, device: Device, 0);
                var logits = Model.forward(srcTensor);  //[batchsize, maxseqlen, 2]
                var splitLogits = logits.split(1, dim: -1);
                var startLogits = splitLogits[0].squeeze(-1).contiguous();  //[batchsize, maxseqlen]
                var endLogits = splitLogits[1].squeeze(-1).contiguous();  //[batchsize, maxseqlen]

                var targetsLong = targetTensor.@long();
                var splitTargets = targetsLong.split(1, dim: -1);
                var startTargets = splitTargets[0].squeeze(-1).contiguous();  //[batchsize]
                var endTargets = splitTargets[1].squeeze(-1).contiguous();  //[batchsize]

                torch.Tensor lossStart = torch.nn.CrossEntropyLoss(reduction: Parent.Option.Reduction).forward(startLogits, startTargets);
                torch.Tensor lossEnd = torch.nn.CrossEntropyLoss(reduction: Parent.Option.Reduction).forward(endLogits, endTargets);

                var loss = ((lossStart + lossEnd) / 2);

                loss.backward();

                Optimizer.Step();
                LearningRateScheduler.step();
                host.CheckAlive();

                if (Updates % Parent.Option.LogEveryNStep == 0)
                {
                    pch.Checkpoint(loss.ToDouble(), Updates);
                    _channel.Info($"Row: {Updates}, Loss: {loss.ToDouble()}");
                }

                return cursorValid;
            }

            private torch.Tensor PrepareBatchTensor(ref List<Tensor> inputTensors, Device device, int padIndex)
            {
                return DataUtils.CollateTokens(inputTensors, padIndex, device: Device);
            }

            private (Tensor image, Tensor Label, bool hasMapping) PrepareData(ValueGetter<ReadOnlyMemory<char>> contextGetter, ValueGetter<ReadOnlyMemory<char>> questionGetter, ValueGetter<ReadOnlyMemory<char>> answerGetter, ValueGetter<int> answerIndexGetter)
            {
                using (var _ = torch.NewDisposeScope())
                {
                    ReadOnlyMemory<char> context = default;
                    ReadOnlyMemory<char> question = default;
                    ReadOnlyMemory<char> answer = default;
                    int answerIndex = default;

                    contextGetter(ref context);
                    questionGetter(ref question);
                    answerGetter(ref answer);
                    answerIndexGetter(ref answerIndex);

                    var contextString = context.ToString();
                    var contextTokens = Tokenizer.EncodeToTokens(contextString, out string normalized);
                    var contextToken = contextTokens.Select(t => t.Value).ToArray();
                    var contextTokenId = Tokenizer.RobertaModel().ConvertIdsToOccurrenceRanks(contextTokens.Select(t => t.Id).ToArray());

                    var mapping = AlignAnswerPosition(contextToken, contextString);
                    if (mapping == null)
                    {
                        return (null, null, false);
                    }
                    var questionTokenId = Tokenizer.EncodeToConverted(question.ToString());

                    var answerEnd = answerIndex + answer.Length - 1;
                    if (!mapping.ContainsKey(answerIndex) || !mapping.ContainsKey(answerEnd))
                    {
                        return (null, null, false);
                    }
                    var targetList = new List<int> { mapping[answerIndex] + 2 + questionTokenId.Count, mapping[answerEnd] + 2 + questionTokenId.Count };

                    var srcTensor = torch.tensor((new[] { 0 /* InitToken */ }).Concat(questionTokenId).Concat(new[] { 2 /* SeparatorToken */ }).Concat(contextTokenId).ToList(), device: Device);
                    // If the end of the answer goes beyond the 512 tokens then set answer start/end index to 0
                    if (targetList[1] > 511)
                    {
                        targetList[0] = 0;
                        targetList[1] = 0;
                    }
                    var labelTensor = torch.tensor(targetList, device: Device);

                    if (srcTensor.NumberOfElements > 512)
                        srcTensor = srcTensor.slice(0, 0, 512, 1);

                    return (srcTensor.MoveToOuterDisposeScope(), labelTensor.MoveToOuterDisposeScope(), true);
                }
            }

            private Dictionary<int, int> AlignAnswerPosition(IReadOnlyList<string> tokens, string text)
            {
                EnglishRobertaTokenizer robertaModel = Tokenizer as EnglishRobertaTokenizer;
                Debug.Assert(robertaModel is not null);

                var mapping = new Dictionary<int, int>();
                int surrogateDeduce = 0;
                for (var (i, j, tid) = (0, 0, 0); i < text.Length && tid < tokens.Count;)
                {
                    // Move to a new token
                    if (j >= tokens[tid].Length)
                    {
                        ++tid;
                        j = 0;
                    }
                    // There are a few UTF-32 chars in corpus, which is considered one char in position
                    else if (i + 1 < text.Length && char.IsSurrogatePair(text[i], text[i + 1]))
                    {
                        i += 2;
                        ++surrogateDeduce;
                    }
                    // White spaces are not included in tokens
                    else if (char.IsWhiteSpace(text[i]))
                    {
                        ++i;
                    }
                    // Chars not included in tokenizer will not appear in tokens
                    else if (!robertaModel.IsSupportedChar(text[i]))
                    {
                        mapping[i - surrogateDeduce] = tid;
                        ++i;
                    }
                    // "\\\"", "``" and "''" converted to "\"" in normalizer
                    else if (i + 1 < text.Length && tokens[tid][j] == '"'
                        && ((text[i] == '`' && text[i + 1] == '`')
                         || (text[i] == '\'' && text[i + 1] == '\'')
                         || (text[i] == '\\' && text[i + 1] == '"')))
                    {
                        mapping[i - surrogateDeduce] = mapping[i + 1 - surrogateDeduce] = tid;
                        i += 2;
                        j += 1;
                    }
                    // Normal match
                    else if (text[i] == tokens[tid][j])
                    {
                        mapping[i - surrogateDeduce] = tid;
                        ++i;
                        ++j;
                    }
                    // There are a few real \u0120 chars in the corpus, so this rule has to be later than text[i] == tokens[tid][j].
                    else if (tokens[tid][j] == '\u0120' && j == 0)
                    {
                        ++j;
                    }
                    else
                    {
                        throw new DataMisalignedException("unmatched!");
                    }
                }

                return mapping;
            }
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);

            var outColumns = inputSchema.ToDictionary(x => x.Name);

            var scoreMetadata = new List<SchemaShape.Column>();

            scoreMetadata.Add(new SchemaShape.Column(Kinds.ScoreColumnKind, SchemaShape.Column.VectorKind.Scalar,
                TextDataViewType.Instance, false));
            scoreMetadata.Add(new SchemaShape.Column(Kinds.ScoreValueKind, SchemaShape.Column.VectorKind.Scalar,
                TextDataViewType.Instance, false));
            scoreMetadata.Add(new SchemaShape.Column(Kinds.ScoreColumnSetId, SchemaShape.Column.VectorKind.Scalar,
                NumberDataViewType.UInt32, true));

            outColumns[Option.PredictedAnswerColumnName] = new SchemaShape.Column(Option.PredictedAnswerColumnName, SchemaShape.Column.VectorKind.VariableVector,
                    TextDataViewType.Instance, false);

            outColumns[Option.ScoreColumnName] = new SchemaShape.Column(Option.ScoreColumnName, SchemaShape.Column.VectorKind.VariableVector,
                NumberDataViewType.Single, false, new SchemaShape(scoreMetadata.ToArray()));

            return new SchemaShape(outColumns.Values);
        }

        private void CheckInputSchema(SchemaShape inputSchema)
        {
            // Verify that all required input columns are present, and are of the same type.
            if (!inputSchema.TryFindColumn(Option.ContextColumnName, out var contextCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "Context", Option.ContextColumnName);
            if (contextCol.Kind != SchemaShape.Column.VectorKind.Scalar || contextCol.ItemType.RawType != typeof(ReadOnlyMemory<char>))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "Context", Option.ContextColumnName,
                    TextDataViewType.Instance.ToString(), contextCol.GetTypeString());

            if (!inputSchema.TryFindColumn(Option.QuestionColumnName, out var questionCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "Question", Option.QuestionColumnName);
            if (questionCol.Kind != SchemaShape.Column.VectorKind.Scalar || questionCol.ItemType.RawType != typeof(ReadOnlyMemory<char>))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "Question", Option.QuestionColumnName,
                    TextDataViewType.Instance.ToString(), questionCol.GetTypeString());

            if (!inputSchema.TryFindColumn(Option.TrainingAnswerColumnName, out var answerCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "TrainingAnswer", Option.TrainingAnswerColumnName);
            if (answerCol.Kind != SchemaShape.Column.VectorKind.Scalar || answerCol.ItemType.RawType != typeof(ReadOnlyMemory<char>))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "TrainingAnswer", Option.TrainingAnswerColumnName,
                    TextDataViewType.Instance.ToString(), answerCol.GetTypeString());

            if (!inputSchema.TryFindColumn(Option.AnswerIndexStartColumnName, out var answerIndexCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "AnswerIndex", Option.AnswerIndexStartColumnName);
            if (answerIndexCol.Kind != SchemaShape.Column.VectorKind.Scalar || answerIndexCol.ItemType.RawType != typeof(int))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "AnswerIndex", Option.AnswerIndexStartColumnName,
                    NumberDataViewType.Int32.ToString(), answerIndexCol.GetTypeString());
        }
    }

    public class QATransformer : RowToRowTransformerBase, IDisposable
    {
        private protected readonly Device Device;
        private protected RobertaModelForQA Model;
        internal readonly QATrainer.Options Options;

        internal const string LoadName = "QATrainer";
        internal const string UserName = "QA Trainer";
        internal const string ShortName = "QA";
        internal const string Summary = "Question and Answer";
        internal const string LoaderSignature = "QATRAIN";

        public Tokenizer Tokenizer;
        private bool _disposedValue;

        internal QATransformer(IHostEnvironment env, QATrainer.Options options, RobertaModelForQA model)
           : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(QATransformer)))
        {
            Device = TorchUtils.InitializeDevice(env);

            Options = options;

            Model = model;
            Model.eval();

            if (Device.type == DeviceType.CUDA)
                Model.cuda();
            using (var ch = Host.Start("Initialize Tokenizer"))
                Tokenizer = TokenizerExtensions.GetInstance(ch);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);

            var outColumns = inputSchema.ToDictionary(x => x.Name);

            var scoreMetadata = new List<SchemaShape.Column>
            {
                new SchemaShape.Column(Kinds.ScoreColumnKind, SchemaShape.Column.VectorKind.Scalar,
                TextDataViewType.Instance, false),
                new SchemaShape.Column(Kinds.ScoreValueKind, SchemaShape.Column.VectorKind.Scalar,
                TextDataViewType.Instance, false),
                new SchemaShape.Column(Kinds.ScoreColumnSetId, SchemaShape.Column.VectorKind.Scalar,
                NumberDataViewType.UInt32, true)
            };

            outColumns[Options.PredictedAnswerColumnName] = new SchemaShape.Column(Options.PredictedAnswerColumnName, SchemaShape.Column.VectorKind.VariableVector,
                TextDataViewType.Instance, false);

            outColumns[Options.ScoreColumnName] = new SchemaShape.Column(Options.ScoreColumnName, SchemaShape.Column.VectorKind.VariableVector,
                NumberDataViewType.Single, false, new SchemaShape(scoreMetadata.ToArray()));

            return new SchemaShape(outColumns.Values);
        }

        private void CheckInputSchema(SchemaShape inputSchema)
        {
            if (!inputSchema.TryFindColumn(Options.ContextColumnName, out var contextCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "Context", Options.ContextColumnName);
            if (contextCol.ItemType != TextDataViewType.Instance)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "Context", Options.ContextColumnName,
                    TextDataViewType.Instance.ToString(), contextCol.GetTypeString());

            if (!inputSchema.TryFindColumn(Options.QuestionColumnName, out var questionCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "Question", Options.QuestionColumnName);
            if (questionCol.ItemType != TextDataViewType.Instance)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "Question", Options.QuestionColumnName,
                    TextDataViewType.Instance.ToString(), questionCol.GetTypeString());
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "QA-ANSWR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(QATransformer).Assembly.FullName);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: id of the context column name
            // int: id of the question column name
            // int: id of the PredictedAnswer column name
            // int: id of the Score name
            // int: topk
            // BinaryStream: TS Model

            ctx.SaveNonEmptyString(Options.ContextColumnName);
            ctx.SaveNonEmptyString(Options.QuestionColumnName);
            ctx.SaveNonEmptyString(Options.PredictedAnswerColumnName);
            ctx.SaveNonEmptyString(Options.ScoreColumnName);
            ctx.Writer.Write(Options.TopKAnswers);

            ctx.SaveBinaryStream("TSModel", w =>
            {
                Model.save(w);
            });
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new QAMapper(this, schema);

        //Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        // Factory method for SignatureLoadModel.
        private static QATransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: id of the context column name
            // int: id of the question column name
            // int: id of the PredictedAnswer column name
            // int: id of the Score name
            // int: topk
            // BinaryStream: TS Model

            var options = new QATrainer.Options()
            {
                ContextColumnName = ctx.LoadString(),
                QuestionColumnName = ctx.LoadString(),
                PredictedAnswerColumnName = ctx.LoadString(),
                ScoreColumnName = ctx.LoadString(),
                TopKAnswers = ctx.Reader.ReadInt32()
            };

            var ch = env.Start("Load Model");

            var model = new RobertaModelForQA(options);

            if (!ctx.TryLoadBinaryStream("TSModel", r => model.load(r)))
                throw env.ExceptDecode();

            return new QATransformer(env, options, model);
        }

        private class QAMapper : MapperBase
        {
            private readonly QATransformer _parent;
            private readonly HashSet<int> _inputColIndices;

            public QAMapper(QATransformer parent, DataViewSchema inputSchema) :
                base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(QAMapper)), inputSchema, parent)
            {
                _parent = parent;
                _inputColIndices = new HashSet<int>();

                if (inputSchema.TryGetColumnIndex(parent.Options.ContextColumnName, out var col))
                    _inputColIndices.Add(col);
                if (inputSchema.TryGetColumnIndex(parent.Options.QuestionColumnName, out col))
                    _inputColIndices.Add(col);

                if (Host is IHostEnvironmentInternal hostInternal)
                {
                    torch.random.manual_seed(hostInternal.Seed ?? 1);
                    torch.cuda.manual_seed(hostInternal.Seed ?? 1);
                }
                else
                {
                    torch.random.manual_seed(1);
                    torch.cuda.manual_seed(1);
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {

                var info = new DataViewSchema.DetachedColumn[2];

                var meta = new DataViewSchema.Annotations.Builder();
                meta.Add(AnnotationUtils.Kinds.ScoreColumnKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreColumnKind.MulticlassClassification.AsMemory(); });
                meta.Add(AnnotationUtils.Kinds.ScoreColumnSetId, AnnotationUtils.ScoreColumnSetIdType, GetScoreColumnSetId(InputSchema));
                meta.Add(AnnotationUtils.Kinds.ScoreValueKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreValueKind.Score.AsMemory(); });


                info[0] = new DataViewSchema.DetachedColumn(_parent.Options.PredictedAnswerColumnName, new VectorDataViewType(TextDataViewType.Instance));

                info[1] = new DataViewSchema.DetachedColumn(_parent.Options.ScoreColumnName, new VectorDataViewType(NumberDataViewType.Single), meta.ToAnnotations());
                return info;
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

            private Delegate CreateGetter(DataViewRow input, int iinfo, TensorCacher outputCacher)
            {
                var ch = Host.Start("Make Getter");
                if (iinfo == 0)
                    return MakePredictedAnswerGetter(input, ch, outputCacher);
                else
                    return MakeScoreGetter(input, ch, outputCacher);
            }

            private Delegate MakeScoreGetter(DataViewRow input, IChannel ch, TensorCacher outputCacher)
            {
                ValueGetter<ReadOnlyMemory<char>> getContext = default;
                ValueGetter<ReadOnlyMemory<char>> getQuestion = default;

                ReadOnlyMemory<char> context = default;
                ReadOnlyMemory<char> question = default;

                getContext = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent.Options.ContextColumnName]);
                getQuestion = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent.Options.QuestionColumnName]);

                ValueGetter<VBuffer<float>> score = (ref VBuffer<float> dst) =>
                {
                    using var disposeScope = torch.NewDisposeScope();
                    UpdateCacheIfNeeded(input.Position, outputCacher, ref context, ref question, ref getContext, ref getQuestion);
                    var editor = VBufferEditor.Create(ref dst, outputCacher.ScoresBuffer.Length);

                    for (var i = 0; i < outputCacher.ScoresBuffer.Length; i++)
                    {
                        editor.Values[i] = outputCacher.ScoresBuffer[i];
                    }
                    dst = editor.Commit();
                };

                return score;
            }

            private Delegate MakePredictedAnswerGetter(DataViewRow input, IChannel ch, TensorCacher outputCacher)
            {
                ValueGetter<ReadOnlyMemory<char>> getContext = default;
                ValueGetter<ReadOnlyMemory<char>> getQuestion = default;

                ReadOnlyMemory<char> context = default;
                ReadOnlyMemory<char> question = default;

                getContext = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent.Options.ContextColumnName]);
                getQuestion = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent.Options.QuestionColumnName]);

                ValueGetter<VBuffer<ReadOnlyMemory<char>>> predictedAnswer = (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                {
                    using var disposeScope = torch.NewDisposeScope();
                    UpdateCacheIfNeeded(input.Position, outputCacher, ref context, ref question, ref getContext, ref getQuestion);
                    var editor = VBufferEditor.Create(ref dst, outputCacher.PredictedAnswersBuffer.Length);

                    for (var i = 0; i < outputCacher.PredictedAnswersBuffer.Length; i++)
                    {
                        editor.Values[i] = outputCacher.PredictedAnswersBuffer[i];
                    }
                    dst = editor.Commit();
                };

                return predictedAnswer;
            }

            public override Delegate[] CreateGetters(DataViewRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Contracts.Assert(input.Schema == base.InputSchema);

                TensorCacher outputCacher = new TensorCacher(_parent.Options.TopKAnswers);
                var ch = Host.Start("Make Getters");
                _parent.Model.eval();

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

            private Tensor PrepInputTensors(ref ReadOnlyMemory<char> context, ref ReadOnlyMemory<char> question, ValueGetter<ReadOnlyMemory<char>> contextGetter, ValueGetter<ReadOnlyMemory<char>> questionGetter, out int contextLength, out int questionLength, out int[] contextIds)
            {

                contextGetter(ref context);
                questionGetter(ref question);

                var contextTokenId = _parent.Tokenizer.RobertaModel().ConvertIdsToOccurrenceRanks(_parent.Tokenizer.EncodeToIds(context.ToString()));

                var questionTokenId = _parent.Tokenizer.RobertaModel().ConvertIdsToOccurrenceRanks(_parent.Tokenizer.EncodeToIds(question.ToString()));

                var srcTensor = torch.tensor((new[] { 0 /* InitToken */ }).Concat(questionTokenId).Concat(new[] { 2 /* SeparatorToken */ }).Concat(contextTokenId).ToList(), device: _parent.Device);

                if (srcTensor.NumberOfElements > 512)
                    srcTensor = srcTensor.slice(0, 0, 512, 1);

                contextLength = contextTokenId.Count;
                questionLength = questionTokenId.Count;
                contextIds = contextTokenId.ToArray();

                return srcTensor.reshape(1, srcTensor.NumberOfElements);
            }

            private Tensor PrepAndRunModel(Tensor inputTensor)
            {
                using (torch.NewDisposeScope())
                {
                    return _parent.Model.forward(inputTensor).MoveToOuterDisposeScope();
                }
            }

            private protected class TensorCacher : IDisposable
            {
                public long Position;

                public int MaxLength;
                public ReadOnlyMemory<char>[] PredictedAnswersBuffer;
                public Single[] ScoresBuffer;

                public TensorCacher(int maxLength)
                {
                    Position = -1;
                    MaxLength = maxLength;

                    PredictedAnswersBuffer = new ReadOnlyMemory<char>[maxLength];
                    ScoresBuffer = new float[maxLength];
                }

                private bool _isDisposed;

                public void Dispose()
                {
                    if (_isDisposed)
                        return;

                    _isDisposed = true;
                }
            }

            private protected void UpdateCacheIfNeeded(long position, TensorCacher outputCache, ref ReadOnlyMemory<char> context, ref ReadOnlyMemory<char> question, ref ValueGetter<ReadOnlyMemory<char>> getContext, ref ValueGetter<ReadOnlyMemory<char>> getQuestion)
            {
                if (outputCache.Position != position)
                {

                    var inputTensor = PrepInputTensors(ref context, ref question, getContext, getQuestion, out int contextLength, out int questionLength, out int[] contextIds);
                    _parent.Model.eval();
                    using (torch.no_grad())
                    {
                        var logits = PrepAndRunModel(inputTensor);

                        var topKSpans = MetricUtils.ComputeTopKSpansWithScore(logits, _parent.Options.TopKAnswers, questionLength, contextLength);
                        int index = 0;
                        foreach (var topKSpan in topKSpans)
                        {
                            var predictStart = topKSpan.start;
                            var predictEnd = topKSpan.end;
                            var score = topKSpan.score;
                            outputCache.PredictedAnswersBuffer[index] = new ReadOnlyMemory<char>(_parent.Tokenizer.Decode(_parent.Tokenizer.RobertaModel().ConvertOccurrenceRanksToIds(contextIds).ToArray().AsSpan(predictStart - questionLength - 2, predictEnd - predictStart).ToArray()).Trim().ToCharArray());
                            outputCache.ScoresBuffer[index++] = score;
                        }

                        logits.Dispose();
                    }
                    outputCache.Position = position;
                }
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => (activeOutput(0) || activeOutput(1)) && _inputColIndices.Any(i => i == col);
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    Model.Dispose();
                    Model = null;
                    _disposedValue = true;
                }
            }
        }

        ~QATransformer()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: false);
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
