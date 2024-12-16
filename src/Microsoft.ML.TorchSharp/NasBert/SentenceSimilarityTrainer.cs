// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Tokenizers;
using Microsoft.ML.TorchSharp.Extensions;
using Microsoft.ML.TorchSharp.NasBert;
using Microsoft.ML.TorchSharp.NasBert.Models;
using TorchSharp;
using static Microsoft.ML.TorchSharp.NasBert.NasBertTrainer;

[assembly: LoadableClass(typeof(SentenceSimilarityTransformer), null, typeof(SignatureLoadModel),
    SentenceSimilarityTransformer.UserName, SentenceSimilarityTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(SentenceSimilarityTransformer), null, typeof(SignatureLoadRowMapper),
    SentenceSimilarityTransformer.UserName, SentenceSimilarityTransformer.LoaderSignature)]

namespace Microsoft.ML.TorchSharp.NasBert
{
    /// <summary>
    /// Represents the <see cref="IEstimator{TTransformer}"/> for training a Deep Neural Network (DNN) to determine sentence similarity.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer, use [TextClassification](xref:Microsoft.ML.TorchSharpCatalog.TextClassification(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,Int32,System.String,System.String,System.String,System.String,Int32,Int32,Int32,Microsoft.ML.TorchSharp.NasBert.BertArchitecture,Microsoft.ML.IDataView)).
    ///
    /// ### Input and output columns
    /// The input label column data must be type <xref:System.Single> and the sentence columns must be of type <xref:Microsoft.ML.Data.TextDataViewType>.
    ///
    /// This trainer outputs the following columns:
    ///
    /// | Output column name | Column type | Description|
    /// | -- | -- | -- |
    /// | `Score` | <xref:System.Single> | The degree of similarity between the two sentences. |
    ///
    /// ### Trainer characteristics
    /// | Characteristic | Value  |
    /// | -- | -- |
    /// | Machine learning task | Regression |
    /// | Is normalization required? | No |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | Microsoft.ML.TorchSharp and libtorch-cpu or libtorch-cuda-11.3 or any of the OS specific variants. |
    /// | Exportable to ONNX | No |
    ///
    /// ### Training algorithm details
    /// Trains a Deep Neural Network (DNN) by leveraging an existing, pretrained NAS-BERT roBERTa model for the purpose of determining sentence similarity.
    /// ]]>
    /// </format>
    /// </remarks>
    ///
    public class SentenceSimilarityTrainer : NasBertTrainer<float, float>
    {

        public class SentenceSimilarityOptions : NasBertOptions
        {
            public SentenceSimilarityOptions()
            {
                BatchSize = 32;
                MaxEpoch = 10;
                TaskType = BertTaskType.SentenceRegression;
                LearningRate = new List<double>() { .0002 };
                WeightDecay = .01;
            }
        }
        internal SentenceSimilarityTrainer(IHostEnvironment env, SentenceSimilarityOptions options) : base(env, options)
        {
        }

        internal SentenceSimilarityTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string scoreColumnName = DefaultColumnNames.Score,
            string sentence1ColumnName = "Sentence1",
            string sentence2ColumnName = default,
            int batchSize = 32,
            int maxEpochs = 10,
            IDataView validationSet = null,
            BertArchitecture architecture = BertArchitecture.Roberta) :
            this(env, new SentenceSimilarityOptions
            {
                ScoreColumnName = scoreColumnName,
                Sentence1ColumnName = sentence1ColumnName,
                Sentence2ColumnName = sentence2ColumnName,
                LabelColumnName = labelColumnName,
                BatchSize = batchSize,
                MaxEpoch = maxEpochs,
                ValidationSet = validationSet,
                TaskType = BertTaskType.SentenceRegression,
                LearningRate = new List<double>() { .0002 },
                WeightDecay = .01
            })
        {
        }

        private protected override TrainerBase CreateTrainer(TorchSharpBaseTrainer<float, float> parent, IChannel ch, IDataView input)
        {
            return new Trainer(parent, ch, input);
        }

        private protected override TorchSharpBaseTransformer<float, float> CreateTransformer(IHost host, Options options, torch.nn.Module model, DataViewSchema.DetachedColumn labelColumn)
        {
            return new SentenceSimilarityTransformer(host, options as NasBertOptions, model as ModelForPrediction, labelColumn);
        }

        private protected class Trainer : NasBertTrainerBase
        {
            private const string ModelUrlString = "models/NasBert2000000.tsm";

            public Trainer(TorchSharpBaseTrainer<float, float> parent, IChannel ch, IDataView input) : base(parent, ch, input, ModelUrlString)
            {
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private protected override float AddToTargets(float target)
            {
                return target;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private protected override torch.Tensor CreateTargetsTensor(ref List<float> targets, torch.Device device)
            {
                return torch.tensor(targets, device: Device).@float();
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private protected override int GetNumCorrect(torch.Tensor predictions, torch.Tensor targets)
            {
                predictions = predictions ?? throw new ArgumentNullException(nameof(predictions));
                return (int)predictions.eq(targets).sum().ToInt64();
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private protected override torch.Tensor GetPredictions(torch.Tensor logits)
            {
                logits = logits ?? throw new ArgumentNullException(nameof(logits));
                return logits.squeeze();
            }

            private protected override int GetRowCountAndSetLabelCount(IDataView input)
            {
                var labelCol = input.GetColumn<float>(Parent.Option.LabelColumnName);
                var rowCount = 0;

                foreach (var label in labelCol)
                {
                    rowCount++;
                }

                // Set 1 class for regression as thats what the model needs.
                Parent.Option.NumberOfClasses = 1;
                return rowCount;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private protected override torch.Tensor GetTargets(torch.Tensor labels)
            {
                return labels.view(-1);
            }
        }
    }

    public sealed class SentenceSimilarityTransformer : NasBertTransformer<float, float>
    {
        internal const string LoadName = "SentSimTrainer";
        internal const string UserName = "Sentence Similarity Trainer";
        internal const string ShortName = "SNTSIMI";
        internal const string Summary = "NLP with NAS-BERT";
        internal const string LoaderSignature = "SNTSIMI";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SNT-SIMI",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // New refactor format
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SentenceSimilarityTransformer).Assembly.FullName);
        }

        internal SentenceSimilarityTransformer(IHostEnvironment env, NasBertOptions options, ModelForPrediction model, DataViewSchema.DetachedColumn labelColumn) : base(env, options, model, labelColumn)
        {
        }

        private protected override IRowMapper GetRowMapper(TorchSharpBaseTransformer<float, float> parent, DataViewSchema schema)
        {
            return new Mapper(parent, schema);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            // *** Binary format ***
            // BaseModel
            //  int: id of label column name
            //  int: id of the score column name
            //  int: id of output column name
            //  int: number of classes
            //  BinaryStream: TS Model
            //  int: id of sentence 1 column name
            //  int: id of sentence 2 column name
            SaveBaseModel(ctx, GetVersionInfo());
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        // Factory method for SignatureLoadModel.
        private static SentenceSimilarityTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // BaseModel
            //  int: id of label column name
            //  int: id of the score column name
            //  int: id of output column name
            //  int: number of classes
            //  BinaryStream: TS Model
            //  int: id of sentence 1 column name
            //  int: id of sentence 2 column name
            var options = new NasBertOptions()
            {
                LabelColumnName = ctx.LoadString(),
                ScoreColumnName = ctx.LoadString(),
                PredictionColumnName = ctx.LoadString(),
                NumberOfClasses = ctx.Reader.ReadInt32(),
            };

            var ch = env.Start("Load Model");
            var tokenizer = TokenizerExtensions.GetInstance(ch);
            EnglishRobertaTokenizer tokenizerModel = tokenizer.RobertaModel();

            var model = new ModelForPrediction(options, tokenizerModel.PadIndex, tokenizerModel.SymbolsCount, options.NumberOfClasses);
            if (!ctx.TryLoadBinaryStream("TSModel", r => model.load(r)))
                throw env.ExceptDecode();

            options.Sentence1ColumnName = ctx.LoadString();
            options.Sentence2ColumnName = ctx.LoadStringOrNull();
            options.TaskType = BertTaskType.SentenceRegression;

            var labelCol = new DataViewSchema.DetachedColumn(options.LabelColumnName, NumberDataViewType.Single);

            return new SentenceSimilarityTransformer(env, options, model, labelCol);
        }

        private sealed class Mapper : NasBertMapper
        {
            public Mapper(TorchSharpBaseTransformer<float, float> parent, DataViewSchema inputSchema) : base(parent, inputSchema)
            {
            }

            private protected override Delegate CreateGetter(DataViewRow input, int iinfo, TensorCacher outputCacher)
            {
                var ch = Host.Start("Make Getter");
                return MakeScoreGetter(input, ch, outputCacher);
            }

            private Delegate MakeScoreGetter(DataViewRow input, IChannel ch, TensorCacher outputCacher)
            {
                ValueGetter<ReadOnlyMemory<char>> getSentence1 = default;
                ValueGetter<ReadOnlyMemory<char>> getSentence2 = default;

                Tokenizer tokenizer = TokenizerExtensions.GetInstance(ch);

                getSentence1 = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[Parent.SentenceColumn.Name]);
                getSentence2 = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[Parent.SentenceColumn2.Name]);

                ReadOnlyMemory<char> sentence1 = default;
                ReadOnlyMemory<char> sentence2 = default;
                var cacher = outputCacher as BertTensorCacher;

                ValueGetter<float> score = (ref float dst) =>
                {
                    using var disposeScope = torch.NewDisposeScope();
                    UpdateCacheIfNeeded(input.Position, outputCacher, ref sentence1, ref sentence2, ref getSentence1, ref getSentence2, tokenizer);
                    dst = cacher.Result.squeeze().cpu().item<float>();
                };

                return score;
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => activeOutput(0) && InputColIndices.Any(i => i == col);
            }
        }
    }

}
