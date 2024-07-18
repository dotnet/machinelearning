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
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Tokenizers;
using Microsoft.ML.TorchSharp.Extensions;
using Microsoft.ML.TorchSharp.NasBert;
using Microsoft.ML.TorchSharp.NasBert.Models;
using TorchSharp;
using static Microsoft.ML.TorchSharp.NasBert.NasBertTrainer;

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
    /// | `Score` | Vector of<xref:System.Single> | The scores of all classes.Higher value means higher probability to fall into the associated class. If the i-th element has the largest value, the predicted label index would be i.Note that i is zero-based index. |
    /// ### Trainer Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Multiclass classification |
    /// | Is normalization required? | No |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | Microsoft.ML.TorchSharp and libtorch-cpu or libtorch-cuda-11.3 or any of the OS specific variants. |
    /// | Exportable to ONNX | No |
    ///
    /// ### Training Algorithm Details
    /// Trains a Deep Neural Network(DNN) by leveraging an existing pre-trained NAS-BERT roBERTa model for the purpose of classifying text.
    /// ]]>
    /// </format>
    /// </remarks>
    ///
    public class TextClassificationTrainer : NasBertTrainer<UInt32, long>
    {
        public class TextClassificationOptions : NasBertTrainer.NasBertOptions
        {
            public TextClassificationOptions()
            {
                TaskType = BertTaskType.TextClassification;
                BatchSize = 32;
                MaxEpoch = 10;
            }
        }

        internal TextClassificationTrainer(IHostEnvironment env, TextClassificationOptions options) : base(env, options)
        {
        }

        internal TextClassificationTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string predictionColumnName = DefaultColumnNames.PredictedLabel,
            string scoreColumnName = DefaultColumnNames.Score,
            string sentence1ColumnName = "Sentence1",
            string sentence2ColumnName = default,
            int batchSize = 32,
            int maxEpochs = 10,
            IDataView validationSet = null,
            BertArchitecture architecture = BertArchitecture.Roberta) :
            this(env, new TextClassificationOptions
            {
                PredictionColumnName = predictionColumnName,
                ScoreColumnName = scoreColumnName,
                Sentence1ColumnName = sentence1ColumnName,
                Sentence2ColumnName = sentence2ColumnName,
                LabelColumnName = labelColumnName,
                BatchSize = batchSize,
                MaxEpoch = maxEpochs,
                ValidationSet = validationSet,
                TaskType = BertTaskType.TextClassification
            })
        {
        }

        private protected override TrainerBase CreateTrainer(TorchSharpBaseTrainer<uint, long> parent, IChannel ch, IDataView input)
        {
            return new Trainer(parent, ch, input);
        }

        private protected override TorchSharpBaseTransformer<uint, long> CreateTransformer(IHost host, Options options, torch.nn.Module model, DataViewSchema.DetachedColumn labelColumn)
        {
            return new TextClassificationTransformer(host, options as NasBertOptions, model as NasBertModel, labelColumn);
        }

        private protected class Trainer : NasBertTrainerBase
        {
            private const string ModelUrlString = "models/NasBert2000000.tsm";

            public Trainer(TorchSharpBaseTrainer<uint, long> parent, IChannel ch, IDataView input) : base(parent, ch, input, ModelUrlString)
            {
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private protected override long AddToTargets(uint target)
            {
                // keys are 1 based but the model is 0 based
                return target - 1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private protected override torch.Tensor CreateTargetsTensor(ref List<long> targets, torch.Device device)
            {
                return torch.tensor(targets, device: Device);
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
                var (_, indexes) = logits.max(-1, false);
                return indexes;
            }

            private protected override int GetRowCountAndSetLabelCount(IDataView input)
            {
                var labelCol = input.GetColumn<uint>(Parent.Option.LabelColumnName);
                var rowCount = 0;
                var uniqueLabels = new HashSet<uint>();

                foreach (var label in labelCol)
                {
                    rowCount++;
                    uniqueLabels.Add(label);
                }

                Parent.Option.NumberOfClasses = uniqueLabels.Count;
                return rowCount;
            }

            private protected override torch.Tensor GetTargets(torch.Tensor labels)
            {
                return labels.view(-1);
            }
        }
    }

    public sealed class TextClassificationTransformer : NasBertTransformer<UInt32, long>
    {
        internal const string LoadName = "TextClassTrainer";
        internal const string UserName = "Text Classification Trainer";
        internal const string ShortName = "TXTCLSS";
        internal const string Summary = "NLP with NAS-BERT";
        internal const string LoaderSignature = "TXTCLSS";

        private static readonly FuncStaticMethodInfo1<object, Delegate> _decodeInitMethodInfo
            = new FuncStaticMethodInfo1<object, Delegate>(DecodeInit<int>);

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TXT-CLSS",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // New refactor format
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TextClassificationTransformer).Assembly.FullName);
        }

        internal TextClassificationTransformer(IHostEnvironment env, NasBertOptions options, NasBertModel model, DataViewSchema.DetachedColumn labelColumn) : base(env, options, model, labelColumn)
        {
        }

        private protected override IRowMapper GetRowMapper(TorchSharpBaseTransformer<uint, long> parent, DataViewSchema schema)
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
            // LabelValues

            SaveBaseModel(ctx, GetVersionInfo());
            var labelColType = LabelColumn.Annotations.Schema[AnnotationUtils.Kinds.KeyValues].Type as VectorDataViewType;
            Microsoft.ML.Internal.Utilities.Utils.MarshalActionInvoke(SaveLabelValues<int>, labelColType.ItemType.RawType, ctx);
        }

        private void SaveLabelValues<T>(ModelSaveContext ctx)
        {
            ValueGetter<VBuffer<T>> getter = LabelColumn.Annotations.GetGetter<VBuffer<T>>(LabelColumn.Annotations.Schema[AnnotationUtils.Kinds.KeyValues]);
            var val = default(VBuffer<T>);
            getter(ref val);

            BinarySaver saver = new BinarySaver(Host, new BinarySaver.Arguments());
            int bytesWritten;
            var labelColType = LabelColumn.Annotations.Schema[AnnotationUtils.Kinds.KeyValues].Type as VectorDataViewType;
            if (!saver.TryWriteTypeAndValue<VBuffer<T>>(ctx.Writer.BaseStream, labelColType, ref val, out bytesWritten))
                throw Host.Except("We do not know how to serialize label names of type '{0}'", labelColType.ItemType);
        }

        //Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        // Factory method for SignatureLoadModel.
        private static TextClassificationTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
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
            // LabelValues

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
            options.TaskType = BertTaskType.TextClassification;

            BinarySaver saver = new BinarySaver(env, new BinarySaver.Arguments());
            DataViewType type;
            object value;
            env.CheckDecode(saver.TryLoadTypeAndValue(ctx.Reader.BaseStream, out type, out value));
            var vecType = type as VectorDataViewType;
            env.CheckDecode(vecType != null);
            env.CheckDecode(value != null);
            var labelGetter = Microsoft.ML.Internal.Utilities.Utils.MarshalInvoke(_decodeInitMethodInfo, vecType.ItemType.RawType, value);

            var meta = new DataViewSchema.Annotations.Builder();
            meta.Add(AnnotationUtils.Kinds.KeyValues, type, labelGetter);

            var labelCol = new DataViewSchema.DetachedColumn(options.LabelColumnName, type, meta.ToAnnotations());

            return new TextClassificationTransformer(env, options, model, labelCol);
        }

        private static Delegate DecodeInit<T>(object value)
        {
            VBuffer<T> buffValue = (VBuffer<T>)value;
            ValueGetter<VBuffer<T>> buffGetter = (ref VBuffer<T> dst) => buffValue.CopyTo(ref dst);
            return buffGetter;
        }

        private sealed class Mapper : NasBertMapper
        {
            public Mapper(TorchSharpBaseTransformer<uint, long> parent, DataViewSchema inputSchema) : base(parent, inputSchema)
            {
            }

            private protected override Delegate CreateGetter(DataViewRow input, int iinfo, TensorCacher outputCacher)
            {
                var ch = Host.Start("Make Getter");
                if (iinfo == 0)
                    return MakePredictedLabelGetter(input, ch, outputCacher);
                else
                    return MakeScoreGetter(input, ch, outputCacher);
            }

            private Delegate MakeScoreGetter(DataViewRow input, IChannel ch, TensorCacher outputCacher)
            {
                ValueGetter<ReadOnlyMemory<char>> getSentence1 = default;
                ValueGetter<ReadOnlyMemory<char>> getSentence2 = default;

                Tokenizer tokenizer = TokenizerExtensions.GetInstance(ch);

                getSentence1 = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[Parent.SentenceColumn.Name]);
                if (Parent.SentenceColumn2.IsValid)
                    getSentence2 = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[Parent.SentenceColumn2.Name]);

                ReadOnlyMemory<char> sentence1 = default;
                ReadOnlyMemory<char> sentence2 = default;

                ValueGetter<VBuffer<float>> score = (ref VBuffer<float> dst) =>
                {
                    using var disposeScope = torch.NewDisposeScope();
                    var editor = VBufferEditor.Create(ref dst, Parent.Options.NumberOfClasses);
                    UpdateCacheIfNeeded(input.Position, outputCacher, ref sentence1, ref sentence2, ref getSentence1, ref getSentence2, tokenizer);
                    var values = (outputCacher as BertTensorCacher).Result.cpu().ToArray<float>();

                    for (var i = 0; i < values.Length; i++)
                    {
                        editor.Values[i] = values[i];
                    }
                    dst = editor.Commit();
                };

                return score;
            }

            private Delegate MakePredictedLabelGetter(DataViewRow input, IChannel ch, TensorCacher outputCacher)
            {
                ValueGetter<ReadOnlyMemory<char>> getSentence1 = default;
                ValueGetter<ReadOnlyMemory<char>> getSentence2 = default;

                Tokenizer tokenizer = TokenizerExtensions.GetInstance(ch);

                getSentence1 = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[Parent.SentenceColumn.Name]);
                if (Parent.SentenceColumn2.IsValid)
                    getSentence2 = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[Parent.SentenceColumn2.Name]);

                ReadOnlyMemory<char> sentence1 = default;
                ReadOnlyMemory<char> sentence2 = default;

                ValueGetter<UInt32> classification = (ref UInt32 dst) =>
                {
                    using var disposeScope = torch.NewDisposeScope();
                    UpdateCacheIfNeeded(input.Position, outputCacher, ref sentence1, ref sentence2, ref getSentence1, ref getSentence2, tokenizer);
                    dst = (UInt32)(outputCacher as BertTensorCacher).Result.argmax(-1).cpu().item<long>() + 1;
                };

                return classification;
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => (activeOutput(0) || activeOutput(1)) && InputColIndices.Any(i => i == col);
            }
        }
    }

}
