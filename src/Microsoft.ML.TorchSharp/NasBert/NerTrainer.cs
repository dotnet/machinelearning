// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
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
using static TorchSharp.torch;

[assembly: LoadableClass(typeof(NerTransformer), null, typeof(SignatureLoadModel),
    NerTransformer.UserName, NerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(NerTransformer), null, typeof(SignatureLoadRowMapper),
    NerTransformer.UserName, NerTransformer.LoaderSignature)]

namespace Microsoft.ML.TorchSharp.NasBert
{
    using TargetType = VBuffer<long>;

    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a Deep Neural Network(DNN) to classify text.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer, use [NER](xref:Microsoft.ML.TorchSharpCatalog.NamedEntityRecognition(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,System.String,System.String,System.String,Int32,Int32,Int32,Microsoft.ML.TorchSharp.NasBert.BertArchitecture,Microsoft.ML.IDataView)).
    ///
    /// ### Input and Output Columns
    /// The input label column data must be a Vector of [string](xref:Microsoft.ML.Data.TextDataViewType) type and the sentence columns must be of type<xref:Microsoft.ML.Data.TextDataViewType>.
    ///
    /// This trainer outputs the following columns:
    ///
    /// | Output Column Name | Column Type | Description|
    /// | -- | -- | -- |
    /// | `PredictedLabel` | Vector of [key](xref:Microsoft.ML.Data.KeyDataViewType) type | The predicted label's index. If its value is i, the actual label would be the i-th category in the key-valued input label type. |
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Multiclass classification |
    /// | Is normalization required? | No |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | Microsoft.ML.TorchSharp and libtorch-cpu or libtorch-cuda-11.3 or any of the OS specific variants. |
    /// | Exportable to ONNX | No |
    ///
    /// ### Training Algorithm Details
    /// Trains a Deep Neural Network(DNN) by leveraging an existing pre-trained NAS-BERT roBERTa model for the purpose of named entity recognition.
    /// ]]>
    /// </format>
    /// </remarks>
    ///
    public class NerTrainer : NasBertTrainer<VBuffer<uint>, TargetType>
    {
        private const char StartChar = (char)(' ' + 256);

        public class NerOptions : NasBertOptions
        {
            public NerOptions()
            {
                LearningRate = new List<double>() { 2e-4 };
                EncoderOutputDim = 384;
                EmbeddingDim = 128;
                Arches = new int[] { 15, 16, 14, 0, 0, 0, 15, 16, 14, 0, 0, 0, 17, 14, 15, 0, 0, 0, 17, 14, 15, 0, 0, 0 };
                TaskType = BertTaskType.NamedEntityRecognition;
            }
        }
        internal NerTrainer(IHostEnvironment env, NerOptions options) : base(env, options)
        {
        }

        internal NerTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string predictionColumnName = DefaultColumnNames.PredictedLabel,
            string sentence1ColumnName = "Sentence1",
            int batchSize = 32,
            int maxEpochs = 10,
            IDataView validationSet = null,
            BertArchitecture architecture = BertArchitecture.Roberta) :
            this(env, new NerOptions
            {
                PredictionColumnName = predictionColumnName,
                ScoreColumnName = default,
                Sentence1ColumnName = sentence1ColumnName,
                Sentence2ColumnName = default,
                LabelColumnName = labelColumnName,
                BatchSize = batchSize,
                MaxEpoch = maxEpochs,
                ValidationSet = validationSet,
            })
        {
        }

        private protected override TrainerBase CreateTrainer(TorchSharpBaseTrainer<VBuffer<uint>, TargetType> parent, IChannel ch, IDataView input)
        {
            return new Trainer(parent, ch, input);
        }

        private protected override TorchSharpBaseTransformer<VBuffer<uint>, TargetType> CreateTransformer(IHost host, Options options, torch.nn.Module model, DataViewSchema.DetachedColumn labelColumn)
        {
            return new NerTransformer(host, options as NasBertOptions, model as NasBertModel, labelColumn);
        }

        internal static bool TokenStartsWithSpace(string token) => token is null || (token.Length != 0 && token[0] == StartChar);

        private protected class Trainer : NasBertTrainerBase
        {
            private const string ModelUrlString = "models/pretrained_NasBert_14M_encoder.tsm";
            internal static readonly int[] ZeroArray = new int[] { 0 /* InitToken */};

            public Trainer(TorchSharpBaseTrainer<VBuffer<uint>, TargetType> parent, IChannel ch, IDataView input) : base(parent, ch, input, ModelUrlString)
            {
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private protected override TargetType AddToTargets(VBuffer<uint> target)
            {
                // keys are 1 based but the model is 0 based
                var tl = target.DenseValues().Select(item => (long)item).ToList();
                tl.Insert(0, 0);
                VBuffer<long> t = new VBuffer<long>(target.Length + 1, tl.ToArray());
                return t;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private protected override torch.Tensor CreateTargetsTensor(ref List<TargetType> targets, torch.Device device)
            {
                var maxLength = 0;
                targets.ForEach(x =>
                    {
                        if (x.Length > maxLength)
                            maxLength = x.Length;
                    }
                );

                long[,] targetArray = new long[targets.Count(), maxLength];

                for (int i = 0; i < targets.Count(); i++)
                {
                    for (int j = 0; j < targets[i].Length; j++)
                    {
                        targetArray[i, j] = targets[i].GetValues()[j];
                    }

                    for (int j = targets[i].Length; j < maxLength; j++)
                    {
                        targetArray[i, j] = 0;
                    }
                }

                return torch.tensor(targetArray, device: Device);
            }

            private protected override torch.Tensor PrepareRowTensor(ref VBuffer<uint> target)
            {
                ReadOnlyMemory<char> sentenceRom = default;
                Sentence1Getter(ref sentenceRom);
                var sentence = sentenceRom.ToString();
                Tensor t;
                IReadOnlyList<EncodedToken> encoding = Tokenizer.EncodeToTokens(sentence, out string normalizedText);

                if (target.Length != encoding.Count)
                {
                    var targetIndex = 0;
                    var targetEditor = VBufferEditor.Create(ref target, encoding.Count);
                    var newValues = targetEditor.Values;
                    for (var i = 0; i < encoding.Count; i++)
                    {
                        if (NerTrainer.TokenStartsWithSpace(encoding[i].Value))
                        {
                            newValues[i] = target.GetItemOrDefault(++targetIndex);
                        }
                        else
                        {
                            newValues[i] = target.GetItemOrDefault(targetIndex);
                        }
                    }
                    target = targetEditor.Commit();
                }
                t = torch.tensor((ZeroArray).Concat(Tokenizer.RobertaModel().ConvertIdsToOccurrenceRanks(encoding.Select(t => t.Id).ToArray())).ToList(), device: Device);

                if (t.NumberOfElements > 512)
                    t = t.slice(0, 0, 512, 1);

                return t;
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
                return indexes.@int();
            }

            private protected override int GetRowCountAndSetLabelCount(IDataView input)
            {
                VBuffer<ReadOnlyMemory<char>> keys = default;
                input.Schema[Parent.Option.LabelColumnName].GetKeyValues(ref keys);
                var labelCol = input.GetColumn<VBuffer<uint>>(Parent.Option.LabelColumnName);
                var rowCount = 0;

                foreach (var label in labelCol)
                {
                    rowCount++;
                }

                Parent.Option.NumberOfClasses = keys.Length + 1;
                return rowCount;
            }

            private protected override torch.Tensor GetTargets(torch.Tensor labels)
            {
                return labels.view(-1);
            }
        }
    }

    public sealed class NerTransformer : NasBertTransformer<VBuffer<uint>, TargetType>
    {
        internal const string LoadName = "NERTrainer";
        internal const string UserName = "NER Trainer";
        internal const string ShortName = "NER";
        internal const string Summary = "NER with NAS-BERT";
        internal const string LoaderSignature = "NER";

        private static readonly FuncStaticMethodInfo1<object, Delegate> _decodeInitMethodInfo
            = new FuncStaticMethodInfo1<object, Delegate>(DecodeInit<int>);

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NER-BERT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TextClassificationTransformer).Assembly.FullName);
        }

        internal NerTransformer(IHostEnvironment env, NasBertOptions options, NasBertModel model, DataViewSchema.DetachedColumn labelColumn) : base(env, options, model, labelColumn)
        {
        }

        private protected override IRowMapper GetRowMapper(TorchSharpBaseTransformer<VBuffer<uint>, TargetType> parent, DataViewSchema schema)
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
        private static NerTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
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

            var options = new NerTrainer.NerOptions()
            {
                LabelColumnName = ctx.LoadString(),
                ScoreColumnName = ctx.LoadStringOrNull(),
                PredictionColumnName = ctx.LoadString(),
                NumberOfClasses = ctx.Reader.ReadInt32(),
            };

            var ch = env.Start("Load Model");
            var tokenizer = TokenizerExtensions.GetInstance(ch);
            EnglishRobertaTokenizer tokenizerModel = tokenizer.RobertaModel();

            var model = new NerModel(options, tokenizerModel.PadIndex, tokenizerModel.SymbolsCount, options.NumberOfClasses);
            if (!ctx.TryLoadBinaryStream("TSModel", r => model.load(r)))
                throw env.ExceptDecode();

            options.Sentence1ColumnName = ctx.LoadString();
            options.Sentence2ColumnName = ctx.LoadStringOrNull();
            options.TaskType = BertTaskType.NamedEntityRecognition;

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

            return new NerTransformer(env, options, model, labelCol);
        }

        private static Delegate DecodeInit<T>(object value)
        {
            VBuffer<T> buffValue = (VBuffer<T>)value;
            ValueGetter<VBuffer<T>> buffGetter = (ref VBuffer<T> dst) => buffValue.CopyTo(ref dst);
            return buffGetter;
        }

        private sealed class Mapper : NasBertMapper
        {
            public Mapper(TorchSharpBaseTransformer<VBuffer<uint>, TargetType> parent, DataViewSchema inputSchema) : base(parent, inputSchema)
            {
            }

            private protected override Delegate CreateGetter(DataViewRow input, int iinfo, TensorCacher outputCacher)
            {
                var ch = Host.Start("Make Getter");
                return MakePredictedLabelGetter(input, ch, outputCacher);

            }

            private void CondenseOutput(ref VBuffer<UInt32> dst, string sentence, Tokenizer tokenizer, TensorCacher outputCacher)
            {
                var pre = tokenizer.PreTokenizer.PreTokenize(sentence);
                IReadOnlyList<EncodedToken> encoding = tokenizer.EncodeToTokens(sentence, out string normalizedText);

                var argmax = (outputCacher as BertTensorCacher).Result.argmax(-1);
                var prediction = argmax.ToArray<long>();

                var targetIndex = 0;
                // Figure out actual count of output tokens
                for (var i = 0; i < encoding.Count; i++)
                {
                    if (NerTrainer.TokenStartsWithSpace(encoding[i].Value))
                    {
                        targetIndex++;
                    }
                }

                var editor = VBufferEditor.Create(ref dst, targetIndex + 1);
                var newValues = editor.Values;
                targetIndex = 0;

                newValues[targetIndex++] = (uint)prediction[0];

                for (var i = 1; i < encoding.Count; i++)
                {
                    if (NerTrainer.TokenStartsWithSpace(encoding[i].Value))
                    {
                        newValues[targetIndex++] = (uint)prediction[i];
                    }
                }

                dst = editor.Commit();
            }

            private Delegate MakePredictedLabelGetter(DataViewRow input, IChannel ch, TensorCacher outputCacher)
            {
                ValueGetter<ReadOnlyMemory<char>> getSentence1 = default;
                ValueGetter<ReadOnlyMemory<char>> getSentence2 = default;

                Tokenizer tokenizer = TokenizerExtensions.GetInstance(ch);

                getSentence1 = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[Parent.SentenceColumn.Name]);

                ReadOnlyMemory<char> sentence1 = default;
                ReadOnlyMemory<char> sentence2 = default;

                ValueGetter<VBuffer<UInt32>> classification = (ref VBuffer<UInt32> dst) =>
                {
                    using var disposeScope = torch.NewDisposeScope();
                    UpdateCacheIfNeeded(input.Position, outputCacher, ref sentence1, ref sentence2, ref getSentence1, ref getSentence2, tokenizer);
                    var argmax = (outputCacher as BertTensorCacher).Result.argmax(-1);
                    var prediction = argmax.ToArray<long>();

                    CondenseOutput(ref dst, sentence1.ToString(), tokenizer, outputCacher);
                };

                return classification;
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => activeOutput(0) && InputColIndices.Any(i => i == col);
            }
        }
    }

}
