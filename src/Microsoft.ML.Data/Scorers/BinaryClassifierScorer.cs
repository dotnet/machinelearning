// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Runtime;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(BinaryClassifierScorer), typeof(BinaryClassifierScorer.Arguments), typeof(SignatureDataScorer),
    "Binary Classifier Scorer", "BinaryClassifierScorer", "BinaryClassifier", "Binary",
    "bin", AnnotationUtils.Const.ScoreColumnKind.BinaryClassification)]

[assembly: LoadableClass(typeof(BinaryClassifierScorer), null, typeof(SignatureLoadDataTransform),
    "Binary Classifier Scorer", BinaryClassifierScorer.LoaderSignature)]

namespace Microsoft.ML.Data
{
    [BestFriend]
    internal sealed class BinaryClassifierScorer : PredictedLabelScorerBase, ITransformCanSaveOnnx
    {
        public sealed class Arguments : ThresholdArgumentsBase
        {
        }

        public const string LoaderSignature = "BinClassScoreTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "BINCLSCR",
                //verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Added threshold, VerWithThreshold
                //verWrittenCur: 0x00010003, // ISchemaBindableMapper
                verWrittenCur: 0x00010004, // ISchemaBindableMapper update
                verReadableCur: 0x00010004,
                verWeCanReadBack: 0x00010004,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(BinaryClassifierScorer).Assembly.FullName);
        }

        private const string RegistrationName = "BinaryClassifierScore";

        private readonly float _threshold;

        /// <summary>
        /// This function performs a number of checks on the inputs and, if appropriate and possible, will produce
        /// a mapper with slots names on the output score column properly mapped. If this is not possible for any
        /// reason, it will just return the input bound mapper.
        /// </summary>
        private static ISchemaBoundMapper WrapIfNeeded(IHostEnvironment env, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(mapper, nameof(mapper));
            env.CheckValueOrNull(trainSchema);

            // The idea is that we will take the key values from the train schema label, and present
            // them as slot name metadata. But there are a number of conditions for this to actually
            // happen, so we test those here. If these are not

            if (trainSchema?.Label == null)
                return mapper; // We don't even have a label identified in a training schema.
            var keyType = trainSchema.Label.Value.Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues)?.Type as VectorDataViewType;
            if (keyType == null || !CanWrap(mapper, keyType))
                return mapper;

            // Great!! All checks pass.
            return Utils.MarshalInvoke(WrapCore<int>, keyType.ItemType.RawType, env, mapper, trainSchema);
        }

        /// <summary>
        /// This is a utility method used to determine whether <see cref="MulticlassClassificationScorer.LabelNameBindableMapper"/>
        /// can or should be used to wrap <paramref name="mapper"/>. This will not throw, since the
        /// desired behavior in the event that it cannot be wrapped, is to just back off to the original
        /// "unwrapped" bound mapper.
        /// </summary>
        /// <param name="mapper">The mapper we are seeing if we can wrap</param>
        /// <param name="labelNameType">The type of the label names from the metadata (either
        /// originating from the key value metadata of the training label column, or deserialized
        /// from the model of a bindable mapper)</param>
        /// <returns>Whether we can call <see cref="MulticlassClassificationScorer.LabelNameBindableMapper.CreateBound{T}"/> with
        /// this mapper and expect it to succeed</returns>
        private static bool CanWrap(ISchemaBoundMapper mapper, DataViewType labelNameType)
        {
            Contracts.AssertValue(mapper);
            Contracts.AssertValue(labelNameType);

            ISchemaBoundRowMapper rowMapper = mapper as ISchemaBoundRowMapper;
            if (rowMapper == null)
                return false; // We could cover this case, but it is of no practical worth as far as I see, so I decline to do so.

            int scoreIdx;
            if (!mapper.OutputSchema.TryGetColumnIndex(AnnotationUtils.Const.ScoreValueKind.Score, out scoreIdx))
                return false; // The mapper doesn't even publish a score column to attach the metadata to.
            if (mapper.OutputSchema[scoreIdx].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.TrainingLabelValues)?.Type != null)
                return false; // The mapper publishes a score column, and already produces its own slot names.

            return labelNameType is VectorDataViewType vectorType && vectorType.Size == 2;
        }

        private static ISchemaBoundMapper WrapCore<T>(IHostEnvironment env, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
        {
            Contracts.AssertValue(env);
            env.AssertValue(mapper);
            env.AssertValue(trainSchema);
            env.Assert(mapper is ISchemaBoundRowMapper);
            env.Assert(trainSchema.Label.HasValue);
            var labelColumn = trainSchema.Label.Value;

            // Key values from the training schema label, will map to slot names of the score output.
            var type = labelColumn.Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues)?.Type as VectorDataViewType;
            env.AssertValue(type);

            // Wrap the fetching of the metadata as a simple getter.
            ValueGetter<VBuffer<T>> getter = (ref VBuffer<T> value) =>
                labelColumn.GetKeyValues(ref value);

            return MulticlassClassificationScorer.LabelNameBindableMapper.CreateBound<T>(env, (ISchemaBoundRowMapper)mapper, type, getter, AnnotationUtils.Kinds.TrainingLabelValues, CanWrap);
        }

        [BestFriend]
        internal BinaryClassifierScorer(IHostEnvironment env, Arguments args, IDataView data, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
            : base(args, env, data, WrapIfNeeded(env, mapper, trainSchema), trainSchema, RegistrationName, AnnotationUtils.Const.ScoreColumnKind.BinaryClassification,
                Contracts.CheckRef(args, nameof(args)).ThresholdColumn, OutputTypeMatches, GetPredColType)
        {
            Contracts.CheckValue(args, nameof(args));
            Contracts.CheckValue(data, nameof(data));
            Contracts.CheckValue(mapper, nameof(mapper));

            _threshold = args.Threshold;
        }

        private BinaryClassifierScorer(IHostEnvironment env, BinaryClassifierScorer transform, IDataView newSource)
            : base(env, transform, newSource, RegistrationName)
        {
            _threshold = transform._threshold;
        }

        private BinaryClassifierScorer(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, OutputTypeMatches, GetPredColType)
        {
            Contracts.AssertValue(ctx);

            // *** Binary format ***
            // <base info>
            // int: sizeof(float)
            // float: threshold

            int cbFloat = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(cbFloat == sizeof(float));
            _threshold = ctx.Reader.ReadFloat();
        }

        public static BinaryClassifierScorer Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));

            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new BinaryClassifierScorer(h, ctx, input));
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base info>
            // int: sizeof(float)
            // float: threshold

            base.SaveCore(ctx);
            ctx.Writer.Write(sizeof(float));
            ctx.Writer.Write(_threshold);
        }

        private protected override void SaveAsOnnxCore(OnnxContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Assert(Bindable is IBindableCanSaveOnnx);
            Host.Assert(Bindings.InfoCount >= 2);

            if (!ctx.ContainsColumn(DefaultColumnNames.Features))
                return;

            base.SaveAsOnnxCore(ctx);
            int delta = Bindings.DerivedColumnCount;

            Host.Assert(delta == 1);

            string[] outColumnNames = new string[Bindings.InfoCount]; //PredictedLabel, Score, Probability.
            for (int iinfo = 0; iinfo < Bindings.InfoCount; ++iinfo)
                outColumnNames[iinfo] = Bindings.GetColumnName(Bindings.MapIinfoToCol(iinfo));

            string scoreColumn = Bindings.RowMapper.OutputSchema[Bindings.ScoreColumnIndex].Name;

            OnnxNode node;
            string opType = "Binarizer";
            var binarizerOutput = ctx.AddIntermediateVariable(NumberDataViewType.Single, "BinarizerOutput", false);
            node = ctx.CreateNode(opType, ctx.GetVariableName(scoreColumn), binarizerOutput, ctx.GetNodeName(opType));
            node.AddAttribute("threshold", _threshold);

            string comparisonOutput = binarizerOutput;
            if (Bindings.PredColType is KeyDataViewType)
            {
                var one = ctx.AddInitializer(1.0f, "one");
                var addOutput = ctx.AddIntermediateVariable(NumberDataViewType.Single, "Add", false);
                opType = "Add";
                ctx.CreateNode(opType, new[] { binarizerOutput, one }, new[] { addOutput }, ctx.GetNodeName(opType), "");
                comparisonOutput = addOutput;
            }

            opType = "Cast";
            node = ctx.CreateNode(opType, comparisonOutput, ctx.GetVariableName(outColumnNames[0]), ctx.GetNodeName(opType), "");
            var predictedLabelCol = OutputSchema.GetColumnOrNull(outColumnNames[0]);
            Host.Assert(predictedLabelCol.HasValue);
            node.AddAttribute("to", predictedLabelCol.Value.Type.RawType);
        }

        private protected override IDataTransform ApplyToDataCore(IHostEnvironment env, IDataView newSource)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(newSource, nameof(newSource));

            return new BinaryClassifierScorer(env, this, newSource);
        }

        protected override Delegate GetPredictedLabelGetter(DataViewRow output, out Delegate scoreGetter)
        {
            Host.AssertValue(output);
            Host.Assert(output.Schema == Bindings.RowMapper.OutputSchema);
            Host.Assert(output.IsColumnActive(output.Schema[Bindings.ScoreColumnIndex]));

            var scoreColumn = output.Schema[Bindings.ScoreColumnIndex];
            ValueGetter<float> mapperScoreGetter = output.GetGetter<float>(scoreColumn);

            long cachedPosition = -1;
            float score = 0;

            ValueGetter<float> scoreFn =
                (ref float dst) =>
                {
                    EnsureCachedPosition(ref cachedPosition, ref score, output, mapperScoreGetter);
                    dst = score;
                };
            scoreGetter = scoreFn;

            if (Bindings.PredColType is KeyDataViewType)
            {
                ValueGetter<uint> predFnAsKey =
                    (ref uint dst) =>
                    {
                        EnsureCachedPosition(ref cachedPosition, ref score, output, mapperScoreGetter);
                        GetPredictedLabelCoreAsKey(score, ref dst);
                    };
                return predFnAsKey;
            }

            ValueGetter<bool> predFn =
                (ref bool dst) =>
                {
                    EnsureCachedPosition(ref cachedPosition, ref score, output, mapperScoreGetter);
                    GetPredictedLabelCore(score, ref dst);
                };
            return predFn;
        }

        private void GetPredictedLabelCore(float score, ref bool value)
        {
            //Behavior for NA values is undefined.
            value = score > _threshold;
        }

        private void GetPredictedLabelCoreAsKey(float score, ref uint value)
        {
            value = (uint)(score > _threshold ? 2 : score <= _threshold ? 1 : 0);
        }

        private protected override JToken PredictedLabelPfa(string[] mapperOutputs)
        {
            Contracts.CheckParam(Utils.Size(mapperOutputs) >= 1, nameof(mapperOutputs));

            var scoreToken = mapperOutputs[0];
            JToken trueVal = 1;
            JToken falseVal = 0;
            JToken nullVal = -1;

            if (!(Bindings.PredColType is KeyDataViewType))
            {
                trueVal = true;
                falseVal = nullVal = false; // Let's pretend those pesky nulls are not there.
            }
            return PfaUtils.If(PfaUtils.Call(">", scoreToken, _threshold), trueVal,
                PfaUtils.If(PfaUtils.Call("<=", scoreToken, _threshold), falseVal, nullVal));
        }

        private static DataViewType GetPredColType(DataViewType scoreType, ISchemaBoundRowMapper mapper)
        {
            var labelNameBindableMapper = mapper.Bindable as MulticlassClassificationScorer.LabelNameBindableMapper;
            if (labelNameBindableMapper == null)
                return BooleanDataViewType.Instance;
            return new KeyDataViewType(typeof(uint), labelNameBindableMapper.Type.Size);
        }

        private static bool OutputTypeMatches(DataViewType scoreType)
            => scoreType == NumberDataViewType.Single;
    }
}
