// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model.Onnx;
using Newtonsoft.Json.Linq;
using Microsoft.ML.Runtime.Model.Pfa;
using System.Collections.Generic;

[assembly: LoadableClass(typeof(BinaryClassifierScorer), typeof(BinaryClassifierScorer.Arguments), typeof(SignatureDataScorer),
    "Binary Classifier Scorer", "BinaryClassifierScorer", "BinaryClassifier", "Binary",
    "bin", MetadataUtils.Const.ScoreColumnKind.BinaryClassification)]

[assembly: LoadableClass(typeof(BinaryClassifierScorer), null, typeof(SignatureLoadDataTransform),
    "Binary Classifier Scorer", BinaryClassifierScorer.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class BinaryClassifierScorer : PredictedLabelScorerBase, ITransformCanSaveOnnx
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

        private readonly Float _threshold;

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
            var keyType = trainSchema.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, trainSchema.Label.Index);
            if (keyType == null || !CanWrap(mapper, keyType))
                return mapper;

            // Great!! All checks pass.
            return Utils.MarshalInvoke(WrapCore<int>, keyType.ItemType.RawType, env, mapper, trainSchema);
        }

        /// <summary>
        /// This is a utility method used to determine whether <see cref="MultiClassClassifierScorer.LabelNameBindableMapper"/>
        /// can or should be used to wrap <paramref name="mapper"/>. This will not throw, since the
        /// desired behavior in the event that it cannot be wrapped, is to just back off to the original
        /// "unwrapped" bound mapper.
        /// </summary>
        /// <param name="mapper">The mapper we are seeing if we can wrap</param>
        /// <param name="labelNameType">The type of the label names from the metadata (either
        /// originating from the key value metadata of the training label column, or deserialized
        /// from the model of a bindable mapper)</param>
        /// <returns>Whether we can call <see cref="MultiClassClassifierScorer.LabelNameBindableMapper.CreateBound{T}"/> with
        /// this mapper and expect it to succeed</returns>
        private static bool CanWrap(ISchemaBoundMapper mapper, ColumnType labelNameType)
        {
            Contracts.AssertValue(mapper);
            Contracts.AssertValue(labelNameType);

            ISchemaBoundRowMapper rowMapper = mapper as ISchemaBoundRowMapper;
            if (rowMapper == null)
                return false; // We could cover this case, but it is of no practical worth as far as I see, so I decline to do so.

            ISchema outSchema = mapper.OutputSchema;
            int scoreIdx;
            if (!outSchema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out scoreIdx))
                return false; // The mapper doesn't even publish a score column to attach the metadata to.
            if (outSchema.GetMetadataTypeOrNull(MetadataUtils.Kinds.TrainingLabelValues, scoreIdx) != null)
                return false; // The mapper publishes a score column, and already produces its own slot names.

            return labelNameType.IsVector && labelNameType.VectorSize == 2;
        }

        private static ISchemaBoundMapper WrapCore<T>(IHostEnvironment env, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
        {
            Contracts.AssertValue(env);
            env.AssertValue(mapper);
            env.AssertValue(trainSchema);
            env.Assert(mapper is ISchemaBoundRowMapper);

            // Key values from the training schema label, will map to slot names of the score output.
            var type = trainSchema.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, trainSchema.Label.Index);
            env.AssertValue(type);
            env.Assert(type.IsVector);

            // Wrap the fetching of the metadata as a simple getter.
            ValueGetter<VBuffer<T>> getter =
                (ref VBuffer<T> value) =>
                {
                    trainSchema.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues,
                        trainSchema.Label.Index, ref value);
                };

            return MultiClassClassifierScorer.LabelNameBindableMapper.CreateBound<T>(env, (ISchemaBoundRowMapper)mapper, type.AsVector, getter, MetadataUtils.Kinds.TrainingLabelValues, CanWrap);
        }

        public BinaryClassifierScorer(IHostEnvironment env, Arguments args, IDataView data, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
            : base(args, env, data, WrapIfNeeded(env, mapper, trainSchema), trainSchema, RegistrationName, MetadataUtils.Const.ScoreColumnKind.BinaryClassification,
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
            // int: sizeof(Float)
            // Float: threshold

            int cbFloat = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(cbFloat == sizeof(Float));
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

        protected override void SaveCore(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base info>
            // int: sizeof(Float)
            // Float: threshold

            base.SaveCore(ctx);
            ctx.Writer.Write(sizeof(Float));
            ctx.Writer.Write(_threshold);
        }

        public override void SaveAsOnnx(OnnxContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Assert(Bindable is IBindableCanSaveOnnx);

            if (!ctx.ContainsColumn(DefaultColumnNames.Features))
                return;

            base.SaveAsOnnx(ctx);
            int delta = Bindings.DerivedColumnCount;

            Host.Assert(delta == 1);

            string[] outColumnNames = new string[Bindings.InfoCount]; //PredictedLabel, Score, Probability.
            for (int iinfo = 0; iinfo < Bindings.InfoCount; ++iinfo)
                outColumnNames[iinfo] = Bindings.GetColumnName(Bindings.MapIinfoToCol(iinfo));

            //Check if "Probability" column was generated by the base class, only then
            //label can be predicted.
            if (Bindings.InfoCount >= 3 && ctx.ContainsColumn(outColumnNames[2]))
            {
                string opType = "Binarizer";
                var node = ctx.CreateNode(opType, new[] { ctx.GetVariableName(outColumnNames[2]) },
                    new[] { ctx.GetVariableName(outColumnNames[0]) }, ctx.GetNodeName(opType));
                node.AddAttribute("threshold", 0.5);
            }
        }

        public override IDataTransform ApplyToData(IHostEnvironment env, IDataView newSource)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(newSource, nameof(newSource));

            return new BinaryClassifierScorer(env, this, newSource);
        }

        protected override Delegate GetPredictedLabelGetter(IRow output, out Delegate scoreGetter)
        {
            Host.AssertValue(output);
            Host.Assert(output.Schema == Bindings.RowMapper.OutputSchema);
            Host.Assert(output.IsColumnActive(Bindings.ScoreColumnIndex));

            ValueGetter<Float> mapperScoreGetter = output.GetGetter<Float>(Bindings.ScoreColumnIndex);

            long cachedPosition = -1;
            Float score = 0;

            ValueGetter<Float> scoreFn =
                (ref Float dst) =>
                {
                    EnsureCachedPosition(ref cachedPosition, ref score, output, mapperScoreGetter);
                    dst = score;
                };
            scoreGetter = scoreFn;

            if (Bindings.PredColType.IsKey)
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

        private void GetPredictedLabelCore(Float score, ref bool value)
        {
            //Behavior for NA values is undefined.
            value = score > _threshold;
        }

        private void GetPredictedLabelCoreAsKey(Float score, ref uint value)
        {
            value = (uint)(score > _threshold ? 2 : score <= _threshold ? 1 : 0);
        }

        protected override JToken PredictedLabelPfa(string[] mapperOutputs)
        {
            Contracts.CheckParam(Utils.Size(mapperOutputs) >= 1, nameof(mapperOutputs));

            var scoreToken = mapperOutputs[0];
            JToken trueVal = 1;
            JToken falseVal = 0;
            JToken nullVal = -1;

            if (!Bindings.PredColType.IsKey)
            {
                trueVal = true;
                falseVal = nullVal = false; // Let's pretend those pesky nulls are not there.
            }
            return PfaUtils.If(PfaUtils.Call(">", scoreToken, _threshold), trueVal,
                PfaUtils.If(PfaUtils.Call("<=", scoreToken, _threshold), falseVal, nullVal));
        }

        private static ColumnType GetPredColType(ColumnType scoreType, ISchemaBoundRowMapper mapper)
        {
            var labelNameBindableMapper = mapper.Bindable as MultiClassClassifierScorer.LabelNameBindableMapper;
            if (labelNameBindableMapper == null)
                return BoolType.Instance;
            return new KeyType(DataKind.U4, 0, labelNameBindableMapper.Type.VectorSize);
        }

        private static bool OutputTypeMatches(ColumnType scoreType)
        {
            return scoreType == NumberType.Float;
        }
    }
}
