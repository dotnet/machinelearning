// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Pfa;
using Microsoft.ML.Runtime.Numeric;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(ClusteringScorer), typeof(ClusteringScorer.Arguments), typeof(SignatureDataScorer),
    "Clustering Scorer", "ClusteringScorer", MetadataUtils.Const.ScoreColumnKind.Clustering)]

[assembly: LoadableClass(typeof(ClusteringScorer), null, typeof(SignatureLoadDataTransform),
    "Clustering Scorer", ClusteringScorer.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class ClusteringScorer : PredictedLabelScorerBase
    {
        public sealed class Arguments : ScorerArgumentsBase
        {
        }

        public const string LoaderSignature = "ClusteringScoreTrans";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CLSTRSCR",
                //verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // ISchemaBindableMapper
                verWrittenCur: 0x00010003, // ISchemaBindableMapper update
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010003,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ClusteringScorer).Assembly.FullName);
        }

        private const string RegistrationName = "ClusteringScore";

        public ClusteringScorer(IHostEnvironment env, Arguments args, IDataView data, ISchemaBoundMapper mapper, RoleMappedSchema trainSchema)
            : base(args, env, data, mapper, trainSchema, RegistrationName, MetadataUtils.Const.ScoreColumnKind.Clustering,
                MetadataUtils.Const.ScoreValueKind.Score, OutputTypeMatches, GetPredColType)
        {
        }

        private ClusteringScorer(IHostEnvironment env, ClusteringScorer transform, IDataView newSource)
            : base(env, transform, newSource, RegistrationName)
        {
        }

        private ClusteringScorer(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, OutputTypeMatches, GetPredColType)
        {
            // *** Binary format ***
            // <base info>
        }

        public static ClusteringScorer Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());

            return h.Apply("Loading Model", ch => new ClusteringScorer(h, ctx, input));
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base info>

            base.SaveCore(ctx);
        }

        public override IDataTransform ApplyToData(IHostEnvironment env, IDataView newSource)
        {
            Contracts.CheckValue(env, nameof(env));
            Contracts.CheckValue(newSource, nameof(newSource));

            return new ClusteringScorer(env, this, newSource);
        }

        protected override Delegate GetPredictedLabelGetter(IRow output, out Delegate scoreGetter)
        {
            Contracts.AssertValue(output);
            Contracts.Assert(output.Schema == Bindings.RowMapper.OutputSchema);
            Contracts.Assert(output.IsColumnActive(Bindings.ScoreColumnIndex));

            ValueGetter<VBuffer<Float>> mapperScoreGetter = output.GetGetter<VBuffer<Float>>(Bindings.ScoreColumnIndex);

            long cachedPosition = -1;
            VBuffer<Float> score = default(VBuffer<Float>);
            int scoreLength = Bindings.PredColType.KeyCount;

            ValueGetter<uint> predFn =
                (ref uint dst) =>
                {
                    EnsureCachedPosition(ref cachedPosition, ref score, output, mapperScoreGetter);
                    Contracts.Check(score.Length == scoreLength);
                    int index = VectorUtils.ArgMin(ref score);
                    if (index < 0)
                        dst = 0;
                    else
                        dst = (uint)index + 1;
                };
            ValueGetter<VBuffer<Float>> scoreFn =
                (ref VBuffer<Float> dst) =>
                {
                    EnsureCachedPosition(ref cachedPosition, ref score, output, mapperScoreGetter);
                    Contracts.Check(score.Length == scoreLength);
                    score.CopyTo(ref dst);
                };

            scoreGetter = scoreFn;
            return predFn;
        }

        protected override JToken PredictedLabelPfa(string[] mapperOutputs)
        {
            Contracts.Assert(Utils.Size(mapperOutputs) == 1);
            return PfaUtils.Call("a.argmax", mapperOutputs[0]);
        }

        private static ColumnType GetPredColType(ColumnType scoreType, ISchemaBoundRowMapper mapper)
        {
            return new KeyType(DataKind.U4, 0, scoreType.VectorSize);
        }

        private static bool OutputTypeMatches(ColumnType scoreType)
        {
            return scoreType.IsKnownSizeVector && scoreType.ItemType == NumberType.Float;
        }
    }
}
