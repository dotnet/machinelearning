// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Numeric;

[assembly: LoadableClass(typeof(MultiVoting), null, typeof(SignatureCombiner), Voting.UserName, MultiVoting.LoadName)]
[assembly: LoadableClass(typeof(MultiVoting), null, typeof(SignatureLoadModel), Voting.UserName, MultiVoting.LoaderSignature)]

namespace Microsoft.ML.Runtime.Ensemble.OutputCombiners
{
    // REVIEW shonk: Why is MultiVoting based on BaseMultiCombiner? Normalizing the model outputs
    // is senseless, so the base adds no real functionality.
    public sealed class MultiVoting : BaseMultiCombiner, ICanSaveModel
    {
        public const string LoadName = "MultiVoting";
        public const string LoaderSignature = "MultiVotingCombiner";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MVOTCOMB",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public MultiVoting(IHostEnvironment env)
            : base(env, LoaderSignature, new Arguments() { Normalize = false })
        {
            Host.Assert(!Normalize);
        }

        private MultiVoting(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
            Host.CheckDecode(!Normalize);
        }

        public static MultiVoting Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new MultiVoting(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            Contracts.Assert(!Normalize);
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        public override Combiner<VBuffer<Single>> GetCombiner()
        {
            return CombineCore;
        }

        private void CombineCore(ref VBuffer<Single> dst, VBuffer<Single>[] src, Single[] weights = null)
        {
            Host.AssertNonEmpty(src);
            Host.Assert(weights == null || Utils.Size(weights) == Utils.Size(src));

            int count = Utils.Size(src);
            if (count == 0)
            {
                dst = new VBuffer<Single>(0, dst.Values, dst.Indices);
                return;
            }

            int len = GetClassCount(src);
            var values = dst.Values;
            if (Utils.Size(values) < len)
                values = new Single[len];
            else
                Array.Clear(values, 0, len);

            int voteCount = 0;
            for (int i = 0; i < count; i++)
            {
                int index = VectorUtils.ArgMax(ref src[i]);
                if (index >= 0)
                {
                    values[index]++;
                    voteCount++;
                }
            }

            // Normalize by dividing by the number of votes.
            for (int i = 0; i < len; i++)
                values[i] /= voteCount;

            // Set the output to values.
            dst = new VBuffer<Single>(len, values, dst.Indices);
        }
    }
}
