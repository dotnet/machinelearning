// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(typeof(Voting), null, typeof(SignatureCombiner), Voting.UserName, Voting.LoadName)]
[assembly: LoadableClass(typeof(Voting), null, typeof(SignatureLoadModel), Voting.UserName, Voting.LoaderSignature)]

namespace Microsoft.ML.Runtime.Ensemble.OutputCombiners
{
    public sealed class Voting : IBinaryOutputCombiner, ICanSaveModel
    {
        private readonly IHost _host;
        public const string UserName = "Voting";
        public const string LoadName = "Voting";
        public const string LoaderSignature = "VotingCombiner";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "VOT COMB",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(Voting).Assembly.FullName);
        }

        public Voting(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);
        }

        private Voting(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.AssertValue(env);
            _host = env.Register(LoaderSignature);
            _host.AssertValue(ctx);

            // *** Binary format ***
            // int: sizeof(Single)
            int cbFloat = ctx.Reader.ReadInt32();
            _host.CheckDecode(cbFloat == sizeof(Single));
        }

        public static Voting Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new Voting(env, ctx);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Single)
            ctx.Writer.Write(sizeof(Single));
        }

        public Combiner<Single> GetCombiner()
        {
            return CombineCore;
        }

        private void CombineCore(ref Single dst, Single[] src, Single[] weights)
        {
            _host.AssertNonEmpty(src);
            _host.Assert(weights == null || Utils.Size(weights) == Utils.Size(src));

            int len = Utils.Size(src);
            int pos = 0;
            int neg = 0;
            for (int i = 0; i < src.Length; i++)
            {
                var v = src[i];
                if (v > 0)
                    pos++;
                else if (v <= 0)
                    neg++;
            }
            dst = (Single)(pos - neg) / (pos + neg);
        }
    }
}
