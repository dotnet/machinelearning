// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(typeof(Median), null, typeof(SignatureCombiner), Median.UserName, Median.LoadName)]
[assembly: LoadableClass(typeof(Median), null, typeof(SignatureLoadModel), Median.UserName, Median.LoaderSignature)]

namespace Microsoft.ML.Runtime.Ensemble.OutputCombiners
{
    /// <summary>
    /// Generic interface for combining outputs of multiple models
    /// </summary>
    public sealed class Median : IRegressionOutputCombiner, IBinaryOutputCombiner, ICanSaveModel
    {
        private readonly IHost _host;
        public const string UserName = "Median";
        public const string LoadName = "Median";
        public const string LoaderSignature = "MedianCombiner";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MEDICOMB",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(Median).Assembly.FullName);
        }

        public Median(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);
        }

        private Median(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.AssertValue(env);
            _host = env.Register(LoaderSignature);

            // *** Binary format ***
            // int: sizeof(Single)
            int cbFloat = ctx.Reader.ReadInt32();
            _host.CheckDecode(cbFloat == sizeof(Single));
        }

        public static Median Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new Median(env, ctx);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            ctx.Writer.Write(sizeof(Single));
        }

        public Combiner<Single> GetCombiner()
        {
            return CombineCore;
        }

        private void CombineCore(ref Single dst, Single[] src, Single[] weights)
        {
            // REVIEW: This mutates "src". We need to ensure that the documentation of
            // combiners makes it clear that combiners are allowed to do this. Note that "normalization"
            // in the multi-class case also mutates.
            _host.AssertNonEmpty(src);
            _host.Assert(weights == null || Utils.Size(weights) == Utils.Size(src));
            dst = MathUtils.GetMedianInPlace(src, src.Length);
        }
    }
}
