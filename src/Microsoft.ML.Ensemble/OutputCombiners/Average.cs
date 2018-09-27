// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(typeof(Average), null, typeof(SignatureCombiner), Average.UserName)]
[assembly: LoadableClass(typeof(Average), null, typeof(SignatureLoadModel), Average.UserName, Average.LoaderSignature)]

namespace Microsoft.ML.Runtime.Ensemble.OutputCombiners
{
    public sealed class Average : BaseAverager, ICanSaveModel, IRegressionOutputCombiner
    {
        public const string UserName = "Average";
        public const string LoadName = "Average";
        public const string LoaderSignature = "AverageCombiner";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "AVG COMB",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(Average).Assembly.FullName);
        }

        public Average(IHostEnvironment env)
            : base(env, LoaderSignature)
        {
        }

        private Average(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
        }

        public static Average Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new Average(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        public override Combiner<Single> GetCombiner()
        {
            // Force the weights to null.
            return(ref Single dst, Single[] src, Single[] weights) =>
                    CombineCore(ref dst, src, null);
        }
    }
}
