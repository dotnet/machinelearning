// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.Ensemble;

[assembly: LoadableClass(typeof(MultiAverage), typeof(MultiAverage.Options), typeof(SignatureCombiner),
    Average.UserName, MultiAverage.LoadName)]
[assembly: LoadableClass(typeof(MultiAverage), null, typeof(SignatureLoadModel), Average.UserName,
    MultiAverage.LoadName, MultiAverage.LoaderSignature)]

namespace Microsoft.ML.Trainers.Ensemble
{
    internal sealed class MultiAverage : BaseMultiAverager
    {
        public const string LoadName = "MultiAverage";
        public const string LoaderSignature = "MultiAverageCombiner";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MAVGCOMB",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MultiAverage).Assembly.FullName);
        }

        [TlcModule.Component(Name = LoadName, FriendlyName = Average.UserName)]
        public sealed class Options : OptionsBase, ISupportMulticlassOutputCombinerFactory
        {
            public IMulticlassOutputCombiner CreateComponent(IHostEnvironment env) => new MultiAverage(env, this);
        }

        public MultiAverage(IHostEnvironment env, Options options)
            : base(env, LoaderSignature, options)
        {
        }

        private MultiAverage(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
        }

        public static MultiAverage Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new MultiAverage(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        public override Combiner<VBuffer<Single>> GetCombiner()
        {
            // Force the weights to null.
            return
                (ref VBuffer<Single> dst, VBuffer<Single>[] src, Single[] weights) =>
                    CombineCore(ref dst, src, null);
        }
    }
}
