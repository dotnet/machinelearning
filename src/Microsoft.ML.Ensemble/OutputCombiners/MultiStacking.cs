// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(typeof(MultiStacking), typeof(MultiStacking.Arguments), typeof(SignatureCombiner),
   Stacking.UserName, MultiStacking.LoadName)]

[assembly: LoadableClass(typeof(MultiStacking), null, typeof(SignatureLoadModel),
    Stacking.UserName, MultiStacking.LoaderSignature)]

namespace Microsoft.ML.Runtime.Ensemble.OutputCombiners
{
    using TVectorPredictor = IPredictorProducing<VBuffer<Single>>;
    public sealed class MultiStacking : BaseStacking<VBuffer<Single>>, ICanSaveModel, IMultiClassOutputCombiner
    {
        public const string LoadName = "MultiStacking";
        public const string LoaderSignature = "MultiStackingCombiner";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MSTACK C",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        [TlcModule.Component(Name = LoadName, FriendlyName = Stacking.UserName)]
        public sealed class Arguments : ArgumentsBase, ISupportMulticlassOutputCombinerFactory
        {
            [Argument(ArgumentType.Multiple, HelpText = "Base predictor for meta learning", ShortName = "bp", SortOrder = 50,
                Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, SignatureType = typeof(SignatureMultiClassClassifierTrainer))]
            [TGUI(Label = "Base predictor")]
            public IComponentFactory<ITrainer<TVectorPredictor>> BasePredictorType;

            internal override IComponentFactory<ITrainer<TVectorPredictor>> GetPredictorFactory() => BasePredictorType;

            public IMultiClassOutputCombiner CreateComponent(IHostEnvironment env) => new MultiStacking(env, this);

            public Arguments()
            {
                // REVIEW: Perhaps we can have a better non-parametetric learner.
                BasePredictorType = ComponentFactoryUtils.CreateFromFunction(
                    env => new Ova(env, new Ova.Arguments()
                    {
                        PredictorType = ComponentFactoryUtils.CreateFromFunction(
                            e => new FastTreeBinaryClassificationTrainer(e, new FastTreeBinaryClassificationTrainer.Arguments()))
                    }));
            }
        }

        public MultiStacking(IHostEnvironment env, Arguments args)
            : base(env, LoaderSignature, args)
        {
        }

        private MultiStacking(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
        }

        public static MultiStacking Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new MultiStacking(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        protected override void FillFeatureBuffer(VBuffer<Single>[] src, ref VBuffer<Single> dst)
        {
            Contracts.AssertNonEmpty(src);

            // REVIEW: Would there be any value in ever making dst sparse?
            int len = 0;
            for (int i = 0; i < src.Length; i++)
                len += src[i].Length;

            var values = dst.Values;
            if (Utils.Size(values) < len)
                values = new Single[len];
            dst = new VBuffer<Single>(len, values, dst.Indices);

            int iv = 0;
            for (int i = 0; i < src.Length; i++)
            {
                src[i].CopyTo(values, iv);
                iv += src[i].Length;
                Contracts.Assert(iv <= len);
            }
            Contracts.Assert(iv == len);
        }
    }
}
