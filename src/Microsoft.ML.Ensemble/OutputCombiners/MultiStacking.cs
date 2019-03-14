// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.Ensemble;

[assembly: LoadableClass(typeof(MultiStacking), typeof(MultiStacking.Arguments), typeof(SignatureCombiner),
   Stacking.UserName, MultiStacking.LoadName)]

[assembly: LoadableClass(typeof(MultiStacking), null, typeof(SignatureLoadModel),
    Stacking.UserName, MultiStacking.LoaderSignature)]

namespace Microsoft.ML.Trainers.Ensemble
{
    using TVectorPredictor = IPredictorProducing<VBuffer<Single>>;
    internal sealed class MultiStacking : BaseStacking<VBuffer<Single>>, IMulticlassOutputCombiner
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
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MultiStacking).Assembly.FullName);
        }

#pragma warning disable CS0649 // The fields will still be set via the reflection driven mechanisms.
        [TlcModule.Component(Name = LoadName, FriendlyName = Stacking.UserName)]
        public sealed class Arguments : ArgumentsBase, ISupportMulticlassOutputCombinerFactory
        {
            // REVIEW: If we make this public again it should be an *estimator* of this type of predictor, rather than the (deprecated) ITrainer.
            [Argument(ArgumentType.Multiple, HelpText = "Base predictor for meta learning", ShortName = "bp", SortOrder = 50,
                Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, SignatureType = typeof(SignatureMulticlassClassifierTrainer))]
            [TGUI(Label = "Base predictor")]
            public IComponentFactory<ITrainer<TVectorPredictor>> BasePredictorType;

            internal override IComponentFactory<ITrainer<TVectorPredictor>> GetPredictorFactory() => BasePredictorType;

            public IMulticlassOutputCombiner CreateComponent(IHostEnvironment env) => new MultiStacking(env, this);
        }
#pragma warning restore CS0649

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

            var editor = VBufferEditor.Create(ref dst, len);

            int iv = 0;
            for (int i = 0; i < src.Length; i++)
            {
                src[i].CopyTo(editor.Values, iv);
                iv += src[i].Length;
                Contracts.Assert(iv <= len);
            }
            Contracts.Assert(iv == len);
            dst = editor.Commit();
        }
    }
}
