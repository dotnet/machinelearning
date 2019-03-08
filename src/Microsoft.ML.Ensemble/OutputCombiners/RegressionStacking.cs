// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.Ensemble;

[assembly: LoadableClass(typeof(RegressionStacking), typeof(RegressionStacking.Arguments), typeof(SignatureCombiner),
    Stacking.UserName, RegressionStacking.LoadName)]

[assembly: LoadableClass(typeof(RegressionStacking), null, typeof(SignatureLoadModel),
    Stacking.UserName, RegressionStacking.LoaderSignature)]

namespace Microsoft.ML.Trainers.Ensemble
{
    using TScalarPredictor = IPredictorProducing<Single>;

    internal sealed class RegressionStacking : BaseScalarStacking, IRegressionOutputCombiner
    {
        public const string LoadName = "RegressionStacking";
        public const string LoaderSignature = "RegressionStacking";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "RSTACK C",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(RegressionStacking).Assembly.FullName);
        }

#pragma warning disable CS0649 // The fields will still be set via the reflection driven mechanisms.
        [TlcModule.Component(Name = LoadName, FriendlyName = Stacking.UserName)]
        public sealed class Arguments : ArgumentsBase, ISupportRegressionOutputCombinerFactory
        {
            // REVIEW: If we make this public again it should be an *estimator* of this type of predictor, rather than the (deprecated) ITrainer.
            [Argument(ArgumentType.Multiple, HelpText = "Base predictor for meta learning", ShortName = "bp", SortOrder = 50,
                Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, SignatureType = typeof(SignatureRegressorTrainer))]
            [TGUI(Label = "Base predictor")]
            public IComponentFactory<ITrainer<TScalarPredictor>> BasePredictorType;

            internal override IComponentFactory<ITrainer<TScalarPredictor>> GetPredictorFactory() => BasePredictorType;

            public IRegressionOutputCombiner CreateComponent(IHostEnvironment env) => new RegressionStacking(env, this);
        }
#pragma warning restore CS0649

        public RegressionStacking(IHostEnvironment env, Arguments args)
            : base(env, LoaderSignature, args)
        {
        }

        private RegressionStacking(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
        }

        public static RegressionStacking Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new RegressionStacking(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }
    }
}
