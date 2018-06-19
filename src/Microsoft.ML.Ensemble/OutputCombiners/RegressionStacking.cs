// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(typeof(RegressionStacking), typeof(RegressionStacking.Arguments), typeof(SignatureCombiner),
    Stacking.UserName, RegressionStacking.LoadName)]
[assembly: LoadableClass(typeof(RegressionStacking), null, typeof(SignatureLoadModel),
    Stacking.UserName, RegressionStacking.LoadName)]

namespace Microsoft.ML.Runtime.Ensemble.OutputCombiners
{
    using TScalarPredictor = IPredictorProducing<Single>;
    public sealed class RegressionStacking : BaseScalarStacking<SignatureRegressorTrainer>, IRegressionOutputCombiner, ICanSaveModel
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
                loaderSignature: LoaderSignature);
        }

        [TlcModule.Component(Name = LoadName, FriendlyName = Stacking.UserName)]
        public sealed class Arguments : ArgumentsBase, ISupportRegressionOutputCombinerFactory
        {
            public Arguments()
            {
                BasePredictorType = new SubComponent<ITrainer<RoleMappedData, TScalarPredictor>, SignatureRegressorTrainer>("FastTreeRegression");
            }

            public IRegressionOutputCombiner CreateComponent(IHostEnvironment env) => new RegressionStacking(env, this);
        }

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
