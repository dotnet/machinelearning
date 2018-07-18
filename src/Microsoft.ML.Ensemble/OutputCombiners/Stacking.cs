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

[assembly: LoadableClass(typeof(Stacking), typeof(Stacking.Arguments), typeof(SignatureCombiner), Stacking.UserName, Stacking.LoadName)]
[assembly: LoadableClass(typeof(Stacking), null, typeof(SignatureLoadModel), Stacking.UserName, Stacking.LoaderSignature)]

namespace Microsoft.ML.Runtime.Ensemble.OutputCombiners
{
    using TScalarPredictor = IPredictorProducing<Single>;
    public sealed class Stacking : BaseScalarStacking<SignatureBinaryClassifierTrainer>, IBinaryOutputCombiner, ICanSaveModel
    {
        public const string UserName = "Stacking";
        public const string LoadName = "Stacking";
        public const string LoaderSignature = "StackingCombiner";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: " STACK C",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        [TlcModule.Component(Name = LoadName, FriendlyName = UserName)]
        public sealed class Arguments : ArgumentsBase, ISupportBinaryOutputCombinerFactory
        {
            public Arguments()
            {
                BasePredictorType = new SubComponent<ITrainer<TScalarPredictor>, SignatureBinaryClassifierTrainer>("FastTreeBinaryClassification");
            }

            public IBinaryOutputCombiner CreateComponent(IHostEnvironment env) => new Stacking(env, this);
        }

        public Stacking(IHostEnvironment env, Arguments args)
            : base(env, LoaderSignature, args)
        {
        }

        private Stacking(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
        }

        public static Stacking Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new Stacking(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }
    }
}
