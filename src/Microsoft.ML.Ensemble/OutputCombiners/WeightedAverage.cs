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

[assembly: LoadableClass(typeof(WeightedAverage), typeof(WeightedAverage.Options), typeof(SignatureCombiner),
    WeightedAverage.UserName, WeightedAverage.LoadName)]

[assembly: LoadableClass(typeof(WeightedAverage), null, typeof(SignatureLoadModel),
     WeightedAverage.UserName, WeightedAverage.LoaderSignature)]

namespace Microsoft.ML.Trainers.Ensemble
{
    internal sealed class WeightedAverage : BaseAverager, IWeightedAverager
    {
        public const string UserName = "Weighted Average";
        public const string LoadName = "WeightedAverage";
        public const string LoaderSignature = "WeightedAverageCombiner";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "WAVGCOMB",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(WeightedAverage).Assembly.FullName);
        }

        [TlcModule.Component(Name = LoadName, FriendlyName = UserName)]
        public sealed class Options : ISupportBinaryOutputCombinerFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The metric type to be used to find the weights for each model", ShortName = "wn", SortOrder = 50)]
            [TGUI(Label = "Weightage Name", Description = "The weights are calculated according to the selected metric")]
            public WeightageKind WeightageName = WeightageKind.Auc;

            public IBinaryOutputCombiner CreateComponent(IHostEnvironment env) => new WeightedAverage(env, this);
        }

        private WeightageKind _weightageKind;

        public string WeightageMetricName { get { return _weightageKind.ToString(); } }

        public WeightedAverage(IHostEnvironment env, Options options)
            : base(env, LoaderSignature)
        {
            _weightageKind = options.WeightageName;
            Host.CheckUserArg(Enum.IsDefined(typeof(WeightageKind), _weightageKind), nameof(options.WeightageName));
        }

        private WeightedAverage(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
            // *** Binary format ***
            // int: _weightageKind
            _weightageKind = (WeightageKind)ctx.Reader.ReadInt32();
            Host.CheckDecode(Enum.IsDefined(typeof(WeightageKind), _weightageKind));
        }

        public static WeightedAverage Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new WeightedAverage(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: _weightageKind

            Contracts.Assert(Enum.IsDefined(typeof(WeightageKind), _weightageKind));
            ctx.Writer.Write((int)_weightageKind);
        }

        public override Combiner<Single> GetCombiner()
        {
            return CombineCore;
        }
    }

    // These values are serialized, so should not be changed.
    internal enum WeightageKind
    {
        [TGUI(Label = BinaryClassifierEvaluator.Accuracy)]
        Accuracy = 0,
        [TGUI(Label = BinaryClassifierEvaluator.Auc)]
        Auc = 1,
        [TGUI(Label = BinaryClassifierEvaluator.PosPrecName)]
        PosPrecision = 2,
        [TGUI(Label = BinaryClassifierEvaluator.PosRecallName)]
        PosRecall = 3,
        [TGUI(Label = BinaryClassifierEvaluator.NegPrecName)]
        NegPrecision = 4,
        [TGUI(Label = BinaryClassifierEvaluator.NegRecallName)]
        NegRecall = 5,
    }

}
