// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(typeof(MultiWeightedAverage), typeof(MultiWeightedAverage.Arguments), typeof(SignatureCombiner),
    MultiWeightedAverage.UserName, MultiWeightedAverage.LoadName)]

[assembly: LoadableClass(typeof(MultiWeightedAverage), null, typeof(SignatureLoadModel),
    MultiWeightedAverage.UserName, MultiWeightedAverage.LoaderSignature)]

namespace Microsoft.ML.Runtime.Ensemble.OutputCombiners
{
    /// <summary>
    /// Generic interface for combining outputs of multiple models
    /// </summary>
    public sealed class MultiWeightedAverage : BaseMultiAverager, IWeightedAverager, ICanSaveModel
    {
        public const string UserName = "Multi Weighted Average";
        public const string LoadName = "MultiWeightedAverage";
        public const string LoaderSignature = "MultiWeightedAverageComb";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MWAVCOMB",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MultiWeightedAverage).Assembly.FullName);
        }

        [TlcModule.Component(Name = LoadName, FriendlyName = UserName)]
        public sealed class Arguments : ArgumentsBase, ISupportMulticlassOutputCombinerFactory
        {
            IMultiClassOutputCombiner IComponentFactory<IMultiClassOutputCombiner>.CreateComponent(IHostEnvironment env) => new MultiWeightedAverage(env, this);

            [Argument(ArgumentType.AtMostOnce, HelpText = "The metric type to be used to find the weights for each model", ShortName = "wn", SortOrder = 50)]
            [TGUI(Label = "Metric Name", Description = "The weights are calculated according to the selected metric")]
            public MultiWeightageKind WeightageName = MultiWeightageKind.AccuracyMicroAvg;
        }

        private readonly MultiWeightageKind _weightageKind;
        public string WeightageMetricName { get { return _weightageKind.ToString(); } }

        public MultiWeightedAverage(IHostEnvironment env, Arguments args)
            : base(env, LoaderSignature, args)
        {
            _weightageKind = args.WeightageName;
            Host.CheckUserArg(Enum.IsDefined(typeof(MultiWeightageKind), _weightageKind), nameof(args.WeightageName));
        }

        private MultiWeightedAverage(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
            // *** Binary format ***
            // int: _weightageKind

            _weightageKind = (MultiWeightageKind)ctx.Reader.ReadInt32();
            Host.CheckDecode(Enum.IsDefined(typeof(MultiWeightageKind), _weightageKind));
        }

        public static MultiWeightedAverage Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new MultiWeightedAverage(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
            // *** Binary format ***
            // int: _weightageKind

            Host.Assert(Enum.IsDefined(typeof(MultiWeightageKind), _weightageKind));
            ctx.Writer.Write((int)_weightageKind);
        }

        public override Combiner<VBuffer<Single>> GetCombiner()
        {
            return CombineCore;
        }
    }

    // These values are serialized, so should not be changed.
    public enum MultiWeightageKind
    {
        [TGUI(Label = MultiClassClassifierEvaluator.AccuracyMicro)]
        AccuracyMicroAvg = 0,
        [TGUI(Label = MultiClassClassifierEvaluator.AccuracyMacro)]
        AccuracyMacroAvg = 1
    }
}
