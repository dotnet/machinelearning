// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.FastTree.Internal;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;

[assembly: LoadableClass(RegressionGamTrainer.Summary,
    typeof(RegressionGamTrainer), typeof(RegressionGamTrainer.Arguments),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    RegressionGamTrainer.UserNameValue,
    RegressionGamTrainer.LoadNameValue,
    RegressionGamTrainer.ShortName, DocName = "trainer/GAM.md")]

[assembly: LoadableClass(typeof(RegressionGamPredictor), null, typeof(SignatureLoadModel),
    "GAM Regression Predictor",
    RegressionGamPredictor.LoaderSignature)]

namespace Microsoft.ML.Runtime.FastTree
{
    public sealed class RegressionGamTrainer :
    GamTrainerBase<RegressionGamTrainer.Arguments, RegressionGamPredictor>
    {
        public partial class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Metric for pruning. (For regression, 1: L1, 2:L2; default L2)", ShortName = "pmetric")]
            [TGUI(Description = "Metric for pruning. (For regression, 1: L1, 2:L2; default L2")]
            public int PruningMetrics = 2;
        }

        internal const string LoadNameValue = "RegressionGamTrainer";
        internal const string UserNameValue = "Generalized Additive Model for Regression";
        internal const string ShortName = "gamr";

        public override PredictionKind PredictionKind => PredictionKind.Regression;

        public RegressionGamTrainer(IHostEnvironment env, Arguments args)
            : base(env, args) { }

        internal override void CheckLabel(RoleMappedData data)
        {
            data.CheckRegressionLabel();
        }

        public override RegressionGamPredictor Train(TrainContext context)
        {
            TrainBase(context);
            return new RegressionGamPredictor(Host, InputLength, TrainSet, MeanEffect, BinEffects, FeatureMap);
        }

        protected override ObjectiveFunctionBase CreateObjectiveFunction()
        {
            return new FastTreeRegressionTrainer.ObjectiveImpl(TrainSet, Args);
        }

        protected override void DefinePruningTest()
        {
            var validTest = new RegressionTest(ValidSetScore, Args.PruningMetrics);
            // Because we specify pruning metrics as L2 by default, the results array will have 1 value
            PruningLossIndex = 0;
            PruningTest = new TestHistory(validTest, PruningLossIndex);
        }
    }

    public class RegressionGamPredictor : GamPredictorBase
    {
        public const string LoaderSignature = "RegressionGamPredictor";
        public override PredictionKind PredictionKind => PredictionKind.Regression;

        public RegressionGamPredictor(IHostEnvironment env, int inputLength, Dataset trainset,
            double meanEffect, double[][] binEffects, int[] featureMap)
            : base(env, LoaderSignature, inputLength, trainset, meanEffect, binEffects, featureMap) { }

        private RegressionGamPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx) { }

        public static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "GAM REGP",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public static RegressionGamPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new RegressionGamPredictor(env, ctx);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            base.Save(ctx);
        }
    }
}
