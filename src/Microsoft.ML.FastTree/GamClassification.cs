// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading.Tasks;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.FastTree.Internal;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;

[assembly: LoadableClass(BinaryClassificationGamTrainer.Summary,
    typeof(BinaryClassificationGamTrainer), typeof(BinaryClassificationGamTrainer.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    BinaryClassificationGamTrainer.UserNameValue,
    BinaryClassificationGamTrainer.LoadNameValue,
    BinaryClassificationGamTrainer.ShortName, DocName = "trainer/GAM.md")]

[assembly: LoadableClass(typeof(BinaryClassGamPredictor), null, typeof(SignatureLoadModel),
    "GAM Binary Class Predictor",
    BinaryClassGamPredictor.LoaderSignature)]

namespace Microsoft.ML.Runtime.FastTree
{
    using Float = System.Single;

    public sealed class BinaryClassificationGamTrainer :
    GamTrainerBase<BinaryClassificationGamTrainer.Arguments, IPredictorProducing<Float>>
    {
        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Should we use derivatives optimized for unbalanced sets", ShortName = "us")]
            [TGUI(Label = "Optimize for unbalanced")]
            public bool UnbalancedSets = false;
        }

        internal const string LoadNameValue = "BinaryClassificationGamTrainer";
        internal const string UserNameValue = "Generalized Additive Model for Binary Classification";
        internal const string ShortName = "gam";
        private readonly double _sigmoidParameter;

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;
        private protected override bool NeedCalibration => true;

        public BinaryClassificationGamTrainer(IHostEnvironment env, Arguments args)
            : base(env, args)
        {
            _sigmoidParameter = 1;
        }

        internal override void CheckLabel(RoleMappedData data)
        {
            data.CheckBinaryLabel();
        }

        public static bool[] ConvertTargetsToBool(double[] targets)
        {
            bool[] boolArray = new bool[targets.Length];
            int innerLoopSize = 1 + targets.Length / BlockingThreadPool.NumThreads;
            var actions = new Action[(int)Math.Ceiling(1.0 * targets.Length / innerLoopSize)];
            var actionIndex = 0;
            for (int d = 0; d < targets.Length; d += innerLoopSize)
            {
                var fromDoc = d;
                var toDoc = Math.Min(d + innerLoopSize, targets.Length);
                actions[actionIndex++] = () =>
                {
                    for (int doc = fromDoc; doc < toDoc; doc++)
                        boolArray[doc] = targets[doc] > 0;
                };
            }
            Parallel.Invoke(new ParallelOptions { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, actions);
            return boolArray;
        }

        public override IPredictorProducing<Float> Train(TrainContext context)
        {
            TrainBase(context);
            var predictor = new BinaryClassGamPredictor(Host, InputLength, TrainSet,
                MeanEffect, BinEffects, FeatureMap, FinalResults);
            var calibrator = new PlattCalibrator(Host, -2 * _sigmoidParameter, 0);
            return new CalibratedPredictor(Host, predictor, calibrator);
        }

        protected override ObjectiveFunctionBase CreateObjectiveFunction()
        {
            return new FastTreeBinaryClassificationTrainer.ObjectiveImpl(
                TrainSet,
                ConvertTargetsToBool(TrainSet.Targets),
                Args.LearningRates,
                0,
                _sigmoidParameter,
                Args.UnbalancedSets,
                Args.MaxOutput,
                Args.GetDerivativesSampleRate,
                false,
                Args.RngSeed,
                ParallelTraining
            );
        }

        protected override void DefinePruningTest()
        {
            var validTest = new BinaryClassificationTest(ValidSetScore,
                ConvertTargetsToBool(ValidSet.Targets), _sigmoidParameter);
            // As per FastTreeClassification.ConstructOptimizationAlgorithm()
            PruningLossIndex = Args.UnbalancedSets ? 3 /*Unbalanced  sets  loss*/ : 1 /*normal loss*/;
            PruningTest = new TestHistory(validTest, PruningLossIndex);
        }
    }

    public class BinaryClassGamPredictor : GamPredictorBase, IPredictorProducing<Float>
    {
        public const string LoaderSignature = "BinaryClassGamPredictor";
        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        public BinaryClassGamPredictor(IHostEnvironment env, int inputLength, Dataset trainset,
            double meanEffect, double[][] binEffects, int[] featureMap, TrainingResults trainingResults)
            : base(env, LoaderSignature, inputLength, trainset, meanEffect, binEffects, featureMap, trainingResults) { }

        private BinaryClassGamPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx) { }

        public static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "GAM BINP",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public static IPredictorProducing<Float> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            var predictor = new BinaryClassGamPredictor(env, ctx);
            ICalibrator calibrator;
            ctx.LoadModelOrNull<ICalibrator, SignatureLoadModel>(env, out calibrator, @"Calibrator");
            if (calibrator == null)
                return predictor;
            return new SchemaBindableCalibratedPredictor(env, predictor, calibrator);
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
