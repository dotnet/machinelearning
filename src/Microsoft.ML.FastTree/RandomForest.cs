// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using Microsoft.ML.Runtime.FastTree.Internal;

namespace Microsoft.ML.Runtime.FastTree
{
    public abstract class RandomForestTrainerBase<TArgs, TPredictor> : FastTreeTrainerBase<TArgs, TPredictor>
        where TArgs : FastForestArgumentsBase, new()
        where TPredictor : IPredictorProducing<Float>
    {
        private readonly bool _quantileEnabled;

        protected RandomForestTrainerBase(IHostEnvironment env, TArgs args, bool quantileEnabled = false)
            : base(env, args)
        {
            _quantileEnabled = quantileEnabled;
        }

        protected override OptimizationAlgorithm ConstructOptimizationAlgorithm(IChannel ch)
        {
            Host.CheckValue(ch, nameof(ch));
            IGradientAdjuster gradientWrapper = MakeGradientWrapper(ch);
            var optimizationAlgorithm = new RandomForestOptimizer(Ensemble, TrainSet, InitTrainScores, gradientWrapper);

            optimizationAlgorithm.TreeLearner = ConstructTreeLearner(ch);
            optimizationAlgorithm.ObjectiveFunction = ConstructObjFunc(ch);
            optimizationAlgorithm.Smoothing = Args.Smoothing;
            // No notion of dropout for non-boosting applications.
            optimizationAlgorithm.DropoutRate = 0;
            optimizationAlgorithm.DropoutRng = null;
            optimizationAlgorithm.PreScoreUpdateEvent += PrintTestGraph;

            return optimizationAlgorithm;
        }

        protected override void InitializeTests()
        {
        }

        protected override TreeLearner ConstructTreeLearner(IChannel ch)
        {
            return new RandomForestLeastSquaresTreeLearner(
                       TrainSet, Args.NumLeaves, Args.MinDocumentsInLeafs, Args.EntropyCoefficient,
                       Args.FeatureFirstUsePenalty, Args.FeatureReusePenalty, Args.SoftmaxTemperature,
                       Args.HistogramPoolSize, Args.RngSeed, Args.SplitFraction,
                       Args.AllowEmptyTrees, Args.GainConfidenceLevel, Args.MaxCategoricalGroupsPerNode,
                       Args.MaxCategoricalSplitPoints, _quantileEnabled, Args.QuantileSampleCount, ParallelTraining,
                       Args.MinDocsPercentageForCategoricalSplit, Args.Bundling, Args.MinDocsForCategoricalSplit, Args.Bias);
        }

        public abstract class RandomForestObjectiveFunction : ObjectiveFunctionBase
        {
            protected RandomForestObjectiveFunction(Dataset trainData, TArgs args, double maxStepSize)
                : base(trainData,
                    1, // No learning rate in random forests.
                    1, // No shrinkage in random forests.
                    maxStepSize,
                    1, // No derivative sampling in random forests.
                    false, // Improvements to quasi-newton step not relevant to RF.
                    args.RngSeed)
            {
            }
        }
    }
}
