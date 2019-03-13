// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.FastTree
{
    public abstract class RandomForestTrainerBase<TOptions, TTransformer, TModel> : FastTreeTrainerBase<TOptions, TTransformer, TModel>
        where TOptions : FastForestOptionsBase, new()
        where TModel : class
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
    {
        private readonly bool _quantileEnabled;

        /// <summary>
        /// Constructor invoked by the maml code-path.
        /// </summary>
        private protected RandomForestTrainerBase(IHostEnvironment env, TOptions options, SchemaShape.Column label, bool quantileEnabled = false)
            : base(env, options, label)
        {
            _quantileEnabled = quantileEnabled;
        }

        /// <summary>
        /// Constructor invoked by the API code-path.
        /// </summary>
        private protected RandomForestTrainerBase(IHostEnvironment env,
            SchemaShape.Column label,
            string featureColumnName,
            string exampleWeightColumnName,
            string rowGroupColumnName,
            int numberOfLeaves,
            int numberOfTrees,
            int minimumExampleCountPerLeaf,
            bool quantileEnabled = false)
            : base(env, label, featureColumnName, exampleWeightColumnName, null, numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf)
        {
            _quantileEnabled = quantileEnabled;
        }

        private protected override OptimizationAlgorithm ConstructOptimizationAlgorithm(IChannel ch)
        {
            Host.CheckValue(ch, nameof(ch));
            IGradientAdjuster gradientWrapper = MakeGradientWrapper(ch);
            var optimizationAlgorithm = new RandomForestOptimizer(Ensemble, TrainSet, InitTrainScores, gradientWrapper);

            optimizationAlgorithm.TreeLearner = ConstructTreeLearner(ch);
            optimizationAlgorithm.ObjectiveFunction = ConstructObjFunc(ch);
            optimizationAlgorithm.Smoothing = FastTreeTrainerOptions.Smoothing;
            // No notion of dropout for non-boosting applications.
            optimizationAlgorithm.DropoutRate = 0;
            optimizationAlgorithm.DropoutRng = null;
            optimizationAlgorithm.PreScoreUpdateEvent += PrintTestGraph;

            return optimizationAlgorithm;
        }

        private protected override void InitializeTests()
        {
        }

        private protected override TreeLearner ConstructTreeLearner(IChannel ch)
        {
            return new RandomForestLeastSquaresTreeLearner(
                       TrainSet, FastTreeTrainerOptions.NumberOfLeaves, FastTreeTrainerOptions.MinimumExampleCountPerLeaf, FastTreeTrainerOptions.EntropyCoefficient,
                       FastTreeTrainerOptions.FeatureFirstUsePenalty, FastTreeTrainerOptions.FeatureReusePenalty, FastTreeTrainerOptions.SoftmaxTemperature,
                       FastTreeTrainerOptions.HistogramPoolSize, FastTreeTrainerOptions.Seed, FastTreeTrainerOptions.FeatureFractionPerSplit,
                       FastTreeTrainerOptions.AllowEmptyTrees, FastTreeTrainerOptions.GainConfidenceLevel, FastTreeTrainerOptions.MaximumCategoricalGroupCountPerNode,
                       FastTreeTrainerOptions.MaximumCategoricalSplitPointCount, _quantileEnabled, FastTreeTrainerOptions.NumberOfQuantileSamples, ParallelTraining,
                       FastTreeTrainerOptions.MinimumExampleFractionForCategoricalSplit, FastTreeTrainerOptions.Bundling, FastTreeTrainerOptions.MinimumExamplesForCategoricalSplit, FastTreeTrainerOptions.Bias);
        }

        internal abstract class RandomForestObjectiveFunction : ObjectiveFunctionBase
        {
            protected RandomForestObjectiveFunction(Dataset trainData, TOptions options, double maxStepSize)
                : base(trainData,
                    1, // No learning rate in random forests.
                    1, // No shrinkage in random forests.
                    maxStepSize,
                    1, // No derivative sampling in random forests.
                    false, // Improvements to quasi-newton step not relevant to RF.
                    options.Seed)
            {
            }
        }
    }
}
