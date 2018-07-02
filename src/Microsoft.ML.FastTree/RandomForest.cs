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
        new internal const string Remarks = @"<remarks>
Decision trees are non-parametric models that perform a sequence of simple tests on inputs. 
This decision procedure maps them to outputs found in the training dataset whose inputs were similar to the instance being processed. 
A decision is made at each node of the binary tree data structure based on a measure of similarity that maps each instance recursively through the branches of the tree until the appropriate leaf node is reached and the output decision returned.
<para>Decision trees have several advantages:</para>
<list type='bullet'>
<item>They are efficient in both computation and memory usage during training and prediction. </item>
<item>They can represent non-linear decision boundaries.</item>
<item>They perform integrated feature selection and classification. </item>
<item>They are resilient in the presence of noisy features.</item>
</list>
Fast forest is a random forest implementation. 
The model consists of an ensemble of decision trees. Each tree in a decision forest outputs a Gaussian distribution by way of prediction. 
An aggregation is performed over the ensemble of trees to find a Gaussian distribution closest to the combined distribution for all trees in the model.
This decision forest classifier consists of an ensemble of decision trees. 
Generally, ensemble models provide better coverage and accuracy than single decision trees. 
Each tree in a decision forest outputs a Gaussian distribution.
<a href='http://en.wikipedia.org/wiki/Random_forest'>Wikipedia: Random forest</a>
<a href='http://jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf'>Quantile regression forest</a>
<a href='https://blogs.technet.microsoft.com/machinelearning/2014/09/10/from-stumps-to-trees-to-forests/'>From Stumps to Trees to Forests</a>
</remarks>";

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
