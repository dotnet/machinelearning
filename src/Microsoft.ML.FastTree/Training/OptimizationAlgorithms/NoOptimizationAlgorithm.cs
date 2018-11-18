// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    /// <summary>
    /// This is dummy optimizer. As Random forest does not have any boosting based optimization, this is place holder to be consistent
    /// with other fast tree based applications
    /// </summary>
    public class RandomForestOptimizer : GradientDescent
    {
        private IGradientAdjuster _gradientWrapper;
        // REVIEW: When the FastTree appliation is decoupled with tree learner and boosting logic, this class should be removed.
        public RandomForestOptimizer(TreeEnsemble ensemble, Dataset trainData, double[] initTrainScores, IGradientAdjuster gradientWrapper)
            : base(ensemble, trainData, initTrainScores, gradientWrapper)
        {
            _gradientWrapper = gradientWrapper;
        }

        protected override ScoreTracker ConstructScoreTracker(string name, Dataset set, double[] initScores)
        {
            //REVIEW: This is not necessary. We can remove this by creating dummy scorer.
            return new ScoreTracker(name, set, initScores);
        }

        public override RegressionTree TrainingIteration(IChannel ch, bool[] activeFeatures)
        {
            Contracts.CheckValue(ch, nameof(ch));

            double[] sampleWeights = null;
            double[] targets = GetGradient(ch);
            double[] weightedTargets = _gradientWrapper.AdjustTargetAndSetWeights(targets, ObjectiveFunction, out sampleWeights);
            RegressionTree tree = ((RandomForestLeastSquaresTreeLearner)TreeLearner).FitTargets(ch, activeFeatures, weightedTargets,
                targets, sampleWeights);

            if (tree != null)
                Ensemble.AddTree(tree);
            return tree;
        }
    }
}
