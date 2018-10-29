// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    //Accelerated gradient descent score tracker
    public class AcceleratedGradientDescent : GradientDescent
    {
        public AcceleratedGradientDescent(Ensemble ensemble, Dataset trainData, double[] initTrainScores, IGradientAdjuster gradientWrapper)
            : base(ensemble, trainData, initTrainScores, gradientWrapper)
        {
            UseFastTrainingScoresUpdate = false;
        }
        protected override ScoreTracker ConstructScoreTracker(string name, Dataset set, double[] initScores)
        {
            return new AgdScoreTracker(name, set, initScores);
        }

        public override RegressionTree TrainingIteration(IChannel ch, bool[] activeFeatures)
        {
            Contracts.CheckValue(ch, nameof(ch));
            AgdScoreTracker trainingScores = TrainingScores as AgdScoreTracker;
            //First Let's make XK=YK as we want to fit YK and LineSearch YK
            // and call base class that uses fits XK (in our case will fir YK thanks to the swap)
            var xk = trainingScores.XK;
            trainingScores.XK = trainingScores.YK;
            trainingScores.YK = null;

            //Invoke standard gradient descent on YK rather than XK(Scores)
            RegressionTree tree = base.TrainingIteration(ch, activeFeatures);

            //Reverse the XK/YK swap
            trainingScores.YK = trainingScores.XK;
            trainingScores.XK = xk;

            if (tree == null)
                return null; // No tree was actually learnt. Give up.

            // ... and update the training scores that we omitted from update
            // in AcceleratedGradientDescent.UpdateScores
            // Here we could use faster way of comuting train scores taking advantage of scores precompited by LineSearch
            // But that would make the code here even more difficult/complex
            trainingScores.AddScores(tree, TreeLearner.Partitioning, 1.0);

            //Now rescale all previous trees based on ratio of new_desired_tree_scale/previous_tree_scale
            for (int t = 0; t < Ensemble.NumTrees - 1; t++)
            {
                Ensemble.GetTreeAt(t).ScaleOutputsBy(AgdScoreTracker.TreeMultiplier(t, Ensemble.NumTrees) / AgdScoreTracker.TreeMultiplier(t, Ensemble.NumTrees - 1));
            }
            return tree;
        }

        public override void UpdateScores(ScoreTracker t, RegressionTree tree)
        {
            if (t == TrainingScores)
            {
                return;
                //Special optimized routine for updating TrainingScores is implemented as part of TrainingItearation
            }
            else
                base.UpdateScores(t, tree);
        }

        public override void FinalizeLearning(int bestIteration)
        {
            if (bestIteration != Ensemble.NumTrees)
            {
                // Restore multiplier for each tree as it was set during bestIteration
                for (int t = 0; t < bestIteration; t++)
                {
                    Ensemble.GetTreeAt(t).ScaleOutputsBy(AgdScoreTracker.TreeMultiplier(t, bestIteration) / AgdScoreTracker.TreeMultiplier(t, Ensemble.NumTrees));
                }
            }
            base.FinalizeLearning(bestIteration);
        }
    }
}
