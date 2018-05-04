// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    public class GradientDescent : OptimizationAlgorithm
    {
        private IGradientAdjuster _gradientWrapper;

        /// number of trees dropped in this iteration
        private int _numberOfDroppedTrees;
        // treeScores stores for every tree the predictions it makes on every training example. This is used
        // to eliminate the need for computing the scores when we drop trees. However, it causes a horrifying
        // memory drain.
        private List<double[]> _treeScores;
        private double[] _droppedScores;
        private double[] _scores;

        public GradientDescent(Ensemble ensemble, Dataset trainData, double[] initTrainScores, IGradientAdjuster gradientWrapper)
            : base(ensemble, trainData, initTrainScores)
        {
            _gradientWrapper = gradientWrapper;
            _treeScores = new List<double[]>();
        }

        protected override ScoreTracker ConstructScoreTracker(string name, Dataset set, double[] initScores)
        {
            return new ScoreTracker(name, set, initScores);
        }

        protected virtual double[] GetGradient(IChannel ch)
        {
            Contracts.AssertValue(ch);
            if (DropoutRate > 0)
            {
                if (_droppedScores == null)
                    _droppedScores = new double[TrainingScores.Scores.Length];
                else
                    Array.Clear(_droppedScores, 0, _droppedScores.Length);
                if (_scores == null)
                    _scores = new double[TrainingScores.Scores.Length];
                int numberOfTrees = Ensemble.NumTrees;
                int[] droppedTrees =
                    Enumerable.Range(0, numberOfTrees).Where(t => (DropoutRng.NextDouble() < DropoutRate)).ToArray();
                _numberOfDroppedTrees = droppedTrees.Length;
                if ((_numberOfDroppedTrees == 0) && (numberOfTrees > 0))
                {
                    droppedTrees = new int[] { DropoutRng.Next(numberOfTrees) };
                    // force at least a single tree to be dropped 
                    _numberOfDroppedTrees = droppedTrees.Length;
                }
                ch.Trace("dropout: Dropping {0} trees of {1} for rate {2}",
                    _numberOfDroppedTrees, numberOfTrees, DropoutRate);
                foreach (int i in droppedTrees)
                {
                    double[] s = _treeScores[i];
                    for (int j = 0; j < _droppedScores.Length; j++)
                    {
                        _droppedScores[j] += s[j]; // summing up the weights of the dropped tree
                        s[j] *= _numberOfDroppedTrees / (1.0 + _numberOfDroppedTrees); // rescaling the dropped tree
                    }
                    Ensemble.GetTreeAt(i).ScaleOutputsBy(_numberOfDroppedTrees / (1.0 + _numberOfDroppedTrees));
                }
                for (int j = 0; j < _scores.Length; j++)
                {
                    _scores[j] = TrainingScores.Scores[j] - _droppedScores[j];
                    TrainingScores.Scores[j] -= _droppedScores[j] / (1.0 + _numberOfDroppedTrees);
                }
                return ObjectiveFunction.GetGradient(ch, _scores);
            }
            else
                return ObjectiveFunction.GetGradient(ch, TrainingScores.Scores);
            }

        protected virtual double[] AdjustTargetsAndSetWeights(IChannel ch)
        {
            if (_gradientWrapper == null)
                return GetGradient(ch);
            else
            {
                double[] targetWeights = null;
                double[] targets = _gradientWrapper.AdjustTargetAndSetWeights(GetGradient(ch), ObjectiveFunction, out targetWeights);
                Dataset.DatasetSkeleton dsSkeleton = TrainingScores.Dataset.Skeleton;
                return targets;
            }
        }

        public override RegressionTree TrainingIteration(IChannel ch, bool[] activeFeatures)
        {
            Contracts.CheckValue(ch, nameof(ch));
            // Fit a regression tree to the gradient using least squares.
            RegressionTree tree = TreeLearner.FitTargets(ch, activeFeatures, AdjustTargetsAndSetWeights(ch));
            if (tree == null)
                return null; // Could not learn a tree. Exit.

            // Adjust output values of tree by performing a Newton step.

            // REVIEW: This should be part of OptimizingAlgorithm.
            using (Timer.Time(TimerEvent.TreeLearnerAdjustTreeOutputs))
            {
                double[] backupScores = null;
                // when doing dropouts we need to replace the TrainingScores with the scores without the dropped trees 
                if (DropoutRate > 0)
                {
                    backupScores = TrainingScores.Scores;
                    TrainingScores.Scores = _scores;
                }

                if (AdjustTreeOutputsOverride != null)
                    AdjustTreeOutputsOverride.AdjustTreeOutputs(ch, tree, TreeLearner.Partitioning, TrainingScores);
                else if (ObjectiveFunction is IStepSearch)
                    (ObjectiveFunction as IStepSearch).AdjustTreeOutputs(ch, tree, TreeLearner.Partitioning, TrainingScores);
                else
                    throw ch.Except("No AdjustTreeOutputs defined. Objective function should define IStepSearch or AdjustTreeOutputsOverride should be set");
                if (DropoutRate > 0)
                {
                    // Returning the original scores.
                    TrainingScores.Scores = backupScores;
                }
            }
            if (Smoothing != 0.0)
            {
                SmoothTree(tree, Smoothing);
                UseFastTrainingScoresUpdate = false;
            }
            if (DropoutRate > 0)
            {
                // Don't do shrinkage if you do dropouts.
                double scaling = (1.0 / (1.0 + _numberOfDroppedTrees));
                tree.ScaleOutputsBy(scaling);
                _treeScores.Add(tree.GetOutputs(TrainingScores.Dataset));
            }
            UpdateAllScores(ch, tree);
            Ensemble.AddTree(tree);
            return tree;
        }
    }

    /// <summary>
    /// Interface for wrapping with weights of gradient target values
    /// </summary>
    public interface IGradientAdjuster
    {
        /// <summary>
        /// Create wrapping of gradient target values
        /// </summary>
        /// <param name="gradient"></param>
        /// <param name="objFunction">Objective functions can be used for constructing weights</param>
        /// <param name="targetWeights"></param>
        double[] AdjustTargetAndSetWeights(double[] gradient, ObjectiveFunctionBase objFunction, out double[] targetWeights);
    }
}
