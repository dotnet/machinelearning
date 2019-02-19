// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Internal.Internallearn;
using Float = System.Single;

namespace Microsoft.ML.Trainers.FastTree
{
    public abstract class BoostingFastTreeTrainerBase<TOptions, TTransformer, TModel> : FastTreeTrainerBase<TOptions, TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TOptions : BoostedTreeOptions, new()
        where TModel : class
    {
        private protected BoostingFastTreeTrainerBase(IHostEnvironment env, TOptions options, SchemaShape.Column label) : base(env, options, label)
        {
        }

        private protected BoostingFastTreeTrainerBase(IHostEnvironment env,
            SchemaShape.Column label,
            string featureColumn,
            string weightColumn,
            string groupIdColumn,
            int numLeaves,
            int numTrees,
            int minDatapointsInLeaves,
            double learningRate)
            : base(env, label, featureColumn, weightColumn, groupIdColumn, numLeaves, numTrees, minDatapointsInLeaves)
        {
            FastTreeTrainerOptions.LearningRates = learningRate;
        }

        private protected override void CheckOptions(IChannel ch)
        {
            if (FastTreeTrainerOptions.OptimizationAlgorithm == BoostedTreeOptions.OptimizationAlgorithmType.AcceleratedGradientDescent)
                FastTreeTrainerOptions.UseLineSearch = true;
            if (FastTreeTrainerOptions.OptimizationAlgorithm == BoostedTreeOptions.OptimizationAlgorithmType.ConjugateGradientDescent)
                FastTreeTrainerOptions.UseLineSearch = true;

            if (FastTreeTrainerOptions.CompressEnsemble && FastTreeTrainerOptions.WriteLastEnsemble)
                throw ch.Except("Ensemble compression cannot be done when forcing to write last ensemble (hl)");

            if (FastTreeTrainerOptions.NumLeaves > 2 && FastTreeTrainerOptions.HistogramPoolSize > FastTreeTrainerOptions.NumLeaves - 1)
                throw ch.Except("Histogram pool size (ps) must be at least 2.");

            if (FastTreeTrainerOptions.NumLeaves > 2 && FastTreeTrainerOptions.HistogramPoolSize > FastTreeTrainerOptions.NumLeaves - 1)
                throw ch.Except("Histogram pool size (ps) must be at most numLeaves - 1.");

            if (FastTreeTrainerOptions.EnablePruning && !HasValidSet)
                throw ch.Except("Cannot perform pruning (pruning) without a validation set (valid).");

            if (FastTreeTrainerOptions.EarlyStoppingRule != null && !HasValidSet)
                throw ch.Except("Cannot perform early stopping without a validation set (valid).");

            if (FastTreeTrainerOptions.UseTolerantPruning && (!FastTreeTrainerOptions.EnablePruning || !HasValidSet))
                throw ch.Except("Cannot perform tolerant pruning (prtol) without pruning (pruning) and a validation set (valid)");

            base.CheckOptions(ch);
        }

        private protected override TreeLearner ConstructTreeLearner(IChannel ch)
        {
            return new LeastSquaresRegressionTreeLearner(
                TrainSet, FastTreeTrainerOptions.NumLeaves, FastTreeTrainerOptions.MinDocumentsInLeafs, FastTreeTrainerOptions.EntropyCoefficient,
                FastTreeTrainerOptions.FeatureFirstUsePenalty, FastTreeTrainerOptions.FeatureReusePenalty, FastTreeTrainerOptions.SoftmaxTemperature,
                FastTreeTrainerOptions.HistogramPoolSize, FastTreeTrainerOptions.RngSeed, FastTreeTrainerOptions.SplitFraction, FastTreeTrainerOptions.FilterZeroLambdas,
                FastTreeTrainerOptions.AllowEmptyTrees, FastTreeTrainerOptions.GainConfidenceLevel, FastTreeTrainerOptions.MaxCategoricalGroupsPerNode,
                FastTreeTrainerOptions.MaxCategoricalSplitPoints, BsrMaxTreeOutput(), ParallelTraining,
                FastTreeTrainerOptions.MinDocsPercentageForCategoricalSplit, FastTreeTrainerOptions.Bundling, FastTreeTrainerOptions.MinDocsForCategoricalSplit, FastTreeTrainerOptions.Bias);
        }

        private protected override OptimizationAlgorithm ConstructOptimizationAlgorithm(IChannel ch)
        {
            Contracts.CheckValue(ch, nameof(ch));
            OptimizationAlgorithm optimizationAlgorithm;
            IGradientAdjuster gradientWrapper = MakeGradientWrapper(ch);

            switch (FastTreeTrainerOptions.OptimizationAlgorithm)
            {
                case BoostedTreeOptions.OptimizationAlgorithmType.GradientDescent:
                    optimizationAlgorithm = new GradientDescent(Ensemble, TrainSet, InitTrainScores, gradientWrapper);
                    break;
                case BoostedTreeOptions.OptimizationAlgorithmType.AcceleratedGradientDescent:
                    optimizationAlgorithm = new AcceleratedGradientDescent(Ensemble, TrainSet, InitTrainScores, gradientWrapper);
                    break;
                case BoostedTreeOptions.OptimizationAlgorithmType.ConjugateGradientDescent:
                    optimizationAlgorithm = new ConjugateGradientDescent(Ensemble, TrainSet, InitTrainScores, gradientWrapper);
                    break;
                default:
                    throw ch.Except("Unknown optimization algorithm '{0}'", FastTreeTrainerOptions.OptimizationAlgorithm);
            }

            optimizationAlgorithm.TreeLearner = ConstructTreeLearner(ch);
            optimizationAlgorithm.ObjectiveFunction = ConstructObjFunc(ch);
            optimizationAlgorithm.Smoothing = FastTreeTrainerOptions.Smoothing;
            optimizationAlgorithm.DropoutRate = FastTreeTrainerOptions.DropoutRate;
            optimizationAlgorithm.DropoutRng = new Random(FastTreeTrainerOptions.RngSeed);
            optimizationAlgorithm.PreScoreUpdateEvent += PrintTestGraph;

            return optimizationAlgorithm;
        }

        private protected override IGradientAdjuster MakeGradientWrapper(IChannel ch)
        {
            if (!FastTreeTrainerOptions.BestStepRankingRegressionTrees)
                return base.MakeGradientWrapper(ch);

            // REVIEW: If this is ranking specific than cmd.bestStepRankingRegressionTrees and
            // this code should be part of Ranking application (and not application).
            if (AreSamplesWeighted(ch))
                return new QueryWeightsBestResressionStepGradientWrapper();
            else
                return new BestStepRegressionGradientWrapper();
        }

        private protected override bool ShouldStop(IChannel ch, ref IEarlyStoppingCriterion earlyStoppingRule, ref int bestIteration)
        {
            if (FastTreeTrainerOptions.EarlyStoppingRule == null)
                return false;

            ch.AssertValue(ValidTest);
            ch.AssertValue(TrainTest);

            var validationResult = ValidTest.ComputeTests().First();
            ch.Assert(validationResult.FinalValue >= 0);
            bool lowerIsBetter = validationResult.LowerIsBetter;

            var trainingResult = TrainTest.ComputeTests().First();
            ch.Assert(trainingResult.FinalValue >= 0);

            // Create early stopping rule.
            if (earlyStoppingRule == null)
            {
                earlyStoppingRule = FastTreeTrainerOptions.EarlyStoppingRule.CreateComponent(Host, lowerIsBetter);
                ch.Assert(earlyStoppingRule != null);
            }

            bool isBestCandidate;
            bool shouldStop = earlyStoppingRule.CheckScore((Float)validationResult.FinalValue,
                (Float)trainingResult.FinalValue, out isBestCandidate);

            if (isBestCandidate)
                bestIteration = Ensemble.NumTrees;

            return shouldStop;
        }

        private protected override int GetBestIteration(IChannel ch)
        {
            int bestIteration = Ensemble.NumTrees;
            if (!FastTreeTrainerOptions.WriteLastEnsemble && PruningTest != null)
            {
                bestIteration = PruningTest.BestIteration;
                ch.Info("Pruning picked iteration {0}", bestIteration);
            }
            return bestIteration;
        }

        /// <summary>
        /// Retrieves max tree output if best regression step option is active or returns negative value otherwise.
        /// </summary>
        private protected double BsrMaxTreeOutput()
        {
            if (FastTreeTrainerOptions.BestStepRankingRegressionTrees)
                return FastTreeTrainerOptions.MaxTreeOutput;
            else
                return -1;
        }

        private protected override bool ShouldRandomStartOptimizer()
        {
            return FastTreeTrainerOptions.RandomStart;
        }
    }
}
