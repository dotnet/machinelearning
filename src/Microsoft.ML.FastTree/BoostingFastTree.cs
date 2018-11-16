// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Trainers.FastTree.Internal;
using System;
using System.Linq;
using Float = System.Single;

namespace Microsoft.ML.Trainers.FastTree
{
    public abstract class BoostingFastTreeTrainerBase<TArgs, TTransformer, TModel> : FastTreeTrainerBase<TArgs, TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TArgs : BoostedTreeArgs, new()
        where TModel : IPredictorProducing<Float>
    {
        protected BoostingFastTreeTrainerBase(IHostEnvironment env, TArgs args, SchemaShape.Column label) : base(env, args, label)
        {
        }

        protected BoostingFastTreeTrainerBase(IHostEnvironment env,
            SchemaShape.Column label,
            string featureColumn,
            string weightColumn,
            string groupIdColumn,
            int numLeaves,
            int numTrees,
            int minDatapointsInLeaves,
            double learningRate,
            Action<TArgs> advancedSettings)
            : base(env, label, featureColumn, weightColumn, groupIdColumn, numLeaves, numTrees, minDatapointsInLeaves, advancedSettings)
        {

            if (Args.LearningRates != learningRate)
            {
                using (var ch = Host.Start($"Setting learning rate to: {learningRate} as supplied in the direct arguments."))
                    Args.LearningRates = learningRate;
            }
        }

        protected override void CheckArgs(IChannel ch)
        {
            if (Args.OptimizationAlgorithm == BoostedTreeArgs.OptimizationAlgorithmType.AcceleratedGradientDescent)
                Args.UseLineSearch = true;
            if (Args.OptimizationAlgorithm == BoostedTreeArgs.OptimizationAlgorithmType.ConjugateGradientDescent)
                Args.UseLineSearch = true;

            if (Args.CompressEnsemble && Args.WriteLastEnsemble)
                throw ch.Except("Ensemble compression cannot be done when forcing to write last ensemble (hl)");

            if (Args.NumLeaves > 2 && Args.HistogramPoolSize > Args.NumLeaves - 1)
                throw ch.Except("Histogram pool size (ps) must be at least 2.");

            if (Args.NumLeaves > 2 && Args.HistogramPoolSize > Args.NumLeaves - 1)
                throw ch.Except("Histogram pool size (ps) must be at most numLeaves - 1.");

            if (Args.EnablePruning && !HasValidSet)
                throw ch.Except("Cannot perform pruning (pruning) without a validation set (valid).");

            if (Args.EarlyStoppingRule != null && !HasValidSet)
                throw ch.Except("Cannot perform early stopping without a validation set (valid).");

            if (Args.UseTolerantPruning && (!Args.EnablePruning || !HasValidSet))
                throw ch.Except("Cannot perform tolerant pruning (prtol) without pruning (pruning) and a validation set (valid)");

            base.CheckArgs(ch);
        }

        protected override TreeLearner ConstructTreeLearner(IChannel ch)
        {
            return new LeastSquaresRegressionTreeLearner(
                TrainSet, Args.NumLeaves, Args.MinDocumentsInLeafs, Args.EntropyCoefficient,
                Args.FeatureFirstUsePenalty, Args.FeatureReusePenalty, Args.SoftmaxTemperature,
                Args.HistogramPoolSize, Args.RngSeed, Args.SplitFraction, Args.FilterZeroLambdas,
                Args.AllowEmptyTrees, Args.GainConfidenceLevel, Args.MaxCategoricalGroupsPerNode,
                Args.MaxCategoricalSplitPoints, BsrMaxTreeOutput(), ParallelTraining,
                Args.MinDocsPercentageForCategoricalSplit, Args.Bundling, Args.MinDocsForCategoricalSplit, Args.Bias);
        }

        protected override OptimizationAlgorithm ConstructOptimizationAlgorithm(IChannel ch)
        {
            Contracts.CheckValue(ch, nameof(ch));
            OptimizationAlgorithm optimizationAlgorithm;
            IGradientAdjuster gradientWrapper = MakeGradientWrapper(ch);

            switch (Args.OptimizationAlgorithm)
            {
                case BoostedTreeArgs.OptimizationAlgorithmType.GradientDescent:
                    optimizationAlgorithm = new GradientDescent(Ensemble, TrainSet, InitTrainScores, gradientWrapper);
                    break;
                case BoostedTreeArgs.OptimizationAlgorithmType.AcceleratedGradientDescent:
                    optimizationAlgorithm = new AcceleratedGradientDescent(Ensemble, TrainSet, InitTrainScores, gradientWrapper);
                    break;
                case BoostedTreeArgs.OptimizationAlgorithmType.ConjugateGradientDescent:
                    optimizationAlgorithm = new ConjugateGradientDescent(Ensemble, TrainSet, InitTrainScores, gradientWrapper);
                    break;
                default:
                    throw ch.Except("Unknown optimization algorithm '{0}'", Args.OptimizationAlgorithm);
            }

            optimizationAlgorithm.TreeLearner = ConstructTreeLearner(ch);
            optimizationAlgorithm.ObjectiveFunction = ConstructObjFunc(ch);
            optimizationAlgorithm.Smoothing = Args.Smoothing;
            optimizationAlgorithm.DropoutRate = Args.DropoutRate;
            optimizationAlgorithm.DropoutRng = new Random(Args.RngSeed);
            optimizationAlgorithm.PreScoreUpdateEvent += PrintTestGraph;

            return optimizationAlgorithm;
        }

        protected override IGradientAdjuster MakeGradientWrapper(IChannel ch)
        {
            if (!Args.BestStepRankingRegressionTrees)
                return base.MakeGradientWrapper(ch);

            // REVIEW: If this is ranking specific than cmd.bestStepRankingRegressionTrees and
            // this code should be part of Ranking application (and not application).
            if (AreSamplesWeighted(ch))
                return new QueryWeightsBestResressionStepGradientWrapper();
            else
                return new BestStepRegressionGradientWrapper();
        }

        protected override bool ShouldStop(IChannel ch, ref IEarlyStoppingCriterion earlyStoppingRule, ref int bestIteration)
        {
            if (Args.EarlyStoppingRule == null)
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
                earlyStoppingRule = Args.EarlyStoppingRule.CreateComponent(Host, lowerIsBetter);
                ch.Assert(earlyStoppingRule != null);
            }

            bool isBestCandidate;
            bool shouldStop = earlyStoppingRule.CheckScore((Float)validationResult.FinalValue,
                (Float)trainingResult.FinalValue, out isBestCandidate);

            if (isBestCandidate)
                bestIteration = Ensemble.NumTrees;

            return shouldStop;
        }

        protected override int GetBestIteration(IChannel ch)
        {
            int bestIteration = Ensemble.NumTrees;
            if (!Args.WriteLastEnsemble && PruningTest != null)
            {
                bestIteration = PruningTest.BestIteration;
                ch.Info("Pruning picked iteration {0}", bestIteration);
            }
            return bestIteration;
        }

        /// <summary>
        /// Retrieves max tree output if best regression step option is active or returns negative value otherwise.
        /// </summary>
        protected double BsrMaxTreeOutput()
        {
            if (Args.BestStepRankingRegressionTrees)
                return Args.MaxTreeOutput;
            else
                return -1;
        }

        protected override bool ShouldRandomStartOptimizer()
        {
            return Args.RandomStart;
        }
    }
}
