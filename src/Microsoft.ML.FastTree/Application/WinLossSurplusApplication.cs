// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.Runtime.CommandLine;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
#if OLD_DATALOAD
    public class WinLossSurplusCommandLineArgs : TrainingCommandLineArgs
    {
        [Argument(ArgumentType.AtMostOnce, HelpText = "Scaling Factor for win loss surplus", ShortName = "wls")]
        public double winlossScaleFactor = 1.0;
    }

    public class WinLossSurplusTrainingApplication : RankingApplication
    {
        new WinLossSurplusCommandLineArgs cmd;
        private const string RegistrationName = "WinLossSurplusApplication";

        public WinLossSurplusTrainingApplication(IHostEnvironment env, string args, TrainingApplicationData data)
            : base(env, RegistrationName, args, data)
        {
            base.cmd = this.cmd = new WinLossSurplusCommandLineArgs();
        }

        public override ObjectiveFunction ConstructObjFunc()
        {
            return new WinLossSurplusObjectiveFunction(TrainSet, TrainSet.Ratings, cmd);
        }

        public override OptimizationAlgorithm ConstructOptimizationAlgorithm(IChannel ch)
        {
            OptimizationAlgorithm optimizationAlgorithm = base.ConstructOptimizationAlgorithm(ch);
            var lossCalculator = new WinLossSurplusTest(optimizationAlgorithm.TrainingScores, TrainSet.Ratings, cmd.sortingAlgorithm, cmd.winlossScaleFactor);
            optimizationAlgorithm.AdjustTreeOutputsOverride = new LineSearch(lossCalculator, 0, cmd.numPostBracketSteps, cmd.minStepSize);
            return optimizationAlgorithm;
        }

        protected override Test ConstructTestForTrainingData()
        {
            return new WinLossSurplusTest(ConstructScoreTracker(TrainSet), TrainSet.Ratings, cmd.sortingAlgorithm, cmd.winlossScaleFactor);
        }

        public override void PrintIterationMessage(IChannel ch, IProgressChannel pch)
        {
            // REVIEW: Shift this to use progress channels.
#if OLD_TRACING
            if (PruningTest != null)
            {
                if (PruningTest is TestWindowWithTolerance)
                {
                    if (PruningTest.BestIteration != -1)
                        ch.Info("Iteration {0} \t(Best tolerated validation moving average WinLossSurplus {1}:{2:00.00}~{3:00.00})",
                                Ensemble.NumTrees,
                                PruningTest.BestIteration,
                                (PruningTest as TestWindowWithTolerance).BestAverageValue,
                                (PruningTest as TestWindowWithTolerance).CurrentAverageValue);
                    else
                        ch.Info("Iteration {0}", Ensemble.NumTrees);
                }
                else
                {
                    ch.Info("Iteration {0} \t(best validation WinLoss {1}:{2:00.00}>{3:00.00})",
                            Ensemble.NumTrees,
                            PruningTest.BestIteration,
                            PruningTest.BestResult.FinalValue,
                            PruningTest.ComputeTests().First().FinalValue);
                }
            }
            else
                base.PrintIterationMessage(ch, pch);
#else
            base.PrintIterationMessage(ch, pch);
#endif
        }

        protected override Test CreateStandardTest(Dataset dataset)
        {
            return new WinLossSurplusTest(
                ConstructScoreTracker(dataset),
                dataset.Ratings,
                cmd.sortingAlgorithm,
                cmd.winlossScaleFactor);
        }

        protected override Test CreateSpecialTrainSetTest()
        {
            return CreateStandardTest(TrainSet);
        }

        protected override Test CreateSpecialValidSetTest()
        {
            return CreateStandardTest(ValidSet);
        }

        protected override string GetTestGraphHeader()
        {
            return "Eval:\tFileName\tMaxSurplus\tSurplus@100\tSurplus@200\tSurplus@300\tSurplus@400\tSurplus@500\tSurplus@1000\tMaxSurplusPos\tPercentTop\n";
        }
    }

    public class WinLossSurplusObjectiveFunction : LambdaRankObjectiveFunction
    {
        public WinLossSurplusObjectiveFunction(Dataset trainSet, short[] labels, WinLossSurplusCommandLineArgs cmd)
            : base(trainSet, labels, cmd)
        {
        }

        protected override void GetGradientInOneQuery(int query, int threadIndex)
        {
            int begin = Dataset.Boundaries[query];
            int numDocuments = Dataset.Boundaries[query + 1] - Dataset.Boundaries[query];

            Array.Clear(_gradient, begin, numDocuments);
            Array.Clear(_weights, begin, numDocuments);

            double inverseMaxDCG = _inverseMaxDCGT[query];

            int[] permutation = _permutationBuffers[threadIndex];

            short[] labels = Labels;
            double[] scoresToUse = _scores;

            // Keep track of top 3 labels for later use
            //GetTopQueryLabels(query, permutation, false);
            unsafe
            {
                fixed (int* pPermutation = permutation)
                fixed (short* pLabels = labels)
                fixed (double* pScores = scoresToUse, pLambdas = _gradient, pWeights = _weights, pDiscount = _discount)
                fixed (double* pGain = _gain, pGainLabels = _gainLabels, pSigmoidTable = _sigmoidTable)
                fixed (int* pOneTwoThree = _oneTwoThree)
                {
                    // calculates the permutation that orders "scores" in descending order, without modifying "scores"
                    Array.Copy(_oneTwoThree, permutation, numDocuments);
#if USE_FASTTREENATIVE
                    double lambdaSum = 0;

                    C_Sort(pPermutation, &pScores[begin], &pLabels[begin], numDocuments);

                    // Keep track of top 3 labels for later use
                    GetTopQueryLabels(query, permutation, true);

                    int numActualResults = numDocuments;

                    C_GetSurplusDerivatives(numDocuments, begin, pPermutation, pLabels,
                            pScores, pLambdas, pWeights, pDiscount,
                            pGainLabels,
                            pSigmoidTable, _minScore, _maxScore, _sigmoidTable.Length, _scoreToSigmoidTableFactor,
                            _costFunctionParam, _distanceWeight2, &lambdaSum, double.MinValue);

                    if (_normalizeQueryLambdas)
                    {
                        if (lambdaSum > 0)
                        {
                            double normFactor = (10 * Math.Log(1 + lambdaSum)) / lambdaSum;

                            for (int i = begin; i < begin + numDocuments; ++i)
                            {
                                pLambdas[i] = pLambdas[i] * normFactor;
                                pWeights[i] = pWeights[i] * normFactor;
                            }
                        }
                    }
#else
                    throw new Exception("Shifted NDCG / ContinuousWeightedRanknet / WinLossSurplus / distanceWeight2 / normalized lambdas are only supported by unmanaged code");
#endif
                }
            }
        }

        [DllImport("FastTreeNative", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        private unsafe static extern void C_GetSurplusDerivatives(int numDocuments, int begin, int* pPermutation, short* pLabels,
                                                           double* pScores, double* pLambdas, double* pWeights, double* pDiscount,
                                                           double* pGainLabels, double* lambdaTable, double minScore, double maxScore,
                                                           int lambdaTableLength, double scoreToLambdaTableFactor,
                                                           char costFunctionParam, [MarshalAs(UnmanagedType.U1)] bool distanceWeight2,
                                                           double* pLambdaSum, double doubleMinValue);

    }
#endif
}
