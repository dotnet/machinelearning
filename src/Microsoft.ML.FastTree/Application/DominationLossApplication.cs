// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.CommandLine;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
#if OLD_DATALOAD
    public class DominationLossCommandLineArgs : TrainingCommandLineArgs
    {
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "The smallest HRS label that maps to a positive (default: 1)", ShortName = "pos")]
        public short smallestPositive = 1;
    }

    public class DominationLossTrainingApplication : ApplicationBase
    {
        DominationLossObjectiveFunction.BestDocsPerQuery TrainSetLabels;
        new DominationLossCommandLineArgs cmd;
        private const string RegistrationName = "DominationLossApplication";

        protected override bool IsRankingApplication { get { return true; } }
        public DominationLossTrainingApplication(IHostEnvironment env, string args, TrainingApplicationData data)
            : base(env, RegistrationName, args, data)
        {
            base.cmd = this.cmd = new DominationLossCommandLineArgs();
        }

        public override ObjectiveFunction ConstructObjFunc()
        {
            return new DominationLossObjectiveFunction(TrainSet, TrainSetLabels, cmd);
        }

        public override OptimizationAlgorithm ConstructOptimizationAlgorithm(IChannel ch)
        {
            OptimizationAlgorithm optimizationAlgorithm = base.ConstructOptimizationAlgorithm(ch);
            var lossCalculator = new DominationLossTest(optimizationAlgorithm.TrainingScores, TrainSetLabels);
            optimizationAlgorithm.AdjustTreeOutputsOverride = new LineSearch(lossCalculator, 0, cmd.numPostBracketSteps, cmd.minStepSize);
            return optimizationAlgorithm;
        }

        protected override void PrepareLabels(IChannel ch)
        {
            TrainSetLabels = new DominationLossObjectiveFunction.BestDocsPerQuery(TrainSet);
        }

        protected override Test ConstructTestForTrainingData()
        {
            return new NDCGTest(ConstructScoreTracker(TrainSet), TrainSet.Ratings, cmd.sortingAlgorithm);
        }

        protected override void InitializeTests()
        {
            if (cmd.testFrequency != int.MaxValue)
            {
                AddFullNDCGTests();
                AddDominationLossTests();
            }

            // TODO add graph

            // TODO add early stopping
        }

        private void AddFullNDCGTests()
        {
            Tests.Add(new NDCGTest(ConstructScoreTracker(TrainSet), TrainSet.Ratings, cmd.sortingAlgorithm));

            if (ValidSet != null)
                Tests.Add(new NDCGTest(ConstructScoreTracker(ValidSet), ValidSet.Ratings, cmd.sortingAlgorithm));

            for (int t = 0; TestSets != null && t < TestSets.Length; ++t)
                Tests.Add(new NDCGTest(ConstructScoreTracker(TestSets[t]), TestSets[t].Ratings, cmd.sortingAlgorithm));
        }

        private void AddDominationLossTests()
        {
            Tests.Add(new DominationLossTest(ConstructScoreTracker(TrainSet), TrainSetLabels));

            if (ValidSet != null)
            {
                var labels = new DominationLossObjectiveFunction.BestDocsPerQuery(ValidSet);
                Tests.Add(new DominationLossTest(ConstructScoreTracker(ValidSet), labels));
            }

            for (int t = 0; TestSets != null && t < TestSets.Length; ++t)
            {
                var labels = new DominationLossObjectiveFunction.BestDocsPerQuery(TestSets[t]);
                Tests.Add(new DominationLossTest(ConstructScoreTracker(TestSets[t]), labels));
            }
        }
    }

    public class DominationLossObjectiveFunction : ObjectiveFunction
    {
        public class BestDocsPerQuery
        {
            public int[] BestDocs;
            public int[] SecondBestDocs;
            public BestDocsPerQuery(Dataset set)
            {
                BestDocs = new int[set.NumQueries];
                SecondBestDocs = new int[set.NumQueries];

                for (int q = 0; q < set.NumQueries; ++q)
                {
                    int best = set.Boundaries[q];
                    int secondBest = set.Boundaries[q];
                    short max = -1;
                    short secondMax = -1;

                    for (int d = set.Boundaries[q]; d < set.Boundaries[q + 1]; ++d)
                    {
                        if (max < set.Ratings[d])
                        {
                            secondMax = max;
                            secondBest = best;
                            max = set.Ratings[d];
                            best = d;
                        }
                        else if (secondMax < set.Ratings[d])
                        {
                            secondMax = set.Ratings[d];
                            secondBest = d;
                        }
                    }
                    BestDocs[q] = best;
                    SecondBestDocs[q] = secondBest;
                }
            }
        }

        private BestDocsPerQuery _bestDocsPerQuery;

        public DominationLossObjectiveFunction(Dataset trainSet, BestDocsPerQuery bestDocsPerQuery, DominationLossCommandLineArgs cmd)
            : base(
                trainSet,
                cmd.learningRates,
                cmd.shrinkage,
                cmd.maxTreeOutput,
                cmd.getDerivativesSampleRate,
                cmd.bestStepRankingRegressionTrees,
                cmd.rngSeed)
        {
            _bestDocsPerQuery = bestDocsPerQuery;
        }

        protected override void GetGradientInOneQuery(int query, int threadIndex)
        {
            int begin = Dataset.Boundaries[query];
            int end = Dataset.Boundaries[query + 1];

            if (end - begin <= 1)
                return;

            int bestDoc = _bestDocsPerQuery.BestDocs[query];
            int secondBestDoc = _bestDocsPerQuery.SecondBestDocs[query];

            // find max score
            double max = double.NegativeInfinity;
            double maxNotBest = double.NegativeInfinity;

            for (int d = begin; d < end; ++d)
            {
                if (max < _scores[d])
                    max = _scores[d];
                if (d != bestDoc && maxNotBest < _scores[d])
                    maxNotBest = _scores[d];
            }

            // sum of exponents and sum of all but best
            double sum = 0.0;
            double sumAllButBest = 0.0;
            for (int d = begin; d < end; ++d)
            {
                sum += Math.Exp(_scores[d] - max);
                if (d != bestDoc)
                    sumAllButBest += Math.Exp(_scores[d] - maxNotBest);
            }

            // calculate gradients
            for (int d = begin; d < end; ++d)
            {
                _gradient[d] = _learningRate * (-Math.Exp(_scores[d] - max) / sum - 0.5 * Math.Exp(_scores[d] - maxNotBest) / sumAllButBest);
            }

            _gradient[bestDoc] = _learningRate * (1.0 - Math.Exp(_scores[bestDoc] - max) / sum);
            _gradient[secondBestDoc] += _learningRate * 0.5;
        }
    }

    public class DominationLossTest : Test
    {
        DominationLossObjectiveFunction.BestDocsPerQuery _bestDocsPerQuery;
        public DominationLossTest(ScoreTracker scoreTracker, DominationLossObjectiveFunction.BestDocsPerQuery bestDocsPerQuery)
            : base(scoreTracker)
        {
            _bestDocsPerQuery = bestDocsPerQuery;
            Contracts.Check(scoreTracker.Dataset.NumQueries == bestDocsPerQuery.BestDocs.Length, "Mismatch between dataset and labels");
        }

        public double ComputeDominationLoss(double[] scores)
        {
            int chunkSize = 1 + Dataset.NumQueries / BlockingThreadPool.NumThreads;   // Minimizes the number of repeat computations in sparse array to have each thread take as big a chunk as possible
            double totalOutput = 0.0;
            var _lock = new Object();
            for (int queryBegin = 0; queryBegin < Dataset.NumQueries; queryBegin += chunkSize)
                BlockingThreadPool.RunOrBlock(delegate(int startQuery, int endQuery)
                {
                    double output = 0.0;
                    for (int query = startQuery; query <= endQuery; query++)
                    {
                        int begin = Dataset.Boundaries[query];
                        int end = Dataset.Boundaries[query + 1];

                        if (end - begin <= 1)
                            continue;

                        int bestDoc = _bestDocsPerQuery.BestDocs[query];
                        int secondBestDoc = _bestDocsPerQuery.SecondBestDocs[query];
                        double bestDocScore = scores[bestDoc];
                        double secondBestDocScore = scores[secondBestDoc];

                        // find max score
                        double max = double.NegativeInfinity;
                        double maxNotBest = double.NegativeInfinity;

                        for (int d = begin; d < end; ++d)
                        {
                            if (max < scores[d])
                                max = scores[d];
                            if (d != bestDoc && maxNotBest < scores[d])
                                maxNotBest = scores[d];
                        }

                        // sum of exponents and sum of all but best
                        double sum = 0.0;
                        double sumAllButBest = 0.0;
                        for (int d = begin; d < end; ++d)
                        {
                            sum += Math.Exp(scores[d] - max);
                            if (d != bestDoc)
                                sumAllButBest += Math.Exp(scores[d] - maxNotBest);
                        }

                        output += max - bestDocScore + Math.Log(sum) + 0.5 * (maxNotBest - secondBestDocScore + Math.Log(sumAllButBest));
                    }
                    lock (_lock)
                    {
                        totalOutput += output;
                    }
                }, queryBegin, Math.Min(queryBegin + chunkSize - 1, Dataset.NumQueries - 1));
            BlockingThreadPool.BlockUntilAllWorkItemsFinish();
            return totalOutput;
        }

        public override IEnumerable<TestResult> ComputeTests(double[] scores)
        {
            List<TestResult> result = new List<TestResult>()
            {
                new TestResult("DominationLoss", ComputeDominationLoss(scores), 1, true, TestResult.ValueOperator.Sum),
            };

            return result;
        }
    }
#endif
}
