// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime.CommandLine;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
#if OLD_DATALOAD
    public class LogLossCommandLineArgs : TrainingCommandLineArgs
    {
        public enum LogLossMode { Pairwise, Wholepage };
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Which style of log loss to use", ShortName = "llm")]
        public LogLossMode loglossmode = LogLossMode.Pairwise;
        [Argument(ArgumentType.LastOccurenceWins, HelpText = "log loss cefficient", ShortName = "llc")]
        public double loglosscoef = 1.0;
    }

    public class LogLossTrainingApplication : ApplicationBase
    {
        new LogLossCommandLineArgs cmd;
        private const string RegistrationName = "LogLossApplication";

        public LogLossTrainingApplication(IHostEnvironment env, string args, TrainingApplicationData data)
            : base(env, RegistrationName, args, data)
        {
            base.cmd = this.cmd = new LogLossCommandLineArgs();
        }

        public override ObjectiveFunction ConstructObjFunc()
        {
            return new LogLossObjectiveFunction(TrainSet, cmd);
        }

        public override OptimizationAlgorithm ConstructOptimizationAlgorithm(IChannel ch)
        {
            OptimizationAlgorithm optimizationAlgorithm = base.ConstructOptimizationAlgorithm(ch);
            //            optimizationAlgorithm.AdjustTreeOutputsOverride = new NoOutputOptimization();
            var lossCalculator = new LogLossTest(optimizationAlgorithm.TrainingScores, cmd.loglosscoef);
            int lossIndex = (cmd.loglossmode == LogLossCommandLineArgs.LogLossMode.Pairwise) ? 0 : 1;

            optimizationAlgorithm.AdjustTreeOutputsOverride = new LineSearch(lossCalculator, lossIndex, cmd.numPostBracketSteps, cmd.minStepSize);

            return optimizationAlgorithm;
        }

        protected override void PrepareLabels(IChannel ch)
        {
        }

        protected override Test ConstructTestForTrainingData()
        {
            return new LogLossTest(ConstructScoreTracker(TrainSet), cmd.loglosscoef);
        }

        protected override void InitializeTests()
        {
            Tests.Add(new LogLossTest(ConstructScoreTracker(TrainSet), cmd.loglosscoef));
            if (ValidSet != null)
                Tests.Add(new LogLossTest(ConstructScoreTracker(ValidSet), cmd.loglosscoef));

            if (TestSets != null)
            {
                for (int t = 0; t < TestSets.Length; ++t)
                {
                    Tests.Add(new LogLossTest(ConstructScoreTracker(TestSets[t]), cmd.loglosscoef));
                }
            }
        }
    }

    public class LogLossObjectiveFunction : RankingObjectiveFunction
    {
        private LogLossCommandLineArgs.LogLossMode _mode;
        private double _coef = 1.0;

        public LogLossObjectiveFunction(Dataset trainSet, LogLossCommandLineArgs cmd)
            : base(trainSet, trainSet.Ratings, cmd)
        {
            _mode = cmd.loglossmode;
            _coef = cmd.loglosscoef;
        }

        protected override void GetGradientInOneQuery(int query, int threadIndex)
        {
            int begin = Dataset.Boundaries[query];
            int end = Dataset.Boundaries[query + 1];
            short[] labels = Dataset.Ratings;

            if (end - begin <= 1)
                return;
            Array.Clear(_gradient, begin, end - begin);

            for (int d1 = begin; d1 < end - 1; ++d1)
            {
                int stop = (_mode == LogLossCommandLineArgs.LogLossMode.Pairwise) ? d1 + 2 : end;
                for (int d2 = d1 + 1; d2 < stop; ++d2)
                {
                    short labelDiff = (short)(labels[d1] - labels[d2]);
                    if (labelDiff == 0)
                        continue;
                    double delta = (_coef * labelDiff) / (1.0 + Math.Exp(_coef * labelDiff * (_scores[d1] - _scores[d2])));

                    _gradient[d1] += delta;
                    _gradient[d2] -= delta;
                }
            }
        }
    }

    public class LogLossTest : Test
    {
        protected double _coef;
        public LogLossTest(ScoreTracker scoreTracker, double coef)
            : base(scoreTracker)
        {
            _coef = coef;
        }

        public override IEnumerable<TestResult> ComputeTests(double[] scores)
        {
            Object _lock = new Object();
            double pairedLoss = 0.0;
            double allPairLoss = 0.0;
            short maxLabel = 0;
            short minLabel = 10000;

            for (int query = 0; query < Dataset.Boundaries.Length - 1; query++)
            {
                int start = Dataset.Boundaries[query];
                int length = Dataset.Boundaries[query + 1] - start;
                for (int i = start; i < start + length; i++)
                    for (int j = i + 1; j < start + length; j++)
                        allPairLoss += Math.Log((1.0 + Math.Exp(-_coef * (Dataset.Ratings[i] - Dataset.Ratings[j]) * (scores[i] - scores[j]))));
                //allPairLoss += Math.Max(0.0, _coef - (Dataset.Ratings[i] - Dataset.Ratings[j]) * (scores[i] - scores[j]));

                for (int i = start; i < start + length - 1; i++)
                {
                    pairedLoss += Math.Log((1.0 + Math.Exp(-_coef * (Dataset.Ratings[i] - Dataset.Ratings[i + 1]) * (scores[i] - scores[i + 1]))));
                    //                     pairedLoss += Math.Max(0.0, _coef - (Dataset.Ratings[i] - Dataset.Ratings[i + 1]) * (scores[i] - scores[i + 1]));
                }
            }

            for (int i = 0; i < Dataset.Ratings.Length; i++)
            {
                if (Dataset.Ratings[i] > maxLabel)
                    maxLabel = Dataset.Ratings[i];
                if (Dataset.Ratings[i] < minLabel)
                    minLabel = Dataset.Ratings[i];
            }
            List<TestResult> result = new List<TestResult>()
            {
                new TestResult("paired loss", pairedLoss, Dataset.NumDocs, true, TestResult.ValueOperator.Average),
                new TestResult("all pairs loss", allPairLoss, Dataset.NumDocs, true, TestResult.ValueOperator.Average),
                new TestResult("coefficient", _coef, 1, true, TestResult.ValueOperator.Constant),
                new TestResult("max Label", maxLabel, 1, false, TestResult.ValueOperator.Max),
                new TestResult("min Label", minLabel, 1, true, TestResult.ValueOperator.Min),
            };

            return result;
        }
    }

    public class NoOutputOptimization : IStepSearch
    {
        public NoOutputOptimization() { }

        public void AdjustTreeOutputs(IChannel ch, RegressionTree tree, DocumentPartitioning partitioning, ScoreTracker trainingScores) { }
    }
#endif
}
