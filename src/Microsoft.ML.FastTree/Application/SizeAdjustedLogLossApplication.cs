// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime.CommandLine;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
#if OLD_DATALOAD
    public class SizeAdjustedLogLossCommandLineArgs : TrainingCommandLineArgs
    {
        public enum LogLossMode
        {
            Pairwise,
            Wholepage
        };

        public enum CostFunctionMode
        {
            SizeAdjustedWinratePredictor,
            SizeAdjustedPageOrdering
        };

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Which style of log loss to use", ShortName = "llm")]
        public LogLossMode loglossmode = LogLossMode.Pairwise;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "Which style of cost function to use", ShortName = "cfm")]
        public CostFunctionMode costFunctionMode = CostFunctionMode.SizeAdjustedPageOrdering;

        // REVIEW: If we ever want to expose this application in TLC, the natural thing would be for these to
        // be loaded from a column from the input data view. However I'll keep them around for now, so that when we
        // do migrate there's some clear idea of how it is to be done.
        [Argument(ArgumentType.AtMostOnce, HelpText = "tab seperated file which contains the max and min score from the model", ShortName = "srange")]
        public string scoreRangeFileName = null;

        [Argument(ArgumentType.MultipleUnique, HelpText = "TSV filename with float size values associated with train bin file", ShortName = "trsl")]
        public string[] trainSizeLabelFilenames = null;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "TSV filename with float size values associated with validation bin file", ShortName = "vasl")]
        public string validSizeLabelFilename = null;

        [Argument(ArgumentType.MultipleUnique, HelpText = "TSV filename with float size values associated with test bin file", ShortName = "tesl")]
        public string[] testSizeLabelFilenames = null;

        [Argument(ArgumentType.LastOccurenceWins, HelpText = "log loss cefficient", ShortName = "llc")]
        public double loglosscoef = 1.0;

    }

    public class SizeAdjustedLogLossUtil
    {
        public const string sizeLabelName = "size";

        public static float[] GetSizeLabels(Dataset set)
        {
            if (set == null)
            {
                return null;
            }
            float[] labels = set.Skeleton.GetData<float>(sizeLabelName);
            if (labels == null)
            {
                labels = set.Ratings.Select(x => (float)x).ToArray();
            }
            return labels;
        }
    }

    public class SizeAdjustedLogLossTrainingApplication : ApplicationBase
    {
        new SizeAdjustedLogLossCommandLineArgs cmd;
        float[] trainSetSizeLabels;
        private const string RegistrationName = "SizeAdjustedLogLossApplication";

        public SizeAdjustedLogLossTrainingApplication(IHostEnvironment env, string args, TrainingApplicationData data)
            : base(env, RegistrationName, args, data)
        {
            base.cmd = this.cmd = new SizeAdjustedLogLossCommandLineArgs();
        }

        public override ObjectiveFunction ConstructObjFunc()
        {
            return new SizeAdjustedLogLossObjectiveFunction(TrainSet, cmd);
        }

        public override OptimizationAlgorithm ConstructOptimizationAlgorithm(IChannel ch)
        {
            OptimizationAlgorithm optimizationAlgorithm = base.ConstructOptimizationAlgorithm(ch);
            //          optimizationAlgorithm.AdjustTreeOutputsOverride = new NoOutputOptimization(); // For testing purposes - this will not use line search and thus the scores won't be scaled.
            var lossCalculator = new SizeAdjustedLogLossTest(optimizationAlgorithm.TrainingScores, cmd.scoreRangeFileName, trainSetSizeLabels, cmd.loglosscoef);

            // The index of the label signifies which index from TestResult would be used as a loss. For every query, we compute both wholepage and pairwise loss with the two cost function modes, this index lets us pick the appropriate one.
            int lossIndex = 0;
            if (cmd.loglossmode == SizeAdjustedLogLossCommandLineArgs.LogLossMode.Wholepage && cmd.costFunctionMode == SizeAdjustedLogLossCommandLineArgs.CostFunctionMode.SizeAdjustedPageOrdering)
            {
                lossIndex = 1;
            }
            else if (cmd.loglossmode == SizeAdjustedLogLossCommandLineArgs.LogLossMode.Pairwise && cmd.costFunctionMode == SizeAdjustedLogLossCommandLineArgs.CostFunctionMode.SizeAdjustedWinratePredictor)
            {
                lossIndex = 2;
            }
            else if (cmd.loglossmode == SizeAdjustedLogLossCommandLineArgs.LogLossMode.Wholepage && cmd.costFunctionMode == SizeAdjustedLogLossCommandLineArgs.CostFunctionMode.SizeAdjustedWinratePredictor)
            {
                lossIndex = 3;
            }

            optimizationAlgorithm.AdjustTreeOutputsOverride = new LineSearch(lossCalculator, lossIndex, cmd.numPostBracketSteps, cmd.minStepSize);

            return optimizationAlgorithm;
        }

        private static IEnumerable<float> LoadSizeLabels(string filename)
        {
            using (StreamReader reader = new StreamReader(new FileStream(filename, FileMode.Open, FileAccess.Read, FileShare.Read)))
            {
                string line = reader.ReadLine();
                Contracts.Check(line != null && line.Trim() == "m:Size", "Regression label file should contain only one column m:Size");
                while ((line = reader.ReadLine()) != null)
                {
                    float val = float.Parse(line.Trim(), CultureInfo.InvariantCulture);
                    yield return val;
                }
            }
        }

        protected override void PrepareLabels(IChannel ch)
        {
            trainSetSizeLabels = SizeAdjustedLogLossUtil.GetSizeLabels(TrainSet);
        }

        protected override Test ConstructTestForTrainingData()
        {
            return new SizeAdjustedLogLossTest(ConstructScoreTracker(TrainSet), cmd.scoreRangeFileName, SizeAdjustedLogLossUtil.GetSizeLabels(TrainSet), cmd.loglosscoef);
        }

        protected override void ProcessBinFile(int trainValidTest, int index, DatasetBinFile bin)
        {
            string labelPath = null;
            switch (trainValidTest)
            {
            case 0:
                if (cmd.trainSizeLabelFilenames != null && index < cmd.trainSetFilenames.Length)
                {
                    labelPath = cmd.trainSizeLabelFilenames[index];
                }
                break;
            case 1:
                if (cmd.validSizeLabelFilename != null)
                {
                    labelPath = cmd.validSizeLabelFilename;
                }
                break;
            case 2:
                if (cmd.testSizeLabelFilenames != null && index < cmd.testSizeLabelFilenames.Length)
                {
                    labelPath = cmd.testSizeLabelFilenames[index];
                }
                break;
            }
            // If we have no labels, return.
            if (labelPath == null)
            {
                return;
            }
            float[] labels = LoadSizeLabels(labelPath).ToArray();
            bin.DatasetSkeleton.SetData(SizeAdjustedLogLossUtil.sizeLabelName, labels, false);
        }

        protected override void InitializeTests()
        {
            Tests.Add(new SizeAdjustedLogLossTest(ConstructScoreTracker(TrainSet), cmd.scoreRangeFileName, SizeAdjustedLogLossUtil.GetSizeLabels(TrainSet), cmd.loglosscoef));
            if (ValidSet != null)
            {
                Tests.Add(new SizeAdjustedLogLossTest(ConstructScoreTracker(ValidSet), cmd.scoreRangeFileName, SizeAdjustedLogLossUtil.GetSizeLabels(ValidSet), cmd.loglosscoef));
            }

            if (TestSets != null && TestSets.Length > 0)
            {
                for (int t = 0; t < TestSets.Length; ++t)
                {
                    Tests.Add(new SizeAdjustedLogLossTest(ConstructScoreTracker(TestSets[t]), cmd.scoreRangeFileName, SizeAdjustedLogLossUtil.GetSizeLabels(TestSets[t]), cmd.loglosscoef));
                }
            }
        }

        public override void PrintIterationMessage(IChannel ch, IProgressChannel pch)
        {
            base.PrintIterationMessage(ch, pch);
        }
    }

    public class SizeAdjustedLogLossObjectiveFunction : RankingObjectiveFunction
    {
        private SizeAdjustedLogLossCommandLineArgs.LogLossMode _mode;
        private SizeAdjustedLogLossCommandLineArgs.CostFunctionMode _algo;
        private double _llc;

        public SizeAdjustedLogLossObjectiveFunction(Dataset trainSet, SizeAdjustedLogLossCommandLineArgs cmd)
            : base(trainSet, trainSet.Ratings, cmd)
        {
            _mode = cmd.loglossmode;
            _algo = cmd.costFunctionMode;
            _llc = cmd.loglosscoef;
        }

        protected override void GetGradientInOneQuery(int query, int threadIndex)
        {
            int begin = Dataset.Boundaries[query];
            int end = Dataset.Boundaries[query + 1];
            short[] labels = Dataset.Ratings;
            float[] sizes = SizeAdjustedLogLossUtil.GetSizeLabels(Dataset);

            Contracts.Check(Dataset.NumDocs == sizes.Length, "Mismatch between dataset and labels");

            if (end - begin <= 1)
            {
                return;
            }
            Array.Clear(_gradient, begin, end - begin);

            for (int d1 = begin; d1 < end - 1; ++d1)
            {
                for (int d2 = d1 + 1; d2 < end; ++d2)
                {
                    float size = sizes[d1];

                    //Compute Lij
                    float sizeAdjustedLoss = 0.0F;
                    for (int d3 = d2; d3 < end; ++d3)
                    {
                        size -= sizes[d3];
                        if (size >= 0.0F && labels[d3] > 0)
                        {
                            sizeAdjustedLoss = 1.0F;
                        }
                        else if (size < 0.0F && labels[d3] > 0)
                        {
                            sizeAdjustedLoss = (1.0F + (size / sizes[d3]));
                        }

                        if (size <= 0.0F || sizeAdjustedLoss > 0.0F)
                        {
                            // Exit condition- we have reached size or size adjusted loss is already populated.
                            break;
                        }
                    }

                    double scoreDiff = _scores[d1] - _scores[d2];
                    float labelDiff = ((float)labels[d1] - sizeAdjustedLoss);
                    double delta = 0.0;
                    if (_algo == SizeAdjustedLogLossCommandLineArgs.CostFunctionMode.SizeAdjustedPageOrdering)
                    {
                        delta = (_llc * labelDiff) / (1.0 + Math.Exp(_llc * labelDiff * scoreDiff));
                    }
                    else
                    {
                        delta = (double)labels[d1] - ((double)(labels[d1] + sizeAdjustedLoss) / (1.0 + Math.Exp(-scoreDiff)));
                    }

                    _gradient[d1] += delta;
                    _gradient[d2] -= delta;

                    if (_mode == SizeAdjustedLogLossCommandLineArgs.LogLossMode.Pairwise)
                    {
                        break;
                    }
                }
            }
        }
    }

    public class SizeAdjustedLogLossTest : Test
    {
        protected string _scoreRangeFileName = null;
        static double maxScore = double.MinValue;
        static double minScore = double.MaxValue;
        private float[] _sizeLabels;
        private double _llc;

        public SizeAdjustedLogLossTest(ScoreTracker scoreTracker, string scoreRangeFileName, float[] sizeLabels, double loglossCoeff)
            : base(scoreTracker)
        {
            Contracts.Check(scoreTracker.Dataset.NumDocs == sizeLabels.Length, "Mismatch between dataset and labels");
            _sizeLabels = sizeLabels;
            _scoreRangeFileName = scoreRangeFileName;
            _llc = loglossCoeff;
        }

        public override IEnumerable<TestResult> ComputeTests(double[] scores)
        {
            Object _lock = new Object();
            double pairedPageOrderingLoss = 0.0;
            double allPairPageOrderingLoss = 0.0;
            double pairedSAWRPredictLoss = 0.0;
            double allPairSAWRPredictLoss = 0.0;

            short maxLabel = 0;
            short minLabel = 10000;
            for (int query = 0; query < Dataset.Boundaries.Length - 1; ++query)
            {
                int begin = Dataset.Boundaries[query];
                int end = Dataset.Boundaries[query + 1];

                if (end - begin <= 1)
                {
                    continue;
                }

                for (int d1 = begin; d1 < end - 1; ++d1)
                {
                    bool firstTime = false;
                    for (int d2 = d1 + 1; d2 < end; ++d2)
                    {
                        float size = _sizeLabels[d1];

                        //Compute Lij
                        float sizeAdjustedLoss = 0.0F;
                        for (int d3 = d2; d3 < end; ++d3)
                        {
                            size -= _sizeLabels[d3];
                            if (size >= 0.0F && Dataset.Ratings[d3] > 0)
                            {
                                sizeAdjustedLoss = 1.0F;
                            }
                            else if (size < 0.0F && Dataset.Ratings[d3] > 0)
                            {
                                sizeAdjustedLoss = (1.0F + (size / _sizeLabels[d3]));
                            }

                            if (size <= 0.0F || sizeAdjustedLoss > 0.0F)
                            {
                                // Exit condition- we have reached size or size adjusted loss is already populated.
                                break;
                            }
                        }
                        //Compute page ordering loss
                        double scoreDiff = scores[d1] - scores[d2];
                        float labelDiff = ((float)Dataset.Ratings[d1] - sizeAdjustedLoss);
                        double pageOrderingLoss = Math.Log(1.0 + Math.Exp(-_llc * labelDiff * scoreDiff));

                        // Compute SAWR predict loss
                        double sawrPredictLoss = (double)((Dataset.Ratings[d1] + sizeAdjustedLoss) * Math.Log(1.0 + Math.Exp(scoreDiff))) - ((double)Dataset.Ratings[d1] * scoreDiff);
                        if (!firstTime)
                        {
                            pairedPageOrderingLoss += pageOrderingLoss;
                            pairedSAWRPredictLoss += sawrPredictLoss;
                            firstTime = true;
                        }
                        allPairPageOrderingLoss += pageOrderingLoss;
                        allPairSAWRPredictLoss += sawrPredictLoss;
                    }
                }
            }

            for (int i = 0; i < Dataset.Ratings.Length; i++)
            {
                if (Dataset.Ratings[i] > maxLabel)
                    maxLabel = Dataset.Ratings[i];
                if (Dataset.Ratings[i] < minLabel)
                    minLabel = Dataset.Ratings[i];
                if (scores[i] > maxScore)
                    maxScore = scores[i];
                if (scores[i] < minScore)
                    minScore = scores[i];
            }

            if (_scoreRangeFileName != null)
            {
                using (StreamWriter sw = File.CreateText(_scoreRangeFileName))
                {
                    sw.WriteLine(string.Format("{0}\t{1}", minScore, maxScore));
                }
            }

            List<TestResult> result = new List<TestResult>()
            {
                // The index of the label signifies which index from TestResult would be used as a loss. For every query, we compute both wholepage and pairwise loss with the two cost function modes, this index lets us pick the appropriate one.
                new TestResult("page ordering paired loss", pairedPageOrderingLoss, Dataset.NumDocs, true, TestResult.ValueOperator.Average),
                new TestResult("page ordering all pairs loss", allPairPageOrderingLoss, Dataset.NumDocs, true, TestResult.ValueOperator.Average),
                new TestResult("SAWR predict paired loss", pairedSAWRPredictLoss, Dataset.NumDocs, true, TestResult.ValueOperator.Average),
                new TestResult("SAWR predict all pairs loss", allPairSAWRPredictLoss, Dataset.NumDocs, true, TestResult.ValueOperator.Average),
                new TestResult("max Label", maxLabel, 1, false, TestResult.ValueOperator.Max),
                new TestResult("min Label", minLabel, 1, true, TestResult.ValueOperator.Min),
            };

            return result;
        }
    }
#endif
}
