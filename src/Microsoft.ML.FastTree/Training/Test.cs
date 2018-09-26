// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    public sealed class TestResult : IComparable<TestResult>
    {
        public enum ValueOperator : int
        {
            None = 0, // the final value will be the raw value,
            // and the test result can't be used for parallel test

            Average, // the final value will be raw value / factor

            SqrtAverage, // the final value will be sqrt(raw value / factor)

            Sum, // the final value will be the raw value for single test result,
            // and the final value of multiple test results will be sum(raw values).

            Max, // the final value will be the raw value for single test result,
            // and the final value of multiple test results will be max(raw values).

            Min, // the final value will be the raw value for single test result,
            // and the final value of multiple test results will be min(raw values).

            Constant, // the final value will be the raw value for single test result, and
            // the raw value should be the same constant for all test results.
        }

        public string LossFunctionName { get; }

        /// <summary>
        /// Raw value used for calculating final test result value.
        /// </summary>
        public double RawValue { get; }

        /// <summary>
        /// The factor used for calculating final test result value.
        /// </summary>
        public double Factor { get; }

        /// <summary>
        /// The operator used for calculating final test result value.
        /// Final value = Operator(RawValue, Factor)
        /// </summary>
        public ValueOperator Operator { get; }

        /// <summary>
        /// Indicates that the lower value of this metric is better
        /// This is used for early stopping (with TestHistory and TestWindowWithTolerance)
        /// </summary>
        public bool LowerIsBetter { get; }

        public double FinalValue { get; }

        public TestResult(string lossFunctionName, double rawValue, double factor, bool lowerIsBetter, ValueOperator valueOperator)
        {
            LossFunctionName = lossFunctionName;
            RawValue = rawValue;
            Factor = factor;
            Operator = valueOperator;
            LowerIsBetter = lowerIsBetter;

            FinalValue = CalculateFinalValue();
        }

        public int CompareTo(TestResult o)
        {
            if (LossFunctionName != o.LossFunctionName)
                throw Contracts.Except("Cannot compare unrelated metrics");

            return FinalValue.CompareTo(o.FinalValue) * (LowerIsBetter ? -1 : 1);
        }

        public int SizeInBytes()
        {
            int lowerIsBetter = LowerIsBetter ? 1 : 0;
            int valueOperator = (int)Operator;

            return RawValue.SizeInBytes()
                + Factor.SizeInBytes()
                + LossFunctionName.SizeInBytes()
                + lowerIsBetter.SizeInBytes()
                + valueOperator.SizeInBytes();
        }

        public void ToByteArray(byte[] buffer, ref int offset)
        {
            LossFunctionName.ToByteArray(buffer, ref offset);
            RawValue.ToByteArray(buffer, ref offset);
            Factor.ToByteArray(buffer, ref offset);

            int lowerIsBetter = LowerIsBetter ? 1 : 0;
            lowerIsBetter.ToByteArray(buffer, ref offset);

            int valueOperator = (int)Operator;
            valueOperator.ToByteArray(buffer, ref offset);
        }

        public static TestResult FromByteArray(byte[] buffer, ref int offset)
        {
            string lossFunctionName = buffer.ToString(ref offset);
            double rawValue = buffer.ToDouble(ref offset);
            double factor = buffer.ToDouble(ref offset);
            int lowerIsBetter = buffer.ToInt(ref offset);
            int valueOperator = buffer.ToInt(ref offset);

            return new TestResult(
                lossFunctionName,
                rawValue,
                factor,
                lowerIsBetter != 0,
                (ValueOperator)valueOperator);
        }

        private double CalculateFinalValue()
        {
            switch (Operator)
            {
                case ValueOperator.Constant:
                case ValueOperator.Max:
                case ValueOperator.Min:
                case ValueOperator.None:
                case ValueOperator.Sum:
                    return RawValue;
                case ValueOperator.Average:
                    return RawValue / Factor;
                case ValueOperator.SqrtAverage:
                    return Math.Sqrt(RawValue / Factor);
                default:
                    throw Contracts.Except("Unsupported value operator: {0}", Operator);
            }
        }
    }

    public abstract class Test
    {
        public ScoreTracker ScoreTracker;
        public Dataset Dataset => ScoreTracker.Dataset;

        //Keeps last returned results by ComputeTests(). UpdateScores invalidates cache.
        protected IEnumerable<TestResult> CachedResults;

        //The method returns one or more losses on a given Dataset
        public abstract IEnumerable<TestResult> ComputeTests(double[] scores);
        private protected Test(ScoreTracker scoreTracker)
        {
            ScoreTracker = scoreTracker;
            if (ScoreTracker != null)
                ScoreTracker.ScoresUpdated += OnScoresUpdated;
        }

        public Test(string datasetName, Dataset set, double[] initScores)
            : this(new ScoreTracker(datasetName, set, initScores)) { }

        public virtual void OnScoresUpdated()
        {
            CachedResults = null;
        }

        public virtual IEnumerable<TestResult> ComputeTests()
        {
            if (CachedResults == null)
                CachedResults = ComputeTests(ScoreTracker.Scores);
            return CachedResults;
        }

        public IEnumerable<TestResult> TestScores(double[] scores)
        {
            return ComputeTests(scores);
        }

        // This is the info string that represnts the cotent in teh most descriptive fashion
        // The main diffrence between ConsoleString is always printed. The caller is responsible for deciding if InfoString is InfoString needs to be printed or not
        public virtual string FormatInfoString()
        {
            var sb = new System.Text.StringBuilder();
            foreach (var r in ComputeTests())
            {
                sb.AppendFormat("{0}.{1}={2}\n", ScoreTracker.DatasetName, r.LossFunctionName, r.FinalValue);
            }
            return sb.ToString();
        }
    };

    // A simple class that tracks history of underlying Test.
    // It captures an iteration that peak on a given metric
    // Each itaratin captures an array of LossFunctions computed by inderlying Test
    public class TestHistory : Test
    {
        public readonly Test SimpleTest;
        public readonly int LossIndex;
        protected IList<TestResult[]> History;
        protected int Iteration { get; private set; }

        public TestResult BestResult { get; private protected set; }
        public int BestIteration { get; private protected set; }

        // scenarioWithoutHistory - simple test scenario we want to track the history and look for best iteration
        // lossIndex - index of lossFunction in case Test returns more than one loss (default should be 0)
        // lower is better: are we looking for minimum or maximum of loss function?
        internal TestHistory(Test scenarioWithoutHistory, int lossIndex)
            : base(null)
        {
            History = new List<TestResult[]>();
            SimpleTest = scenarioWithoutHistory;
            LossIndex = lossIndex;
            BestIteration = -1;
            SimpleTest.ScoreTracker.ScoresUpdated += OnScoresUpdated;
        }

        public sealed override void OnScoresUpdated()
        {
            Iteration++;
            var results = SimpleTest.ComputeTests().ToArray();
            UpdateBest(results[LossIndex]);
            History.Add(results);
        }

        protected virtual void UpdateBest(TestResult r)
        {
            if (BestResult == null || BestResult.CompareTo(r) == -1)
            {
                BestResult = r;
                BestIteration = Iteration;
            }
        }

        public sealed override IEnumerable<TestResult> ComputeTests()
        {
            //We assume that a _simpleTest keeps ownership of scores
            //We don't double cache he results let the SimpleTest cache
            return SimpleTest.ComputeTests();
        }

        public sealed override IEnumerable<TestResult> ComputeTests(double[] scores)
        {
            yield return SimpleTest.ComputeTests(scores).ToArray()[LossIndex];
        }

        public sealed override string FormatInfoString()
        {
            return SimpleTest.FormatInfoString();
        }
    }

    // A class that tracks history of underlying Test.
    // Can capture an iteration that peak on a given metric
    // Each itaratin captures an array of LossFunctions computed by inderlying Test
    public class TestWindowWithTolerance : TestHistory
    {
        // Struct to keep information for tolerant early stopping
        private struct ValueIterationPair
        {
            public int Iteration;
            public double Sum;

            public ValueIterationPair(int iteration, double sum)
            {
                Iteration = iteration;
                Sum = sum;
            }
        }

        private readonly int _windowSize;
        private readonly double _tolerance;
        // Queue for moving window
        private LinkedList<double> _window;

        // This queue keeps track of the iterations which are within tolerance from the best iteration
        // The first element of the queue is the early stopping candidate
        private LinkedList<ValueIterationPair> _toleratedQueue;

        // Average validation for the current window
        private double _currentWindowSum;

        public double BestAverageValue => _toleratedQueue.Count == 0 ? 0.0 : _toleratedQueue.First().Sum / _windowSize;
        public double CurrentAverageValue => _currentWindowSum / _windowSize;

        // windowSize - number of iterations of average
        // tolerance - how much off we can be from the best average (0.04 stand that we consider the best itration the average over the window is 4% worse than the best average)
        public TestWindowWithTolerance(Test scenarioWithoutHistory, int lossIndex,
                                       int windowSize, double tolerance)
            : base(scenarioWithoutHistory, lossIndex)
        {
            _window = new LinkedList<double>();
            _toleratedQueue = new LinkedList<ValueIterationPair>();
            _windowSize = windowSize;
            _tolerance = tolerance;
        }

        protected override void UpdateBest(TestResult r)
        {
            if (BestResult != null && r.LowerIsBetter != BestResult.LowerIsBetter)
                throw Contracts.Except("TestResult don't match");

            double currentValue = ComputeTests().First().FinalValue * (r.LowerIsBetter ? -1.0 : 1.0);
            double toleranceFactor = 1.0 - (_tolerance * (r.LowerIsBetter ? -1.0 : 1.0));
            if (_window.Count == _windowSize)
            {
                double outValue = _window.First();
                _window.RemoveFirst();
                _window.AddLast(currentValue);
                _currentWindowSum = _currentWindowSum - outValue + currentValue;
            }
            else
            {
                _currentWindowSum = _currentWindowSum + currentValue;
                _window.AddLast(currentValue);
            }

            // Add to queue if higher than the current best
            if (_window.Count == _windowSize &&
                 (_toleratedQueue.Count == 0 || _currentWindowSum > _toleratedQueue.Last().Sum))
            {
                _toleratedQueue.AddLast(new ValueIterationPair(Iteration - _windowSize / 2, _currentWindowSum));

                // If the earliest candidate iteration is beyond tolerance, pop it out
                while (_toleratedQueue.First().Sum < _currentWindowSum * toleranceFactor)
                {
                    _toleratedQueue.RemoveFirst();
                }

                BestIteration = _toleratedQueue.First().Iteration;
                BestResult = History[BestIteration - 1][LossIndex];
            }
        }
    }

    public class NdcgTest : Test
    {
        protected readonly DcgCalculator DcgCalculator;
        private readonly string _sortingAlgorithm;
        protected readonly short[] Labels;

        internal NdcgTest(ScoreTracker scoreTracker, short[] labels, string sortingAlgorithm)
            : base(scoreTracker)
        {
            Labels = labels;
            Contracts.Check(scoreTracker.Dataset.NumDocs == labels.Length, "Mismatch between dataset and labels");
            _sortingAlgorithm = sortingAlgorithm;
            DcgCalculator = new DcgCalculator(Dataset.MaxDocsPerQuery, _sortingAlgorithm);
        }

        public override IEnumerable<TestResult> ComputeTests(double[] scores)
        {
            IList<TestResult> result = new List<TestResult>();
            double[] ndcg = DcgCalculator.NdcgRangeFromScores(Dataset, Labels, scores);
            for (int i = 0; i < ndcg.Length; i++)
            {
                result.Add(new TestResult("NDCG@" + (i + 1).ToString(), ndcg[i] * Dataset.NumQueries, Dataset.NumQueries, false, TestResult.ValueOperator.Average));
            }
            return result;
        }

        public override string FormatInfoString()
        {
            var sb = new System.Text.StringBuilder();
            sb.Append(ScoreTracker.DatasetName);
            sb.Append("NDCG:\t");
            int i = 1;
            foreach (var t in ComputeTests())
            {
                if (i > 1)
                    sb.Append("\t");
                sb.AppendFormat("@{0}:{1:00.00}", i++, 100.0 * t.FinalValue);
            }
            sb.AppendLine();
            return sb.ToString();
        }
    }

    public class FastNdcgTest : NdcgTest
    {
        protected readonly int NdcgTruncation;

        public FastNdcgTest(ScoreTracker scoreTracker, short[] labels, string sortingAlgorithm, int ndcgTruncation)
            : base(scoreTracker, labels, sortingAlgorithm)
        {
            Contracts.CheckParam(ndcgTruncation == 1 || ndcgTruncation == 3, nameof(ndcgTruncation),
                nameof(FastNdcgTest) + " only supports NDCG1 & NDCG3");
            NdcgTruncation = ndcgTruncation;
        }

        public override IEnumerable<TestResult> ComputeTests(double[] scores)
        {
            double fastNdcg = 0;
            switch (NdcgTruncation)
            {
                case 1:
                    fastNdcg = DcgCalculator.Ndcg1(Dataset, Labels, scores);
                    break;
                case 3:
                    fastNdcg = DcgCalculator.Ndcg3(Dataset, Labels, scores);
                    break;
                default:
                    Contracts.Assert(false);
                    throw Contracts.Except();
            }

            List<TestResult> result = new List<TestResult>()
            {
                new TestResult("NDCG@" + NdcgTruncation.ToString(), fastNdcg * Dataset.NumQueries, Dataset.NumQueries, false, TestResult.ValueOperator.Average),
            };

            return result;
        }
    }

    public sealed class FastNdcgTestForTrainSet : FastNdcgTest
    {
        private readonly ScoreTracker _trainingScores;
        private readonly FastTreeRankingTrainer.LambdaRankObjectiveFunction _rankingObjectiveFunction;

        public FastNdcgTestForTrainSet(ScoreTracker trainingScores, FastTreeRankingTrainer.LambdaRankObjectiveFunction rankingObjectiveFunction, short[] labels, string sortingAlgorithm, int ndcgTruncation)
            : base(trainingScores, labels, sortingAlgorithm, ndcgTruncation)
        {
            _trainingScores = trainingScores;
            _rankingObjectiveFunction = rankingObjectiveFunction;
        }

        public override IEnumerable<TestResult> ComputeTests()
        {
            if (CachedResults == null)
                CachedResults = ComputeTests(_trainingScores.Scores);
            return CachedResults;
        }

        public override IEnumerable<TestResult> ComputeTests(double[] scores)
        {
            short[][] trainQueriesTopLabels = _rankingObjectiveFunction.TrainQueriesTopLabels;
            double fastNdcg = 0;
            switch (NdcgTruncation)
            {
                case 1:
                    fastNdcg = DcgCalculator.Ndcg1(Dataset, trainQueriesTopLabels);
                    break;
                case 3:
                    fastNdcg = DcgCalculator.Ndcg3(Dataset, trainQueriesTopLabels);
                    break;
                default:
                    throw Contracts.Except("FastNDCGTest only supports NDCG1 & NDCG3");
            }
            List<TestResult> result = new List<TestResult>()
            {
                new TestResult("NDCG@" + NdcgTruncation.ToString(), fastNdcg * Dataset.NumQueries, Dataset.NumQueries, false, TestResult.ValueOperator.Average),
            };

            return result;
        }
    }

    public sealed class WinLossSurplusTest : Test
    {
        private readonly Lazy<WinLossCalculator> _winLossCalculator;

        private readonly double _scaleFactor;
        private readonly string _sortingAlgorithm;
        private readonly short[] _labels;

        public WinLossSurplusTest(ScoreTracker scoreTracker, short[] labels, string sortingAlgorithm, double scaleFactor)
            : base(scoreTracker)
        {
            _labels = labels;
            _sortingAlgorithm = sortingAlgorithm;
            _scaleFactor = scaleFactor;
            _winLossCalculator = new Lazy<WinLossCalculator>(
                () => new WinLossCalculator(Dataset.MaxDocsPerQuery, _sortingAlgorithm));
        }

        public override IEnumerable<TestResult> ComputeTests(double[] scores)
        {
            double[] surplus = _winLossCalculator.Value.WinLossRangeFromScores(Dataset, _labels, scores);

            IList<TestResult> result = new List<TestResult>()
            {
                new TestResult("MaxSurplus", surplus[6] * _scaleFactor, 1.0, false, TestResult.ValueOperator.Sum),
                new TestResult("Surplus@100", surplus[0] * _scaleFactor * Dataset.NumQueries, Dataset.NumQueries, false, TestResult.ValueOperator.Average),
                new TestResult("Surplus@200", surplus[1] * _scaleFactor * Dataset.NumQueries, Dataset.NumQueries, false, TestResult.ValueOperator.Average),
                new TestResult("Surplus@300", surplus[2] * _scaleFactor * Dataset.NumQueries, Dataset.NumQueries, false, TestResult.ValueOperator.Average),
                new TestResult("Surplus@400", surplus[3] * _scaleFactor * Dataset.NumQueries, Dataset.NumQueries, false, TestResult.ValueOperator.Average),
                new TestResult("Surplus@500", surplus[4] * _scaleFactor * Dataset.NumQueries, Dataset.NumQueries, false, TestResult.ValueOperator.Average),
                new TestResult("Surplus@1000", surplus[5] * _scaleFactor * Dataset.NumQueries, Dataset.NumQueries, false, TestResult.ValueOperator.Average),
                new TestResult("MaxSurplusPos", surplus[7], 1, false, TestResult.ValueOperator.Sum),
                new TestResult("PercentTop", surplus[7], surplus[8], false, TestResult.ValueOperator.Average),
            };

            return result;
        }

        public override string FormatInfoString()
        {
            var sb = new System.Text.StringBuilder();
            sb.Append(ScoreTracker.DatasetName);
            sb.Append("WinLossSurplus:\t");
            int i = 1;
            foreach (var t in ComputeTests())
            {
                if (i > 1)
                    sb.Append("\t");
                sb.AppendFormat("{0}:{1:00.00}", t.LossFunctionName, t.FinalValue);
                i++;
            }
            sb.AppendLine();
            return sb.ToString();
        }
    }

    public sealed class RegressionTest : Test
    {
        private readonly float[] _labels;
        private readonly int? _resultType;

        ///<param name="scoreTracker"></param>
        /// <param name="resultType">1: L1, 2: L2. Otherwise, return all.</param>
        public RegressionTest(ScoreTracker scoreTracker, int? resultType = null)
            : base(scoreTracker)
        {
            _labels = FastTreeRegressionTrainer.GetDatasetRegressionLabels(scoreTracker.Dataset);
            Contracts.Check(scoreTracker.Dataset.NumDocs == _labels.Length, "Mismatch between dataset and labels");
            _resultType = resultType;
        }

        public override IEnumerable<TestResult> ComputeTests(double[] scores)
        {
            Object testLock = new Object();
            double[] weights = Dataset.SampleWeights;
            double totalL1Error = 0.0;
            double totalL2Error = 0.0;
            int chunkSize = 1 + Dataset.NumDocs / BlockingThreadPool.NumThreads;   // Minimizes the number of repeat computations in sparse array to have each thread take as big a chunk as possible
            // REVIEW: This partitioning doesn't look optimal.
            // Probably make sence to investigate better ways of splitting data?
            var actions = new Action[(int)Math.Ceiling(1.0 * Dataset.NumDocs / chunkSize)];
            var actionIndex = 0;
            for (int documentStart = 0; documentStart < Dataset.NumDocs; documentStart += chunkSize)
            {
                var startDoc = documentStart;
                var endDoc = Math.Min(documentStart + chunkSize - 1, Dataset.NumDocs - 1);
                actions[actionIndex++] = () =>
                {
                    double l1Error = 0.0;
                    double l2Error = 0.0;
                    for (int i = startDoc; i <= endDoc; i++)
                    {
                        double error = _labels[i] - scores[i];
                        double weight = (weights != null) ? weights[i] : 1.0;
                        l1Error += weight * Math.Abs(error);
                        l2Error += weight * error * error;
                    }
                    lock (testLock)
                    {
                        totalL1Error += l1Error;
                        totalL2Error += l2Error;
                    }
                };
            }
            Parallel.Invoke(new ParallelOptions() { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, actions);

            List<TestResult> result = new List<TestResult>();

            Contracts.Assert(_resultType == null || _resultType == 1 || _resultType == 2);

            switch (_resultType)
            {
                case 1:
                    result.Add(new TestResult("L1", totalL1Error, Dataset.NumDocs, true, TestResult.ValueOperator.Average));
                    break;
                case 2:
                    result.Add(new TestResult("L2", totalL2Error, Dataset.NumDocs, true, TestResult.ValueOperator.SqrtAverage));
                    break;
                default:
                    result.Add(new TestResult("L1", totalL1Error, Dataset.NumDocs, true, TestResult.ValueOperator.Average));
                    result.Add(new TestResult("L2", totalL2Error, Dataset.NumDocs, true, TestResult.ValueOperator.SqrtAverage));
                    break;
            }

            return result;
        }
    }

    public sealed class BinaryClassificationTest : Test
    {
        private readonly bool[] _binaryLabels;
        private readonly double _recipNpos;
        private readonly double _recipNneg;
        private readonly double _sigmoidParameter;

        public BinaryClassificationTest(ScoreTracker scoreTracker, bool[] binaryLabels, double sigmoidParameter)
            : base(scoreTracker)
        {
            _binaryLabels = binaryLabels;
            _sigmoidParameter = sigmoidParameter;

            Contracts.Check(scoreTracker.Dataset.NumDocs == binaryLabels.Length, "Mismatch between dataset and labels");

            long npos;
            long nneg;
            ComputeExampleCounts(_binaryLabels, out npos, out nneg);
            _recipNpos = 1.0 / npos;
            _recipNneg = 1.0 / nneg;
        }

        public static void ComputeExampleCounts(bool[] binaryLabels, out long npos, out long nneg)
        {
            long totalNpos = 0;
            long totalNneg = 0;
            // Compute number number of positives and number of negative examples
            int chunkSize = 1 + binaryLabels.Length / BlockingThreadPool.NumThreads;   // Minimizes the number of repeat computations in sparse array to have each thread take as big a chunk as possible
            // REVIEW: This partitioning doesn't look optimal.
            // Probably make sence to investigate better ways of splitting data?
            var actions = new Action[(int)Math.Ceiling(1.0 * binaryLabels.Length / chunkSize)];
            var actionIndex = 0;
            for (int documentStart = 0; documentStart < binaryLabels.Length; documentStart += chunkSize)
            {
                var startDoc = documentStart;
                var endDoc = Math.Min(documentStart + chunkSize - 1, binaryLabels.Length - 1);
                actions[actionIndex++] = () =>
                {
                    long localNpos = 0;
                    long localNneg = 0;
                    for (int i = startDoc; i <= endDoc; i++)
                    {
                        if (binaryLabels[i])
                            localNpos++;
                        else
                            localNneg++;
                    }
                    Interlocked.Add(ref totalNpos, localNpos);
                    Interlocked.Add(ref totalNneg, localNneg);
                };
            }
            Parallel.Invoke(new ParallelOptions() { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, actions);

            npos = totalNpos;
            nneg = totalNneg;
        }

        public override IEnumerable<TestResult> ComputeTests(double[] scores)
        {
            var testLock = new Object();

            double totalErrorRate = 0.0;
            double totalLossRate = 0.0;
            double totalBalancedErrorRate = 0.0;
            double totalBalancedLossRate = 0.0;
            double totalAllDocumentsWeight = 0.0;

            int chunkSize = 1 + Dataset.NumDocs / BlockingThreadPool.NumThreads;   // Minimizes the number of repeat computations in sparse array to have each thread take as big a chunk as possible
            // REVIEW: This partitioning doesn't look optimal.
            // Probably make sence to investigate better ways of splitting data?
            var actions = new Action[(int)Math.Ceiling(1.0 * Dataset.NumDocs / chunkSize)];
            var actionIndex = 0;
            for (int documentStart = 0; documentStart < Dataset.NumDocs; documentStart += chunkSize)
            {
                var startDoc = documentStart;
                var endDoc = Math.Min(documentStart + chunkSize - 1, Dataset.NumDocs - 1);
                actions[actionIndex++] = () =>
                {
                    double errorRate = 0.0;
                    double lossRate = 0.0;
                    double balancedErrorRate = 0.0;
                    double balancedLossRate = 0.0;
                    double allDocumentsWeight = 0.0;
                    for (int i = startDoc; i <= endDoc; i++)
                    {
                        bool label = _binaryLabels[i];
                        bool predictedClass = scores[i] > 0.0;
                        double balancedRecip = label ? _recipNpos : _recipNneg;

                        const double documentWeight = 1.0;
                        bool correct = !(label ^ predictedClass);
                        double loss = Math.Log(1.0 + Math.Exp(-1.0 * _sigmoidParameter * (label ? 1 : -1) * scores[i]));

                        errorRate += (correct ? 0 : 1) * documentWeight;
                        lossRate += loss * documentWeight;
                        balancedErrorRate += correct ? 0.0 : documentWeight * balancedRecip;
                        balancedLossRate += loss * documentWeight * balancedRecip;
                        allDocumentsWeight += documentWeight;
                    }
                    lock (testLock)
                    {
                        totalErrorRate += errorRate;
                        totalLossRate += lossRate;
                        totalBalancedErrorRate += balancedErrorRate;
                        totalBalancedLossRate += balancedLossRate;
                        totalAllDocumentsWeight += allDocumentsWeight;
                    }
                };
            }
            Parallel.Invoke(new ParallelOptions() { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, actions);
            totalErrorRate /= totalAllDocumentsWeight;
            totalLossRate /= totalAllDocumentsWeight;
            //BalancedErrorRate already included reciprocal part of number of documents but we need to scale it to (0-1) range
            totalBalancedErrorRate /= 2;
            //BalancedLoosRate  already included reciprocal part of number of documents

            List<TestResult> result = new List<TestResult>()
            {
                new TestResult("ErrorRate", totalErrorRate * totalAllDocumentsWeight, totalAllDocumentsWeight, true, TestResult.ValueOperator.Average),
                new TestResult("LossRate", totalLossRate * totalAllDocumentsWeight, totalAllDocumentsWeight, true, TestResult.ValueOperator.Average),
                new TestResult("BalancedErrorRate", totalBalancedErrorRate, 1, true, TestResult.ValueOperator.None),
                new TestResult("BalancedLossRate", totalBalancedLossRate, 1, true, TestResult.ValueOperator.None),
            };

            return result;

        }
    }
}
