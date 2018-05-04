// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    public sealed class DcgCalculator
    {
        // This should be exposed to outside classes as constants
        public static double[] LabelMap = new double[] { 0.0, 3.0, 7.0, 15.0, 31.0 };
        public static readonly double[] DiscountMap;
        private readonly int[] _oneTwoThree;

        // reusable memory
        private int[][] _permutationBuffers;
        private double[][] _scoreBuffers;
        private DcgPermutationComparer[] _comparers;

        /// <summary>
        /// Contains the instances for a second Level comparer, which gets applied after the initial rank
        /// based ordering has happened. The array stores one second level comparer per thread.
        /// </summary>
        private DescendingStableIdealComparer[] _secondLevelcomparers;

        private double _result;

        // Pre-compute discount maps. It is done per all instances
        static DcgCalculator()
        {
            DiscountMap = new double[200];
            for (int i = 0; i < DiscountMap.Length; i++)
            {
                DiscountMap[i] = 1.0 / Math.Log(2 + i);
            }
        }

        public static double[] LabelGainMap
        {
            get { return LabelMap; }
            set { LabelMap = value; }
        }

        /// <summary>
        /// Constructs a DCG calculator
        /// </summary>
        /// <param name="maxDocsPerQuery">the maximum number of documents per query</param>
        /// <param name="sortingAlgorithm">a string describing the sorting algorithm to use</param>
        /// <param name="topNDocsForIdealDcg">specifies the ideal DCG@ computation.</param>
        public DcgCalculator(int maxDocsPerQuery, string sortingAlgorithm, int topNDocsForIdealDcg = 0)
        {
            int numThreads = BlockingThreadPool.NumThreads;
            _oneTwoThree = Utils.GetIdentityPermutation(maxDocsPerQuery);
            _permutationBuffers = new int[numThreads][];
            for (int i = 0; i < numThreads; ++i)
            {
                _permutationBuffers[i] = new int[maxDocsPerQuery];
            }

            _scoreBuffers = new double[numThreads][];
            for (int i = 0; i < numThreads; ++i)
            {
                _scoreBuffers[i] = new double[maxDocsPerQuery];
            }

            _comparers = new DcgPermutationComparer[numThreads];

            for (int i = 0; i < numThreads; ++i)
            {
                _comparers[i] = DcgPermutationComparerFactory.GetDcgPermutationFactory(sortingAlgorithm);

                // only reorder query/URL pairs, if we have at least two query/URL pairs for reordering
                if (topNDocsForIdealDcg > 1)
                {
                    // using lazy initialize for the _secondLevelComparers to make it cheap and easy
                    // later to test, if re-ordering needs to be computed. This way it also only allocates
                    // the memory it really needs
                    if (_secondLevelcomparers == null)
                    {
                        _secondLevelcomparers = new DescendingStableIdealComparer[numThreads];
                    }

                    _secondLevelcomparers[i] = new DescendingStableIdealComparer(topNDocsForIdealDcg);
                }
            }
        }

        /// <summary>
        /// Calculates the natural-based max DCG at a given truncation
        /// </summary>
        /// <param name="labels">vector of labels</param>
        /// <param name="boundaries">vector of query boundaries</param>
        /// <param name="trunc">truncation to use</param>
        /// <param name="labelCounts"></param>
        public static double[] MaxDcg(short[] labels, int[] boundaries, int trunc, int[][] labelCounts)
        {
            double[] maxDcg = new double[boundaries.Length - 1];

            for (int q = 0; q < boundaries.Length - 1; ++q)
            {
                maxDcg[q] = MaxDcgQuery(labels, boundaries[q], boundaries[q + 1] - boundaries[q], trunc, labelCounts[q]);
            }

            return maxDcg;
        }

        /// <summary>
        /// Calculates the natural-based max DCG at a given truncation for a query
        /// </summary>
        /// <param name="labels">vector of labels</param>
        /// <param name="begin">Index of the first document</param>
        /// <param name="labelCounts"></param>
        /// <param name="trunc">truncation to use</param>
        /// <param name="numDocuments"></param>
        public static double MaxDcgQuery(short[] labels, int begin, int numDocuments, int trunc, int[] labelCounts)
        {
            int maxTrunc = Math.Min(trunc, numDocuments);

            if (maxTrunc == 0)
                return double.NaN;

            Array.Clear(labelCounts, 0, LabelMap.Length);

            for (int i = begin; i < begin + numDocuments; ++i)
            {
                short label = labels[i];
                labelCounts[label]++;
            }

            int topLabel = LabelMap.Length - 1;
            double maxDcg = 0;

            for (int t = 0; t < maxTrunc; ++t)
            {
                while (labelCounts[topLabel] <= 0 && topLabel > 0)
                {
                    topLabel--;
                }

                maxDcg += LabelMap[topLabel] / Math.Log(2.0 + t);
                labelCounts[topLabel]--;
            }

            return maxDcg;
        }

        /// <summary>
        /// Efficient computation of average NDCG@3 for the entire dataset
        /// Note that it is virtual and MPI provides faster implementations for MPI
        /// </summary>
        /// <param name="dataset">the dataset</param>
        /// <param name="scores">vector of scores</param>
        /// <param name="labels"></param>
        public double Ndcg3(Dataset dataset, short[] labels, double[] scores)
        {
            if (Utils.Size(dataset.MaxDcg) < 3)
                dataset.Skeleton.RecomputeMaxDcg(3);
            double[] maxDCG3 = dataset.MaxDcg[2];

            _result = 0.0;
            Parallel.ForEach(Enumerable.Range(0, dataset.NumQueries).Where(query => maxDCG3[query] > 0),
                new ParallelOptions() { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads },
                (query) =>
                 {
                     int begin = dataset.Boundaries[query];
                     int end = dataset.Boundaries[query + 1];
                     Ndcg3Worker(scores, labels, begin, end, maxDCG3[query]);
                 });
            return _result / dataset.NumQueries;
        }

        // thread worker
        // Also called by MPI NDCG3
        private void Ndcg3Worker(double[] scores, short[] labels, int begin, int end, double maxDCG3)
        {
            Utils.InterlockedAdd(ref _result, DCG3(scores, labels, begin, end) / maxDCG3);
        }

        /// <summary>
        /// Efficient computation of natural-based pessimistic DCG@3 for a given query
        /// </summary>
        /// <param name="scores">vector of scores</param>
        /// <param name="labels">vector of labels</param>
        /// <param name="begin">index of first document in query</param>
        /// <param name="end">index of first document in next query</param>
        public static unsafe double DCG3(double[] scores, short[] labels, int begin, int end)
        {
            if (begin >= end)
                throw Contracts.ExceptParam(nameof(begin));

            double maxScore1 = double.NegativeInfinity;
            double maxScore2 = double.NegativeInfinity;
            double maxScore3 = double.NegativeInfinity;
            int maxLabel1 = -1;
            int maxLabel2 = -1;
            int maxLabel3 = -1;

            fixed (double* pScores = scores)
            {
                fixed (short* pLabels = labels)
                {
                    for (int d = begin; d < end; ++d)
                    {
                        double score = pScores[d];
                        short label = pLabels[d];

                        // check if the current document should be in the top 3
                        if (score > maxScore3 || (score == maxScore3 && label < maxLabel3))
                        {
                            if (score > maxScore2 || (score == maxScore2 && label < maxLabel2))
                            {
                                maxScore3 = maxScore2;
                                maxLabel3 = maxLabel2;

                                if (score > maxScore1 || (score == maxScore1 && label < maxLabel1))
                                {
                                    maxScore2 = maxScore1;
                                    maxLabel2 = maxLabel1;
                                    maxScore1 = score;
                                    maxLabel1 = label;
                                }
                                else
                                {
                                    maxScore2 = score;
                                    maxLabel2 = label;
                                }
                            }
                            else
                            {
                                maxScore3 = score;
                                maxLabel3 = label;
                            }
                        }
                    }
                }
            }

            // calculate the dcg
            double dcg = LabelMap[maxLabel1] * DiscountMap[0];
            if (maxScore2 > double.NegativeInfinity)
                dcg += LabelMap[maxLabel2] * DiscountMap[1];
            if (maxScore3 > double.NegativeInfinity)
                dcg += LabelMap[maxLabel3] * DiscountMap[2];

            return dcg;
        }

        /// <summary>
        /// Efficient computation of average NDCG@1 for the entire dataset
        /// Note that it is virtual and MPI provides faster implemetations for MPI
        /// </summary>
        /// <param name="dataset">the dataset</param>
        /// <param name="labels"></param>
        /// <param name="scores">the vector of score from previous rounds</param>
        /// <returns>average NDCG@1 for an entire dataset</returns>
        public double Ndcg1(Dataset dataset, short[] labels, double[] scores)
        {
            if (Utils.Size(dataset.MaxDcg) < 1)
                dataset.Skeleton.RecomputeMaxDcg(1);
            double[] maxDCG1 = dataset.MaxDcg[0];
            _result = 0.0;
            Parallel.ForEach(Enumerable.Range(0, dataset.NumQueries).Where(query => maxDCG1[query] > 0),
               new ParallelOptions() { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads },
               (query) =>
               {
                   int begin = dataset.Boundaries[query];
                   int end = dataset.Boundaries[query + 1];
                   Ndcg1Worker(scores, labels, begin, end, maxDCG1[query]);
               });
            return _result / dataset.NumQueries;
        }

        // Computation of NDCG@3 for pre sorted Dataset
        public double Ndcg3(Dataset dataset, short[][] labelsSorted)
        {
            if (Utils.Size(dataset.MaxDcg) < 3)
                dataset.Skeleton.RecomputeMaxDcg(3);
            double[] maxDCG3 = dataset.MaxDcg[2];
            double result = 0.0;
            for (int query = 0; query < dataset.NumQueries; ++query)
            {
                if (maxDCG3[query] <= 0)
                    continue;
                short[] queryLabels = labelsSorted[query];
                double dcg = LabelMap[queryLabels[0]] * DiscountMap[0] +
                    LabelMap[queryLabels[1]] * DiscountMap[1] +
                    LabelMap[queryLabels[2]] * DiscountMap[2];
                result += dcg / maxDCG3[query];
            }

            return result / dataset.NumQueries;
        }

        // Computation of NDCG@1 for pre sorted Dataset
        public double Ndcg1(Dataset dataset, short[][] labelsSorted)
        {
            if (Utils.Size(dataset.MaxDcg) < 1)
                dataset.Skeleton.RecomputeMaxDcg(1);
            double[] maxDCG1 = dataset.MaxDcg[0];
            double result = 0.0;
            for (int query = 0; query < dataset.NumQueries; ++query)
            {
                if (maxDCG1[query] <= 0)
                    continue;
                result += LabelMap[labelsSorted[query][0]] / maxDCG1[query];
            }

            return result * DiscountMap[0] / dataset.NumQueries;
        }

        // thread worker
        // Also used by MPI NDCG1
        private void Ndcg1Worker(double[] scores, short[] labels, int begin, int end, double maxDCG1)
        {
            Utils.InterlockedAdd(ref _result, DCG1(scores, labels, begin, end) / maxDCG1);
        }

        /// <summary>
        /// Calculates the natural-based pessimistic DCG@1 of scores(query)
        /// </summary>
        /// <param name="scores">vector of scores</param>
        /// <param name="labels">vector of labels</param>
        /// <param name="begin">index of first document in query</param>
        /// <param name="end">index of first document in next query</param>
        /// <returns>DCG@1</returns>
        public static unsafe double DCG1(double[] scores, short[] labels, int begin, int end)
        {
            double maxScore = scores[begin];
            int argMaxLabel = labels[begin];

            fixed (double* pScores = scores)
            {
                fixed (short* pLabels = labels)
                {
                    for (int d = begin + 1; d < end; ++d)
                    {
                        double score = pScores[d];

                        // check if the current document should be in the top 3
                        if (score > maxScore || (score == maxScore && labels[d] < argMaxLabel))
                        {
                            maxScore = score;
                            argMaxLabel = pLabels[d];
                        }
                    }

                    // calculate the dcg
                    return LabelMap[argMaxLabel] * DiscountMap[0];
                }
            }
        }

        /// <summary>
        /// calculates the average NDCG given the scores array
        /// For performance reason it duplicates some
        /// </summary>
        public double[] NdcgRangeFromScores(Dataset dataset, short[] labels, double[] scores)
        {
            int truncation = dataset.MaxDcg.Length;

            double[] result = new double[truncation];

            int chunkSize = 1 + dataset.NumQueries / BlockingThreadPool.NumThreads;   // Minimizes the number of repeat computations in sparse array to have each thread take as big a chunk as possible
            // REVIEW: This partitioning doesn't look optimal.
            // Probably make sence to investigate better ways of splitting data?
            var actions = new Action[(int)Math.Ceiling(1.0 * dataset.NumQueries / chunkSize)];
            var actionIndex = 0;
            for (int q = 0; q < dataset.NumQueries; q += chunkSize)
            {
                var start = q;
                var threadIndex = actionIndex;
                actions[actionIndex++] = () =>
                {
                    NdcgRangeWorkerChunkFromScores(dataset, labels, scores, result, start, Math.Min(dataset.NumQueries - start, chunkSize), threadIndex);
                };
            }

            Parallel.Invoke(new ParallelOptions() { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, actions);

            for (int t = 0; t < truncation; ++t)
            {
                result[t] /= dataset.NumQueries;
            }

            return result;
        }

        // thread worker per-chunk
        private void NdcgRangeWorkerChunkFromScores(Dataset dataset, short[] labels, double[] scores, double[] result, int startQuery, int numQueries, int threadIndex)
        {
            for (int q = startQuery; q < startQuery + numQueries; q++)
            {
                NdcgRangeWorkerFromScores(dataset, labels, scores, result, q, threadIndex);
            }
        }

        // thread worker
        private void NdcgRangeWorkerFromScores(Dataset dataset, short[] labels, double[] scores, double[] result, int query, int threadIndex)
        {
            int begin = dataset.Boundaries[query];
            int count = dataset.Boundaries[query + 1] - begin;

            int[] permutation = _permutationBuffers[threadIndex];

            // get labels
            double[][] maxDcg = dataset.MaxDcg;

            int truncation = maxDcg.Length;

            SortRankingResults(labels, threadIndex, begin, begin, count, permutation, scores);

            if (count > truncation)
                count = truncation;
            double dcg = 0;
            for (int t = 0; t < count; ++t)
            {
                dcg = dcg + LabelMap[labels[begin + permutation[t]]] * DiscountMap[t];
                if (dcg > 0)
                    Utils.InterlockedAdd(ref result[t], dcg / maxDcg[t][query]);
            }

            if (dcg > 0)
                for (int t = count; t < truncation; ++t)
                {
                    Utils.InterlockedAdd(ref result[t], dcg / maxDcg[t][query]);
                }
        }

        /// <summary>
        /// Orders the queries based on the given comparer.
        /// </summary>
        /// <param name="labels">The label for all query URL pairs</param>
        /// <param name="threadIndex">Specifies the thread which is executing this code</param>
        /// <param name="scoreBegin">position of the first query-URL pair to sort in the score array</param>
        /// <param name="labelBegin">position of the first query-URL pair to sort in the label array</param>
        /// <param name="count">number of query-URL pairs</param>
        /// <param name="permutation">resulting query order array</param>
        /// <param name="scores">The scores for all query URL pairs</param>
        private void SortRankingResults(short[] labels, int threadIndex, int scoreBegin, int labelBegin, int count, int[] permutation, double[] scores)
        {
            Array.Copy(_oneTwoThree, permutation, count);

            DcgPermutationComparer comparer = _comparers[threadIndex];
            // set values for the comparer
            comparer.Scores = scores;
            comparer.Labels = labels;
            comparer.ScoresOffset = scoreBegin;
            comparer.LabelsOffset = labelBegin;

            // calculate the permutation
            Array.Sort(permutation, 0, count, comparer);

            // check if there is topN re-sorter specified. If so,
            // change the order of the TOP N results
            if (_secondLevelcomparers != null)
            {
                // set values for the comparer
                _secondLevelcomparers[threadIndex].Labels = labels;
                _secondLevelcomparers[threadIndex].LabelsOffset = labelBegin;

                // calculate the permutation
                Array.Sort(permutation, 0, Math.Min(count, _secondLevelcomparers[threadIndex].CompareFirstN), _secondLevelcomparers[threadIndex]);
            }
        }

        public double[] DcgFromScores(Dataset dataset, double[] scores, double[] discount)
        {
            short[] ratings = dataset.Ratings;
            double[] result = new double[dataset.NumQueries];
            int[] order = OrderingFromScores(dataset, scores);

            for (int q = 0; q < dataset.NumQueries; ++q)
            {
                int begin = dataset.Boundaries[q];
                int end = dataset.Boundaries[q + 1];
                double dcg = 0.0;
                for (int d = begin; d < end; ++d)
                {
                    dcg += discount[d - begin] * LabelMap[ratings[begin + order[d]]];
                }

                result[q] = dcg;
            }
            return result;
        }

        /// <summary>
        /// Calculates the order of documents. This returns an array with as many elements
        /// as there are documents, where the subarray in a query's boundary will contain
        /// elements from 0 up to but not including the number of documents in the query.
        /// The first value in this subarray will contain the index of the document in the
        /// subarray at top position (highest ranked), and the last value the index of the
        /// document with bottom position (lowest ranked).
        /// </summary>
        /// <param name="dataset">The dataset over which to calculate the DCG.</param>
        /// <param name="scores">The scores for all documents within the dataset.</param>
        /// <returns></returns>
        public int[] OrderingFromScores(Dataset dataset, double[] scores)
        {
            int[] result = new int[dataset.NumDocs];

            int chunkSize = 1 + dataset.NumQueries / BlockingThreadPool.NumThreads;   // Minimizes the number of repeat computations in sparse array to have each thread take as big a chunk as possible
            // REVIEW: This partitioning doesn't look optimal.
            // Probably make sence to investigate better ways of splitting data?
            var actions = new Action[(int)Math.Ceiling(1.0 * dataset.NumQueries / chunkSize)];
            var actionIndex = 0;
            for (int q = 0; q < dataset.NumQueries; q += chunkSize)
            {
                var start = q;
                var threadIndex = actionIndex;
                actions[actionIndex++] = () =>
                  {
                      OrderingRangeWorkerFromScores(dataset, scores, result, start, Math.Min(dataset.NumQueries - start, chunkSize), threadIndex);
                  };
            }
            Parallel.Invoke(new ParallelOptions() { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, actions);
            return result;
        }

        // thread worker per-chunk
        private void OrderingRangeWorkerFromScores(Dataset dataset, double[] scores, int[] result, int startQuery, int numQueries, int threadIndex)
        {
            for (int q = startQuery; q < startQuery + numQueries; q++)
            {
                OrderingRangeWorkerPerQueryFromScores(dataset, scores, result, q, threadIndex);
            }
        }

        // thread worker
        private void OrderingRangeWorkerPerQueryFromScores(Dataset dataset, double[] scores, int[] result, int query, int threadIndex)
        {
            int begin = dataset.Boundaries[query];
            int count = dataset.Boundaries[query + 1] - begin;

            int[] permutation = _permutationBuffers[threadIndex];

            // get labels
            short[] labels = dataset.Ratings;

            SortRankingResults(labels, threadIndex, begin, begin, count, permutation, scores);

            Array.Copy(permutation, 0, result, begin, count);
        }
    }
}
