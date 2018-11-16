// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    public sealed class WinLossCalculator
    {
        private readonly int[] _oneTwoThree;

        // reusable memory
        private readonly int[][] _permutationBuffers;
        private readonly double[][] _scoreBuffers;
        private readonly DcgPermutationComparer[] _comparers;

        /// <summary>
        /// Constructs a WinLoss calculator
        /// </summary>
        /// <param name="maxDocsPerQuery">the maximum number of documents per query</param>
        /// <param name="sortingAlgorithm">a string describing the sorting algorithm to use</param>
        public WinLossCalculator(int maxDocsPerQuery, string sortingAlgorithm)
        {
            int numThreads = BlockingThreadPool.NumThreads;
            _oneTwoThree = Utils.GetIdentityPermutation(maxDocsPerQuery);
            _permutationBuffers = new int[numThreads][];
            for (int i = 0; i < numThreads; ++i)
                _permutationBuffers[i] = new int[maxDocsPerQuery];
            _scoreBuffers = new double[numThreads][];
            for (int i = 0; i < numThreads; ++i)
                _scoreBuffers[i] = new double[maxDocsPerQuery];
            _comparers = new DcgPermutationComparer[numThreads];
            for (int i = 0; i < numThreads; ++i)
                _comparers[i] = DcgPermutationComparerFactory.GetDcgPermutationFactory(sortingAlgorithm);
        }

        /// <summary>
        /// calculates the average WinLoss given the scores array
        /// For performance reason it duplicates some
        /// </summary>
        public double[] WinLossRangeFromScores(Dataset dataset, short[] labels, double[] scores)
        {
            double[] result = new double[9];

            int chunkSize = 1 + dataset.NumQueries / BlockingThreadPool.NumThreads;   // Minimizes the number of repeat computations in sparse array to have each thread take as big a chunk as possible
            // REVIEW: This partitioning doesn't look optimal.
            // Probably make sence to investigate better ways of splitting data?
            var actions = new Action[(int)Math.Ceiling(1.0 * dataset.NumQueries / chunkSize)];
            var actionIndex = 0;
            var queue = new ConcurrentQueue<int>(Enumerable.Range(0, BlockingThreadPool.NumThreads));
            for (int q = 0; q < dataset.NumQueries; q += chunkSize)
            {
                var start = q;
                var threadIndex = actionIndex;
                actions[actionIndex++] = (() =>
                    WinLossRangeWorkerChunkFromScores(dataset, labels, scores, result, start, Math.Min(dataset.NumQueries - start, chunkSize), threadIndex));
            }
            Parallel.Invoke(new ParallelOptions { MaxDegreeOfParallelism = BlockingThreadPool.NumThreads }, actions);

            for (int t = 0; t < 6; ++t)
                result[t] /= dataset.NumQueries;

            return result;
        }

        // thread worker per-chunk
        private void WinLossRangeWorkerChunkFromScores(Dataset dataset, short[] labels, double[] scores, double[] result, int startQuery, int numQueries, int threadIndex)
        {
            for (int q = startQuery; q < startQuery + numQueries; q++)
                WinLossRangeWorkerFromScores(dataset, labels, scores, result, q, threadIndex);
        }

        // thread worker
        private void WinLossRangeWorkerFromScores(Dataset dataset, short[] labels, double[] scores, double[] result, int query, int threadIndex)
        {
            int begin = dataset.Boundaries[query];
            int count = dataset.Boundaries[query + 1] - begin;

            int[] permutation = _permutationBuffers[threadIndex];
            DcgPermutationComparer comparer = _comparers[threadIndex];

            // set values for the comparer
            comparer.Scores = scores;
            comparer.Labels = labels;
            comparer.ScoresOffset = begin;
            comparer.LabelsOffset = begin;

            // calculate the permutation
            Array.Copy(_oneTwoThree, permutation, count);
            Array.Sort(permutation, 0, count, comparer);

            int surplus = 0;
            int maxsurplus = 0;
            int maxsurpluspos = 0;
            for (int t = 0; t < count; ++t)
            {
                if (labels[begin + permutation[t]] > 0 ||
                    labels[begin + permutation[t]] < 0)
                {
                    surplus += labels[begin + permutation[t]];
                }
                else
                {
                    surplus--;
                }
                if (surplus > maxsurplus)
                {
                    maxsurplus = surplus;
                    maxsurpluspos = t;
                }
                if (t == 100)
                    Utils.InterlockedAdd(ref result[0], surplus);
                if (t == 200)
                    Utils.InterlockedAdd(ref result[1], surplus);
                if (t == 300)
                    Utils.InterlockedAdd(ref result[2], surplus);
                if (t == 400)
                    Utils.InterlockedAdd(ref result[3], surplus);
                if (t == 500)
                    Utils.InterlockedAdd(ref result[4], surplus);
                if (t == 1000)
                    Utils.InterlockedAdd(ref result[5], surplus);
            }
            Utils.InterlockedAdd(ref result[6], maxsurplus);
            Utils.InterlockedAdd(ref result[7], maxsurpluspos);
            Utils.InterlockedAdd(ref result[8], count);
        }
    }
}
