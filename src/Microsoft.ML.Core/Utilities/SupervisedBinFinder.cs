// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    /// <summary>
    /// This class performs discretization of (value, label) pairs into bins in a way that minimizes
    /// the target function "minimum description length".
    /// The algorithm is outlineed in an article
    /// "Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning"
    /// [Fayyad, Usama M.; Irani, Keki B. (1993)] https://ijcai.org/Past%20Proceedings/IJCAI-93-VOL2/PDF/022.pdf
    ///
    /// The class can be used several times sequentially, it is stateful and not thread-safe.
    /// Both Single and Double precision processing is implemented, and is identical.
    /// </summary>
    public sealed class SupervisedBinFinder
    {
        private readonly struct ValuePair<T> : IComparable<ValuePair<T>>
            where T : IComparable<T>
        {
            public readonly T Value;
            public readonly int Label;

            public ValuePair(T value, int label)
            {
                Value = value;
                Label = label;
            }

            public int CompareTo(ValuePair<T> other)
            {
                return Value.CompareTo(other.Value);
            }
        }

        private int _valueCount;
        private int _distinctValueCount;
        private int _labelCardinality;
        private int _maxBins;
        private int _minBinSize;

        // cumulative counts for distinct values. Dimensions: _distinctValueCount X (_labelCardinality + 1) (last column is the total counts)
        // REVIEW: optimize memory allocation in sequential use case (don't re-allocate if we have a large enough array already)
        private int[,] _cumulativeCounts;

        /// <summary>
        /// Finds the bins for Single values (and integer labels)
        /// </summary>
        /// <param name="maxBins">Maximum number of bins</param>
        /// <param name="minBinSize">Minimum number of values per bin (stopping condition for greedy bin splitting)</param>
        /// <param name="nLabels">Cardinality of the labels</param>
        /// <param name="values">The feature values</param>
        /// <param name="labels">The corresponding label values</param>
        /// <returns>An array of split points, no more than <paramref name="maxBins"/> total (but maybe less), ending with PositiveInfinity</returns>
        public Single[] FindBins(int maxBins, int minBinSize, int nLabels, IList<Single> values, IList<int> labels)
        {
            // prepare the values: count distinct values and populate the value pair array
            _valueCount = values.Count;
            _labelCardinality = nLabels;
            _maxBins = maxBins;
            _minBinSize = minBinSize;
            Contracts.Assert(_valueCount == labels.Count);
            _distinctValueCount = 0;
            var seenValues = new HashSet<Single>();
            var valuePairs = new ValuePair<Single>[_valueCount];
            for (int i = 0; i < _valueCount; i++)
            {
                valuePairs[i] = new ValuePair<Single>(values[i], labels[i]);
                if (seenValues.Add(values[i]))
                    _distinctValueCount++;
            }
            Array.Sort(valuePairs);

            // populate the cumulative counts with unique values
            _cumulativeCounts = new int[_distinctValueCount, _labelCardinality + 1];
            var distinctValues = new Single[_distinctValueCount];
            Single curValue = Single.NegativeInfinity;
            int curIndex = -1;
            foreach (var pair in valuePairs)
            {
                Contracts.Assert(pair.Value >= curValue);
                if (pair.Value > curValue || curIndex < 0)
                {
                    curValue = pair.Value;
                    curIndex++;
                    distinctValues[curIndex] = curValue;
                    if (curIndex > 0)
                    {
                        for (int i = 0; i < _labelCardinality + 1; i++)
                            _cumulativeCounts[curIndex, i] = _cumulativeCounts[curIndex - 1, i];
                    }
                }
                _cumulativeCounts[curIndex, pair.Label]++;
                _cumulativeCounts[curIndex, _labelCardinality]++;
            }

            Contracts.Assert(curIndex == _distinctValueCount - 1);

            var boundaries = FindBinsCore();
            Contracts.Assert(Utils.Size(boundaries) > 0);
            Contracts.Assert(boundaries.Length == 1 && boundaries[0] == 0 || boundaries[0] > 0, "boundaries are exclusive, can't have 0");
            Contracts.Assert(boundaries[boundaries.Length - 1] == _distinctValueCount);

            // transform boundary indices back into bin upper bounds
            var numUpperBounds = boundaries.Length;
            Single[] result = new Single[numUpperBounds];
            for (int i = 0; i < numUpperBounds - 1; i++)
            {
                var split = boundaries[i];
                result[i] = BinFinderBase.GetSplitValue(distinctValues[split - 1], distinctValues[split]);

                // Even though distinctValues may contain infinities, the boundaries may not be infinite:
                // GetSplitValue(a,b) only returns +-inf if a==b==+-inf,
                // and distinctValues won't contain more than one +inf or -inf.
                Contracts.Assert(FloatUtils.IsFinite(result[i]));
            }

            result[numUpperBounds - 1] = Single.PositiveInfinity;
            AssertStrictlyIncreasing(result);

            return result;
        }

        /// <summary>
        /// Finds the bins for Double values (and integer labels)
        /// </summary>
        /// <param name="maxBins">Maximum number of bins</param>
        /// <param name="minBinSize">Minimum number of values per bin (stopping condition for greedy bin splitting)</param>
        /// <param name="nLabels">Cardinality of the labels</param>
        /// <param name="values">The feature values</param>
        /// <param name="labels">The corresponding label values</param>
        /// <returns>An array of split points, no more than <paramref name="maxBins"/> total (but maybe less), ending with PositiveInfinity</returns>
        public Double[] FindBins(int maxBins, int minBinSize, int nLabels, IList<Double> values, IList<int> labels)
        {
            // prepare the values: count distinct values and populate the value pair array
            _valueCount = values.Count;
            _labelCardinality = nLabels;
            _maxBins = maxBins;
            _minBinSize = minBinSize;
            Contracts.Assert(_valueCount == labels.Count);
            _distinctValueCount = 0;
            var seenValues = new HashSet<Double>();
            var valuePairs = new ValuePair<Double>[_valueCount];
            for (int i = 0; i < _valueCount; i++)
            {
                valuePairs[i] = new ValuePair<Double>(values[i], labels[i]);
                if (seenValues.Add(values[i]))
                    _distinctValueCount++;
            }
            Array.Sort(valuePairs);

            // populate the cumulative counts with unique values
            _cumulativeCounts = new int[_distinctValueCount, _labelCardinality + 1];
            var distinctValues = new Double[_distinctValueCount];
            Double curValue = Double.NegativeInfinity;
            int curIndex = -1;
            foreach (var pair in valuePairs)
            {
                Contracts.Assert(pair.Value >= curValue);
                if (pair.Value > curValue || curIndex < 0)
                {
                    curValue = pair.Value;
                    curIndex++;
                    distinctValues[curIndex] = curValue;
                    if (curIndex > 0)
                    {
                        for (int i = 0; i < _labelCardinality + 1; i++)
                            _cumulativeCounts[curIndex, i] = _cumulativeCounts[curIndex - 1, i];
                    }
                }
                _cumulativeCounts[curIndex, pair.Label]++;
                _cumulativeCounts[curIndex, _labelCardinality]++;
            }

            Contracts.Assert(curIndex == _distinctValueCount - 1);

            var boundaries = FindBinsCore();
            Contracts.Assert(Utils.Size(boundaries) > 0);
            Contracts.Assert(boundaries.Length == 1 && boundaries[0] == 0 || boundaries[0] > 0, "boundaries are exclusive, can't have 0");
            Contracts.Assert(boundaries[boundaries.Length - 1] == _distinctValueCount);

            // transform boundary indices back into bin upper bounds
            var numUpperBounds = boundaries.Length;
            Double[] result = new Double[numUpperBounds];
            for (int i = 0; i < numUpperBounds - 1; i++)
            {
                var split = boundaries[i];
                result[i] = BinFinderBase.GetSplitValue(distinctValues[split - 1], distinctValues[split]);

                // Even though distinctValues may contain infinities, the boundaries may not be infinite:
                // GetSplitValue(a,b) only returns +-inf if a==b==+-inf,
                // and distinctValues won't contain more than one +inf or -inf.
                Contracts.Assert(FloatUtils.IsFinite(result[i]));
            }

            result[numUpperBounds - 1] = Double.PositiveInfinity;
            AssertStrictlyIncreasing(result);

            return result;
        }

        [Conditional("DEBUG")]
        private void AssertStrictlyIncreasing(Single[] result)
        {
#if DEBUG
            for (int i = 1; i < result.Length; i++)
                Contracts.Assert(result[i] > result[i - 1]);
#endif
        }

        [Conditional("DEBUG")]
        private void AssertStrictlyIncreasing(Double[] result)
        {
#if DEBUG
            for (int i = 1; i < result.Length; i++)
                Contracts.Assert(result[i] > result[i - 1]);
#endif
        }

        private class SplitInterval
        {
            public readonly int Min;
            public readonly int Lim;

            public readonly Double Gain;
            public readonly int SplitLim;

            public SplitInterval(SupervisedBinFinder binFinder, int min, int lim, bool skipSplitCalculation)
            {
                Min = min;
                Lim = lim;
                Gain = -1;

                if (skipSplitCalculation)
                    return;// no split is done

                // calculate best split and associated gain
                int totalCount;
                Double totalEntropy = binFinder.GetEntropy(min, lim, out totalCount);
                if (totalCount < binFinder._minBinSize) // too small bin, won't split
                    return;
                if (totalEntropy <= 0) // we achieved perfect entropy, no need to split any further
                    return;

                Double logN = Math.Log(lim - min);
                for (int split = min + 1; split < lim; split++)
                {
                    int leftCount;
                    int rightCount;
                    var leftEntropy = binFinder.GetEntropy(min, split, out leftCount);
                    var rightEntropy = binFinder.GetEntropy(split, lim, out rightCount);
                    Contracts.Assert(leftCount + rightCount == totalCount);

                    // This term corresponds to the 'fixed cost associated with a split'
                    // It's a simplification of a Delta(A,T;S) term calculated in the paper
                    var delta = logN - binFinder._labelCardinality * (totalEntropy - leftEntropy - rightEntropy);

                    var curGain = totalCount * totalEntropy // total cost of transmitting non-split content
                               - leftCount * leftEntropy // cost of transmitting left part of the split
                               - rightCount * rightEntropy // cost of transmitting right part of the split
                               - delta; // fixed cost of transmitting additional codebook
                    if (curGain > Gain)
                    {
                        Gain = curGain;
                        SplitLim = split;
                    }
                }
            }
        }

        /// <summary>
        /// Calculate the entropy and label cardinality for a given interval within the data
        /// </summary>
        private Double GetEntropy(int min, int lim, out int totalCount)
        {
            Double entropy = 0;
            totalCount = _cumulativeCounts[lim - 1, _labelCardinality];
            if (min > 0)
                totalCount -= _cumulativeCounts[min - 1, _labelCardinality];
            for (int i = 0; i < _labelCardinality; i++)
            {
                var count = _cumulativeCounts[lim - 1, i];
                if (min > 0)
                    count -= _cumulativeCounts[min - 1, i];
                if (count == 0 || count == totalCount)
                    continue;
                var p = (Double)count / totalCount;
                entropy -= p * Math.Log(p);
            }

            return entropy;
        }

        /// <summary>
        /// Finds the optimum bins with respect to <see cref="_cumulativeCounts"/>
        /// </summary>
        /// <returns>The sorted array of indices that are exclusive upper bounds of the respective bins</returns>
        private int[] FindBinsCore()
        {
            if (_distinctValueCount == 0)
                return new int[] { _distinctValueCount };

            // we will put intervals into a heap so that the one with maximum gain is at the top
            var intervals = new Heap<SplitInterval>((x, y) => x.Gain < y.Gain);

            // start with a single interval covering all points
            intervals.Add(new SplitInterval(this, 0, _distinctValueCount, false));

            // while we haven't reached max # of bins and there's still gain in splitting (best interval's gain is positive)
            while (intervals.Count < _maxBins && intervals.Top.Gain > 0)
            {
                // take the interval with the best split gain
                var toSplit = intervals.Pop();

                // make the split
                bool isLastSplit = intervals.Count == _maxBins - 1;
                var left = new SplitInterval(this, toSplit.Min, toSplit.SplitLim, isLastSplit);
                var right = new SplitInterval(this, toSplit.SplitLim, toSplit.Lim, isLastSplit);

                // put the results back into the heap
                intervals.Add(left);
                intervals.Add(right);
            }

            var binCount = intervals.Count;
            var results = new int[binCount];
            for (int i = 0; i < binCount; i++)
                results[i] = intervals.Pop().Lim;

            Contracts.Assert(intervals.Count == 0);

            Array.Sort(results);
            return results;
        }
    }
}
