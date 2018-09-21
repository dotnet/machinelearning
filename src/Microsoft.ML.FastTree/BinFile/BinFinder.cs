// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    /// <summary>
    /// A class that bins vectors of doubles into a specified number of equal mass bins.
    /// </summary>
    internal sealed class BinFinder
    {
        private readonly GreedyBinFinder _finder;
        private double[] _distinctValues;
        private int[] _counts;

        private static double[] _trivialBinUpperBounds; // Will be initialized to a single element positive infinity array.

        private static double[] TrivialBinUpperBounds
        {
            get
            {
                if (_trivialBinUpperBounds == null)
                    Interlocked.CompareExchange(ref _trivialBinUpperBounds, new double[] { double.PositiveInfinity }, null);
                Contracts.AssertValue(_trivialBinUpperBounds);
                return _trivialBinUpperBounds;
            }
        }

        public BinFinder()
        {
            _finder = new GreedyBinFinder();
        }

        /// <summary>
        /// Finds the distinct values in the <paramref name="values"/>. You must have
        /// pre-allocated <paramref name="distinctValues"/> and <paramref name="counts"/> yourself.
        /// The scheme is destructive, because it modifies the arrays within <paramref name="values"/>.
        /// </summary>
        /// <param name="values">The values we are binning</param>
        /// <param name="distinctValues">This working array will be filled with a sorted list of the
        /// distinct values detected within <paramref name="values"/></param>
        /// <param name="counts">This working array will be filled with a sorted list of the distinct
        /// values detected within <paramref name="values"/></param>
        /// <returns>The logical length of both <paramref name="distinctValues"/> and
        /// <paramref name="counts"/></returns>
        private int FindDistinctCounts(ref VBuffer<Double> values, double[] distinctValues, int[] counts)
        {
            if (values.Count == 0)
            {
                if (values.Length == 0)
                    return 0;
                distinctValues[0] = 0;
                counts[0] = values.Length;
                return 1;
            }
            var valArray = values.Values;

            // Get histogram of values
            Array.Sort(valArray, 0, values.Count);
            // Note that Array.Sort will, by MSDN documentation, make NaN be the first item of a sorted
            // list (that is, NaN is considered to be ordered "below" any other value for the purpose of
            // a sort, including negative infinity). So when checking if values contains no NaN values, it
            // suffices to check only the first item.
            if (double.IsNaN(valArray[0]))
                return -1;
            int idist = 0; // Index into the "distinct" arrays.
            if (!values.IsDense && valArray[0] > 0)
            {
                // Implicit zeros at the head.
                distinctValues[0] = 0;
                counts[0] = values.Length - values.Count;
                idist = 1;
            }
            double last = distinctValues[idist] = valArray[0];
            counts[idist] = 1;

            for (int i = 1; i < values.Count; ++i)
            {
                double curr = valArray[i];
                if (curr != last)
                {
                    Contracts.Assert(curr > last);
                    // We are at a boundary. We will be filling in the next entry.
                    idist++;
                    if (last < 0 && curr >= 0 && !values.IsDense)
                    {
                        // This boundary is going from negative, to non-negative, and there are "implicit" zeros.
                        distinctValues[idist] = 0;
                        counts[idist] = values.Length - values.Count;
                        if (curr == 0)
                        {
                            // No need to do any more work.
                            ++counts[idist];
                            last = curr;
                            continue;
                        }
                        Contracts.Assert(curr > 0);
                        idist++; // Fall through to the general case now.
                    }
                    distinctValues[idist] = curr;
                    counts[idist] = 1;
                    last = curr;
                }
                else
                {
                    Contracts.Assert(curr == distinctValues[idist]);
                    ++counts[idist];
                }
            }
            if (!values.IsDense && distinctValues[idist] < 0)
            {
                // Implicit zeros at the tail.
                distinctValues[++idist] = 0;
                counts[idist] = values.Length - values.Count;
            }

            return idist + 1;
        }

        /// <summary>
        /// Finds the bins.
        /// </summary>
        private void FindBinsFromDistinctCounts(double[] distinctValues, int[] counts, int numValues, int maxBins, out double[] binUpperBounds, out int firstBinCount)
        {
            Contracts.Assert(0 <= numValues && numValues <= distinctValues.Length);
            Contracts.Assert(numValues <= counts.Length);

#if DEBUG
            int inv = 0;
            int bad = 0;
            var prev = double.NegativeInfinity;
            for (int i = 0; i < numValues; i++)
            {
                var v = distinctValues[i];
                if (!FloatUtils.IsFinite(v))
                    bad++;
                else
                {
                    if (!(prev < v))
                        inv++;
                    prev = v;
                }
            }
            Contracts.Assert(bad == 0, "distinctValues passed to FindBinsFromDistinctCounts contains non-finite values");
            Contracts.Assert(inv == 0, "distinctValues passed to FindBinsFromDistinctCounts is not sorted");
#endif

            if (numValues <= maxBins)
            {
                binUpperBounds = new double[Math.Max(1, numValues)];
                for (int i = 1; i < binUpperBounds.Length; i++)
                    binUpperBounds[i - 1] = GetSplitValue(distinctValues[i - 1], distinctValues[i]);
                binUpperBounds[binUpperBounds.Length - 1] = double.PositiveInfinity;

                firstBinCount = numValues > 0 ? counts[0] : 0;
                return;
            }

            var path = new int[maxBins + 1];
            _finder.FindBinsWithCounts(counts, numValues, maxBins, path);
            binUpperBounds = new double[maxBins];
            for (int i = 1; i < binUpperBounds.Length; i++)
                binUpperBounds[i - 1] = GetSplitValue(distinctValues[path[i] - 1], distinctValues[path[i]]);
            binUpperBounds[binUpperBounds.Length - 1] = double.PositiveInfinity;

            // Compute the first bin count.
            firstBinCount = 0;
            var firstBinUpperBound = binUpperBounds[0];
            for (int v = 0; v < numValues; ++v)
            {
                if (distinctValues[v] > firstBinUpperBound)
                    firstBinCount += counts[v];
            }
        }

        /// <summary>
        /// Check to see if we can "trivialize" this feature, because it would
        /// be impossible to split with the indicated minimum examples per leaf.
        /// </summary>
        /// <param name="distinctCounts">The counts of each distinct bin value</param>
        /// <param name="numDistinct">The logical length of <paramref name="distinctCounts"/></param>
        /// <param name="minPerLeaf">The minimum examples per leaf we are filtering on</param>
        /// <returns>Whether this feature is trivial, that is, it would be impossible to split on it</returns>
        private bool IsTrivial(int[] distinctCounts, int numDistinct, int minPerLeaf)
        {
            Contracts.Assert(0 <= numDistinct && numDistinct <= Utils.Size(distinctCounts));
            Contracts.Assert(minPerLeaf >= 0);

            if (minPerLeaf == 0)
                return false;

            int thresh = 0;
            int count = 0;
            while (thresh < numDistinct && count < minPerLeaf)
                count += distinctCounts[thresh++];
            // Now we've reached the earliest possible split point.
            // Reset, and continue counting.
            count = 0;
            while (thresh < numDistinct)
            {
                if ((count += distinctCounts[thresh++]) >= minPerLeaf)
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Finds the bins.
        /// </summary>
        /// <param name="values">The values we are binning</param>
        /// <param name="maxBins">The maximum number of bins to find</param>
        /// <param name="minPerLeaf">The minimum number of documents per leaf</param>
        /// <param name="binUpperBounds">The calculated upper bound of each bin</param>
        /// <returns>Whether finding the bins is successful. If there were NaN values in <paramref name="values"/>,
        /// this will return false and the output arrays will be <c>null</c>. Otherwise it will return true.</returns>
        public bool FindBins(ref VBuffer<Double> values, int maxBins, int minPerLeaf, out double[] binUpperBounds)
        {
            Contracts.Assert(maxBins > 0);
            Contracts.Assert(minPerLeaf >= 0);

            if (values.Count == 0)
            {
                binUpperBounds = TrivialBinUpperBounds;
                return true;
            }

            int arraySize = values.IsDense ? values.Count : values.Count + 1;
            Utils.EnsureSize(ref _distinctValues, arraySize, arraySize, keepOld: false);
            Utils.EnsureSize(ref _counts, arraySize, arraySize, keepOld: false);

            int numValues = FindDistinctCounts(ref values, _distinctValues, _counts);
            if (numValues < 0)
            {
                binUpperBounds = null;
                return false;
            }
            if (IsTrivial(_counts, numValues, minPerLeaf))
            {
                binUpperBounds = TrivialBinUpperBounds;
                return true;
            }
            int firstBinCount;
            FindBinsFromDistinctCounts(_distinctValues, _counts, numValues, maxBins, out binUpperBounds, out firstBinCount);
            return true;
        }

        private static double GetSplitValue(double a, double b)
        {
            // REVIEW: I am unconvinced this splitting scheme is sensible. Everything else about
            // the bin finding procedure is non-parametric in that it only depends on the order of the values
            // and not their distribution, so why do we muddy it by suddenly introducing parametricism here
            // by taking an average? Otherwise FastTree would be completely opaque to the distribution of the
            // data, which is a very nice property. (With things as they stand now, FastTree 99% doesn't
            // benefit from normalization of data, but by just returning "a" we'd 100% not benefit from
            // normalization of data.
            Contracts.Assert(a < b);
            var ave = a / 2 + b / 2;
            Contracts.Assert(a <= ave);
            return a < ave ? ave : b;
        }
    }
}
