// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    [BestFriend]
    internal abstract class BinFinderBase
    {
        private Single[] _valuesSng; // distinct values
        private Double[] _valuesDbl; // distinct values
        private List<int> _counts; // counts for each value
        private int[] _path; // current set of pegs

        protected int CountBins { get; private set; } // number of bins
        protected int CountValues { get; private set; } // number of distinct values

        protected BinFinderBase()
        {
            _counts = new List<int>();
        }

        // PositiveInfinity and NegativeInfinity are accepted, but NaN is not allowed.
        public Single[] FindBins(int cbin, IList<Single> values, int numZeroes = 0)
        {
            Contracts.Check(cbin >= 2);
            Contracts.Check(numZeroes >= 0);
            var numValues = values.Count + ((numZeroes > 0) ? 1 : 0);
            if (numValues == 0)
                return new[] { Single.PositiveInfinity };

            _counts.Clear();
            CountBins = cbin;
            if (Utils.Size(_path) < CountBins + 1)
            {
                _path = new int[CountBins + 1];
            }

            if (Utils.Size(_valuesSng) < numValues)
            {
                _valuesSng = new Single[numValues];
            }

            values.CopyTo(_valuesSng, 0);
            if (numZeroes > 0)
            {
                Contracts.Assert(numValues == values.Count + 1);
                _valuesSng[values.Count] = 0;
                --numZeroes;
            }

            Array.Sort(_valuesSng, 0, numValues);

            // Get the distinct _values and counts.
            int count = 1;
            for (int i = 1; i < numValues; i++)
            {
                if (_valuesSng[i] > _valuesSng[i - 1])
                {
                    if (_valuesSng[i - 1] == 0)
                        count += numZeroes;
                    _counts.Add(count);
                    _valuesSng[_counts.Count] = _valuesSng[i];
                    count = 1;
                }
                else
                    count++;
            }
            if (_valuesSng[numValues - 1] == 0)
                count += numZeroes;
            _counts.Add(count);

            Single[] bounds;
            CountValues = _counts.Count;
            if (CountValues <= CountBins)
            {
                bounds = new Single[CountValues];
                for (int i = 1; i < bounds.Length; i++)
                    bounds[i - 1] = GetSplitValue(_valuesSng[i - 1], _valuesSng[i]);
            }
            else
            {
                FindBinsCore(_counts, _path);
                Contracts.Assert(_path[0] == 0);
                Contracts.Assert(_path[CountBins] == CountValues);
                bounds = new Single[CountBins];
                for (int i = 1; i < bounds.Length; i++)
                    bounds[i - 1] = GetSplitValue(_valuesSng[_path[i] - 1], _valuesSng[_path[i]]);
            }

            bounds[bounds.Length - 1] = Single.PositiveInfinity;

#if DEBUG
            for (int i = 1; i < bounds.Length; i++)
                Contracts.Assert(bounds[i] > bounds[i - 1]);
#endif

            return bounds;
        }

        // PositiveInfinity and NegativeInfinity are accepted, but NaN is not allowed.
        public Double[] FindBins(int cbin, IList<Double> values, int numZeroes = 0)
        {
            Contracts.Check(cbin >= 2);
            Contracts.Check(numZeroes >= 0);
            var numValues = values.Count + ((numZeroes > 0) ? 1 : 0);
            if (numValues == 0)
                return new[] { Double.PositiveInfinity };

            _counts.Clear();
            CountBins = cbin;
            if (Utils.Size(_path) < CountBins + 1)
            {
                _path = new int[CountBins + 1];
            }

            if (Utils.Size(_valuesDbl) < numValues)
            {
                _valuesDbl = new Double[numValues];
            }

            values.CopyTo(_valuesDbl, 0);
            if (numZeroes > 0)
            {
                Contracts.Assert(numValues == values.Count + 1);
                _valuesDbl[values.Count] = 0;
                --numZeroes;
            }

            Array.Sort(_valuesDbl, 0, numValues);

            // Get the distinct _values and counts.
            int count = 1;
            for (int i = 1; i < numValues; i++)
            {
                if (_valuesDbl[i] > _valuesDbl[i - 1])
                {
                    if (_valuesDbl[i - 1] == 0)
                        count += numZeroes;
                    _counts.Add(count);
                    _valuesDbl[_counts.Count] = _valuesDbl[i];
                    count = 1;
                }
                else
                    count++;
            }
            if (_valuesDbl[numValues - 1] == 0)
                count += numZeroes;
            _counts.Add(count);

            Double[] bounds;
            CountValues = _counts.Count;
            if (CountValues <= CountBins)
            {
                bounds = new Double[CountValues];
                for (int i = 1; i < bounds.Length; i++)
                    bounds[i - 1] = GetSplitValue(_valuesDbl[i - 1], _valuesDbl[i]);
            }
            else
            {
                FindBinsCore(_counts, _path);
                Contracts.Assert(_path[0] == 0);
                Contracts.Assert(_path[CountBins] == CountValues);
                bounds = new Double[CountBins];
                for (int i = 1; i < bounds.Length; i++)
                    bounds[i - 1] = GetSplitValue(_valuesDbl[_path[i] - 1], _valuesDbl[_path[i]]);
            }

            bounds[bounds.Length - 1] = Double.PositiveInfinity;

#if DEBUG
            for (int i = 1; i < bounds.Length; i++)
                Contracts.Assert(bounds[i] > bounds[i - 1]);
#endif

            return bounds;
        }

        public void FindBinsWithCounts(IList<int> counts, int numValues, int cbin, int[] path)
        {
            Contracts.Check(cbin >= 2);
            Contracts.Check(numValues > cbin);
            Contracts.Check(counts.Count >= numValues);
            Contracts.Check(path.Length >= cbin + 1);

            CountValues = numValues;
            CountBins = cbin;
            _counts.Clear();
            // REVIEW: counts is a buffer that can be larger than numValues, hence the need to copy it here.
            // A better option would be to pass down the numValues and avoid the copy
            for (int i = 0; i < numValues; i++)
            {
                var count = counts[i];
                _counts.Add(count);
            }

            FindBinsCore(_counts, path);

            Contracts.Assert(path[0] == 0);
            Contracts.Assert(path[CountBins] == CountValues);
        }

        /// <summary>
        /// This should normally be just (a + b) / 2, except in one interesting case:
        /// If a and b are 'consecutive' floats (they differ only in the least significant bit),
        /// the above expression is possible to be rounded to a.
        /// This can lead to bin bounds that are not strictly increasing!
        /// Also note that the simple (a + b) / 2 can overflow, that's the reason for a / 2 + b / 2.
        /// </summary>
        public static Single GetSplitValue(Single a, Single b)
        {
            Contracts.Assert(a < b);

            if (Single.IsNegativeInfinity(a))
            {
                if (b == Single.MinValue)
                    return Single.MinValue;
                a = Single.MinValue;
            }
            if (Single.IsPositiveInfinity(b))
            {
                if (a == Single.MaxValue)
                    return Single.PositiveInfinity;
                b = Single.MaxValue;
            }

            var ave = a / 2 + b / 2;
            Contracts.Assert(a <= ave);
            return a < ave ? ave : b;
        }

        /// <summary>
        /// This should normally be just (a + b) / 2, except in one interesting case:
        /// If a and b are 'consecutive' floats (they differ only in the least significant bit),
        /// the above expression is possible to be rounded to a.
        /// This can lead to bin bounds that are not strictly increasing!
        /// Also note that the simple (a + b) / 2 can overflow, that's the reason for a / 2 + b / 2.
        /// </summary>
        public static Double GetSplitValue(Double a, Double b)
        {
            Contracts.Assert(a < b);

            if (Double.IsNegativeInfinity(a))
            {
                if (b == Double.MinValue)
                    return Double.MinValue;
                a = Double.MinValue;
            }
            if (Double.IsPositiveInfinity(b))
            {
                if (a == Double.MaxValue)
                    return Double.PositiveInfinity;
                b = Double.MaxValue;
            }

            var ave = a / 2 + b / 2;
            Contracts.Assert(a <= ave);
            return a < ave ? ave : b;
        }

        // Given the counts of the distinct values (in order), produce the boundary indices
        // as a "path" array. The returned path should satisfy:
        // * path.Length = _cbin+ 1
        // * path[0] = 0
        // * path[_cbin] = _cval
        // * path is strictly increasing.
        protected abstract void FindBinsCore(List<int> counts, int[] path);
    }
}

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    // This needs to be large enough to represent a product of 2 ints without losing precision
    using EnergyType = System.Int64;

    // Uses the energy function: sum(1,N) dx^2 where dx is the difference in accum values.
    [BestFriend]
    internal sealed class GreedyBinFinder : BinFinderBase
    {
        // Potential drop location for another peg, together with its energy improvement.
        // PlacePegs uses a heap of these. Note that this is a struct so size matters.
        private readonly struct Segment
        {
            public readonly int Min;
            public readonly int Split;
            public readonly int Max;
            public readonly EnergyType Energy;

            public Segment(int min, int split, int max, EnergyType energy)
            {
                Min = min;
                Split = split;
                Max = max;
                Energy = energy;
            }
        }

        // Potential peg move, together with its energy decrease over the current peg location.
        // ReduceEnergy uses a heap (with deletion) of these.
        private class Peg : HeapNode
        {
            public readonly int Index;
            public int Split;
            public EnergyType Energy;

            public Peg(int index, int split)
            {
                Index = index;
                Split = split;
            }
        }

        private Heap<Segment> _segmentHeap; // heap used for dropping initial peg placement
        private HeapNode.Heap<Peg> _pegHeap; // heap used for selecting the largest energy decrease
        private int[] _accum; // integral of counts
        private int[] _path; // current set of pegs
        private Float _meanBinSize;

        public GreedyBinFinder()
        {
            _segmentHeap = new Heap<Segment>((a, b) => a.Energy < b.Energy);
            _pegHeap = new HeapNode.Heap<Peg>((a, b) => a.Energy < b.Energy);
        }

        protected override void FindBinsCore(List<int> counts, int[] path)
        {
            Contracts.Assert(CountValues > CountBins);
            Contracts.Assert(counts.Count == CountValues);
            Contracts.Assert(path.Length >= CountBins + 1);

            _path = path;
            // Integrate counts into _accum.
            if (Utils.Size(_accum) < CountValues + 1)
                _accum = new int[CountValues + 1];
            for (int i = 0; i < CountValues; i++)
                _accum[i + 1] = _accum[i] + counts[i];
            _meanBinSize = (Float)_accum[CountValues] / CountBins;

            PlacePegs();

            ReduceEnergy();
        }

        /// <summary>
        /// Initial placement of the pegs.
        /// Places pegs one by one and always picks the largest existing segment to split.
        /// </summary>
        private void PlacePegs()
        {
            _segmentHeap.Clear();

            Segment seg = GetSegmentSplit(0, CountValues);
            _segmentHeap.Add(seg);

            _path[0] = 0;
            for (int i = 1; i < CountBins; i++)
            {
                Contracts.Assert(_segmentHeap.Count > 0);
                seg = _segmentHeap.Pop();
                _path[i] = seg.Split;
                if (seg.Min < seg.Split - 1)
                    _segmentHeap.Add(GetSegmentSplit(seg.Min, seg.Split));
                if (seg.Split < seg.Max - 1)
                    _segmentHeap.Add(GetSegmentSplit(seg.Split, seg.Max));
            }
            _path[CountBins] = CountValues;

            Array.Sort(_path, 0, CountBins + 1);
            Contracts.Assert(_path[0] == 0);
            Contracts.Assert(_path[CountBins] == CountValues);
        }

        /// <summary>
        /// Gets the best split for a peg between min and max.
        /// In case of a tie, use 'pos' to pick the location closer to the more natural split.
        /// </summary>
        private Segment GetSegmentSplit(int min, int max)
        {
            EnergyType energy;
            int split = FindSplitPosition(out energy, min, max);

            return new Segment(min, split, max, energy);
        }

        private int FindSplitPosition(out EnergyType energy, int min, int max, int pos = -1)
        {
            Contracts.Assert(min < max - 1);
            int key = (int)((uint)(_accum[min] + _accum[max]) / 2);
            int split = Utils.FindIndexSorted(_accum, min, max, key);
            if (split == min)
            {
                split++;
                energy = GetSplitEnergy(min, split, max);
            }
            else if (split == max)
            {
                split--;
                energy = GetSplitEnergy(min, split, max);
            }
            else
            {
                energy = GetSplitEnergy(min, split, max);
                EnergyType e;
                if (split < max - 1 && (e = GetSplitEnergy(min, split + 1, max)) >= energy)
                {
                    if (e > energy || BetterPlacement(split + 1, split, pos))
                    {
                        energy = e;
                        split++;
                    }
                }
                else if (split > min + 1 && (e = GetSplitEnergy(min, split - 1, max)) >= energy)
                {
                    if (e > energy || BetterPlacement(split - 1, split, pos))
                    {
                        energy = e;
                        split--;
                    }
                }
            }

            return split;
        }

        /// <summary>
        /// Computes the energy reduction for splitting segment [min, max] at 'split' point
        /// </summary>
        private EnergyType GetSplitEnergy(int min, int split, int max)
        {
            Contracts.Assert(0 <= min && min < split && split < max && max <= CountValues);
            int a = _accum[min];
            int b = _accum[split];
            int c = _accum[max];

            // With x = c - b and y = b - a, the energy reduction is:
            //   (x + y)^2 - x^2 - y^2 == 2*x*y,
            // so x*y = (c-b)*(b-a) is a good measure of energy reduction.
            Contracts.Assert(a < b && b < c);
            return (EnergyType)(c - b) * (b - a);
        }

        /// <summary>
        /// Returns true if 'i' is a better peg placement than 'j', which means it is closer to the ideal position for peg 'pos'
        /// </summary>
        private bool BetterPlacement(int i, int j, int pos)
        {
            if (pos > 0)
            {
                var ideal = _meanBinSize * pos;
                return Math.Abs(_accum[i] - ideal) < Math.Abs(_accum[j] - ideal);
            }

            return false;
        }

        /// <summary>
        /// After the initial peg placement,
        /// </summary>
        private void ReduceEnergy()
        {
            // Initializes the heap with the best moves for each peg.
            var pegs = new Peg[CountBins + 1];
            _pegHeap.Clear();
            for (int i = 1; i < CountBins; i++)
            {
                int min = _path[i - 1];
                int cur = _path[i];
                int max = _path[i + 1];

                EnergyType energy;
                int split = FindSplitPosition(out energy, min, max, i);
                var peg = new Peg(i, split);
                // The segment energy is now just the delta from the previous position
                pegs[i] = peg;
                if (peg.Split != cur)
                {
                    peg.Energy = energy - GetSplitEnergy(min, cur, max);
                    Contracts.Assert(peg.Energy >= 0);
                    _pegHeap.Add(peg);
                }
            }

            // While we can still reduce energy, do the best move and update if necessary the left and right moves.
            while (_pegHeap.Count > 0)
            {
                var peg = _pegHeap.Pop();
                Contracts.Assert(pegs[peg.Index] == peg);
                Contracts.Assert(_path[peg.Index] != peg.Split);
                _path[peg.Index] = peg.Split;
                EnergyType e;
                Contracts.Assert(FindSplitPosition(out e, _path[peg.Index - 1], _path[peg.Index + 1], peg.Index) == peg.Split);

                if (peg.Index > 1)
                    UpdatePeg(pegs[peg.Index - 1]);

                if (peg.Index < CountBins - 1)
                    UpdatePeg(pegs[peg.Index + 1]);
            }
        }

        private void UpdatePeg(Peg peg)
        {
            if (peg.InHeap)
                _pegHeap.Delete(peg);

            int min = _path[peg.Index - 1];
            int cur = _path[peg.Index];
            int max = _path[peg.Index + 1];
            EnergyType energy;
            int split = FindSplitPosition(out energy, min, max, peg.Index);

            if (split != cur)
            {
                peg.Split = split;
                peg.Energy = energy - GetSplitEnergy(min, cur, max);
                Contracts.Assert(peg.Energy >= 0);
                _pegHeap.Add(peg);
            }
        }
    }
}

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    // Reasonable choices are Double and System.Int64.
    using EnergyType = System.Double;

    // Uses dynamic programming.
    [BestFriend]
    internal sealed class DynamicBinFinder : BinFinderBase
    {
        private int[] _accum; // integral of counts

        // Number of "holes" that will be skipped.
        private int _cskip;

        // We use (row,col) dimensions where:
        // * 0 <= row < _cbin
        // * 0 <= col <= _cskip
        // * Each row corresponds to placing one additional peg (left to right)
        // * Each column corresponds to the number of holes that have been skipped (to the left of this "position").
        // * (row,col) has its latest peg in hole number row+1+col.

        // This tracks the best predecessor peg location. It stores values for each (row,col) with
        // 1 <= row < _cbin and 0 <= col <= _cskip in row major order. Note that it doesn't store values for row = 0.
        // The value at location (row,col) is the column of the predecessor peg that gives minimum energy of all
        // possible left configurations that have placed row+1 pegs with the latest peg in position (row+1)+col.
        private int[] _pathInfo;

        // This tracks the minimum energy achieved for the current row in _pathInfo. Note that to compute
        // the energy for (row, col), we only use energies for (row-1, colPrev) where colPrev <= col.
        // Thus a single row of values is sufficient.
        // REVIEW: Consider storing energies in reverse order to match the common access pattern.
        // REVEIW: What should we use for the energy type?
        private EnergyType[] _energies;
        private EnergyType[] _energiesBest;

        private int[] _cols;
        private int[] _colsNew;
        private int[] _path;

        protected override void FindBinsCore(List<int> counts, int[] path)
        {
            Contracts.Assert(CountBins >= 2);
            Contracts.Assert(CountValues > CountBins);
            Contracts.Assert(counts.Count == CountValues);
            Contracts.Assert(path.Length >= CountBins + 1);

            _path = path;
            // Integrate counts into _accum.
            if (Utils.Size(_accum) < CountValues + 1)
                _accum = new int[CountValues + 1];
            for (int i = 0; i < CountValues; i++)
                _accum[i + 1] = _accum[i] + counts[i];

            // Initialize the energy table.
            _cskip = CountValues - CountBins;
            int width = _cskip + 1;
            int height = CountBins - 1;

            int sizeEnergy = checked(width);
            if (Utils.Size(_energies) < sizeEnergy)
            {
                _energiesBest = new EnergyType[sizeEnergy];
                _energies = new EnergyType[sizeEnergy];
            }
            int sizeInfo = checked(width * (height - 1));
            if (Utils.Size(_pathInfo) < sizeInfo)
                _pathInfo = new int[sizeInfo];
            if (Utils.Size(_cols) < width)
            {
                _colsNew = new int[width];
                _cols = new int[width];
            }

            // Row zero is special.
            EnergyType bestWorst = EnergyType.MaxValue;
            for (int col = width; --col >= 0; )
            {
                _energies[col] = Square(_accum[1 + col]);
                EnergyType worst;
                GetEnergyBounds(0, col, _energies[col], out _energiesBest[col], out worst);
                if (bestWorst > worst)
                    bestWorst = worst;
            }

            Contracts.Assert(bestWorst < EnergyType.MaxValue);
            int ccol = 0;
            for (int col = 0; col < width; col++)
            {
                if (_energiesBest[col] <= bestWorst)
                    _cols[ccol++] = col;
            }
            Contracts.Assert(ccol > 0);

            int colMin = _cols[0];

            int colBest;
            EnergyType eBest;
            int ivBase = 0;
            int icolSrc = width;
            for (int row = 1; row < height; row++)
            {
                Contracts.Assert(ivBase == (row - 1) * width);
                for (int col = width; --col >= colMin; )
                {
                    int accum = _accum[row + 1 + col];
                    eBest = EnergyType.MaxValue;
                    colBest = -1;
                    for (int icol = 0; icol < ccol; icol++)
                    {
                        var colPrev = _cols[icol];
                        if (colPrev > col)
                            break;
                        var e = _energies[colPrev];
                        if (eBest <= e)
                            continue;
                        e += Square(accum - _accum[row + colPrev]);
                        if (eBest > e)
                        {
                            eBest = e;
                            colBest = colPrev;
                        }
                    }
                    if (eBest <= bestWorst)
                    {
                        Contracts.Assert(eBest > 0);
                        Contracts.Assert(0 <= colBest && colBest <= col);
                        _energies[col] = eBest;
                        _pathInfo[ivBase + col] = colBest;
                        EnergyType worst;
                        GetEnergyBounds(row, col, eBest, out _energiesBest[col], out worst);
                        if (bestWorst > worst)
                            bestWorst = worst;
                        _colsNew[--icolSrc] = col;
                    }
                }
                ivBase += width;

                // Determine new set of columns.
                Contracts.Assert(bestWorst < EnergyType.MaxValue);
                ccol = 0;
                while (icolSrc < width)
                {
                    int col = _colsNew[icolSrc++];
                    if (_energiesBest[col] <= bestWorst)
                        _cols[ccol++] = col;
                }
                Contracts.Assert(ccol > 0);
            }

            // The last peg must be in the last column.
            Contracts.Assert(height + width == CountValues);
            int total = _accum[CountValues];
            eBest = EnergyType.MaxValue;
            colBest = -1;
            for (int icol = 0; icol < ccol; icol++)
            {
                var colPrev = _cols[icol];
                var e = _energies[colPrev] + Square(total - _accum[height + colPrev]);
                if (eBest > e)
                {
                    eBest = e;
                    colBest = colPrev;
                }
            }
            Contracts.Assert(eBest < EnergyType.MaxValue);

            _path[0] = 0;
            _path[CountBins] = CountValues;
            Contracts.Assert(height == CountBins - 1);
            _path[height] = height + colBest;

            // Fill in the rest of the path.
            ivBase = (height - 1) * width;
            for (int row = height; --row > 0; )
            {
                // Recall that the _pathInfo table doesn't have row zero.
                ivBase -= width;
                Contracts.Assert(ivBase == (row - 1) * width);

                Contracts.Assert(_pathInfo[ivBase + colBest] <= colBest);
                colBest = _pathInfo[ivBase + colBest];
                _path[row] = row + colBest;
                Contracts.Assert(_path[row] < _path[row + 1]);
            }
            Contracts.Assert(ivBase == 0);
        }

        private static EnergyType Square(int d)
        {
            Contracts.Assert(d > 0);
            return (EnergyType)d * d;
        }

        /// <summary>
        /// For the remaining bins, compute the best energy distribution and the worst energy distribution
        /// The best energy distribution:
        ///     - make the distances as equal as possible;
        ///     - some of them will be 'ave' and the rest will be 'ave+1'
        /// The worst energy distribution:
        ///     - all except one distance will be 1
        /// </summary>
        private void GetEnergyBounds(int row, int col, EnergyType cur, out EnergyType best, out EnergyType worst)
        {
            Contracts.Assert(0 <= row && row < CountBins - 1);
            Contracts.Assert(0 <= col && col <= _cskip);

            // The distance to span.
            int span = _accum[CountValues] - _accum[row + 1 + col];

            // The number of remaining bins.
            int cbin = CountBins - row - 1;
            Contracts.Assert(0 < cbin && cbin <= span);

            // Best case is that the remaining pegs are all equally spaced. Of course, they have
            // to be spaced integer distances apart.
            int ave = span / cbin;
            int rem = span - ave * cbin;
            Contracts.Assert(0 <= rem && rem < cbin);
            best = cbin * Square(ave);
            if (rem > 0)
                best += rem * (2 * ave + 1);
            best += cur;

            // Worst case is all 1's except for one big bin.
            int ones = cbin - 1;
            worst = Square(span - ones) + ones;
            worst += cur;

            Contracts.Assert(worst >= best);
        }
    }
}
