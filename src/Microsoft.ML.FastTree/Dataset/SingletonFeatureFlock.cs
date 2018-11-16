// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Internal.Utilities;
using System.Linq;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    /// <summary>
    /// The singleton feature flock is the simplest possible sort of flock, that is, a flock
    /// over one feature.
    /// </summary>
    internal sealed class SingletonFeatureFlock : FeatureFlockBase
    {
        private readonly IntArray _bins;
        private readonly double[] _binUpperBounds;

        public override int Examples => _bins.Length;

        public SingletonFeatureFlock(IntArray bins, double[] binUpperBounds)
            : base(1)
        {
            Contracts.AssertValue(bins);
            Contracts.AssertValue(binUpperBounds);
            Contracts.Assert(bins.Length == 0 || bins.Max() < binUpperBounds.Length);

            _bins = bins;
            _binUpperBounds = binUpperBounds;
        }

        public override long SizeInBytes()
        {
            return _bins.SizeInBytes() + sizeof(double) * _binUpperBounds.Length;
        }

        public override SufficientStatsBase CreateSufficientStats(bool hasWeights)
        {
            return new SufficientStats(this, hasWeights);
        }

        public override IIntArrayForwardIndexer GetIndexer(int featureIndex)
        {
            Contracts.Assert(featureIndex == 0);
            return _bins.GetIndexer();
        }

        public override int BinCount(int featureIndex)
        {
            Contracts.Assert(featureIndex == 0);
            return _binUpperBounds.Length;
        }

        public override FlockForwardIndexerBase GetFlockIndexer()
        {
            return new Indexer(this);
        }

        public override FeatureFlockBase[] Split(int[][] assignment)
        {
            return _bins.Split(assignment)
                .Select(bins => new SingletonFeatureFlock(bins, _binUpperBounds)).ToArray();
        }

        public override double Trust(int featureIndex)
        {
            Contracts.Assert(featureIndex == 0);
            return 1;
        }

        public override double[] BinUpperBounds(int featureIndex)
        {
            Contracts.Assert(featureIndex == 0);
            return _binUpperBounds;
        }

        private sealed class Indexer : FlockForwardIndexerBase
        {
            private readonly SingletonFeatureFlock _flock;
            private readonly IIntArrayForwardIndexer _indexer;

            public override FeatureFlockBase Flock { get { return _flock; } }

            public override int this[int featureIndex, int rowIndex]
            {
                get
                {
                    Contracts.Assert(featureIndex == 0);
                    return _indexer[rowIndex];
                }
            }

            public Indexer(SingletonFeatureFlock flock)
            {
                Contracts.AssertValue(flock);
                _flock = flock;
                _indexer = _flock.GetIndexer(0);
            }
        }

        private sealed class SufficientStats : SufficientStatsBase<SufficientStats>
        {
            private readonly SingletonFeatureFlock _flock;
            private readonly FeatureHistogram _hist;

            public override FeatureFlockBase Flock
            {
                get { return _flock; }
            }

            public SufficientStats(SingletonFeatureFlock flock, bool hasWeights)
                : base(flock.Count)
            {
                Contracts.AssertValue(flock);
                _flock = flock;
                _hist = new FeatureHistogram(_flock._bins, _flock._binUpperBounds.Length, hasWeights);
            }

            protected override void SubtractCore(SufficientStats other)
            {
                _hist.Subtract(other._hist);
            }

            protected override void SumupCore(int featureOffset, bool[] active,
                int numDocsInLeaf, double sumTargets, double sumWeights,
                double[] outputs, double[] weights, int[] docIndices)
            {
                Contracts.AssertValueOrNull(active);
                Contracts.Assert(active == null || (0 <= featureOffset && featureOffset <= Utils.Size(active) - Flock.Count));
                if (active != null && !active[featureOffset])
                    return;
                _hist.SumupWeighted(numDocsInLeaf, sumTargets, sumWeights, outputs, weights, docIndices);
            }

            public override long SizeInBytes()
            {
                return FeatureHistogram.EstimateMemoryUsedForFeatureHistogram(_hist.NumFeatureValues,
                    _hist.SumWeightsByBin != null);
            }

            protected override int GetMaxBorder(int featureIndex)
            {
                return _hist.NumFeatureValues - 1;
            }

            protected override int GetMinBorder(int featureIndex)
            {
                return 1;
            }

            protected override PerBinStats GetBinStats(int featureIndex)
            {
                if (_hist.SumWeightsByBin != null)
                    return new PerBinStats(_hist.SumTargetsByBin[featureIndex], _hist.SumWeightsByBin[featureIndex], _hist.CountByBin[featureIndex]);
                else
                    return new PerBinStats(_hist.SumTargetsByBin[featureIndex], 0, _hist.CountByBin[featureIndex]);
            }

            protected override double GetBinGradient(int featureIndex, double bias)
            {
                if (_hist.SumWeightsByBin != null)
                    return _hist.SumTargetsByBin[featureIndex] / (_hist.SumWeightsByBin[featureIndex] + bias);
                else
                    return _hist.SumTargetsByBin[featureIndex] / (_hist.CountByBin[featureIndex] + bias);
            }
        }
    }
}
