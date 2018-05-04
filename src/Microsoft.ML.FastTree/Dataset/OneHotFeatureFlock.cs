// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    /// <summary>
    /// A feature flock for a set of features where per example at most one of the features has a
    /// non-zero bin.
    /// </summary>
    internal sealed class OneHotFeatureFlock : SinglePartitionedIntArrayFlockBase<IntArray>
    {
        public OneHotFeatureFlock(IntArray bins, int[] hotFeatureStarts, double[][] binUpperBounds, bool categorical)
            : base(bins, hotFeatureStarts, binUpperBounds, categorical)
        {
        }

        public override FeatureFlockBase[] Split(int[][] assignment)
        {
            //REVIEW: Does it make sense to call a flock categorical here?
            return Bins.Split(assignment)
                .Select(bins => new OneHotFeatureFlock(bins, HotFeatureStarts, AllBinUpperBounds, false))
                .ToArray();
        }

        public override IIntArrayForwardIndexer GetIndexer(int featureIndex)
        {
            Contracts.Assert(0 <= featureIndex && featureIndex < Count);
            if (HotFeatureStarts[featureIndex] == HotFeatureStarts[featureIndex + 1])
                return new Dense0BitIntArray(Bins.Length);
            return new Indexer(Bins.GetIndexer(), HotFeatureStarts[featureIndex],
                HotFeatureStarts[featureIndex + 1]);
        }

        public override FlockForwardIndexerBase GetFlockIndexer()
        {
            return new FlockIndexer(this);
        }

        private sealed class Indexer : IIntArrayForwardIndexer
        {
            private readonly IIntArrayForwardIndexer _indexer;
            private readonly int _minMinusOne;
            private readonly int _lim;

            public int this[int index]
            {
                get
                {
                    int val = _indexer[index];
                    if (_minMinusOne < val && val < _lim)
                        return val - _minMinusOne;
                    return 0;
                }
            }

            /// <summary>
            /// Instantiates an indexer that translates from the "concatenated" bin space across all features,
            /// into the original logical space for each individual feature.
            /// </summary>
            /// <param name="indexer">The indexer into the "shared" <see cref="IntArray"/>, that we
            /// are translating into the original logical space for this feature, where values in the
            /// range of [<paramref name="min"/>,<paramref name="lim"/>) will map from 1 onwards, and all
            /// other values will map to 0</param>
            /// <param name="min">The minimum value from the indexer that will map to 1</param>
            /// <param name="lim">The exclusive upper bound on values from the indexer</param>
            public Indexer(IIntArrayForwardIndexer indexer, int min, int lim)
            {
                Contracts.AssertValue(indexer);
                Contracts.Assert(1 <= min && min < lim);
                _indexer = indexer;
                _minMinusOne = min - 1;
                _lim = lim;
            }
        }

        private sealed class FlockIndexer : FlockForwardIndexerBase
        {
            private readonly OneHotFeatureFlock _flock;
            private readonly IIntArrayForwardIndexer _indexer;

            public override FeatureFlockBase Flock { get { return _flock; } }

            public override int this[int featureIndex, int rowIndex]
            {
                get
                {
                    Contracts.Assert(0 <= featureIndex && featureIndex < _flock.Count);
                    int val = _indexer[rowIndex];
                    int min = _flock.HotFeatureStarts[featureIndex];
                    if (min <= val && val < _flock.HotFeatureStarts[featureIndex + 1])
                        return val - min + 1;
                    return 0;
                }
            }

            public FlockIndexer(OneHotFeatureFlock flock)
            {
                Contracts.AssertValue(flock);
                _flock = flock;
                _indexer = _flock.Bins.GetIndexer();
            }
        }
    }
}
