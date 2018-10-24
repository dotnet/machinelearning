// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    /// <summary>
    /// This is a feature flock that misuses a property of <see cref="DeltaSparseIntArray.Sumup"/>
    /// that it works just fine with zero deltas, to "overload" rows into having their targets, thus
    /// allowing a single sumup to accumulate multiple "features" on a single row.
    /// </summary>
    internal sealed class NHotFeatureFlock : SinglePartitionedIntArrayFlockBase<DeltaSparseIntArray>
    {
        // We abuse a property of the sparse int array that it never actually
        // checks or enforces that the deltas are non-zero.

        // These are the same as the internal structures to the delta sparse int array.
        private readonly DenseIntArray _values;
        private readonly byte[] _deltas;

        public NHotFeatureFlock(DenseIntArray values, byte[] deltas, int len,
            int[] hotFeatureStarts, double[][] binUpperBounds)
            : base(new DeltaSparseIntArray(values, deltas, len), hotFeatureStarts, binUpperBounds)
        {
            _values = values;
            _deltas = deltas;
        }

        public override FeatureFlockBase[] Split(int[][] assignment)
        {
            // REVIEW: This is not implemented because no code actually ever calls split, but this
            // may change in the future. If it does we'll need to be a little less stupid here of course.
            throw Contracts.ExceptNotImpl("Lazy Tom");
        }

        public override FlockForwardIndexerBase GetFlockIndexer()
        {
            return new FlockIndexer(this);
        }

        private sealed class FlockIndexer : FlockForwardIndexerBase
        {
            private readonly NHotFeatureFlock _flock;
            private int _pos;
            private int _nextIndex; // Next non-zero index.

            public override FeatureFlockBase Flock
            {
                get { return _flock; }
            }

            public FlockIndexer(NHotFeatureFlock flock)
            {
                Contracts.AssertValue(flock);
                _flock = flock;
                if (_flock._deltas.Length > 0)
                    _nextIndex = _flock._deltas[0];
                else
                    _nextIndex = _flock.Bins.Length;
            }

            public override int this[int featureIndex, int rowIndex]
            {
                get
                {
                    Contracts.Assert(0 <= featureIndex && featureIndex < _flock.Count);
                    Contracts.Assert(0 <= rowIndex && rowIndex < _flock.Bins.Length);
                    // Forward to the next rowIndex.
                    while (rowIndex > _nextIndex)
                    {
                        if (++_pos < _flock._deltas.Length)
                            _nextIndex += _flock._deltas[_pos];
                        else
                            _nextIndex = _flock.Bins.Length;
                    }
                    if (_nextIndex > rowIndex)
                        return 0;
                    Contracts.Assert(_nextIndex == rowIndex);
                    Contracts.Assert(_pos < _flock._deltas.Length);
                    Contracts.Assert(_pos < _flock._values.Length);
                    int min = _flock.HotFeatureStarts[featureIndex];
                    int lim = _flock.HotFeatureStarts[featureIndex + 1];
                    int p = _pos;
                    do
                    {
                        int v = _flock._values[p];
                        if (v >= lim) // We've past the valid range.
                            return 0;
                        if (v >= min) // We've found the right range!
                            return v - min + 1;
                    } while (++p < _flock._deltas.Length && _flock._deltas[p] == 0);
                    return 0;
                }
            }
        }
    }
}
