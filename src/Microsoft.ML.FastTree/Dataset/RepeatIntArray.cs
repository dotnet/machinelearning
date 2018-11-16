// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
#if USE_SINGLE_PRECISION
    using FloatType = Single;
#else
    using FloatType = Double;
#endif

    internal sealed class DeltaRepeatIntArray : IntArray
    {
        private readonly DenseIntArray _values;
        private readonly int _length;
        private byte[] _deltas;
        private readonly int _deltasActualLength;

        public DeltaRepeatIntArray(int length, IntArrayBits bitsPerItem, IEnumerable<int> values)
        {
            using (Timer.Time(TimerEvent.SparseConstruction))
            {
                List<int> tempValueList = new List<int>();
                List<byte> tempDeltaList = new List<byte>();

                _length = 0;

                byte delta = 0;
                int lastVal = -1;

                foreach (int val in values)
                {
                    if (val != lastVal || delta == byte.MaxValue)
                    {
                        tempValueList.Add(val);
                        lastVal = val;
                        if (_length != 0)
                            tempDeltaList.Add(delta);
                        delta = 0;
                    }
                    ++delta;
                    ++_length;
                }
                if (delta > 0)
                    tempDeltaList.Add(delta);

                if (_length != length)
                    throw Contracts.Except("Length provided to repeat vector is inconsistent with value enumeration");

                // It is faster not to use a 4-bit dense array here. The memory difference is minor, since it's just
                //  the sparse values that are saved on.
                // TODO: Implement a special iterator for 4-bit array, and change this code to use the iterator, which
                //          may be faster
                if (bitsPerItem == IntArrayBits.Bits0)
                    throw Contracts.Except("Use dense arrays for 0 bits");
                if (bitsPerItem <= IntArrayBits.Bits8)
                    bitsPerItem = IntArrayBits.Bits8;

                _values = IntArray.New(tempValueList.Count, IntArrayType.Dense, bitsPerItem, tempValueList) as DenseIntArray;
                _deltas = tempDeltaList.ToArray();
            }
        }

        private DeltaRepeatIntArray(DenseIntArray values, byte[] deltas, int length)
        {
            _values = values;
            _deltas = deltas;
            _length = length;
            _deltasActualLength = length;

            if (BitsPerItem == IntArrayBits.Bits0)
                throw Contracts.Except("Use dense arrays for 0 bits");
        }

        public DeltaRepeatIntArray(byte[] buffer, ref int position)
        {
            _length = buffer.ToInt(ref position);
            _deltasActualLength = position;
            _deltasActualLength = buffer.ToInt(ref _deltasActualLength);
            _deltas = buffer.ToByteArray(ref position);
            _values = IntArray.New(buffer, ref position) as DenseIntArray;
        }

        /// <summary>
        /// Writes a binary representation of this class to a byte buffer, at a given position.
        /// The position is incremented to the end of the representation
        /// </summary>
        /// <param name="buffer">a byte array where the binary represenaion is written</param>
        /// <param name="position">the position in the byte array</param>
        public override void ToByteArray(byte[] buffer, ref int position)
        {
            base.ToByteArray(buffer, ref position);
            _length.ToByteArray(buffer, ref position);
            _deltas.ToByteArray(buffer, ref position);
            _values.ToByteArray(buffer, ref position);
        }

        public override int SizeInBytes()
        {
            return _values.SizeInBytes() + _deltas.SizeInBytes() + sizeof(int) + base.SizeInBytes();
        }

        public override MD5Hash MD5Hash
        {
            get { return MD5Hasher.Hash(_deltas) ^ _values.MD5Hash; }
        }

        public override int Length { get { return _length; } }

        public override IntArray Clone(IntArrayBits bitsPerItem, IntArrayType type)
        {
            return IntArray.New(_length, type, bitsPerItem, this);
        }

        public override IntArrayBits BitsPerItem { get { return _values.BitsPerItem; } }

        public override IntArrayType Type { get { return IntArrayType.Repeat; } }

        public override IEnumerator<int> GetEnumerator()
        {
            for (int i = 0; i < _deltas.Length; ++i)
            {
                int val = _values[i];
                for (int j = _deltas[i]; j > 0; --j)
                    yield return val;
            }
        }

        private class DeltaRepeatIntArrayIndexer : IIntArrayForwardIndexer
        {
            private readonly DeltaRepeatIntArray _array;
            private int _pos;
            private int _nextIndex; // next different index
            private int _lastVal; // The last value.

            public DeltaRepeatIntArrayIndexer(DeltaRepeatIntArray array)
            {
                _array = array;
                if (array._deltas.Length > 0)
                {
                    _pos = 0;
                    _nextIndex = array._deltas[0];
                    _lastVal = array._values[0];
                }
                else
                {
                    _pos = -1;
                    _nextIndex = 0;
                }
            }

            #region IIntArrayForwardIndexer Members

            public unsafe int this[int virtualIndex]
            {
                get
                {
                    if (virtualIndex < _nextIndex)
                        return _lastVal;

                    fixed (byte* pDeltas = _array._deltas)
                    {
                        while (++_pos < _array._values.Length)
                        {
                            //_index = _nextIndex;
                            _nextIndex += pDeltas[_pos];
                            if (virtualIndex < _nextIndex)
                                return (_lastVal = _array._values[_pos]);
                        }
                    }
                    return -1;
                }
            }

            #endregion
        }

        /// <summary>
        /// Clone an IntArray containing only the items indexed by <paramref name="itemIndices"/>
        /// </summary>
        /// <param name="itemIndices"> item indices will be contained in the cloned IntArray  </param>
        /// <returns> The cloned IntArray </returns>
        public override IntArray Clone(int[] itemIndices)
        {
            IIntArrayForwardIndexer indexer = GetIndexer();

            return new DeltaRepeatIntArray(itemIndices.Length, BitsPerItem, itemIndices.Select(i => indexer[i]));
        }

        public override IIntArrayForwardIndexer GetIndexer()
        {
            return new DeltaRepeatIntArrayIndexer(this);
        }

        public override IntArray[] Split(int[][] assignment)
        {
            return assignment.Select(a =>
            {
                IIntArrayForwardIndexer indexer = GetIndexer();
                return new DeltaRepeatIntArray(a.Length, BitsPerItem, a.Select(i => indexer[i]));
            }).ToArray();
        }

        public override void Sumup(SumupInputData input, FeatureHistogram histogram)
        {
            using (Timer.Time(TimerEvent.SumupRepeat))
            {
                if (input.DocIndices == null)
                {
                    SumupRoot(input, histogram);
                }
                else
                {
                    SumupLeaf(input, histogram);
                }
            }
        }

        private unsafe void SumupRoot(SumupInputData input, FeatureHistogram histogram)
        {
            fixed (FloatType* pOutputsFixed = input.Outputs)
            fixed (FloatType* pSumTargetsFixed = histogram.SumTargetsByBin)
            fixed (double* pWeightsFixed = input.Weights)
            fixed (double* pSumWeightsFixed = histogram.SumWeightsByBin)
            {
                FloatType* pOutputs = pOutputsFixed;
                double* pWeights = pWeightsFixed;
                for (int i = 0; i < _values.Length; i++)
                {
                    int featureBin = _values[i];
                    //FloatType* pSumTargets = pSumTargetsFixed + featureBin;
                    FloatType subsum = pSumTargetsFixed[featureBin];

                    for (int j = 0; j < _deltas[i]; ++j)
                        subsum += pOutputs[j];
                    pSumTargetsFixed[featureBin] = subsum;
                    if (pWeightsFixed != null)
                    {
                        double subweightsum = pSumWeightsFixed[featureBin];
                        for (int j = 0; j < _deltas[i]; ++j)
                            subweightsum += pWeights[j];
                        pSumWeightsFixed[featureBin] = subweightsum;
                        pWeights += _deltas[i];
                    }
                    pOutputs += _deltas[i];
                    histogram.CountByBin[featureBin] += _deltas[i];
                }
            }
        }

        private unsafe void SumupLeaf(SumupInputData input, FeatureHistogram histogram)
        {
            if (_length == 0)
                return;
            int nextStep = _deltas[0];
            int pos = 0;

            fixed (int* pDocIndicesFixed = input.DocIndices)
            fixed (FloatType* pOutputsFixed = input.Outputs)
            fixed (double* pWeightsFixed = input.Weights)
            fixed (double* pSumWeightsFixed = histogram.SumWeightsByBin)
            {
                int* pdoc = pDocIndicesFixed;
                int* end = pDocIndicesFixed + input.TotalCount;
                FloatType* pOutputs = pOutputsFixed;
                double* pWeights = pWeightsFixed;
                while (pdoc < end)
                {
                    while (nextStep <= *pdoc)
                        nextStep += _deltas[++pos];
                    int bin = _values[pos];
                    int count = 0;
                    FloatType subsum = histogram.SumTargetsByBin[bin];
                    if (pWeightsFixed != null)
                    {
                        double subweightsum = histogram.SumWeightsByBin[bin];
                        while (pdoc < end && nextStep > *pdoc)
                        {
                            subsum += *(pOutputs++);
                            subweightsum += *(pWeights++);
                            count++;
                            pdoc++;
                        }
                        histogram.SumWeightsByBin[bin] = subweightsum;
                    }
                    else
                    {
                        while (pdoc < end && nextStep > *pdoc)
                        {
                            subsum += *(pOutputs++);
                            count++;
                            pdoc++;
                        }
                    }
                    histogram.SumTargetsByBin[bin] = subsum;
                    histogram.CountByBin[bin] += count;
                }
            }
        }
    }
}
