// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
#if USE_SINGLE_PRECISION
    using FloatType = System.Single;
#else
    using FloatType = System.Double;
#endif

    /// <summary>
    /// This <see cref="IntArray"/> implementation represents a sequence of values using parallel
    /// arrays of both values, as well as deltas indicating the number of values to the next
    /// explicit value. Values "between" these deltas are implicitly zero.
    /// 
    /// Note that it is possible to misuse the deltas by making some of them themselves 0, allowing
    /// us to represent multiple values per row. In this case, <see cref="IntArray.GetEnumerator"/>
    /// and <see cref="IntArray.GetIndexer"/> will not have sensible values, but
    /// <see cref="IntArray.Sumup"/> will work and accumulate the same target into multiple bins.
    /// This trick should be used with caution, and is useful for the <see cref="NHotFeatureFlock"/>.
    /// </summary>
    internal sealed class DeltaSparseIntArray : IntArray
    {
        private readonly DenseIntArray _values;
        private readonly byte[] _deltas;
        private readonly int _length;

        public DeltaSparseIntArray(int length, IntArrayBits bitsPerItem)
        {
            if (bitsPerItem == IntArrayBits.Bits0)
                throw Contracts.Except("Use dense arrays for 0 bits");
            if (bitsPerItem <= IntArrayBits.Bits8)
                bitsPerItem = IntArrayBits.Bits8;

            _length = length;
        }

        public DeltaSparseIntArray(int length, IntArrayBits bitsPerItem, IEnumerable<int> values)
        {
            using (Timer.Time(TimerEvent.SparseConstruction))
            {
                List<int> tempValueList = new List<int>();
                List<byte> tempDeltaList = new List<byte>();

                _length = 0;

                byte delta = 0;

                foreach (int val in values)
                {
                    if (val != 0 || delta == byte.MaxValue)
                    {
                        tempDeltaList.Add(delta);
                        tempValueList.Add(val);
                        _length += delta;
                        delta = 0;
                    }
                    ++delta;
                }

                _length += delta;

                if (_length != length)
                    throw Contracts.Except("Length provided to sparse vector is inconsistent with value enumeration");

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

        public DeltaSparseIntArray(byte[] buffer, ref int position)
        {
            _length = buffer.ToInt(ref position);
            // REVIEW: The two lines below is as it actually appeared. I have no earthly idea of what
            // it was trying to accomplish. It appears to function as a no-op, resulting in no valid results
            // inside _deltasActualLength.
            //_deltasActualLength = position;
            //_deltasActualLength = buffer.ToInt(ref _deltasActualLength);
            _deltas = buffer.ToByteArray(ref position);
            _values = IntArray.New(buffer, ref position) as DenseIntArray;
        }

        public DeltaSparseIntArray(DenseIntArray values, byte[] deltas, int length)
        {
            Contracts.AssertValue(values);
            Contracts.AssertValue(deltas);
            Contracts.Assert(values.Length == deltas.Length);
            Contracts.Assert(deltas.Sum(d => (long)d) < length);

            _values = values;
            _deltas = deltas;
            _length = length;

            if (BitsPerItem == IntArrayBits.Bits0)
                throw Contracts.Except("Use dense arrays for 0 bits");
        }

        /// <summary>
        /// Construct a sparse int array from index, value pairs.
        /// </summary>
        /// <param name="length">The total length of the constructed array.</param>
        /// <param name="bitsPerItem">The number of bits required to store the values.</param>
        /// <param name="nonZeroValues">An ordered enumerable of (index,value) pairs.
        /// Each index should be strictly increasing as the iterable proceeds.</param>
        public DeltaSparseIntArray(int length, IntArrayBits bitsPerItem, IEnumerable<KeyValuePair<int, int>> nonZeroValues)
        {
            using (Timer.Time(TimerEvent.SparseConstruction))
            {
                List<int> tempValueList = new List<int>();
                List<byte> tempDeltaList = new List<byte>();

                int currentIndex = 0;
                foreach (KeyValuePair<int, int> pair in nonZeroValues)
                {
                    int index = pair.Key;
                    int value = pair.Value;
                    if (index <= currentIndex && (index < 0 || tempValueList.Count > 0))
                        throw Contracts.Except("index {0} occurred after {1}", index, currentIndex);
                    while (index - currentIndex > byte.MaxValue)
                    {
                        tempDeltaList.Add(byte.MaxValue);
                        tempValueList.Add(0);
                        currentIndex += byte.MaxValue;
                    }
                    tempDeltaList.Add((byte)(index - currentIndex));
                    tempValueList.Add(value);
                    currentIndex = index;
                }
                // Add the final chunks of 0's if it ended early
                while (length - currentIndex > byte.MaxValue)
                {
                    tempDeltaList.Add(byte.MaxValue);
                    tempValueList.Add(0);
                    currentIndex += byte.MaxValue;
                }
                if (currentIndex >= length && currentIndex > 0)
                    throw Contracts.Except("Index {0} inconsistent with length {1}", currentIndex, length);
                _length = length;

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

        public struct IndexPair
        {
            public int Index;
            public int Value;
        }

        public IEnumerable<IndexPair> GetIndexPairs()
        {
            IndexPair pair = new IndexPair();
            int currIndex = 0;
            int sparseIndex = 0;
            foreach (int val in _values)
            {
                currIndex += _deltas[sparseIndex++];
                if (val != 0)
                {
                    pair.Index = currIndex;
                    pair.Value = val;
                    yield return pair;
                }
            }
        }

        public int DeltaLength => _deltas.Length;

        #region IIntArray Members

        public override int Length => _length;

        public override IntArray[] Split(int[][] assignment)
        {
            IntArray[] parts = new IntArray[assignment.Length];
            for (int i = 0; i < assignment.Length; ++i)
            {
                IIntArrayForwardIndexer indexer = GetIndexer();
                parts[i] = IntArray.New(assignment[i].Length, IntArrayType.Sparse, BitsPerItem, assignment[i].Select(x => indexer[x]));
            }

            return parts;
        }

        /// <summary>
        /// Clone an IntArray containing only the items indexed by <paramref name="itemIndices"/>
        /// </summary>
        /// <param name="itemIndices"> item indices will be contained in the cloned IntArray  </param>
        /// <returns> The cloned IntArray </returns>
        public override IntArray Clone(int[] itemIndices)
        {
            IIntArrayForwardIndexer indexer = GetIndexer();

            return IntArray.New(itemIndices.Length, IntArrayType.Sparse, BitsPerItem, itemIndices.Select(x => indexer[x]));
        }

        /// <summary>
        /// Returns the number of bytes written by the member ToByteArray()
        /// </summary>
        public override int SizeInBytes()
        {
            return _values.SizeInBytes()
                + _deltas.SizeInBytes()
                + sizeof(int) + base.SizeInBytes();
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

        public override IntArrayBits BitsPerItem
        {
            get { return _values.BitsPerItem; }
        }

        public override IntArrayType Type { get { return IntArrayType.Sparse; } }

        public override MD5Hash MD5Hash
        {
            get { return MD5Hasher.Hash(_deltas) ^ _values.MD5Hash; }
        }

        public override IntArray Clone(IntArrayBits bitsPerItem, IntArrayType type)
        {
            if (type == IntArrayType.Sparse || type == IntArrayType.Current)
            {
                if (bitsPerItem <= IntArrayBits.Bits8)
                    bitsPerItem = IntArrayBits.Bits8;
                DenseIntArray newValues = _values.Clone(bitsPerItem, IntArrayType.Dense) as DenseIntArray;
                return new DeltaSparseIntArray(newValues, _deltas, _length);
            }
            else
            {
                DenseIntArray dense = IntArray.New(Length, IntArrayType.Dense, BitsPerItem) as DenseIntArray;
                int index = 0;
                for (int i = 0; i < _values.Length; ++i)
                {
                    index += _deltas[i];
                    dense[index] = _values[i];
                }
                return dense;
            }
        }

        private void SumupRoot(SumupInputData input, FeatureHistogram histogram)
        {
            // Sum up the non-zero values, then subtract from total to get the zero values
            if (histogram.SumWeightsByBin != null)
            {
                SumupRootWeighted(input, histogram);
                return;
            }
            double totalOutput = 0;
            int currentPos = 0;

            for (int i = 0; i < _values.Length; i++)
            {
                currentPos += _deltas[i];

                int featureBin = _values[i];
                FloatType output = (FloatType)input.Outputs[currentPos];

                histogram.SumTargetsByBin[featureBin] = (FloatType)(histogram.SumTargetsByBin[featureBin] + output);
                ++histogram.CountByBin[featureBin];
                totalOutput += output;
            }
            // Fixup the zeros. There were some zero items already placed in the zero-th entry, just add the remainder
            histogram.SumTargetsByBin[0] += (FloatType)(input.SumTargets - totalOutput);
            histogram.CountByBin[0] += input.TotalCount - _values.Length;
        }

        private void SumupRootWeighted(SumupInputData input, FeatureHistogram histogram)
        {
            // Sum up the non-zero values, then subtract from total to get the zero values
            Contracts.Assert(histogram.SumWeightsByBin != null);
            Contracts.Assert(input.Weights != null);
            double totalOutput = 0;
            double totalWeight = 0;
            int currentPos = 0;

            for (int i = 0; i < _values.Length; i++)
            {
                currentPos += _deltas[i];

                int featureBin = _values[i];
                FloatType output = (FloatType)input.Outputs[currentPos];
                FloatType weight = (FloatType)input.Weights[currentPos];

                histogram.SumTargetsByBin[featureBin] = (FloatType)(histogram.SumTargetsByBin[featureBin] + output);
                histogram.SumWeightsByBin[featureBin] = (FloatType)(histogram.SumWeightsByBin[featureBin] + weight);
                ++histogram.CountByBin[featureBin];
                totalOutput += output;
                totalWeight += weight;
            }
            // Fixup the zeros. There were some zero items already placed in the zero-th entry, just add the remainder
            histogram.SumTargetsByBin[0] += (FloatType)(input.SumTargets - totalOutput);
            histogram.SumWeightsByBin[0] += (FloatType)(input.SumWeights - totalWeight);
            histogram.CountByBin[0] += input.TotalCount - _values.Length;
        }

        // Fixing the arrays and using unsafe accesses may give a slight speedup, but it is hard to tell.
        // OPTIMIZE: Another two methods would be doing binary search or using a hashtable -- binary search
        //  when there are very few docs in the leaf
        private unsafe void SumupLeaf(SumupInputData input, FeatureHistogram histogram)
        {
            if (histogram.SumWeightsByBin != null)
            {
                SumupLeafWeighted(input, histogram);
                return;
            }
            int iDocIndices = 0;
            int iSparse = 0;
            int totalCount = 0;
            FloatType totalOutput = 0;
            int currentPos = _deltas.Length > 0 ? _deltas[iSparse] : _length;

            fixed (int* pDocIndices = input.DocIndices)
            fixed (byte* pDeltas = _deltas)
            fixed (FloatType* pOutputs = input.Outputs)
            {
                for (; ; )
                {
                    if (currentPos < pDocIndices[iDocIndices])
                    {
                        if (++iSparse >= _deltas.Length)
                            break;
                        currentPos += pDeltas[iSparse];
                    }
                    else if (currentPos > pDocIndices[iDocIndices])
                    {
                        if (++iDocIndices >= input.TotalCount)
                            break;
                    }
                    else
                    {
                        // A nonzero entry matched one of the docs in the leaf, add it to the histogram.
                        int featureBin = _values[iSparse];
                        FloatType output = pOutputs[iDocIndices];
                        histogram.SumTargetsByBin[featureBin] += output;
                        totalOutput += output;
                        ++histogram.CountByBin[featureBin];

                        totalCount++;

                        if (++iSparse >= _deltas.Length)
                            break;

                        // Note that if the delta is 0, we will "stay" on this document, thus
                        // allowing the sumup to work to accumulate multiple bins per document.
                        if (pDeltas[iSparse] > 0)
                        {
                            currentPos += pDeltas[iSparse];
                            if (++iDocIndices >= input.TotalCount)
                                break;
                        }
                    }
                }
            }
            // Fixup the zeros. There were some zero items already placed in the zero-th entry, just add the remainder
            histogram.SumTargetsByBin[0] += (FloatType)(input.SumTargets - totalOutput);
            histogram.CountByBin[0] += input.TotalCount - totalCount;
        }

        private unsafe void SumupLeafWeighted(SumupInputData input, FeatureHistogram histogram)
        {
            Contracts.Assert(histogram.SumWeightsByBin != null);
            Contracts.Assert(input.Weights != null);

            int iDocIndices = 0;
            int iSparse = 0;
            int totalCount = 0;
            FloatType totalOutput = 0;
            double totalWeights = 0;
            int currentPos = _deltas.Length > 0 ? _deltas[iSparse] : _length;

            fixed (int* pDocIndices = input.DocIndices)
            fixed (byte* pDeltas = _deltas)
            fixed (FloatType* pOutputs = input.Outputs)
            fixed (double* pWeights = input.Weights)
            {
                while (true)
                {
                    if (currentPos < pDocIndices[iDocIndices])
                    {
                        if (++iSparse >= _deltas.Length)
                            break;
                        currentPos += pDeltas[iSparse];
                    }
                    else if (currentPos > pDocIndices[iDocIndices])
                    {
                        if (++iDocIndices >= input.TotalCount)
                            break;
                    }
                    else
                    {
                        // a nonzero entry matched one of the docs in the leaf, add it to the histogram
                        int featureBin = _values[iSparse];
                        FloatType output = pOutputs[iDocIndices];
                        histogram.SumTargetsByBin[featureBin] += output;
                        totalOutput += output;
                        double weights = pWeights[iDocIndices];
                        histogram.SumWeightsByBin[featureBin] += weights;
                        totalWeights += weights;
                        ++histogram.CountByBin[featureBin];

                        totalCount++;

                        if (++iSparse >= _deltas.Length)
                            break;

                        if (pDeltas[iSparse] > 0)
                        {
                            currentPos += pDeltas[iSparse];
                            if (++iDocIndices >= input.TotalCount)
                                break;
                        }
                    }
                }
            }
            // Fixup the zeros. There were some zero items already placed in the zero-th entry, just add the remainder
            histogram.SumTargetsByBin[0] += (FloatType)(input.SumTargets - totalOutput);
            histogram.SumWeightsByBin[0] += (FloatType)(input.SumWeights - totalWeights);
            histogram.CountByBin[0] += input.TotalCount - totalCount;
        }

        public override void Sumup(SumupInputData input, FeatureHistogram histogram)
        {
            using (Timer.Time(TimerEvent.SumupSparse))
            {
#if USE_FASTTREENATIVE
                var callbackIntArray = _values as DenseDataCallbackIntArray;
                if (callbackIntArray != null)
                {
                    unsafe
                    {
                        fixed (byte* pDeltas = _deltas)
                        {
                            byte* pDeltas2 = pDeltas;
                            callbackIntArray.Callback(pValues =>
                            {
                                SumupCPlusPlusSparse(input, histogram, (byte*)pValues, pDeltas2, _deltas.Length,
                                    _values.BitsPerItem);
                            });
                        }
                    }
                    return;
                }
#endif
                if (input.DocIndices == null)
                    SumupRoot(input, histogram);
                else
                    SumupLeaf(input, histogram);
            }
        }

#if USE_FASTTREENATIVE
        [DllImport("FastTreeNative", CallingConvention = CallingConvention.StdCall)]
        private unsafe static extern int C_SumupDeltaSparse_float(int numBits, byte* pValues, byte* pDeltas, int numDeltas, int* pIndices, float* pSampleOutputs, double* pSampleOutputWeights,
                                  float* pSumTargetsByBin, double* pSumTargets2ByBin, int* pCountByBin,
                                  int totalCount, double totalSampleOutputs, double totalSampleOutputWeights);

        [DllImport("FastTreeNative", CallingConvention = CallingConvention.StdCall)]
        private unsafe static extern int C_SumupDeltaSparse_double(int numBits, byte* pValues, byte* pDeltas, int numDeltas, int* pIndices, double* pSampleOutputs, double* pSampleOutputWeights,
                                  double* pSumTargetsByBin, double* pSumTargets2ByBin, int* pCountByBin,
                                  int totalCount, double totalSampleOutputs, double totalSampleOutputWeights);

        private unsafe void SumupCPlusPlusSparse(SumupInputData input, FeatureHistogram histogram, byte* pValues, byte* pDeltas, int numDeltas, IntArrayBits bitsPerItem)
        {
            fixed (FloatType* pSumTargetsByBin = histogram.SumTargetsByBin)
            fixed (FloatType* pSampleOutputs = input.Outputs)
            fixed (double* pSumWeightsByBin = histogram.SumWeightsByBin)
            fixed (double* pSampleWeights = input.Weights)
            fixed (int* pIndices = input.DocIndices)
            fixed (int* pCountByBin = histogram.CountByBin)
            {
                int rv =
#if USE_SINGLE_PRECISION
                    C_SumupDeltaSparse_float
#else
                    C_SumupDeltaSparse_double
#endif
                        ((int)BitsPerItem, pValues, pDeltas, numDeltas, pIndices, pSampleOutputs, pSampleWeights,
                            pSumTargetsByBin, pSumWeightsByBin, pCountByBin,
                            input.TotalCount, input.SumTargets, input.SumWeights);
                if (rv < 0)
                    throw Contracts.Except("CSumup sumupdeltasparse {0}", rv);
            }
        }
#endif

        private class DeltaSparseIntArrayIndexer : IIntArrayForwardIndexer
        {
            private readonly DeltaSparseIntArray _array;
            private int _pos;
            private int _nextIndex; // Next non-zero index.

            public DeltaSparseIntArrayIndexer(DeltaSparseIntArray array)
            {
                Contracts.AssertValue(array);

                _array = array;
                if (_array._deltas.Length > 0)
                    _nextIndex = _array._deltas[0];
                else
                    _nextIndex = _array.Length;
            }

            #region IIntArrayForwardIndexer Members

            public unsafe int this[int virtualIndex]
            {
                get
                {
                    if (virtualIndex < _nextIndex)
                        return 0;

                    if (virtualIndex == _nextIndex)
                        return _array._values[_pos];

                    ++_pos;
                    fixed (byte* pDeltas = _array._deltas)
                    {
                        while (_pos < _array._values.Length)
                        {
                            _nextIndex += pDeltas[_pos];
                            if (virtualIndex < _nextIndex)
                                return 0;
                            if (virtualIndex == _nextIndex)
                                return _array._values[_pos];

                            ++_pos;
                        }
                    }

                    _nextIndex = _array._length;
                    return 0;
                }
            }

            #endregion
        }

        public override IIntArrayForwardIndexer GetIndexer()
        {
            return new DeltaSparseIntArrayIndexer(this);
        }

        #endregion

        #region IEnumerable<int> Members

        public override IEnumerator<int> GetEnumerator()
        {
            int curr = -1;

            for (int i = 0; i < _deltas.Length; ++i)
            {
                int next = i == 0 ? _deltas[i] : curr + _deltas[i];
                while (++curr < next)
                    yield return 0;
                yield return _values[i];
            }

            while (++curr < _length)
                yield return 0;
        }

        #endregion

    }
}
