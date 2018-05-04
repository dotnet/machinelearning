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

    internal sealed class SegmentIntArray : IntArray
    {
        private readonly byte[] _segType;
        private readonly int[] _segLength;
        private readonly uint[] _data;
        private readonly int _length;
        private readonly IntArrayBits _bpi;

        /// <summary>
        /// The cost of a transition between segments in bits.
        /// </summary>
        public const long TransitionCost = (sizeof(byte) + sizeof(int)) << 3;
        public const ushort U16TransitionCost = (ushort)TransitionCost;

        public override IntArrayBits BitsPerItem
        {
            get { return _bpi; }
        }

        public override MD5Hash MD5Hash
        {
            get { return MD5Hasher.Hash(_data) ^ MD5Hasher.Hash(_segLength) ^ MD5Hasher.Hash(_segType); }
        }

        public override IntArrayType Type
        {
            get { return IntArrayType.Segmented; }
        }

        public SegmentIntArray(int length, IEnumerable<int> values)
        {
            using (Timer.Time(TimerEvent.SparseConstruction))
            {
                uint[] vals = new uint[length];
                uint pos = 0;
                uint max = 0;
                foreach (int v in values)
                {
                    if (pos >= length)
                    {
                        throw Contracts.Except("Length provided to segment vector is inconsistent with value enumeration");
                    }
                    vals[pos++] = (uint)v;
                    if ((uint)v > max)
                        max = (uint)v;
                }
                if (pos != length)
                {
                    throw Contracts.Except("Length provided to segment vector is inconsistent with value enumeration");
                }

                int maxbits = BitsForValue(max);
                int transitions;
                long bits;
                SegmentFindOptimalPath(vals, vals.Length, maxbits, out bits, out transitions);
                var b = FromWorkArray(vals, vals.Length, bits, transitions);
                _segType = b._segType;
                _segLength = b._segLength;
                _data = b._data;
                _length = b._length;
                _bpi = b._bpi;
            }
        }

        public SegmentIntArray(byte[] buffer, ref int position)
        {
            _bpi = (IntArrayBits)(buffer.ToInt(ref position));
            _length = buffer.ToInt(ref position);
            _segType = buffer.ToByteArray(ref position);
            _segLength = buffer.ToIntArray(ref position);
            _data = buffer.ToUIntArray(ref position);
        }

        private SegmentIntArray(byte[] segType, int[] segLen, uint[] data, int len)
        {
            _segType = segType;
            _segLength = segLen;
            _data = data;
            _length = len;
            _bpi = IntArrayBits.Bits32;
        }

        public override void ToByteArray(byte[] buffer, ref int position)
        {
            base.ToByteArray(buffer, ref position);
            ((int)_bpi).ToByteArray(buffer, ref position);
            _length.ToByteArray(buffer, ref position);
            _segType.ToByteArray(buffer, ref position);
            _segLength.ToByteArray(buffer, ref position);
            _data.ToByteArray(buffer, ref position);
        }

        public override int SizeInBytes()
        {
            return base.SizeInBytes() + sizeof(int) + sizeof(int) +
                _segType.SizeInBytes() + _segLength.SizeInBytes() +
                _data.SizeInBytes();
        }

        private int Get(long offset, byte bits)
        {
            return Get(offset, ~((uint)((-1) << bits)));
        }

        private int Get(long offset, uint mask)
        {
            int minor = (int)(offset & 0x1f);
            int major = (int)(offset >> 5);
            return (int)((uint)((((ulong)_data[major] >> minor) | ((((ulong)_data[major + 1]) << 32) >> minor))) & mask);
        }

        public static int BitsForValue(uint val)
        {
            int firstvalid;
            for (firstvalid = 0; val > 0; val >>= 1, firstvalid++)
                ;
            return firstvalid;
        }

        /// <summary>
        /// Finds the bits necessary for the optimal variable bit encoding of this
        /// array. If we are also finding the actual optimal path, it can only work
        ///
        /// This is a considerably less efficienct managed analogue to the
        /// C_SegmentFindOptimalPath and C_SegmentFindOptimalCost functions.
        /// It is used by the class only when not using the unmanaged library.
        /// </summary>
        /// <param name="ivalues">The values for which we should find the optimal cost. If
        /// findPath is active, the most significant 5 bits will be used to store the bitness
        /// with which this path should be chosen.</param>
        /// <param name="bitsForMaxItem">This should be the maximum number of bits necessary
        /// to encode the largest item in that array, or a higher value. Owing to the nature
        /// of the values as 32 bit quantities this value should be in the range [0,32], or
        /// 21 if we are finding the
        /// cannot exceed 31.</param>
        /// <param name="findPath">Whether we should find the best path, by also storing the
        /// optimal path in the most 5 significant digits.</param>
        /// <param name="bits">The number of bits necessary for the optimal encoding.</param>
        /// <param name="transitions">The number of transitions necessary in the
        /// optimal encoding (only if findPath is true).</param>
        /// <param name="max">The maximum element in the ivalues array.</param>
        public static void StatsOfBestEncoding(uint[] ivalues, int bitsForMaxItem, bool findPath, out long bits, out int transitions, out uint max)
        {
            // The cost of the state.
            byte[] state = new byte[bitsForMaxItem + 1];
            byte firstvalid;

            if (bitsForMaxItem > 32 || bitsForMaxItem < 0)
                throw Contracts.Except("Bits for max item must be in range [0,32], {0} is illegal", bitsForMaxItem);
            else if (bitsForMaxItem > 21 && findPath)
                throw Contracts.Except("Cannot use more than 21 bits if also storing the actual optimal path");

            max = 0;
            bits = TransitionCost;
            //IEnumerator<int> ienum = ivalues.GetEnumerator();
            //while (ienum.MoveNext())
            for (int i = 0; i < ivalues.Length; ++i)
            {
                //uint val = (uint)ienum.Current;
                uint val = (uint)ivalues[i];
                if (val > max)
                    max = val;
                uint transmap = 0;
                for (firstvalid = 0; val > 0; val >>= 1, firstvalid++)
                {
                    state[firstvalid] = 0xff;
                }
                byte beststate = 0;
                byte bestcost = 0xff;
                for (byte b = firstvalid; b <= bitsForMaxItem; ++b)
                {
                    if (state[b] <= TransitionCost)
                    {
                        // We should stay.
                        state[b] += b;
                    }
                    else
                    {
                        // We should transition.
                        state[b] = (byte)(TransitionCost + b);
                        transmap |= (uint)(1 << b);
                    }
                    if (bestcost > state[b])
                    {
                        bestcost = state[beststate = b];
                    }
                }
                for (byte b = firstvalid; b <= bitsForMaxItem; ++b)
                {
                    state[b] -= bestcost;
                }
                bits += bestcost;
                if (findPath)
                {
                    ivalues[i] = ((((uint)beststate) << 27) | (((uint)firstvalid) << 22) | transmap | (uint)ivalues[i]);
                }
            }
            if (bitsForMaxItem < 32 && (((uint)1) << bitsForMaxItem) <= max)
            {
                throw Contracts.Except(
                    "Maximum specified bits {0} was not actually sufficient to encode maximum value {1}",
                    bitsForMaxItem, max);
            }
            transitions = 0;
            if (findPath)
            {
                int back = 1;
                int bitness = 0;
                for (int i = ivalues.Length - 1; i >= 0; --i)
                {
                    bitness = back != 0 ? (int)(ivalues[i] >> 27) : bitness;
                    transitions += back;
                    back = (int)((ivalues[i] >> bitness) & 1);
                    ivalues[i] &= (uint)((1 << ((int)(ivalues[i] >> 22))) - 1);
                    ivalues[i] |= (uint)(bitness << 27);
                }
            }
        }

        public override IntArray Clone(IntArrayBits bitsPerItem, IntArrayType type)
        {
            throw Contracts.ExceptNotImpl();
        }

        public override IEnumerator<int> GetEnumerator()
        {
            long boffset = 0;
            for (int s = 0; s < _segType.Length; ++s)
            {
                int segLen = _segLength[s];
                byte segType = _segType[s];
                if (segType == 0)
                {
                    while (segLen-- > 0)
                    {
                        // This tiny optimization makes a *huge* difference
                        // for our often sparse features.
                        yield return 0;
                    }
                }
                else
                {
                    while (segLen-- > 0)
                    {
                        yield return Get(boffset, segType);
                        boffset += segType;
                    }
                }
            }
        }

        public override IIntArrayForwardIndexer GetIndexer()
        {
            return new SegmentIntArrayIndexer(this);
        }

        public override int Length
        {
            get { return _length; }
        }

        public override IntArray[] Split(int[][] assignment)
        {
            return assignment.Select(a =>
            {
                SegmentIntArrayIndexer ind = GetIndexer() as SegmentIntArrayIndexer;
                return new SegmentIntArray(a.Length, a.Select(i => ind[i]));
            }).ToArray();
        }

        /// <summary>
        /// Clone an IntArray containing only the items indexed by <paramref name="itemIndices"/>
        /// </summary>
        /// <param name="itemIndices"> item indices will be contained in the cloned IntArray  </param>
        /// <returns> The cloned IntArray </returns>
        public override IntArray Clone(int[] itemIndices)
        {
            SegmentIntArrayIndexer indexer = GetIndexer() as SegmentIntArrayIndexer;

            return new SegmentIntArray(itemIndices.Length, itemIndices.Select(i => indexer[i]));
        }

        private class SegmentIntArrayIndexer : IIntArrayForwardIndexer
        {
            private readonly SegmentIntArray _array;
            private int _nextIndex; // index where the next segment begins

            private long _currentBit; // the bit offset
            private int _currentIndex; // the index where the current segment begins
            private byte _currentType; // the type of the current segment
            private int _currentSegment;

            public SegmentIntArrayIndexer(SegmentIntArray array)
            {
                _array = array;
                _currentSegment = 0;
                _currentBit = 0;

                if (_array._segType.Length > 0)
                {
                    _currentIndex = 0;
                    _currentType = _array._segType[0];
                    _nextIndex = _array._segLength[0];
                }
                else
                {
                    // Handle the edge case where we have a completely empty array.
                    _currentIndex = _array.Length;
                    _currentType = 0;
                    _nextIndex = _currentIndex;
                }
            }

            #region IIntArrayForwardIndexer Members

            public unsafe int this[int virtualIndex]
            {
                get
                {
                    while (_nextIndex <= virtualIndex)
                    {
                        _currentBit += (_nextIndex - _currentIndex) * _currentType;
                        _currentIndex = _nextIndex;
                        _currentType = _array._segType[++_currentSegment];
                        _nextIndex += _array._segLength[_currentSegment];
                    }
                    long bitoffset = _currentBit + (virtualIndex - _currentIndex) * _currentType;
                    int major = (int)(bitoffset >> 5);
                    return (int)(((long)_array._data[major] | (((long)_array._data[major + 1]) << 32)) >> (int)(bitoffset & 0x1f)) & ((1 << _currentType) - 1);
                }
            }

            #endregion
        }

        public static SegmentIntArray FromWorkArray(uint[] workArray, int len, long bits, int transitions)
        {
            long databits = bits - (long)transitions * (long)TransitionCost;
            byte[] st = new byte[transitions];
            int[] sl = new int[transitions];
            uint[] data = new uint[(databits >> 5) + 2];

            int curroffset = 0;
            int localoffset = 0;
            int lastbits = -1;
            int runlen = 0;
            int segoffset = 0;
            ulong currdata = 0;

            for (int i = 0; i < len; ++i)
            {
                uint val = workArray[i];
                currdata |= (((ulong)(val & 0x07ffffff)) << localoffset);
                int thisbits = (int)(val >> 27);
                localoffset += thisbits;
                if (localoffset >= 32)
                {
                    data[curroffset++] = (uint)currdata;
                    localoffset -= 32;
                    currdata >>= 32;
                }
                if (lastbits != thisbits)
                {
                    st[segoffset++] = (byte)thisbits;
                    if (runlen > 0)
                        sl[segoffset - 2] = runlen;

                    lastbits = thisbits;
                    runlen = 0;
                }
                runlen++;
            }
            if (runlen > 0)
                sl[segoffset - 1] = runlen;
            data[curroffset] = (uint)currdata;
            data[curroffset + 1] = (uint)(currdata >> 32);

            return new SegmentIntArray(st, sl, data, len);
        }

#if USE_FASTTREENATIVE
        public static void SegmentFindOptimalPath(uint[] array, int len, int bitsNeeded, out long bits, out int transitions)
        {
            if (bitsNeeded <= 15)
            {
                SegmentFindOptimalPath15(array, len, out bits, out transitions);
            }
            else if (bitsNeeded <= 21)
            {
                SegmentFindOptimalPath21(array, len, out bits, out transitions);
            }
            else if (bitsNeeded <= 31)
            {
                throw Contracts.ExceptNotImpl("Segment array pathfinder currently does not support more than 21 bits");
            }
            else
            {
                throw Contracts.Except("Segment array cannot represent more than 31 bits");
            }
        }

        public static void SegmentFindOptimalCost(uint[] array, int len, int bitsNeeded, out long bits)
        {
            if (bitsNeeded <= 15)
            {
                SegmentFindOptimalCost15(array, len, out bits);
            }
            else if (bitsNeeded <= 31)
            {
                SegmentFindOptimalCost31(array, len, out bits);
            }
            else
            {
                throw Contracts.Except("Segment array cannot represent more than 31 bits");
            }
        }

        public unsafe static void SegmentFindOptimalPath7(uint[] array, int len, out long bits, out int transitions)
        {
            long b = 0;
            int t = 0;
            fixed (uint* pArray = array)
            {
                bits = 0;
                C_SegmentFindOptimalPath7(pArray, len, &b, &t);
            }
            bits = b;
            transitions = t;
        }

        public unsafe static void SegmentFindOptimalPath15(uint[] array, int len, out long bits, out int transitions)
        {
            long b = 0;
            int t = 0;
            fixed (uint* pArray = array)
            {
                bits = 0;
                C_SegmentFindOptimalPath15(pArray, len, &b, &t);
            }
            bits = b;
            transitions = t;
        }

        public unsafe static void SegmentFindOptimalPath21(uint[] array, int len, out long bits, out int transitions)
        {
            long b = 0;
            int t = 0;
            fixed (uint* pArray = array)
            {
                bits = 0;
                C_SegmentFindOptimalPath21(pArray, len, &b, &t);
            }
            bits = b;
            transitions = t;
        }

        public unsafe static void SegmentFindOptimalCost15(uint[] array, int len, out long bits)
        {
            long b = 0;
            fixed (uint* pArray = array)
            {
                bits = 0;
                C_SegmentFindOptimalCost15(pArray, len, &b);
            }
            bits = b;
        }

        public unsafe static void SegmentFindOptimalCost31(uint[] array, int len, out long bits)
        {
            long b = 0;
            fixed (uint* pArray = array)
            {
                bits = 0;
                C_SegmentFindOptimalCost31(pArray, len, &b);
            }
            bits = b;
        }

#pragma warning disable TLC_GeneralName // Externs follow their own rules.
        [DllImport("FastTreeNative", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        private unsafe static extern void C_SegmentFindOptimalPath21(uint* valv, int valc, long* pBits, int* pTransitions);

        [DllImport("FastTreeNative", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        private unsafe static extern void C_SegmentFindOptimalPath15(uint* valv, int valc, long* pBits, int* pTransitions);

        [DllImport("FastTreeNative", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        private unsafe static extern void C_SegmentFindOptimalPath7(uint* valv, int valc, long* pBits, int* pTransitions);

        [DllImport("FastTreeNative", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        private unsafe static extern void C_SegmentFindOptimalCost15(uint* valv, int valc, long* pBits);

        [DllImport("FastTreeNative", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        private unsafe static extern void C_SegmentFindOptimalCost31(uint* valv, int valc, long* pBits);

        [DllImport("FastTreeNative", CallingConvention = CallingConvention.StdCall)]
        private unsafe static extern int C_SumupSegment_float(
            uint* pData, byte* pSegType, int* pSegLength, int* pIndices,
            float* pSampleOutputs, double* pSampleOutputWeights,
            float* pSumTargetsByBin, double* pSumWeightsByBin,
            int* pCountByBin, int totalCount, double totalSampleOutputs);

        [DllImport("FastTreeNative", CallingConvention = CallingConvention.StdCall)]
        private unsafe static extern int C_SumupSegment_double(
            uint* pData, byte* pSegType, int* pSegLength, int* pIndices,
            double* pSampleOutputs, double* pSampleOutputWeights,
            double* pSumTargetsByBin, double* pSumWeightsByBin,
            int* pCountByBin, int totalCount, double totalSampleOutputs);
#pragma warning restore TLC_GeneralName

        public unsafe void SumupCPlusPlus(SumupInputData input, FeatureHistogram histogram)
        {
            using (Timer.Time(TimerEvent.SumupSegment))
            {
                fixed (FloatType* pSumTargetsByBin = histogram.SumTargetsByBin)
                fixed (FloatType* pSampleOutputs = input.Outputs)
                fixed (double* pSumWeightsByBin = histogram.SumWeightsByBin)
                fixed (double* pSampleOuputWeights = input.Weights)
                fixed (uint* pData = _data)
                fixed (byte* pSegType = _segType)
                fixed (int* pSegLength = _segLength)
                fixed (int* pIndices = input.DocIndices)
                fixed (int* pCountByBin = histogram.CountByBin)
                {
                    int rv =
#if USE_SINGLE_PRECISION
                        C_SumupSegment_float
#else
                        C_SumupSegment_double
#endif
                            (pData, pSegType, pSegLength, pIndices, pSampleOutputs, pSampleOuputWeights,
                             pSumTargetsByBin,
                             pSumWeightsByBin, pCountByBin, input.TotalCount,
                             input.SumTargets);
                    if (rv < 0)
                        throw Contracts.Except("CSumup returned error {0}", rv);
                }
            }
        }
#else // when not USE_FASTTREENATIVE
        public static void SegmentFindOptimalPath(uint[] array, int len, int bitsNeeded, out long bits, out int transitions)
        {
            uint max;
            StatsOfBestEncoding(array, bitsNeeded, true, out bits, out transitions, out max);
        }

        public static void SegmentFindOptimalCost(uint[] array, int len, int bitsNeeded, out long bits)
        {
            int transitions;
            uint max;
            StatsOfBestEncoding(array, bitsNeeded, false, out bits, out transitions, out max);
        }
#endif // USE_FASTTREENATIVE

        public override void Sumup(SumupInputData input, FeatureHistogram histogram)
        {
            using (Timer.Time(TimerEvent.SumupSegment))
            {
                if (_length == 0)
                    return;
#if USE_FASTTREENATIVE
                SumupCPlusPlus(input, histogram);
#else
                base.Sumup(input, histogram);
#endif
            }
        }

    }
}
