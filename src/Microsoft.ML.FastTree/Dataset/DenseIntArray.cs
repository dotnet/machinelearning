// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
#if USE_SINGLE_PRECISION
    using FloatType = System.Single;
#else
    using FloatType = System.Double;
#endif

    /// <summary>
    /// Abstract class implementing some common functions of the dense int array types.
    /// </summary>
    internal abstract class DenseIntArray : IntArray, IIntArrayForwardIndexer
    {
        public override IntArrayType Type { get { return IntArrayType.Dense; } }

        protected DenseIntArray(int length)
        {
            Contracts.Assert(length >= 0);
            Length = length;
        }

        public override int Length { get; }

        /// <summary>
        /// Gets or sets the value at this index.
        /// Value must be in legal range 0...((2^<see cref="IntArray.BitsPerItem"/>)-1).
        /// </summary>
        /// <param name="index">Index of value to get or set</param>
        /// <returns>The value at this index</returns>
        public abstract int this[int index] { get; set; }

        public override IntArray Clone(IntArrayBits bitsPerItem, IntArrayType type)
        {
            if (type == IntArrayType.Current)
                type = IntArrayType.Dense;
            return New(Length, type, bitsPerItem, this);
        }

        /// <summary>
        /// Clone an IntArray containing only the items indexed by <paramref name="itemIndices"/>
        /// </summary>
        /// <param name="itemIndices"> item indices will be contained in the cloned IntArray  </param>
        /// <returns> The cloned IntArray </returns>
        public override IntArray Clone(int[] itemIndices)
        {
            return IntArray.New(itemIndices.Length, IntArrayType.Dense, BitsPerItem, itemIndices.Select(x => this[x]));
        }

        public override IntArray[] Split(int[][] assignment)
        {
            int numParts = assignment.Length;
            IntArray[] newArrays = new IntArray[numParts];

            for (int p = 0; p < numParts; ++p)
            {
                newArrays[p] = IntArray.New(assignment[p].Length, IntArrayType.Dense, BitsPerItem, assignment[p].Select(x => this[x]));
            }

            return newArrays;
        }

#if USE_FASTTREENATIVE
        [DllImport("FastTreeNative", CallingConvention = CallingConvention.StdCall)]
        private static extern unsafe int C_Sumup_float(
            int numBits, byte* pData, int* pIndices, float* pSampleOutputs, double* pSampleOutputWeights,
            FloatType* pSumTargetsByBin, double* pSumTargets2ByBin, int* pCountByBin,
            int totalCount, double totalSampleOutputs, double totalSampleOutputWeights);

        [DllImport("FastTreeNative", CallingConvention = CallingConvention.StdCall)]
        private static extern unsafe int C_Sumup_double(
            int numBits, byte* pData, int* pIndices, double* pSampleOutputs, double* pSampleOutputWeights,
            FloatType* pSumTargetsByBin, double* pSumTargets2ByBin, int* pCountByBin,
            int totalCount, double totalSampleOutputs, double totalSampleOutputWeights);

        protected static unsafe void SumupCPlusPlusDense(SumupInputData input, FeatureHistogram histogram,
            byte* data, int numBits)
        {
            using (Timer.Time(TimerEvent.SumupCppDense))
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
                        C_Sumup_float
#else
                        C_Sumup_double
#endif
                        (numBits, data, pIndices, pSampleOutputs, pSampleWeights,
                         pSumTargetsByBin, pSumWeightsByBin, pCountByBin,
                         input.TotalCount, input.SumTargets, input.SumWeights);
                    if (rv < 0)
                        throw Contracts.Except("CSumup returned error {0}", rv);
                }
            }
        }

#endif

        public override IIntArrayForwardIndexer GetIndexer()
        {
            return this;
        }

        #region IEnumerable<int> Members

        public override IEnumerator<int> GetEnumerator()
        {
            for (int i = 0; i < Length; ++i)
                yield return this[i];
        }

        #endregion
    }

    internal abstract class DenseDataCallbackIntArray : DenseIntArray
    {
        protected DenseDataCallbackIntArray(int length)
            : base(length)
        {
        }

        public abstract void Callback(Action<IntPtr> callback);
    }

    /// <summary>
    /// A "null" feature representing only zeros.
    /// </summary>
    internal sealed class Dense0BitIntArray : DenseIntArray
    {
        public override IntArrayBits BitsPerItem { get { return IntArrayBits.Bits0; } }

        public Dense0BitIntArray(int length)
            : base(length)
        {
        }

        public Dense0BitIntArray(byte[] buffer, ref int position)
            : base(buffer.ToInt(ref position))
        {
        }

        public override MD5Hash MD5Hash {
            get { return MD5Hasher.Hash(Length); }
        }

        /// <summary>
        /// Returns the number of bytes written by the member ToByteArray()
        /// </summary>
        public override int SizeInBytes()
        {
            return sizeof(int) + base.SizeInBytes();
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
            Length.ToByteArray(buffer, ref position);
        }

        public override int this[int index] {
            get {
                Contracts.Assert(0 <= index && index < Length);
                return 0;
            }

            set {
                Contracts.Assert(0 <= index && index < Length);
                Contracts.Assert(value == 0);
            }
        }

        public override void Sumup(SumupInputData input, FeatureHistogram histogram)
        {
            histogram.SumTargetsByBin[0] = input.SumTargets;
            if (histogram.SumWeightsByBin != null)
                histogram.SumWeightsByBin[0] = input.SumWeights;
            histogram.CountByBin[0] = input.TotalCount;
        }
    }

    /// <summary>
    /// A class to represent features using 10 bits.
    /// </summary>
    internal sealed class Dense10BitIntArray : DenseIntArray
    {
        private const int _bits = 10;
        private const int _mask = (1 << _bits) - 1;
        private uint[] _data;

        public override IntArrayBits BitsPerItem { get { return IntArrayBits.Bits10; } }

        public Dense10BitIntArray(int len)
            : base(len)
        {
            _data = new uint[((((long)len) * _bits) >> 5) + 2];
        }

        public Dense10BitIntArray(byte[] buffer, ref int position)
            : base(buffer.ToInt(ref position))
        {
            _data = buffer.ToUIntArray(ref position);
        }

        public Dense10BitIntArray(int len, IEnumerable<int> values)
            : this(len)
        {
            int i = 0;
            long offset = 0;
            foreach (int val in values)
            {
                if (i++ > len)
                    break;
                Set(offset, _mask, val);
                offset += _bits;
            }
        }

        private int Get(long offset, byte bits)
        {
            return Get(offset, ~((uint)((-1) << bits)));
        }

        private int Get(long offset, uint mask)
        {
            int minor = (int)(offset & 0x1f);
            int major = (int)(offset >> 5);
            return (int)((uint)((_data[major] | (((ulong)_data[major + 1]) << 32)) >> minor) & mask);
        }

        private void Set(long offset, byte bits, int value)
        {
            Set(offset, ~((uint)((-1) << bits)), value);
        }

        private void Set(long offset, uint mask, int value)
        {
            int minor = (int)(offset & 0x1f);
            int major = (int)(offset >> 5);

            uint major0Mask = mask << minor;
            uint major1Mask = (uint)((((ulong)mask) << minor) >> 32);

            ulong val = ((((ulong)value) & mask) << minor);
            _data[major] = (_data[major] & ~major0Mask) | (uint)val;
            _data[major + 1] = (_data[major + 1] & ~major1Mask) | (uint)(val >> 32);
        }

        public override MD5Hash MD5Hash {
            get { return MD5Hasher.Hash(_data); }
        }

        /// <summary>
        /// Returns the number of bytes written by the member ToByteArray()
        /// </summary>
        public override int SizeInBytes()
        {
            return _data.SizeInBytes() + sizeof(int) + base.SizeInBytes();
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
            Length.ToByteArray(buffer, ref position);
            _data.ToByteArray(buffer, ref position);
        }

        public sealed override unsafe int this[int index] {
            get {
                long offset = index;
                offset = (offset << 3) + (offset << 1);
                int minor = (int)(offset & 0x1f);
                int major = (int)(offset >> 5);
                fixed (uint* pData = _data)
                    return (int)(((*(ulong*)(pData + major)) >> minor) & _mask);
            }

            set {
                Contracts.Assert(0 <= value && value < (1 << 10));
                Set(((long)index) * 10, _mask, value);
            }
        }

        private void SumupRoot(FeatureHistogram histogram, FloatType[] outputs, double[] weights)
        {
            int fval;
            long offset = 0;
            for (int i = 0; i < Length; ++i)
            {
                fval = Get(offset, _mask);
                histogram.SumTargetsByBin[fval] += outputs[i];
                if (histogram.SumWeightsByBin != null)
                    histogram.SumWeightsByBin[fval] += weights[i];
                ++histogram.CountByBin[fval];
                offset += _bits;
            }
        }

        public override unsafe void Sumup(SumupInputData input, FeatureHistogram histogram)
        {
            using (Timer.Time(TimerEvent.SumupDense10))
            {
                if (input.DocIndices == null)
                {
                    SumupRoot(histogram, input.Outputs, input.Weights);
                    return;
                }

                int fval = 0;
                fixed (uint* pData = _data)
                fixed (int* pCountByBin = histogram.CountByBin)
                fixed (int* pDocIndicies = input.DocIndices)
                fixed (FloatType* pSumTargetsByBin = histogram.SumTargetsByBin)
                fixed (FloatType* pTargets = input.Outputs)
                {
                    if (histogram.SumWeightsByBin != null)
                    {
                        fixed (double* pSumWeightsByBin = histogram.SumWeightsByBin)
                        fixed (double* pWeights = input.Weights)
                        {
                            for (int ii = 0; ii < input.TotalCount; ++ii)
                            {
                                long offset = pDocIndicies[ii];
                                offset = (offset << 3) + (offset << 1);
                                int minor = (int)(offset & 0x1f);
                                int major = (int)(offset >> 5);
                                fval = (int)(((*(ulong*)(pData + major)) >> minor) & _mask);
                                pSumTargetsByBin[fval] += pTargets[ii];
                                pSumWeightsByBin[fval] += pWeights[ii];
                                ++pCountByBin[fval];
                            }
                        }
                    }
                    else
                    {
                        int end = input.TotalCount;
                        for (int ii = 0; ii < end; ++ii)
                        {
                            long offset = pDocIndicies[ii];
                            offset = (offset << 3) + (offset << 1);
                            int minor = (int)(offset & 0x1f);
                            int major = (int)(offset >> 5);
                            fval = (int)(((*(ulong*)(pData + major)) >> minor) & _mask);
                            pSumTargetsByBin[fval] += pTargets[ii];
                            ++pCountByBin[fval];
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// A class to represent features using 8 bits
    /// </summary>
    /// <remarks>Represents values -1...(2^s-2)
    /// 0-bit array only represents the value -1</remarks>
    internal sealed class Dense8BitIntArray : DenseDataCallbackIntArray
    {
        private byte[] _data;

        public override IntArrayBits BitsPerItem { get { return IntArrayBits.Bits8; } }

        public Dense8BitIntArray(int len)
            : base(len)
        {
            _data = new byte[len];
        }

        public Dense8BitIntArray(byte[] buffer, ref int position)
            : base(buffer.ToInt(ref position))
        {
            _data = buffer.ToByteArray(ref position);
        }

        public Dense8BitIntArray(int len, IEnumerable<int> values)
            : base(len)
        {
            _data = values.Select(i => (byte)i).ToArray(len);
        }

        public override MD5Hash MD5Hash => MD5Hasher.Hash(_data);

        /// <summary>
        /// Returns the number of bytes written by the member ToByteArray()
        /// </summary>
        public override int SizeInBytes()
        {
            return _data.SizeInBytes() + sizeof(int) + base.SizeInBytes();
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
            Length.ToByteArray(buffer, ref position);
            _data.ToByteArray(buffer, ref position);
        }

        public override unsafe void Callback(Action<IntPtr> callback)
        {
            fixed (byte* pData = _data)
            {
                callback((IntPtr)pData);
            }
        }

        public override unsafe int this[int index] {
            get { return _data[index]; }

            set {
                Contracts.Assert(0 <= value && value <= byte.MaxValue);
                _data[index] = (byte)value;
            }
        }

#if USE_FASTTREENATIVE
        public override void Sumup(SumupInputData input, FeatureHistogram histogram)
        {
            unsafe
            {
                fixed (byte* pData = _data)
                {
                    SumupCPlusPlusDense(input, histogram, pData, 8);
                }
            }
        }
#endif
    }

    /// <summary>
    /// A class to represent features using 4 bits.
    /// </summary>
    internal sealed class Dense4BitIntArray : DenseIntArray
    {
        /// <summary>
        /// For a given byte, the high 4 bits is the first value, the low 4 bits is the next value.
        /// </summary>
        private byte[] _data;

        public override IntArrayBits BitsPerItem { get { return IntArrayBits.Bits4; } }

        public override MD5Hash MD5Hash {
            get { return MD5Hasher.Hash(_data); }
        }

        public Dense4BitIntArray(int len)
            : base(len)
        {
            _data = new byte[(len + 1) / 2]; // Even length = half the bytes. Odd length = half the bytes+0.5.
        }

        public Dense4BitIntArray(int len, IEnumerable<int> values)
            : base(len)
        {
            _data = new byte[(len + 1) / 2];

            int currentIndex = 0;
            bool upper = true;
            foreach (int value in values)
            {
                byte b = (byte)value;
                if (upper)
                {
                    _data[currentIndex] = (byte)(b << 4);
                    upper = false;
                }
                else
                {
                    _data[currentIndex] |= (byte)(b & 0x0f);
                    currentIndex++;
                    upper = true;
                }
            }
        }

        public Dense4BitIntArray(byte[] buffer, ref int position)
            : base(buffer.ToInt(ref position))
        {
            _data = buffer.ToByteArray(ref position);
        }

        /// <summary>
        /// Returns the number of bytes written by the member ToByteArray()
        /// </summary>
        public override int SizeInBytes()
        {
            return _data.SizeInBytes() + sizeof(int) + base.SizeInBytes();
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
            Length.ToByteArray(buffer, ref position);
            _data.ToByteArray(buffer, ref position);
        }

        public override unsafe int this[int index] {
            get {
                int dataIndex = index / 2;
                bool highBits = (index % 2 == 0);

                byte v = _data[dataIndex];
                if (highBits)
                    v >>= 4;
                else
                    v &= 0x0f;

                return v;
            }

            set {
                Contracts.Assert(0 <= value && value < (1 << 4));
                byte v;
                v = (byte)value;

                int dataIndex = index / 2;
                bool highBits = (index % 2 == 0);
                if (highBits)
                {
                    _data[dataIndex] &= 0x0f;
                    _data[dataIndex] |= (byte)(v << 4);
                }
                else
                {
                    _data[dataIndex] &= 0xf0;
                    _data[dataIndex] |= v;
                }
            }
        }

#if USE_FASTTREENATIVE
        public override void Sumup(SumupInputData input, FeatureHistogram histogram)
        {
            unsafe
            {
                fixed (byte* pData = _data)
                {
                    SumupCPlusPlusDense(input, histogram, pData, 4);
                }
            }
        }
#endif
    }

    /// <summary>
    /// A class to represent features using 16 bits.
    /// </summary>
    internal sealed class Dense16BitIntArray : DenseDataCallbackIntArray
    {
        private ushort[] _data;

        public override IntArrayBits BitsPerItem { get { return IntArrayBits.Bits16; } }

        public Dense16BitIntArray(int len)
            : base(len)
        {
            _data = new ushort[len];
        }

        public Dense16BitIntArray(int len, IEnumerable<int> values)
            : base(len)
        {
            _data = values.Select(i => (ushort)i).ToArray(len);
        }

        public Dense16BitIntArray(byte[] buffer, ref int position)
            : base(buffer.ToInt(ref position))
        {
            _data = buffer.ToUShortArray(ref position);
        }

        public override MD5Hash MD5Hash {
            get { return MD5Hasher.Hash(_data); }
        }

        public override unsafe void Callback(Action<IntPtr> callback)
        {
            fixed (ushort* pData = _data)
            {
                callback((IntPtr)pData);
            }
        }

        /// <summary>
        /// Returns the number of bytes written by the member ToByteArray()
        /// </summary>
        public override int SizeInBytes()
        {
            return _data.SizeInBytes() + sizeof(int) + base.SizeInBytes();
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
            Length.ToByteArray(buffer, ref position);
            _data.ToByteArray(buffer, ref position);
        }

        public override unsafe int this[int index] {
            get {
                return _data[index];
            }

            set {
                Contracts.Assert(0 <= value && value <= ushort.MaxValue);
                _data[index] = (ushort)value;
            }
        }
#if USE_FASTTREENATIVE
        public override void Sumup(SumupInputData input, FeatureHistogram histogram)
        {
            unsafe
            {
                fixed (ushort* pData = _data)
                {
                    byte* pDataBytes = (byte*)pData;
                    SumupCPlusPlusDense(input, histogram, pDataBytes, 16);
                }
            }
        }
#endif
    }

    /// <summary>
    /// A class to represent features using 32 bits.
    /// </summary>
    internal sealed class Dense32BitIntArray : DenseDataCallbackIntArray
    {
        private int[] _data;

        public override IntArrayBits BitsPerItem { get { return IntArrayBits.Bits32; } }

        public Dense32BitIntArray(int len)
            : base(len)
        {
            _data = new int[len];
        }

        public Dense32BitIntArray(int len, IEnumerable<int> values)
            : base(len)
        {
            _data = values.ToArray(len);
        }

        public Dense32BitIntArray(byte[] buffer, ref int position)
            : base(buffer.ToInt(ref position))
        {
            _data = buffer.ToIntArray(ref position);
        }

        public override unsafe void Callback(Action<IntPtr> callback)
        {
            fixed (int* pData = _data)
            {
                callback((IntPtr)pData);
            }
        }

        public override MD5Hash MD5Hash {
            get { return MD5Hasher.Hash(_data); }
        }

        /// <summary>
        /// Returns the number of bytes written by the member ToByteArray()
        /// </summary>
        public override int SizeInBytes()
        {
            return _data.SizeInBytes() + sizeof(int) + base.SizeInBytes();
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
            Length.ToByteArray(buffer, ref position);
            _data.ToByteArray(buffer, ref position);
        }

        public override int this[int index] {
            get {
                return _data[index];
            }

            set {
                Contracts.Assert(value >= 0);
                _data[index] = value;
            }
        }

#if USE_FASTTREENATIVE
        public override void Sumup(SumupInputData input, FeatureHistogram histogram)
        {
            unsafe
            {
                fixed (int* pData = _data)
                {
                    byte* pDataBytes = (byte*)pData;
                    SumupCPlusPlusDense(input, histogram, pDataBytes, 32);
                }
            }
        }
#endif
    }
}
