// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime.Internal.CpuMath
{
    using Float = System.Single;

    /// <summary>
    /// This implements a logical array of Floats that is automatically aligned for SSE/AVX operations.
    /// To pin and force alignment, call the GetPin method, typically wrapped in a using (since it
    /// returns a Pin struct that is IDisposable). From the pin, you can get the IntPtr to pass to
    /// native code.
    /// 
    /// The ctor takes an alignment value, which must be a power of two at least sizeof(Float).
    /// </summary>
    public sealed class AlignedArray
    {
        // Items includes "head" items filled with NaN, followed by _size entries, followed by "tail"
        // items, also filled with NaN. Note that _size * sizeof(Float) is divisible by _cbAlign.
        // It is illegal to access any slot outsize [_base, _base + _size). This is internal so clients
        // can easily pin it.
        internal Float[] Items;

        private readonly int _size; // Must be divisible by (_cbAlign / sizeof(Float)).
        private readonly int _cbAlign; // The alignment in bytes, a power of two, divisible by sizeof(Float).
        private int _base; // Where the values start in Items (changes to ensure alignment).

        private object _lock; // Used to make sure only one thread can re-align the values.

        /// <summary>
        /// Allocate an aligned vector with the given alignment (in bytes).
        /// The alignment must be a power of two and at least sizeof(Float).
        /// </summary>
        public AlignedArray(int size, int cbAlign)
        {
            Contracts.Assert(0 < size);
            // cbAlign should be a power of two.
            Contracts.Assert(sizeof(Float) <= cbAlign);
            Contracts.Assert((cbAlign & (cbAlign - 1)) == 0);
            // cbAlign / sizeof(Float) should divide size.
            Contracts.Assert((size * sizeof(Float)) % cbAlign == 0);

            Items = new Float[size + cbAlign / sizeof(Float)];
            _size = size;
            _cbAlign = cbAlign;
            _lock = new object();
        }

        internal unsafe int GetBase(long addr)
        {
#if DEBUG
            fixed (Float* pv = Items)
                Contracts.Assert((Float*)addr == pv);
#endif

            int cbLow = (int)(addr & (_cbAlign - 1));
            int ibMin = cbLow == 0 ? 0 : _cbAlign - cbLow;
            Contracts.Assert(ibMin % sizeof(Float) == 0);

            int ifltMin = ibMin / sizeof(Float);
            if (ifltMin == _base)
                return _base;

            MoveData(ifltMin);
#if DEBUG
            // Anything outsize [_base, _base + _size) should not be accessed, so
            // set them to NaN, for debug validation.
            for (int i = 0; i < _base; i++)
                Items[i] = Float.NaN;
            for (int i = _base + _size; i < Items.Length; i++)
                Items[i] = Float.NaN;
#endif
            return _base;
        }

        private void MoveData(int newBase)
        {
            lock (_lock)
            {
                // Since the array is already pinned, addr and ifltMin in GetBase() cannot change
                // so all we need is to make sure the array is moved only once.
                if (_base != newBase)
                {
                    Array.Copy(Items, _base, Items, newBase, _size);
                    _base = newBase;
                }
            }
        }

        public int Size { get { return _size; } }

        public int CbAlign { get { return _cbAlign; } }

        public Float this[int index]
        {
            get
            {
                Contracts.Assert(0 <= index && index < _size);
                return Items[index + _base];
            }
            set
            {
                Contracts.Assert(0 <= index && index < _size);
                Items[index + _base] = value;
            }
        }

        public void CopyTo(Float[] dst, int index, int count)
        {
            Contracts.Assert(0 <= count && count <= _size);
            Contracts.Assert(dst != null);
            Contracts.Assert(0 <= index && index <= dst.Length - count);
            Array.Copy(Items, _base, dst, index, count);
        }

        public void CopyTo(int start, Float[] dst, int index, int count)
        {
            Contracts.Assert(0 <= count);
            Contracts.Assert(0 <= start && start <= _size - count);
            Contracts.Assert(dst != null);
            Contracts.Assert(0 <= index && index <= dst.Length - count);
            Array.Copy(Items, start + _base, dst, index, count);
        }

        public void CopyFrom(Float[] src, int index, int count)
        {
            Contracts.Assert(0 <= count && count <= _size);
            Contracts.Assert(src != null);
            Contracts.Assert(0 <= index && index <= src.Length - count);
            Array.Copy(src, index, Items, _base, count);
        }

        public void CopyFrom(int start, Float[] src, int index, int count)
        {
            Contracts.Assert(0 <= count);
            Contracts.Assert(0 <= start && start <= _size - count);
            Contracts.Assert(src != null);
            Contracts.Assert(0 <= index && index <= src.Length - count);
            Array.Copy(src, index, Items, start + _base, count);
        }

        // Copies values from a sparse vector.
        // valuesSrc contains only the non-zero entries. Those are copied into their logical positions in the dense array.
        // rgposSrc contains the logical positions + offset of the non-zero entries in the dense array.
        // rgposSrc runs parallel to the valuesSrc array.
        public void CopyFrom(int[] rgposSrc, Float[] valuesSrc, int posMin, int iposMin, int iposLim, bool zeroItems)
        {
            Contracts.Assert(rgposSrc != null);
            Contracts.Assert(valuesSrc != null);
            Contracts.Assert(rgposSrc.Length <= valuesSrc.Length);
            Contracts.Assert(0 <= iposMin && iposMin <= iposLim && iposLim <= rgposSrc.Length);

            // Zeroing-out and setting the values in one-pass does not seem to give any perf benefit.
            // So explicitly zeroing and then setting the values.
            if (zeroItems)
                ZeroItems();

            for (int ipos = iposMin; ipos < iposLim; ++ipos)
            {
                Contracts.Assert(posMin <= rgposSrc[ipos]);
                int iv = _base + rgposSrc[ipos] - posMin;
                Contracts.Assert(iv < _size + _base);
                Items[iv] = valuesSrc[ipos];
            }
        }

        public void CopyFrom(AlignedArray src)
        {
            Contracts.Assert(src != null);
            Contracts.Assert(src._size == _size);
            Contracts.Assert(src._cbAlign == _cbAlign);
            Array.Copy(src.Items, src._base, Items, _base, _size);
        }

        public void ZeroItems()
        {
            Array.Clear(Items, _base, _size);
        }

        public void ZeroItems(int[] rgposSrc, int posMin, int iposMin, int iposLim)
        {
            Contracts.Assert(rgposSrc != null);
            Contracts.Assert(0 <= iposMin && iposMin <= iposLim && iposLim <= rgposSrc.Length);
            Contracts.Assert(iposLim - iposMin <= _size);

            int ivCur = 0;
            for (int ipos = iposMin; ipos < iposLim; ++ipos)
            {
                int ivNextNonZero = rgposSrc[ipos] - posMin;
                Contracts.Assert(ivCur <= ivNextNonZero && ivNextNonZero < _size);
                while (ivCur < ivNextNonZero)
                    Items[_base + ivCur++] = 0;
                Contracts.Assert(ivCur == ivNextNonZero);
                // Skip the non-zero element at ivNextNonZero.
                ivCur++;
            }

            while (ivCur < _size)
                Items[_base + ivCur++] = 0;
        }

        // REVIEW: This is hackish and slightly dangerous. Perhaps we should wrap this in an
        // IDisposable that "locks" this, prohibiting GetBase from being called, while the buffer
        // is "checked out".
        public void GetRawBuffer(out Float[] items, out int offset)
        {
            items = Items;
            offset = _base;
        }
    }
}