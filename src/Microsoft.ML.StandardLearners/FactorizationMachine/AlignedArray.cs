// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Internal.CpuMath.Core;
using System;

namespace Microsoft.ML.Runtime.FactorizationMachine
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
    internal sealed class AlignedArray
    {
        // Items includes "head" items filled with NaN, followed by _size entries, followed by "tail"
        // items, also filled with NaN. Note that _size * sizeof(Float) is divisible by _cbAlign.
        // It is illegal to access any slot outsize [_base, _base + _size). This is internal so clients
        // can easily pin it.
        public Float[] Items;

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

        public unsafe int GetBase(long addr)
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

        public void CopyFrom(AlignedArray src)
        {
            Contracts.Assert(src != null);
            Contracts.Assert(src._size == _size);
            Contracts.Assert(src._cbAlign == _cbAlign);
            Array.Copy(src.Items, src._base, Items, _base, _size);
        }
    }
}