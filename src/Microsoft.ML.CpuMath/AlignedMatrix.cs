// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using Microsoft.ML.Internal.CpuMath.Core;

namespace Microsoft.ML.Internal.CpuMath
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    /// <summary>
    /// This implements a logical array of Floats that is automatically aligned for SSE/AVX operations.
    /// This is a thin wrapper around the AlignedArray type implemented in C++. This simply couples
    /// the AlignedArray with a logical size, which does not include padding, while the AlignedArray
    /// size does include padding.
    /// </summary>
    [BestFriend]
    internal sealed class CpuAlignedVector : ICpuVector
    {
        private readonly AlignedArray _items;
        private readonly int _size; // The logical size.

        /// <summary>
        /// The value count.
        /// </summary>
        public int ValueCount { get { return _size; } }

        /// <summary>
        /// The logical size of the vector.
        /// </summary>
        public int VectorSize { get { return _size; } }

        // Round cflt up to a multiple of cfltAlign.
        private static int RoundUp(int cflt, int cfltAlign)
        {
            Contracts.Assert(0 < cflt);
            // cfltAlign should be a power of two.
            Contracts.Assert(0 < cfltAlign && (cfltAlign & (cfltAlign - 1)) == 0);

            // Determine the number of "blobs" of size cfltAlign.
            int cblob = (cflt + cfltAlign - 1) / cfltAlign;
            return cblob * cfltAlign;
        }

        /// <summary>
        /// Allocate an aligned vector with the given alignment (in bytes).
        /// The alignment must be a power of two and at least sizeof(Float).
        /// </summary>
        public CpuAlignedVector(int size, int cbAlign)
        {
            Contracts.Assert(0 < size);
            // cbAlign should be a power of two.
            Contracts.Assert(sizeof(float) <= cbAlign);
            Contracts.Assert((cbAlign & (cbAlign - 1)) == 0);

            int cfltAlign = cbAlign / sizeof(float);
            int cflt = RoundUp(size, cfltAlign);
            _items = new AlignedArray(cflt, cbAlign);
            _size = size;
            AssertValid();
        }

        public void Dispose()
        {
        }

        [Conditional("DEBUG")]
        private void AssertValid()
        {
#if DEBUG
            Contracts.Assert(0 < _size && _size <= _items.Size);

            // The padding, [_size, _items.Size), should contain zeros.
            for (int i = _size; i < _items.Size; i++)
                Contracts.Assert(_items[i] == 0);
#endif
        }

        /// <summary>
        /// The physical AligenedArray items.
        /// </summary>
        public AlignedArray Items { get { return _items; } }

        /// <summary>
        /// The alignment.
        /// </summary>
        public int CbAlign
        {
            get { return _items.CbAlign; }
        }

        /// <summary>
        /// Set and get the value of the vector at the given index.
        /// </summary>
        /// <param name="index">The index</param>
        /// <returns>The value at the given index</returns>
        public float this[int index]
        {
            get
            {
                Contracts.Assert(0 <= index && index < _size);
                return _items[index];
            }
            set
            {
                Contracts.Assert(0 <= index && index < _size);
                _items[index] = value;
            }
        }

        /// <summary>
        /// Get the value of the vector at the given index.
        /// </summary>
        /// <param name="i">The index</param>
        /// <returns>The value at the given index</returns>
        public float GetValue(int i)
        {
            Contracts.Assert(0 <= i && i < _size);
            return _items[i];
        }

        /// <summary>
        /// Assign randomized values to the vector elements via the input function.
        /// </summary>
        /// <param name="rand">The input rand om function that takes no arguments and returns a float value</param>
        public void Randomize(Func<float> rand)
        {
            Contracts.AssertValue(rand);
            for (int i = 0; i < _size; i++)
                _items[i] = rand();
        }

        /// <summary>
        /// Assign zeros to the vector elements.
        /// </summary>
        public void Zero()
        {
            _items.ZeroItems();
        }

        /// <summary>
        /// Copy the values into dst, starting at slot ivDst and advancing ivDst.
        /// </summary>
        /// <param name="dst">The destination array</param>
        /// <param name="ivDst">The starting index in the destination array</param>
        public void CopyTo(float[] dst, ref int ivDst)
        {
            Contracts.AssertValue(dst);
            Contracts.Assert(0 <= ivDst && ivDst <= dst.Length - _size);
            _items.CopyTo(dst, ivDst, _size);
            ivDst += _size;
        }

        /// <summary>
        /// Copy the values from this vector starting at slot ivSrc into dst, starting at slot ivDst.
        /// The number of values that are copied is determined by count.
        /// </summary>
        /// <param name="ivSrc">The staring index in this vector</param>
        /// <param name="dst">The destination array</param>
        /// <param name="ivDst">The starting index in the destination array</param>
        /// <param name="count">The number of elements to be copied</param>
        public void CopyTo(int ivSrc, float[] dst, int ivDst, int count)
        {
            Contracts.AssertValue(dst);
            Contracts.Assert(0 <= count && count <= dst.Length);
            Contracts.Assert(0 <= ivSrc && ivSrc <= _size - count);
            Contracts.Assert(0 <= ivDst && ivDst <= dst.Length - count);
            _items.CopyTo(ivSrc, dst, ivDst, count);
        }

        /// <summary>
        /// Copy the values from src, starting at slot index and advancing index, into this vector.
        /// </summary>
        /// <param name="src">The source array</param>
        /// <param name="index">The starting index in the source array</param>
        public void CopyFrom(float[] src, ref int index)
        {
            Contracts.AssertValue(src);
            Contracts.Assert(0 <= index && index <= src.Length - _size);
            _items.CopyFrom(src.AsSpan(index, _size));
            index += _size;
        }

        /// <summary>
        /// Copy the values from src, starting at slot index and advancing index, into this vector, starting at slot ivDst.
        /// The number of values that are copied is determined by count.
        /// </summary>
        /// <param name="ivDst">The staring index in this vector</param>
        /// <param name="src">The source array</param>
        /// <param name="ivSrc">The starting index in the source array</param>
        /// <param name="count">The number of elements to be copied</param>
        public void CopyFrom(int ivDst, float[] src, int ivSrc, int count)
        {
            Contracts.AssertValue(src);
            Contracts.Assert(0 <= count && count <= src.Length);
            Contracts.Assert(0 <= ivDst && ivDst <= _size - count);
            Contracts.Assert(0 <= ivSrc && ivSrc <= src.Length - count);
            _items.CopyFrom(ivDst, src.AsSpan(ivSrc, _size));
        }

        /// <summary>
        /// Copy the values of src vector into this vector. The src vector must have the same size as this vector.
        /// </summary>
        /// <param name="src">The source vector</param>
        public void CopyFrom(CpuAlignedVector src)
        {
            Contracts.AssertValue(src);
            Contracts.Assert(src._size == _size);
            _items.CopyFrom(src._items);
        }

        /// <summary>
        /// Get the underlying AlignedArray as IEnumerator&lt;Float&gt;.
        /// </summary>
        public IEnumerator<float> GetEnumerator()
        {
            for (int i = 0; i < _size; i++)
                yield return _items[i];
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    /// <summary>
    /// This implements a logical matrix of Floats that is automatically aligned for SSE/AVX operations.
    /// The ctor takes an alignment value, which must be a power of two at least sizeof(Float).
    /// </summary>
    [BestFriend]
    internal abstract class CpuAlignedMatrixBase
    {
        // _items includes "head" items filled with NaN, followed by RunLenPhy * RunCntPhy entries, followed by
        // "tail" items, also filled with NaN. Note that RunLenPhy and RunCntPhy are divisible by the alignment
        // specified in the ctor and are >= RunLen and RunCnt, respectively. It is illegal to access any slot
        // outsize [_base, _base + RunLenPhy * RunCntPhy). The padding should all be zero (and maintained as such).
        // The items are arranged in "runs" of length RunLen. There are RunCnt such runs. Each run ends with
        // (RunLenPhy - RunLen) padding slots. There are an addition (RunCntPhy - RunCnt) padding runs of length
        // RunLenPhy, which are entirely zero. Any native code should be able to assume and should maintain
        // these invariants.
        public AlignedArray Items { get; }

        protected readonly int FloatAlign; // The alignment.

        // Since FloatAlign is a power of two, shifting by Shift = log_2(FloatAlign) is the same as multiplying/dividing by FloatAlign.
        protected readonly int Shift;
        // Since FloatAlign is a power of two, bitwise and with Mask = FloatAlign - 1 will be the same as modding by FloatAlign.
        protected readonly int Mask;

        // Logical length of runs (RunLen) and number of runs (RunCnt).
        public readonly int RunLen;
        public readonly int RunCnt;

        // Physical (padded) length and number of runs.
        public readonly int RunLenPhy;
        public readonly int RunCntPhy;

        /// <summary>
        /// The logical number values in the matrix
        /// </summary>
        public int ValueCount => RunLen * RunCnt;

        /// <summary>
        /// The logical number of rows
        /// </summary>
        public abstract int RowCount { get; }

        /// <summary>
        /// The logical number of columns
        /// </summary>
        public abstract int ColCount { get; }

        /// <summary>
        /// The physical number of rows
        /// </summary>
        public abstract int RowCountPhy { get; }

        /// <summary>
        /// The physical number of columns
        /// </summary>
        public abstract int ColCountPhy { get; }

        // Round cflt up to a multiple of cfltAlign.
        protected static int RoundUp(int cflt, int cfltAlign)
        {
            Contracts.Assert(0 < cflt);
            // cfltAlign should be a power of two.
            Contracts.Assert(0 < cfltAlign && (cfltAlign & (cfltAlign - 1)) == 0);

            // Determine the number of "blobs" of size cfltAlign.
            int cblob = (cflt + cfltAlign - 1) / cfltAlign;
            return cblob * cfltAlign;
        }

        /// <summary>
        /// Allocate an aligned matrix with the given alignment (in bytes).
        /// </summary>
        protected CpuAlignedMatrixBase(int runLen, int runCnt, int cbAlign)
        {
            Contracts.Assert(0 < runLen);
            Contracts.Assert(0 < runCnt);
            // cbAlign should be a power of two.
            Contracts.Assert(sizeof(float) <= cbAlign);
            Contracts.Assert((cbAlign & (cbAlign - 1)) == 0);

            RunLen = runLen;
            RunCnt = runCnt;

            FloatAlign = cbAlign / sizeof(float);
            Shift = GeneralUtils.CbitLowZero((uint)FloatAlign);
            Mask = FloatAlign - 1;

            RunLenPhy = RoundUp(runLen, FloatAlign);
            RunCntPhy = RoundUp(runCnt, FloatAlign);
            Items = new AlignedArray(RunLenPhy * RunCntPhy, cbAlign);

            AssertValid();
        }

        [Conditional("DEBUG")]
        protected void AssertValid()
        {
#if DEBUG
            Contracts.Assert(0 < RunLen && RunLen <= RunLenPhy);
            Contracts.Assert(0 < RunCnt && RunCnt <= RunCntPhy);
            Contracts.Assert(RunLenPhy * RunCntPhy == Items.Size);

            // Assert that the padding at the end of each run contains zeros.
            for (int i = 0; i < RunCnt; i++)
            {
                for (int j = RunLen; j < RunLenPhy; j++)
                    Contracts.Assert(Items[i * RunLenPhy + j] == 0);
            }

            // Assert that the padding runs contain zeros.
            for (int i = RunCnt; i < RunCntPhy; i++)
            {
                for (int j = 0; j < RunLenPhy; j++)
                    Contracts.Assert(Items[i * RunLenPhy + j] == 0);
            }
#endif
        }

        public void Dispose()
        {
        }

        /// <summary>
        /// Assign randomized values to the matrix elements via the input function.
        /// </summary>
        /// <param name="rand">The input rand om function that takes no arguments and returns a float value</param>
        public void Randomize(Func<float> rand)
        {
            Contracts.AssertValue(rand);
            for (int i = 0, k = 0; i < RunCnt; i++)
            {
                Contracts.Assert(k == i * RunLenPhy);
                for (int j = 0; j < RunLen; j++)
                    Items[k + j] = rand();
                k += RunLenPhy;
            }
        }

        /// <summary>
        /// Assign zeros to the matrix elements.
        /// </summary>
        public void Zero()
        {
            Items.ZeroItems();
        }

        /// <summary>
        /// Copy the values of src matrix into this matrix. The src matrix must have the same physical and logical size as this matrix.
        /// </summary>
        /// <param name="src">The source matrix</param>
        public void CopyFrom(CpuAlignedMatrixBase src)
        {
            AssertValid();
            Contracts.AssertValue(src);
            src.AssertValid();
            Contracts.Assert(src.RunLen == RunLen);
            Contracts.Assert(src.RunCnt == RunCnt);
            Contracts.Assert(src.RunLenPhy == RunLenPhy);
            Contracts.Assert(src.RunCntPhy == RunCntPhy);
            Items.CopyFrom(src.Items);
        }
    }

    /// <summary>
    /// This implements a logical row-major matrix of Floats that is automatically aligned for SSE/AVX operations.
    /// The ctor takes an alignment value, which must be a power of two at least sizeof(Float).
    /// </summary>
    [BestFriend]
    internal abstract class CpuAlignedMatrixRowBase : CpuAlignedMatrixBase, ICpuBuffer<float>
    {
        protected CpuAlignedMatrixRowBase(int crow, int ccol, int cbAlign)
            : base(ccol, crow, cbAlign)
        {
        }

        /// <summary>
        /// The logical number of rows
        /// </summary>
        public override int RowCount => RunCnt;

        /// <summary>
        /// The logical number of columns
        /// </summary>
        public override int ColCount { get { return RunLen; } }

        /// <summary>
        /// The physical number of rows
        /// </summary>
        public override int RowCountPhy { get { return RunCntPhy; } }

        /// <summary>
        /// The physical number of columns
        /// </summary>
        public override int ColCountPhy { get { return RunLenPhy; } }

        /// <summary>
        /// Copy the values into dst, starting at slot ivDst and advancing ivDst.
        /// </summary>
        /// <param name="dst">The destination array</param>
        /// <param name="ivDst">The starting index in the destination array</param>
        public void CopyTo(float[] dst, ref int ivDst)
        {
            Contracts.AssertValue(dst);
            Contracts.Assert(0 <= ivDst && ivDst <= dst.Length - ValueCount);

            if (ColCount == ColCountPhy)
            {
                // Can copy all at once.
                Items.CopyTo(0, dst, ivDst, ValueCount);
                ivDst += ValueCount;
            }
            else
            {
                // Copy each row.
                int ivSrc = 0;
                for (int row = 0; row < RowCount; row++)
                {
                    Items.CopyTo(ivSrc, dst, ivDst, ColCount);
                    ivSrc += ColCountPhy;
                    ivDst += ColCount;
                }
            }
        }

        /// <summary>
        /// Copy the values from src, starting at slot ivSrc and advancing ivSrc.
        /// </summary>
        /// <param name="src">The source array</param>
        /// <param name="ivSrc">The starting index in the source array</param>
        public void CopyFrom(float[] src, ref int ivSrc)
        {
            Contracts.AssertValue(src);
            Contracts.Assert(0 <= ivSrc && ivSrc <= src.Length - ValueCount);

            if (ColCount == ColCountPhy)
            {
                Items.CopyFrom(src.AsSpan(ivSrc, ValueCount));
                ivSrc += ValueCount;
            }
            else
            {
                for (int row = 0; row < RowCount; row++)
                {
                    Items.CopyFrom(row * ColCountPhy, src.AsSpan(ivSrc, ColCount));
                    ivSrc += ColCount;
                }
            }
        }

        /// <summary>
        /// Get the underlying AlignedArray as IEnumerator&lt;Float&gt;.
        /// </summary>
        public IEnumerator<float> GetEnumerator()
        {
            for (int row = 0; row < RowCount; row++)
            {
                int ivBase = row * ColCountPhy;
                for (int col = 0; col < ColCount; col++)
                    yield return Items[ivBase + col];
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    /// <summary>
    /// This implements a row-major matrix of Floats that is automatically aligned for SSE/AVX operations.
    /// The ctor takes an alignment value, which must be a power of two at least sizeof(Float).
    /// </summary>
    [BestFriend]
    internal sealed class CpuAlignedMatrixRow : CpuAlignedMatrixRowBase, ICpuFullMatrix
    {
        public CpuAlignedMatrixRow(int crow, int ccol, int cbAlign)
            : base(crow, ccol, cbAlign)
        {
        }

        /// <summary>
        /// The logical number of rows
        /// </summary>
        public override int RowCount { get { return RunCnt; } }

        /// <summary>
        /// The logical number of columns
        /// </summary>
        public override int ColCount { get { return RunLen; } }

        /// <summary>
        /// The physical number of rows
        /// </summary>
        public override int RowCountPhy { get { return RunCntPhy; } }

        /// <summary>
        /// The physical number of columns
        /// </summary>
        public override int ColCountPhy { get { return RunLenPhy; } }

        /// <summary>
        /// Copy the values from this matrix, starting from the row into dst, starting at slot ivDst and advancing ivDst.
        /// </summary>
        /// <param name="row">The starting row in this matrix</param>
        /// <param name="dst">The destination array</param>
        /// <param name="ivDst">The starting index in the destination array</param>
        public void CopyTo(int row, float[] dst, ref int ivDst)
        {
            Contracts.AssertValue(dst);
            Contracts.Assert(0 <= row && row < RowCount);
            Contracts.Assert(0 <= ivDst && ivDst <= dst.Length - ColCount);

            Items.CopyTo(row * ColCountPhy, dst, ivDst, ColCount);
            ivDst += ColCount;
        }

        /// <summary>
        /// Assign zeros to the values at the indices
        /// </summary>
        /// <param name="indices">The indices</param>
        public void ZeroItems(int[] indices)
        {
            Contracts.AssertValue(indices);

            // REVIEW: Ideally, we'd adjust the indices once so we wouldn't need to
            // repeatedly deal with padding adjustments.
            CpuMathUtils.ZeroMatrixItems(Items, ColCount, ColCountPhy, indices);
        }
    }

    /// <summary>
    /// This implements a logical matrix of Floats that is automatically aligned for SSE/AVX operations.
    /// The ctor takes an alignment value, which must be a power of two at least sizeof(Float).
    /// </summary>
    [BestFriend]
    internal sealed class CpuAlignedMatrixCol : CpuAlignedMatrixBase, ICpuFullMatrix
    {
        /// <summary>
        /// Allocate an aligned matrix with the given alignment (in bytes).
        /// </summary>
        public CpuAlignedMatrixCol(int crow, int ccol, int cbAlign)
            : base(crow, ccol, cbAlign)
        {
        }

        /// <summary>
        /// The logical number of rows
        /// </summary>
        public override int RowCount { get { return RunCnt; } }

        /// <summary>
        /// The logical number of columns
        /// </summary>
        public override int ColCount { get { return RunLen; } }

        /// <summary>
        /// The physical number of rows
        /// </summary>
        public override int RowCountPhy { get { return RunCntPhy; } }

        /// <summary>
        /// The physical number of columns
        /// </summary>
        public override int ColCountPhy { get { return RunLenPhy; } }

        /// <summary>
        /// Copy the values into dst, starting at slot ivDst and advancing ivDst.
        /// </summary>
        /// <param name="dst">The destination array</param>
        /// <param name="ivDst">The starting index in the destination array</param>
        public void CopyTo(float[] dst, ref int ivDst)
        {
            Contracts.AssertValue(dst);
            Contracts.Assert(0 <= ivDst && ivDst <= dst.Length - ValueCount);

            for (int row = 0; row < RowCount; row++)
            {
                for (int col = 0; col < ColCount; col++)
                    dst[ivDst++] = Items[row + col * RowCountPhy];
            }
        }

        /// <summary>
        /// Copy the values from this matrix, starting from the row into dst, starting at slot ivDst and advancing ivDst.
        /// </summary>
        /// <param name="row">The starting row in this matrix</param>
        /// <param name="dst">The destination array</param>
        /// <param name="ivDst">The starting index in the destination array</param>
        public void CopyTo(int row, float[] dst, ref int ivDst)
        {
            Contracts.AssertValue(dst);
            Contracts.Assert(0 <= row && row < RowCount);
            Contracts.Assert(0 <= ivDst && ivDst <= dst.Length - ColCount);

            for (int col = 0; col < ColCount; col++)
                dst[ivDst++] = Items[row + col * RowCountPhy];
        }

        /// <summary>
        /// Copy the values from src, starting at slot ivSrc and advancing ivSrc.
        /// </summary>
        /// <param name="src">The source array</param>
        /// <param name="ivSrc">The starting index in the source array</param>
        public void CopyFrom(float[] src, ref int ivSrc)
        {
            Contracts.AssertValue(src);
            Contracts.Assert(0 <= ivSrc && ivSrc <= src.Length - ValueCount);
            for (int row = 0; row < RowCount; row++)
            {
                for (int col = 0; col < ColCount; col++)
                    Items[row + col * RowCountPhy] = src[ivSrc++];
            }
        }

        /// <summary>
        /// Assign zeros to the values at the indices
        /// </summary>
        /// <param name="indices">The indices</param>
        public void ZeroItems(int[] indices)
        {
            Contracts.AssertValue(indices);

            // REVIEW: Ideally, we'd adjust the indices once so we wouldn't need to
            // repeatedly deal with padding adjustments.
            foreach (int iv in indices)
            {
                int row = iv / ColCount;
                int col = iv % ColCount;
                Items[row + col * ColCountPhy] = 0;
            }
        }

        /// <summary>
        /// Get the underlying AlignedArray as IEnumerator&lt;Float&gt;.
        /// </summary>
        public IEnumerator<float> GetEnumerator()
        {
            for (int row = 0; row < RowCount; row++)
            {
                for (int col = 0; col < ColCount; col++)
                    yield return Items[row + col * RowCountPhy];
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}