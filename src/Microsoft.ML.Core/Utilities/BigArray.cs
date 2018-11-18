// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    /// <summary>
    /// An array-like data structure that supports storing more than
    /// <see cref="Utils.ArrayMaxSize"/> many entries, up to 0x7FEFFFFF00000L.
    /// The entries are indexed by 64-bit integers, and a single entry can be accessed by
    /// the indexer if no modifications to the entries is desired, or the <see cref="ApplyAt"/>
    /// method. Efficient looping can be accomplished by calling the <see cref="ApplyRange"/> method.
    /// This data structure employs the "length and capacity" pattern. The logical length
    /// can be retrieved from the <see cref="Length"/> property, which can possibly be strictly less
    /// than the total capacity.
    /// </summary>
    /// <typeparam name="T">The type of entries.</typeparam>
    [BestFriend]
    internal sealed class BigArray<T> : IEnumerable<T>
    {
        // REVIEW: This class merges and replaces the original private BigArray implementation in CacheDataView.
        // There the block size was 25 bits. Need to understand the performance implication of this 32x change.
        private const int BlockSizeBits = 20;
        private const int BlockSize = 1 << BlockSizeBits;

        // This field works both as the largest valid index in the big array and
        // a bit mask used to determine block index. This is only valid if BlockSize
        // is a power of two.
        private const int BlockSizeMinusOne = BlockSize - 1;
        private const long MaxSize = (long)Utils.ArrayMaxSize << BlockSizeBits;

        // In the first few iterations, we are conservative with our
        // memory allocations, but beyond this many subarrays, we go
        // for full BlockSize allocation.
        private const int FullAllocationBeyond = 4;

        // The 2-D jagged array containing the entries.
        // Its total size is larger than or equal to _length, but
        // less than Length + BlockSize.
        // Each one-dimension subarray has length equal to BlockSize,
        // except for the last one, which has a positive length
        // less than or equal to BlockSize.
        private T[][] _entries;

        // The logical length of the array. May be strictly less than
        // the actual total size of _entries.
        private long _length;

        /// <summary>
        /// Gets the logical length of the big array.
        /// </summary>
        public long Length { get { return _length; } }

        /// <summary>
        /// Gets or sets the entry at <paramref name="index"/>.
        /// </summary>
        /// <remarks>
        /// This indexer is not efficient for looping. If looping access to entries is desired,
        /// use the <see cref="ApplyRange"/> method instead.
        /// Note that unlike a normal array, the value returned from this indexer getter cannot be modified
        /// (for example, by ++ operator or passing into a method as a ref parameter). To modify an entry, use
        /// the <see cref="ApplyAt"/> method instead.
        /// </remarks>
        public T this[long index]
        {
            get
            {
                Contracts.CheckParam(0 <= index && index < _length, nameof(index), "Index out of range.");
                int bI = (int)(index >> BlockSizeBits);
                int idx = (int)(index & BlockSizeMinusOne);
                return _entries[bI][idx];
            }
            set
            {
                Contracts.CheckParam(0 <= index && index < _length, nameof(index), "Index out of range.");
                int bI = (int)(index >> BlockSizeBits);
                int idx = (int)(index & BlockSizeMinusOne);
                _entries[bI][idx] = value;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BigArray{T}"/> class with a specified size.
        /// </summary>
        public BigArray(long size = 0)
        {
            // Verifies the preconditional invariant that BlockSize is a power of two.
            Contracts.Assert(BlockSize > 1 & (BlockSize & (BlockSize - 1)) == 0, "Block size is not a power of two.");

            Contracts.CheckParam(size >= 0, nameof(size), "Must be non-negative.");
            if (size == 0)
            {
                _entries = new T[0][];
                return;
            }

            Contracts.CheckParam(size <= MaxSize, nameof(size), "Size of BigArray is too large.");
            var longBlockCount = ((size - 1) >> BlockSizeBits) + 1;
            Contracts.Assert(longBlockCount <= Utils.ArrayMaxSize);
            int blockCount = (int)longBlockCount;
            int lastBlockSize = (int)(((size - 1) & BlockSizeMinusOne) + 1);
            Contracts.Assert(blockCount > 0);
            Contracts.Assert(0 < lastBlockSize & lastBlockSize <= BlockSize);
            _length = size;
            _entries = new T[blockCount][];
            for (int i = 0; i < blockCount - 1; i++)
                _entries[i] = new T[BlockSize];

            _entries[blockCount - 1] = new T[lastBlockSize];
        }

        public delegate void Visitor(long index, ref T item);

        /// <summary>
        /// Applies a <see cref="Visitor"/> method at a given <paramref name="index"/>.
        /// </summary>
        public void ApplyAt(long index, Visitor manip)
        {
            Contracts.CheckValue(manip, nameof(manip));
            Contracts.CheckParam(0 <= index && index < _length, nameof(index), "Index out of range.");
            int bI = (int)(index >> BlockSizeBits);
            int idx = (int)(index & BlockSizeMinusOne);
            manip(index, ref _entries[bI][idx]);
        }

        /// <summary>
        /// Implements a more efficient way to loop over index range in [min, lim) and apply
        /// the specified method delegate.
        /// </summary>
        public void ApplyRange(long min, long lim, Visitor manip)
        {
            Contracts.CheckValue(manip, nameof(manip));
            Contracts.CheckParam(min >= 0, nameof(min), "Specified minimum index must be non-negative.");
            Contracts.CheckParam(lim <= _length, nameof(lim), "Specified limit index must be no more than length of the array.");
            if (min >= lim)
                return;

            long max = lim - 1;
            int minBlockIndex = (int)(min >> BlockSizeBits);
            int minIndexInBlock = (int)(min & BlockSizeMinusOne);
            int maxBlockIndex = (int)(max >> BlockSizeBits);
            int maxIndexInBlock = (int)(max & BlockSizeMinusOne);
            long index = min;
            for (int bI = minBlockIndex; bI <= maxBlockIndex; bI++)
            {
                int idxMin = bI == minBlockIndex ? minIndexInBlock : 0;
                int idxMax = bI == maxBlockIndex ? maxIndexInBlock : BlockSizeMinusOne;
                var block = _entries[bI];
                for (int idx = idxMin; idx <= idxMax; idx++)
                    manip(index++, ref block[idx]);
            }
            Contracts.Assert(index == lim);
        }

        /// <summary>
        /// Fills the entries with index in [min, lim) with the given value.
        /// </summary>
        public void FillRange(long min, long lim, T value)
        {
            Contracts.CheckParam(min >= 0, nameof(min), "Specified minimum index must be non-negative.");
            Contracts.CheckParam(lim <= _length, nameof(lim), "Specified limit index must be no more than length of the array.");
            if (min >= lim)
                return;

            long max = lim - 1;
            int minBlockIndex = (int)(min >> BlockSizeBits);
            int minIndexInBlock = (int)(min & BlockSizeMinusOne);
            int maxBlockIndex = (int)(max >> BlockSizeBits);
            int maxIndexInBlock = (int)(max & BlockSizeMinusOne);
#if DEBUG
            long index = min;
#endif
            for (int bI = minBlockIndex; bI <= maxBlockIndex; bI++)
            {
                int idxMin = bI == minBlockIndex ? minIndexInBlock : 0;
                int idxMax = bI == maxBlockIndex ? maxIndexInBlock : BlockSizeMinusOne;
                var block = _entries[bI];
                for (int idx = idxMin; idx <= idxMax; idx++)
                {
                    block[idx] = value;
#if DEBUG
                    index++;
#endif
                }
            }
#if DEBUG
            Contracts.Assert(index == lim);
#endif
        }

        /// <summary>
        /// Resizes the array so that its logical length equals <paramref name="newLength"/>. This method
        /// is more efficient than initialize another array and copy the entries because it preserves
        /// existing blocks. The actual capacity of the array may become larger than <paramref name="newLength"/>.
        /// If <paramref name="newLength"/> equals <see cref="Length"/>, then no operation is done.
        /// If <paramref name="newLength"/> is less than <see cref="Length"/>, the array shrinks in size
        /// so that both its length and its capacity equal <paramref name="newLength"/>.
        /// If <paramref name="newLength"/> is larger than <see cref="Length"/>, the array capacity grows
        /// to the smallest integral multiple of <see cref="BlockSize"/> that is larger than <paramref name="newLength"/>,
        /// unless <paramref name="newLength"/> is less than <see cref="BlockSize"/>, in which case the capacity
        /// grows to double its current capacity or <paramref name="newLength"/>, which ever is larger,
        /// but up to <see cref="BlockSize"/>.
        /// </summary>
        public void Resize(long newLength)
        {
            Contracts.CheckParam(newLength >= 0, nameof(newLength), "Specified new size must be non-negative.");
            Contracts.CheckParam(newLength <= MaxSize, nameof(newLength), "Specified new size is too large.");

            if (newLength == _length)
                return;

            if (newLength == 0)
            {
                // Shrink to empty.
                _entries = new T[0][];
                _length = newLength;
                return;
            }

            var longBlockCount = ((newLength - 1) >> BlockSizeBits) + 1;
            Contracts.Assert(0 < longBlockCount & longBlockCount <= Utils.ArrayMaxSize);
            int newBlockCount = (int)longBlockCount;
            int newLastBlockLength = (int)(((newLength - 1) & BlockSizeMinusOne) + 1);
            Contracts.Assert(0 < newLastBlockLength & newLastBlockLength <= BlockSize);

            if (_length == 0)
            {
                _entries = new T[newBlockCount][];
                for (int i = 0; i < newBlockCount - 1; i++)
                    _entries[i] = new T[BlockSize];

                _entries[newBlockCount - 1] = new T[newLastBlockLength];
                _length = newLength;
                return;
            }

            int curBlockCount = _entries.GetLength(0);
            Contracts.Assert(curBlockCount > 0);
            int curLastBlockSize = Utils.Size(_entries[curBlockCount - 1]);
            int curLastBlockLength = (int)(((_length - 1) & BlockSizeMinusOne) + 1);
            Contracts.Assert(0 < curLastBlockLength & curLastBlockLength <= curLastBlockSize & curLastBlockSize <= BlockSize);

            if (newLength < _length)
            {
                // Shrink to a smaller array
                Contracts.Assert(newBlockCount < curBlockCount | (newBlockCount == curBlockCount & newLastBlockLength < curLastBlockLength));
                Array.Resize(ref _entries, newBlockCount);
                Array.Resize(ref _entries[newBlockCount - 1], newLastBlockLength);
            }
            else if (newLength <= ((long)curBlockCount) << BlockSizeBits)
            {
                // Grow to a larger array, but with the same number of blocks.
                // So only need to grow the size of the last block if necessary.
                Contracts.Assert(curBlockCount == newBlockCount);
                if (newLastBlockLength > curLastBlockSize)
                {
                    if (newBlockCount == 1)
                    {
                        Contracts.Assert(_length == curLastBlockLength);
                        Contracts.Assert(newLength == newLastBlockLength);
                        Contracts.Assert(_entries.Length == 1);
                        Array.Resize(ref _entries[0], Math.Min(BlockSize, Math.Max(2 * curLastBlockSize, newLastBlockLength)));
                    }
                    else
                    {
                        // Grow the last block to full size if there are more than one blocks.
                        Array.Resize(ref _entries[newBlockCount - 1], BlockSize);
                    }
                }
            }
            else
            {
                // Need more blocks.
                Contracts.Assert(newBlockCount > curBlockCount);
                Array.Resize(ref _entries, newBlockCount);
                Array.Resize(ref _entries[curBlockCount - 1], BlockSize);
                for (int bI = curBlockCount; bI < newBlockCount; bI++)
                    _entries[bI] = new T[BlockSize];
            }

            _length = newLength;
        }

        /// <summary>
        /// Trims the capacity to logical length.
        /// </summary>
        public void TrimCapacity()
        {
            if (_length == 0)
            {
                Contracts.Assert(Utils.Size(_entries) == 0);
                return;
            }

            int maMax;
            int miLim;
            LongLimToMajorMaxMinorLim(_length, out maMax, out miLim);
            Contracts.Assert(maMax >= 0);
            Contracts.Assert(0 < miLim && miLim <= Utils.Size(_entries[maMax]));
            if (Utils.Size(_entries[maMax]) != miLim)
                Array.Resize(ref _entries[maMax], miLim);
            Array.Resize(ref _entries, maMax + 1);
        }

        /// <summary>
        /// Appends the elements of <paramref name="src"/> to the end.
        /// This method is thread safe related to calls to <see cref="M:CopyTo"/> (assuming those copy operations
        /// are happening over ranges already added), but concurrent calls to
        /// <see cref="M:AddRange"/> should not be attempted. Intended usage is that
        /// one thread will call this method, while multiple threads may access
        /// previously added ranges from <see cref="M:CopyTo"/>, concurrently with
        /// this method or themselves.
        /// </summary>
        public void AddRange(ReadOnlySpan<T> src)
        {
            if (src.IsEmpty)
                return;

            int maMin;
            int miMin;
            int maMax;
            int miLim;
            LongMinToMajorMinorMin(_length, out maMin, out miMin);
            LongLimToMajorMaxMinorLim(_length + src.Length, out maMax, out miLim);

            Contracts.Assert(maMin <= maMax); // Could be violated if length == 0, but we already took care of this.
            Utils.EnsureSize(ref _entries, maMax + 1, BlockSize);
            switch (maMax - maMin)
            {
            case 0:
                // Spans only one subarray, most common case and simplest implementation.
                Contracts.Assert(miLim - miMin == src.Length);
                Utils.EnsureSize(ref _entries[maMax], maMax >= FullAllocationBeyond ? BlockSize : miLim, BlockSize);
                src.CopyTo(_entries[maMax].AsSpan(miMin));
                break;
            case 1:
                // Spans two subarrays.
                Contracts.Assert((BlockSize - miMin) + miLim == src.Length);
                Utils.EnsureSize(ref _entries[maMin], BlockSize, BlockSize);
                int firstSubArrayCapacity = BlockSize - miMin;
                src.Slice(0, firstSubArrayCapacity).CopyTo(_entries[maMin].AsSpan(miMin));
                Contracts.Assert(_entries[maMax] == null);
                Utils.EnsureSize(ref _entries[maMax], maMax >= FullAllocationBeyond ? BlockSize : miLim, BlockSize);
                src.Slice(firstSubArrayCapacity, miLim).CopyTo(_entries[maMax]);
                break;
            default:
                // Spans three or more subarrays. Very rare.
                int miSubMin = miMin;

                // Copy the first segment.
                Utils.EnsureSize(ref _entries[maMin], BlockSize, BlockSize);
                int srcSoFar = BlockSize - miMin;
                src.Slice(0, srcSoFar).CopyTo(_entries[maMin].AsSpan(miMin));
                // Copy the internal segments.
                for (int major = maMin + 1; major < maMax; ++major)
                {
                    Contracts.Assert(_entries[major] == null);
                    _entries[major] = new T[BlockSize];
                    src.Slice(srcSoFar, BlockSize).CopyTo(_entries[major]);
                    srcSoFar += BlockSize;
                    Contracts.Assert(srcSoFar < src.Length);
                }
                // Copy the last segment.
                Contracts.Assert(src.Length - srcSoFar == miLim);
                Contracts.Assert(_entries[maMax] == null);
                Utils.EnsureSize(ref _entries[maMax], maMax >= FullAllocationBeyond ? BlockSize : miLim, BlockSize);
                src.Slice(srcSoFar, miLim).CopyTo(_entries[maMax]);
                break;
            }
            _length += src.Length;
        }

        /// <summary>
        /// Copies the subarray starting from index <paramref name="idx"/> of length
        /// <paramref name="length"/> to the destination array <paramref name="dst"/>.
        /// Concurrent calls to this method is valid even with one single concurrent call
        /// to <see cref="M:AddRange"/>.
        /// </summary>
        public void CopyTo(long idx, T[] dst, int length)
        {
            // Accesses on the internal arrays of this class should be valid even if
            // some other thread is utilizing AddRange, since Utils.EnsureSize(...) will
            // not replace the array until any allocation or copying has already happened.
            Contracts.Assert(0 <= length && length <= Utils.Size(dst));
            Contracts.Assert(idx <= Length && length <= Length - idx);
            if (length == 0)
                return;

            int maMin;
            int miMin;
            int maMax;
            int miLim;
            LongMinToMajorMinorMin(idx, out maMin, out miMin);
            LongLimToMajorMaxMinorLim(idx + length, out maMax, out miLim);

            Contracts.Assert(maMin <= maMax); // Could happen if length == 0, but we already took care of this.
            switch (maMax - maMin)
            {
            case 0:
                // Spans only one subarray, most common case and simplest implementation.
                Contracts.Assert(miLim - miMin == length);
                Contracts.Assert(miLim <= Utils.Size(_entries[maMax]));
                Array.Copy(_entries[maMax], miMin, dst, 0, length);
                break;
            case 1:
                // Spans two subarrays.
                Contracts.Assert((BlockSize - miMin) + miLim == length);
                Contracts.Assert(BlockSize <= Utils.Size(_entries[maMin]));
                Array.Copy(_entries[maMin], miMin, dst, 0, BlockSize - miMin);
                Contracts.Assert(miLim <= Utils.Size(_entries[maMax]));
                Array.Copy(_entries[maMax], 0, dst, BlockSize - miMin, miLim);
                break;
            default:
                // Spans three or more subarrays. Very rare.
                int miSubMin = miMin;

                // Copy the first segment.
                Contracts.Assert(BlockSize <= Utils.Size(_entries[maMin]));
                int dstSoFar = BlockSize - miMin;
                Array.Copy(_entries[maMin], miMin, dst, 0, dstSoFar);
                // Copy the internal segments.
                for (int major = maMin + 1; major < maMax; ++major)
                {
                    Contracts.Assert(BlockSize <= Utils.Size(_entries[major]));
                    Array.Copy(_entries[major], 0, dst, dstSoFar, BlockSize);
                    dstSoFar += BlockSize;
                    Contracts.Assert(dstSoFar < length);
                }
                // Copy the last segment.
                Contracts.Assert(length - dstSoFar == miLim);
                Contracts.Assert(miLim <= Utils.Size(_entries[maMax]));
                Array.Copy(_entries[maMax], 0, dst, dstSoFar, miLim);
                break;
            }
        }

        private static void LongMinToMajorMinorMin(long min, out int major, out int minor)
        {
            Contracts.Assert(min >= 0);
            Contracts.Assert((min >> BlockSizeBits) < int.MaxValue);
            major = (int)(min >> BlockSizeBits);
            minor = (int)(min & BlockSizeMinusOne);
            Contracts.Assert((long)major * BlockSize + minor == min);
        }

        private static void LongLimToMajorMaxMinorLim(long lim, out int major, out int minor)
        {
            Contracts.Assert(lim > 0);
            Contracts.Assert((lim >> BlockSizeBits) < int.MaxValue);
            // Note that lim below this point is the original lim minus 1.
            major = (int)((--lim) >> BlockSizeBits);
            minor = (int)((lim & BlockSizeMinusOne) + 1);
            Contracts.Assert((long)major * BlockSize + minor == lim + 1);
        }

        public IEnumerator<T> GetEnumerator()
        {
            long cur = 0;
            while (cur < _length)
            {
                yield return this[cur];
                cur++;
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
