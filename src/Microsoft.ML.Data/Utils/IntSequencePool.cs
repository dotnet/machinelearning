// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections;
using System.IO;
using System.Linq;
using System.Threading;
using System.Text;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    /// <summary>
    /// A dictionary of uint sequences of variable length. Stores the sequences as
    /// byte sequences encoded with LEB128. Empty sequences (or null) are also valid.
    /// </summary>
    public sealed class SequencePool
    {
        // uint sequences are hashed into _mask+1 buckets. _buckets contains the ID of the first
        // sequence that falls in it (or -1 if it is empty).
        // We store the sequences by using LEB128 encoding, and storing the resulting byte sequence.
        // For each sequence, we store its start index in the byte array and its hash value.
        // This information for sequence with ID id is stored in _start[id] and _hash[id]
        // respectively. In addition, we store the ID of the next sequence in the same bucket
        // in _next[id].

        // Number of buckets minus 1. The number of buckets must be a power of two.
        private int _mask;
        // The i'th entry contains the ID of the first sequence in the i'th bucket.
        private int[] _buckets;

        // The number of uint sequences. The arrays _next, _start and _hash have logical
        // length _idLim, and are indexed by the ID's of the sequences.
        private int _idLim;
        // For 0 <= id < _idLim, the ID of the next uint sequence in the same bucket as sequence id.
        private int[] _next;
        // Indexed by id. Contains the starting index in _bytes of the sequences. The logical length
        // of this array is _idLim+1, with the last item being the total count of bytes.
        private int[] _start;
        // Indexed by id. Contains the hashes of the sequences. The hashing is performed on the uncompressed uint
        // sequences.
        private uint[] _hash;

        // The byte sequences. The logical length of _bytes is _start[_idLim].
        private byte[] _bytes;

        public int Count { get { return _idLim; } }

        public SequencePool()
        {
            _mask = 31;
            _buckets = Utils.CreateArray<int>(_mask + 1, -1);

            _next = new int[10];
            _start = new int[11];
            _hash = new uint[10];
            _bytes = new byte[40];

            AssertValid();
        }

        public SequencePool(BinaryReader reader)
        {
            // *** Binary format ***
            // int: _idLim (the number of sequences)
            // int[]: _start (length is _idLim+1)
            // byte[]: _bytes (length is _start[_idLim])

            _idLim = reader.ReadInt32();
            Contracts.CheckDecode(0 <= _idLim && _idLim < int.MaxValue);
            _start = reader.ReadIntArray(_idLim + 1);
            Contracts.CheckDecode(Utils.Size(_start) > 0 && _start[0] == 0);
            Contracts.CheckDecode(_start[_idLim] >= 0);
            _bytes = reader.ReadByteArray(_start[_idLim]);
            if (_idLim < 10)
                Array.Resize(ref _start, 11);
            if (Utils.Size(_bytes) < 40)
                Array.Resize(ref _bytes, 40);

            // Find the smallest power of 2 that is greater than _idLim.
            int ibit = Utils.IbitHigh((uint)Math.Max(_idLim, 31));
            Contracts.Assert(4 <= ibit && ibit <= 31);
            if (ibit < 31)
                ibit++;
            _mask = (1 << ibit) - 1;

            _buckets = Utils.CreateArray<int>(_mask + 1, -1);

            _hash = new uint[Math.Max(_idLim, 10)];
            _next = new int[Math.Max(_idLim, 10)];

            uint[] sequence = null;
            var cb = _start[_idLim];
            for (int id = 0; id < _idLim; id++)
            {
                Contracts.CheckDecode(_start[id] <= _start[id + 1] && _start[id + 1] <= cb);
                int count = Leb128ToUIntArray(_bytes, _start[id], _start[id + 1], ref sequence);
                _hash[id] = Hashing.HashSequence(sequence, 0, count);
                int i = GetBucketIndex(_hash[id]);
                _next[id] = _buckets[i];
                _buckets[i] = id;
            }

            AssertValid();
        }

        public void Save(BinaryWriter writer)
        {
            AssertValid();

            // *** Binary format ***
            // int: _idLim (the number of sequences)
            // int[]: _start (length is _idLim+1)
            // byte[]: _bytes (length is _start[_idLim])

            writer.Write(_idLim);
#if DEBUG
            for (int id = 0; id < _idLim; id++)
                Contracts.Assert(_start[id] <= _start[id + 1]);
#endif
            writer.WriteIntsNoCount(_start, _idLim + 1);
            writer.WriteBytesNoCount(_bytes, _start[_idLim]);
        }

        [Conditional("DEBUG")]
        private void AssertValid()
        {
            // Number of buckets must be a power of two.
            Contracts.AssertValue(_buckets);
            Contracts.Assert(_buckets.Length == _mask + 1);
            Contracts.Assert(Utils.IsPowerOfTwo(_mask + 1));

            Contracts.Assert(0 <= _idLim && Math.Max(10, _idLim) <= Utils.Size(_start) - 1);
            Contracts.Assert(Math.Max(10, _idLim) <= Utils.Size(_hash));
            Contracts.Assert(Math.Max(10, _idLim) <= Utils.Size(_next));
            Contracts.Assert(_start[0] == 0);
            Contracts.Assert(0 <= _start[_idLim] && Math.Max(40, _start[_idLim]) <= Utils.Size(_bytes));
        }

        private int GetFirstIdInBucket(uint hash)
        {
            return _buckets[(int)hash & _mask];
        }

        private int GetBucketIndex(uint hash)
        {
            return (int)hash & _mask;
        }

        // Returns the ID of the requested sequence, or -1 if it is not found.
        private int GetCore(uint[] sequence, int min, int lim, out uint hash)
        {
            AssertValid();
            Contracts.Assert(0 <= min && min <= lim && lim <= Utils.Size(sequence));

            hash = Hashing.HashSequence(sequence, min, lim);

            for (int idCur = GetFirstIdInBucket(hash); idCur >= 0; idCur = _next[idCur])
            {
                Contracts.Assert(0 <= idCur && idCur < _idLim);
                if (_hash[idCur] != hash)
                    continue;

                var ibCur = _start[idCur];
                var ibLim = _start[idCur + 1];
                for (int i = min; ; i++)
                {
                    Contracts.Assert(ibCur <= ibLim);
                    if (i >= lim)
                    {
                        // Need to make sure that we have reached the end of the sequence in the pool at the
                        // same time that we reached the end of sequence.
                        if (ibCur == ibLim)
                            return idCur;
                        break;
                    }
                    if (ibCur >= ibLim)
                        break;
                    uint decoded;
                    var success = TryDecodeOne(_bytes, ref ibCur, _start[idCur + 1], out decoded);
                    Contracts.Assert(success);
                    if (sequence[i] != decoded)
                        break;
                }
            }
            return -1;
        }

        /// <summary>
        /// Returns true if the sequence was added, or false if it was already in the pool.
        /// </summary>
        /// <param name="sequence">The array containing the sequence to add to the pool.</param>
        /// <param name="min">The location in the array of the first sequence element.</param>
        /// <param name="lim">The exclusive end of the sequence.</param>
        /// <param name="id">To be populated with the id of the added sequence.</param>
        /// <returns>True if the sequence was added, false if the sequence was already present in the pool.</returns>
        public bool TryAdd(uint[] sequence, int min, int lim, out int id)
        {
            Contracts.Check(0 <= min && min <= lim && lim <= Utils.Size(sequence));

            uint hash;
            id = GetCore(sequence, min, lim, out hash);
            if (id >= 0)
                return false;
            id = _idLim;
            AddCore(sequence, min, lim, hash);
            Contracts.Assert(id == _idLim - 1);
            return true;
        }

        /// <summary>
        /// Find the given sequence in the pool. If not found, returns -1.
        /// </summary>
        /// <param name="sequence">An integer sequence</param>
        /// <param name="min">The starting index of the sequence to find in the pool</param>
        /// <param name="lim">The length of the sequence to find in the pool</param>
        /// <returns>The ID of the sequence if it is found, -1 otherwise</returns>
        public int Get(uint[] sequence, int min, int lim)
        {
            Contracts.Check(0 <= min && min <= lim && lim <= Utils.Size(sequence));

            uint hash;
            return GetCore(sequence, min, lim, out hash);
        }

        /// <summary>
        /// Adds the item. Does NOT check for whether the item is already present.
        /// </summary>
        private void AddCore(uint[] sequence, int min, int lim, uint hash)
        {
            Contracts.Assert(0 <= min && min <= lim && lim <= Utils.Size(sequence));
            Contracts.Assert(Hashing.HashSequence(sequence, min, lim) == hash);

            if (_idLim + 1 >= _start.Length)
            {
                Contracts.Check(_start.Length != Utils.ArrayMaxSize, "Cannot allocate memory for the sequence pool");
                Contracts.Assert(_idLim + 1 == _start.Length);
                long newSize = (long)_start.Length + _start.Length / 2;
                int size = (newSize > Utils.ArrayMaxSize) ? Utils.ArrayMaxSize : (int)newSize;
                Array.Resize(ref _start, size);
            }

            Contracts.Assert(_hash.Length >= _next.Length);
            if (_idLim >= _next.Length)
            {
                Contracts.Check(_next.Length != Utils.ArrayMaxSize, "Cannot allocate memory for the sequence pool");
                Contracts.Assert(_idLim == _next.Length);
                long newSize = (long)_next.Length + _next.Length / 2;
                int size = (newSize > Utils.ArrayMaxSize) ? Utils.ArrayMaxSize : (int)newSize;
                Array.Resize(ref _hash, size);
                Array.Resize(ref _next, size);
            }

            var cbMax = checked(5 * (lim - min));
            var ibLim = _start[_idLim];
            if (ibLim > _bytes.Length - cbMax)
            {
                Contracts.Check(_bytes.Length != Utils.ArrayMaxSize, "Cannot allocate memory for the sequence pool");
                long newSize = Math.Max((long)_bytes.Length + _bytes.Length / 2, ibLim + cbMax);
                int size = (newSize > Utils.ArrayMaxSize) ? Utils.ArrayMaxSize : (int)newSize;
                Array.Resize(ref _bytes, size);
            }
            Contracts.Assert(_idLim < _next.Length);
            Contracts.Assert(ibLim <= _bytes.Length - cbMax);

            int i = GetBucketIndex(hash);
            _next[_idLim] = _buckets[i];
            _hash[_idLim] = hash;
            _buckets[i] = _idLim;
            _idLim++;
            _start[_idLim] = _start[_idLim - 1];
            UIntArrayToLeb128(sequence, min, lim, _bytes, ref _start[_idLim]);

            if (_idLim >= _buckets.Length)
                GrowTable();

            AssertValid();
        }

        private void GrowTable()
        {
            AssertValid();

            int size = checked(2 * _buckets.Length);
            _buckets = Utils.CreateArray<int>(size, -1);
            _mask = size - 1;

            for (int id = 0; id < _idLim; id++)
            {
                int i = GetBucketIndex(_hash[id]);
                _next[id] = _buckets[i];
                _buckets[i] = id;
            }

            AssertValid();
        }

        // populates sequence with the integers in sequence number id, and returns the count.
        public int GetById(int id, ref uint[] sequence)
        {
            Contracts.Check(0 <= id && id < _idLim);
            return Leb128ToUIntArray(_bytes, _start[id], _start[id + 1], ref sequence);
        }

        // Asserts that byteSequences is big enough.
        private static void UIntArrayToLeb128(uint[] values, int min, int lim, byte[] bytes, ref int ib)
        {
            Contracts.Assert(bytes.Length >= ib + 5 * (lim - min));
            uint value;
            for (int i = min; i < lim; i++)
            {
                value = values[i];
                // Copied from Utils.WriteLEB128Int
                while (value >= 0x80)
                {
                    bytes[ib++] = (byte)(value | 0x80);
                    value >>= 7;
                }
                bytes[ib++] = (byte)value;
            }
        }

        private static bool TryDecodeOne(byte[] bytes, ref int ib, int ibLim, out uint value)
        {
            value = 0;
            int shift = 0;
            for (; ib < ibLim; ib++)
            {
                uint bCur = bytes[ib];
                if (shift == 4 * 7 && bCur > 0x0F)
                    return false;

                value |= (((bCur & (uint)0x7F)) << shift);
                shift += 7;
                if ((bCur & 0x80) == 0)
                {
                    ib++;
                    return true;
                }
            }
            return false;
        }

        private static int Leb128ToUIntArray(byte[] bytes, int min, int lim, ref uint[] sequence)
        {
            Contracts.Assert(0 <= min && min <= lim && lim <= Utils.Size(bytes));

            int cur = min;
            int count = 0;
            while (cur < lim)
            {
                if (Utils.Size(sequence) <= count)
                {
                    Contracts.Assert(count < lim - min);
                    Array.Resize(ref sequence, lim - min);
                }
                Contracts.CheckDecode(TryDecodeOne(bytes, ref cur, lim, out sequence[count]));
                Contracts.Assert(cur <= lim);
                count++;
            }
            return count;
        }
    }
}
