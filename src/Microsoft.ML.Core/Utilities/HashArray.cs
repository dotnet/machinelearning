// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

//#define DUMP_STATS

using System;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    // REVIEW: May want to add an IEnumerable<KeyValuePair<int, TItem>>.

    using Conditional = System.Diagnostics.ConditionalAttribute;

    /// <summary>
    /// A Hash Array that implements both val -> index lookup and index -> val lookup.
    /// Also implements memory efficient sorting.
    /// Note: Supports adding and looking up of items but does not support removal of items.
    /// </summary>
    public sealed class HashArray<TItem>
        // REVIEW: May want to not consider not making TItem have to be IComparable but instead
        // could support user specified sort order.
        where TItem : IEquatable<TItem>, IComparable<TItem>
    {
        private int[] _rgit; // Buckets of prime size.

        // Utilizing this struct gives better cache behavior than using parallel arrays. Note that this is a
        // mutable struct so the fields can be assigned individually.
        private struct Entry : IComparable<Entry>
        {
            public int ItNext;
            public TItem Value;

            // This is needed for sorting. It only uses Value.
            public int CompareTo(Entry other)
            {
                return Value.CompareTo(other.Value);
            }
        }

        // Count of TItems stored.
        private int _ct;

        private Entry[] _entries;

        public int Count { get { return _ct; } }

        public HashArray()
        {
            _rgit = new int[HashHelpers.Primes[0]];
            for (int i = 0; i < _rgit.Length; i++)
                _rgit[i] = -1;

            AssertValid();
        }

        [Conditional("DEBUG")]
        private void AssertValid()
        {
            Contracts.AssertValue(_rgit);
            Contracts.AssertNonEmpty(_rgit);

            Contracts.Assert(0 <= _ct & _ct <= Utils.Size(_entries));

            // The number of buckets should be at least the number of items, unless we're reached the
            // biggest number of buckets allowed.
            Contracts.Assert(Utils.Size(_rgit) >= _ct || Utils.Size(_rgit) == HashHelpers.MaxPrimeArrayLength);
        }

        private int GetIit(int hash)
        {
            return (int)((uint)hash % _rgit.Length);
        }

        public TItem GetItem(int it)
        {
            Contracts.Assert(0 <= it && it < _ct);
            return _entries[it].Value;
        }

        /// <summary>
        /// Find the index of the given val in the hash array.
        /// Returns a bool representing if val is present.
        /// Index outputs the index that the val is at in the array, -1 otherwise.
        /// </summary>
        public bool TryGetIndex(TItem val, out int index)
        {
            AssertValid();
            Contracts.Assert(val != null);

            index = GetIndexCore(val, GetIit(val.GetHashCode()));
            return index >= 0;
        }

        // Return the index of value, -1 if it is not present.
        private int GetIndexCore(TItem val, int iit)
        {
            Contracts.Assert(0 <= iit && iit < _rgit.Length);
            int it = _rgit[iit];
            while (it >= 0)
            {
                Contracts.Assert(it < _ct);
                if (_entries[it].Value.Equals(val))
                    return it;
                // Get the next item in the bucket.
                it = _entries[it].ItNext;
            }
            Contracts.Assert(it == -1);
            return -1;
        }

        /// <summary>
        /// Make sure the given value has an equivalent TItem in the hashArray
        /// and return the index of the value.
        /// </summary>
        public int Add(TItem val)
        {
            int iit = GetIit(val.GetHashCode());
            int index = GetIndexCore(val, iit);
            if (index >= 0)
                return index;

            return AddCore(val, iit);
        }

        /// <summary>
        /// Make sure the given value has an equivalent TItem in the hashArray
        /// and return the index of the value.
        /// </summary>
        public bool TryAdd(TItem val)
        {
            int iit = GetIit(val.GetHashCode());
            int index = GetIndexCore(val, iit);
            if (index >= 0)
                return false;

            AddCore(val, iit);
            return true;
        }

        /// <summary>
        /// Adds the value as a TItem. Does not check whether the TItem is already present.
        /// Returns the index of the added value.
        /// </summary>
        private int AddCore(TItem val, int iit)
        {
            AssertValid();
            Contracts.Assert(val != null);
            Contracts.Assert(0 <= iit && iit < _rgit.Length);

            if (_ct >= Utils.Size(_entries))
            {
                Contracts.Assert(_ct == Utils.Size(_entries));
                Utils.EnsureSize(ref _entries, _ct + 1);
            }
            Contracts.Assert(_ct < _entries.Length);

            _entries[_ct].Value = val;
            _entries[_ct].ItNext = _rgit[iit];
            _rgit[iit] = _ct;

            if (++_ct >= _rgit.Length)
                GrowTable();

            AssertValid();

            // Return the index of the added value.
            return _ct - 1;
        }

        private void GrowTable()
        {
            AssertValid();

            int size = HashHelpers.ExpandPrime(_ct);
            Contracts.Assert(size >= _rgit.Length);
            if (size <= _rgit.Length)
                return;

            // Populate new buckets.
            DumpStats();
            _rgit = new int[size];
            FillTable();
            DumpStats();

            AssertValid();
        }

        public void Sort()
        {
            AssertValid();

            // Sort _rgt, keeping _rghash in parallel.
            Array.Sort(_entries, 0, _ct);

            // Reconstruct _rgit from now sorted _rgt and _rghash.
            FillTable();
            AssertValid();
        }

        /// <summary>
        /// Appropriately fills _rgnext and _rgit based on _rgt and _rghash.
        /// </summary>
        private void FillTable()
        {
            for (int i = 0; i < _rgit.Length; i++)
                _rgit[i] = -1;

            for (int it = 0; it < _ct; it++)
            {
                int iit = GetIit(_entries[it].Value.GetHashCode());
                _entries[it].ItNext = _rgit[iit];
                _rgit[iit] = it;
            }
        }

        [Conditional("DUMP_STATS")]
        private void DumpStats()
        {
            int c = 0;
            for (int i = 0; i < _rgit.Length; i++)
            {
                if (_rgit[i] >= 0)
                    c++;
            }
            Console.WriteLine("Table: {0} out of {1}", c, _rgit.Length);
        }

        /// <summary>
        /// Copies all items to the passed in array. Requires the passed in array to be at least the
        /// same length as Count.
        /// </summary>
        public void CopyTo(TItem[] array)
        {
            Contracts.Check(array != null);
            Contracts.Check(array.Length >= _ct);

            for (int i = 0; i < _ct; i++)
                array[i] = _entries[i].Value;
        }

        private static class HashHelpers
        {
            // Note: This HashHelpers class was adapted from the BCL code base.

            // This is the maximum prime smaller than Array.MaxArrayLength
            public const int MaxPrimeArrayLength = 0x7FEFFFFD;

            // Table of prime numbers to use as hash table sizes.
            // Each subsequent prime ensures that the table will at least double in size upon each growth
            // in order to improve the efficiency of the hash table.
            public static readonly int[] Primes =
            {
                3, 7, 17, 37, 79, 163, 331, 673, 1361, 2729, 5471, 10949, 21911, 43853, 87719, 175447, 350899,
                701819, 1403641, 2807303, 5614657, 11229331, 22458671, 44917381, 89834777, 179669557, 359339171,
                718678369, 1437356741, 2146435069,
            };

            public static int GetPrime(int min)
            {
                Contracts.Assert(0 <= min && min < MaxPrimeArrayLength);

                for (int i = 0; i < Primes.Length; i++)
                {
                    int prime = Primes[i];
                    if (prime >= min)
                        return prime;
                }

                Contracts.Assert(false);
                return min + 1;
            }

            // Returns size of hashtable to grow to.
            public static int ExpandPrime(int oldSize)
            {
                int newSize = 2 * oldSize;

                // Allow the hashtables to grow to maximum possible size (~2G elements) before encoutering capacity overflow.
                // Note that this check works even when _items.Length overflowed thanks to the (uint) cast .
                if ((uint)newSize >= MaxPrimeArrayLength)
                    return MaxPrimeArrayLength;

                return GetPrime(newSize);
            }
        }
    }
}