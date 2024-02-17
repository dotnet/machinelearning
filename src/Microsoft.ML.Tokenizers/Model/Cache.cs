// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace Microsoft.ML.Tokenizers
{
    internal sealed class Cache<TKey, TValue> where TKey : notnull where TValue : notnull
    {
        internal Cache() : this(Bpe.DefaultCacheCapacity) { }

        internal Cache(int capacity)
        {
            Capacity = capacity;
            Map = new Dictionary<TKey, TValue>(Capacity);
        }

        private readonly ReaderWriterLockSlim _cacheLock = new ReaderWriterLockSlim();

        internal Dictionary<TKey, TValue> Map { get; set; }

        internal int Capacity { get; set; }

        internal void Fresh() => Map = new Dictionary<TKey, TValue>(Capacity);

        internal void Clear()
        {
            _cacheLock.EnterWriteLock();
            try
            {
                Map.Clear();
            }
            finally { _cacheLock.ExitWriteLock(); }
        }

        internal List<TValue> GetValues(IEnumerable<TKey> keys)
        {
            List<TValue> values = new();
            _cacheLock.EnterReadLock();
            try
            {
                foreach (TKey key in keys)
                {
                    if (Map.TryGetValue(key, out TValue? value))
                    {
                        values.Add(value);
                    }
                }
            }
            finally { _cacheLock.ExitReadLock(); }

            return values;
        }

        internal bool TryGet(TKey key, out TValue value)
        {
            _cacheLock.EnterReadLock();
            try
            {
                return Map.TryGetValue(key, out value!);
            }
            finally { _cacheLock.ExitReadLock(); }
        }

        internal void SetValues(IEnumerable<(TKey, TValue)> entries)
        {
            _cacheLock.EnterWriteLock();
            try
            {
                foreach ((TKey, TValue) entry in entries)
                {
                    if (Capacity <= Map.Count)
                    {
                        break;
                    }
                    Map[entry.Item1] = entry.Item2;
                }
            }
            finally { _cacheLock.ExitWriteLock(); }
        }

        internal void Set(TKey k, TValue v)
        {
            _cacheLock.EnterWriteLock();
            try
            {
                if (Capacity > Map.Count)
                {
                    Map[k] = v;
                }
            }
            finally { _cacheLock.ExitWriteLock(); }
        }

        internal KeyValuePair<TKey, TValue>[] ToArray()
        {
            _cacheLock.EnterReadLock();
            try
            {
                return Map.ToArray();
            }
            finally { _cacheLock.ExitReadLock(); }
        }

        internal TValue GetOrAdd(TKey key, TValue value)
        {
            _cacheLock.EnterUpgradeableReadLock();
            try
            {
                if (Map.TryGetValue(key, out TValue? v))
                {
                    return v;
                }

                _cacheLock.EnterWriteLock();
                try
                {
                    if (Capacity > Map.Count)
                    {
                        Map[key] = value;
                    }
                }
                finally { _cacheLock.ExitWriteLock(); }

                return value;
            }
            finally { _cacheLock.ExitUpgradeableReadLock(); }
        }
    }
}
