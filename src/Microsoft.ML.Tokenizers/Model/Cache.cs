// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

namespace Microsoft.ML.Tokenizers
{
    internal sealed class Cache<TKey, TValue>
    {
        internal Cache() : this(Bpe.DefaultCacheCapacity) { }

        internal Cache(int capacity)
        {
            Capacity = capacity;
            Map = new Dictionary<TKey, TValue>((int)Capacity);
        }

        private readonly ReaderWriterLockSlim _cacheLock = new ReaderWriterLockSlim();

        internal Dictionary<TKey, TValue> Map { get; set; }

        internal int Capacity { get; set; }

        internal void Fresh() => Map = new Dictionary<TKey, TValue>((int)Capacity);

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
            List<TValue>? values = new();
            _cacheLock.EnterReadLock();
            try
            {
                foreach (TKey key in keys)
                {
                    if (Map.TryGetValue(key, out TValue value))
                    {
                        values.Add(value);
                    }
                }
            }
            finally { _cacheLock.ExitReadLock(); }

            return values;
        }

        internal TValue? Get(TKey key)
        {
            _cacheLock.EnterReadLock();
            try
            {
                if (Map.TryGetValue(key, out TValue value))
                {
                    return value;
                }
            }
            finally { _cacheLock.ExitReadLock(); }

            return default;
        }

        internal void SetValues(IEnumerable<(TKey, TValue)> enteries)
        {
            _cacheLock.EnterWriteLock();
            try
            {
                foreach ((TKey, TValue) entry in enteries)
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
    }
}
