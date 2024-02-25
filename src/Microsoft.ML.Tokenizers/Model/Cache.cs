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

        private readonly object _lock = new();

        internal Dictionary<TKey, TValue> Map { get; set; }

        internal int Capacity { get; set; }

        internal void Fresh() => Map = new Dictionary<TKey, TValue>(Capacity);

        internal void Clear()
        {
            lock (_lock)
            {
                Map.Clear();
            }
        }

        internal List<TValue> GetValues(IEnumerable<TKey> keys)
        {
            List<TValue> values = new();
            lock (_lock)
            {
                foreach (TKey key in keys)
                {
                    if (Map.TryGetValue(key, out TValue? value))
                    {
                        values.Add(value);
                    }
                }
            }

            return values;
        }

        internal bool TryGet(TKey key, out TValue value)
        {
            lock (_lock)
            {
                return Map.TryGetValue(key, out value!);
            }
        }

        internal void SetValues(IEnumerable<(TKey, TValue)> entries)
        {
            lock (_lock)
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
        }

        internal void Set(TKey k, TValue v)
        {
            lock (_lock)
            {
                if (Capacity > Map.Count)
                {
                    Map[k] = v;
                }
            }
        }

        internal KeyValuePair<TKey, TValue>[] ToArray()
        {
            lock (_lock)
            {
                return Map.ToArray();
            }
        }

        internal TValue GetOrAdd(TKey key, TValue value)
        {
            lock (_lock)
            {
                if (Map.TryGetValue(key, out TValue? v))
                {
                    return v;
                }

                if (Capacity > Map.Count)
                {
                    Map[key] = value;
                }

                return value;
            }
        }
    }
}
