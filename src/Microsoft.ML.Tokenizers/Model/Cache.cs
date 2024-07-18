// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Tokenizers
{
    internal sealed class Cache<TKey, TValue> where TKey : notnull where TValue : notnull
    {
        private readonly int _capacity;
        private readonly Dictionary<TKey, TValue> _map;
        private object SyncObj => _map;

        internal Cache() : this(BpeTokenizer.DefaultCacheCapacity) { }

        internal Cache(int capacity)
        {
            _capacity = capacity;
            _map = new Dictionary<TKey, TValue>(capacity);
        }

        internal bool TryGetValue(TKey key, out TValue value)
        {
            lock (SyncObj)
            {
                return _map.TryGetValue(key, out value!);
            }
        }

        internal TValue GetOrAdd(TKey key, TValue value)
        {
            lock (SyncObj)
            {
                if (_map.TryGetValue(key, out TValue? v))
                {
                    return v!;
                }

                _map[key] = value;
                return value;
            }
        }

        internal void Set(TKey key, TValue value)
        {
            lock (SyncObj)
            {
                if (_map.Count < _capacity)
                {
                    _map[key] = value;
                }
            }
        }
    }
}
