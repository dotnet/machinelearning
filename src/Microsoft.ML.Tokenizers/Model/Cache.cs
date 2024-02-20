// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Tokenizers
{
    internal sealed class Cache<TValue>
    {
        private readonly int _capacity;
        private readonly Dictionary<StringSpanOrdinalKey, TValue> _map;

        private object SyncObj => _map;

        internal Cache() : this(Bpe.DefaultCacheCapacity) { }

        internal Cache(int capacity)
        {
            _capacity = capacity;
            _map = new Dictionary<StringSpanOrdinalKey, TValue>(capacity);
        }

        internal bool TryGetValue(string key, out TValue value)
        {
            lock (SyncObj)
            {
                return _map.TryGetValue(new StringSpanOrdinalKey(key), out value!);
            }
        }

        internal unsafe bool TryGetValue(ReadOnlySpan<char> key, out TValue value)
        {
            lock (SyncObj)
            {
                fixed (char* ptr = key)
                {
                    return _map.TryGetValue(new StringSpanOrdinalKey(ptr, key.Length), out value!);
                }
            }
        }

        internal void Set(string k, TValue v)
        {
            lock (SyncObj)
            {
                if (_map.Count < _capacity)
                {
                    _map[new StringSpanOrdinalKey(k)] = v;
                }
            }
        }
    }
}
