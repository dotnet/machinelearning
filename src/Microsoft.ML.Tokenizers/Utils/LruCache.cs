// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Tokenizers
{
    internal sealed class LruCache<TValue>
    {
        /// <summary>
        /// The default LRU cache size.
        /// </summary>
        public const int DefaultCacheSize = 8192;

        private readonly Dictionary<StringSpanOrdinalKey, LinkedListNode<KeyValuePair<string, TValue>>> _cache = new();
        private readonly LinkedList<KeyValuePair<string, TValue>> _lruList = new();
        private readonly int _cacheSize;

        private object SyncObj => _cache;

        /// <summary>
        /// Constructs an <see cref="LruCache{TValue}" /> object.
        /// </summary>
        /// <param name="cacheSize">
        /// The maximum number of mappings that can be cached. This defaults to <see cref="DefaultCacheSize" />, which is set to <value>8192</value>.
        /// </param>
        public LruCache(int cacheSize = DefaultCacheSize)
        {
            if (cacheSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(cacheSize), "Cache size must be a positive number.");
            }

            _cacheSize = cacheSize;
        }

        /// <summary>
        /// Retrieves the value associated with the specified key /> object.
        /// </summary>
        /// <param name="key">The object to be used as a key.</param>
        /// <param name="value">An out parameter that is set to the value of the key if key contains a mapping in the cache.</param>
        /// <returns>
        /// true if the cache contains a mapping for key, false otherwise.
        /// </returns>
        public bool TryGetValue(string key, out TValue value)
        {
            lock (SyncObj)
            {
                if (_cache.TryGetValue(new StringSpanOrdinalKey(key), out LinkedListNode<KeyValuePair<string, TValue>>? cached))
                {
                    _lruList.Remove(cached);
                    _lruList.AddFirst(cached);
                    value = cached.Value.Value;
                    return true;
                }

                value = default!;
                return false;
            }
        }

        /// <summary>
        /// Retrieves the value associated with the specified key /> object.
        /// </summary>
        /// <param name="key">The object to be used as a key.</param>
        /// <param name="value">An out parameter that is set to the value of the key if key contains a mapping in the cache.</param>
        /// <returns>
        /// true if the cache contains a mapping for key, false otherwise.
        /// </returns>
        public unsafe bool TryGetValue(ReadOnlySpan<char> key, out TValue value)
        {
            lock (SyncObj)
            {
                fixed (char* ptr = key)
                {
                    if (_cache.TryGetValue(new StringSpanOrdinalKey(ptr, key.Length), out LinkedListNode<KeyValuePair<string, TValue>>? cached))
                    {
                        _lruList.Remove(cached);
                        _lruList.AddFirst(cached);
                        value = cached.Value.Value;
                        return true;
                    }
                }

                value = default!;
                return false;
            }
        }

        /// <summary>
        /// Adds or replaces a mapping in the cache.
        /// </summary>
        /// <param name="key">The key whose mapped <paramref name="value" /> is to be created or replaced.</param>
        /// <param name="value">The new value to be mapped to the <paramref name="key" />.</param>
        public void Add(string key, TValue value)
        {
            lock (SyncObj)
            {
                if (_cache.TryGetValue(new StringSpanOrdinalKey(key), out LinkedListNode<KeyValuePair<string, TValue>>? cached))
                {
                    cached.Value = new KeyValuePair<string, TValue>(key, value);
                    _lruList.Remove(cached);
                    _lruList.AddFirst(cached);
                    return;
                }

                while (_cache.Count >= _cacheSize)
                {
                    LinkedListNode<KeyValuePair<string, TValue>>? nodeToEvict = _lruList.Last;
                    _lruList.RemoveLast();
                    _cache.Remove(new StringSpanOrdinalKey(nodeToEvict!.Value.Key));
                }

                var node = new LinkedListNode<KeyValuePair<string, TValue>>(new KeyValuePair<string, TValue>(key, value));
                _cache[new StringSpanOrdinalKey(key)] = node;
                _lruList.AddFirst(node);
            }
        }
    }
}
