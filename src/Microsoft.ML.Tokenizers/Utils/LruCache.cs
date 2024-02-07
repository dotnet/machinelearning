// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Tokenizers
{
    internal class LruCache<TKey, TValue> where TKey : notnull where TValue : notnull
    {
        /// <summary>
        /// The default LRU cache size.
        /// </summary>
        public const int DefaultCacheSize = 8192; // 4096;

        private readonly object _lockObject = new object();

        private class CacheItem
        {
            public readonly TKey Key;
            public TValue Value;

            public CacheItem(TKey key, TValue value)
            {
                Key = key;
                Value = value;
            }
        }

        private readonly Dictionary<TKey, LinkedListNode<CacheItem>> _cache;
        private readonly LinkedList<CacheItem> _lruList;
        private readonly int _cacheSize;

        /// <summary>
        /// Constructs an <see cref="LruCache{TKey,TValue}" /> object.
        /// </summary>
        /// <param name="cacheSize">
        /// The maximum number of <typeparamref name="TKey" /> to <typeparamref name="TValue" /> mappings
        /// that can be cached. This defaults to <see cref="DefaultCacheSize" />, which is set to
        /// <value>4096</value>.
        /// </param>
        public LruCache(int cacheSize = DefaultCacheSize)
        {
            _cache = new Dictionary<TKey, LinkedListNode<CacheItem>>();
            _lruList = new LinkedList<CacheItem>();
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
        public bool Lookup(TKey key, out TValue value)
        {
            lock (_lockObject)
            {
                if (_cache.TryGetValue(key, out LinkedListNode<CacheItem>? cached))
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

        protected virtual void OnEviction(TValue evictedValue) { }

        private void EvictIfNeeded()
        {
            while (_cache.Count >= _cacheSize)
            {
                LinkedListNode<CacheItem>? nodeToEvict = _lruList.Last;
                _lruList.RemoveLast();
                _cache.Remove(nodeToEvict!.Value.Key);
                OnEviction(nodeToEvict.Value.Value);
            }
        }

        /// <summary>
        /// Adds or replaces a mapping in the cache.
        /// </summary>
        /// <param name="key">The key whose mapped <paramref name="value" /> is to be created or replaced.</param>
        /// <param name="value">The new value to be mapped to the <paramref name="key" />.</param>
        public void Add(TKey key, TValue value) => Replace(key, value, out _);

        public bool Replace(TKey key, TValue value, out TValue oldValue)
        {
            lock (_lockObject)
            {
                return ReplaceInternal(key, value, out oldValue);
            }
        }

        private bool ReplaceInternal(TKey key, TValue value, out TValue oldValue)
        {
            if (_cache.TryGetValue(key, out LinkedListNode<CacheItem>? cached))
            {
                oldValue = cached.Value.Value;
                cached.Value.Value = value;
                _lruList.Remove(cached);
                _lruList.AddFirst(cached);
                return true;
            }
            EvictIfNeeded();
            var node = new LinkedListNode<CacheItem>(new CacheItem(key, value));
            _cache[key] = node;
            _lruList.AddFirst(node);
            oldValue = default!;
            return false;
        }

        /// <summary>
        /// The number of entries currently present in the cache.
        /// </summary>
        public int Count => _cache.Count;

        /// <summary>
        /// Clears the contents of this cache.
        /// </summary>
        public void Clear()
        {
            _cache.Clear();
            _lruList.Clear();
        }
    }
}