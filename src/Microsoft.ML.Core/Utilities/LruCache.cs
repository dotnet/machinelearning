// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    /// <summary>
    /// Implements a least recently used cache.
    /// </summary>
    public sealed class LruCache<TKey, TValue>
    {
        private readonly int _size;
        private readonly Dictionary<TKey, LinkedListNode<KeyValuePair<TKey, TValue>>> _cache;
        private readonly LinkedList<KeyValuePair<TKey, TValue>> _lru;

        /// <summary>
        /// Returns the keys of the items stored in the cache, sorted from the most recently used
        /// to the least recently used.
        /// </summary>
        public IEnumerable<TKey> Keys => _lru.Select(kvp => kvp.Key);

        /// <summary>
        /// Initializes a new LRU cache with a given size.
        /// The class is not thread safe.
        /// </summary>
        public LruCache(int size)
        {
            Contracts.CheckParam(size > 0, nameof(size), "Must be positive");
            _size = size;
            _cache = new Dictionary<TKey, LinkedListNode<KeyValuePair<TKey, TValue>>>(_size);
            _lru = new LinkedList<KeyValuePair<TKey, TValue>>();
        }

        /// <summary>
        /// Looks up and returns an item. If the item is found, mark it as recently used.
        /// </summary>
        public bool TryGetValue(TKey key, out TValue value)
        {
            LinkedListNode<KeyValuePair<TKey, TValue>> node;
            if (_cache.TryGetValue(key, out node))
            {
                Contracts.Assert(key.GetHashCode() == node.Value.Key.GetHashCode());
                _lru.Remove(node);
                _lru.AddFirst(node);
                value = node.Value.Value;
                return true;
            }

            value = default(TValue);
            return false;
        }

        /// <summary>
        /// Adds a new item in the cache. It will be marked as recently used. If the cache
        /// would grow over the max size, the least recently used item is removed.
        /// </summary>
        public void Add(TKey key, TValue value)
        {
            Contracts.Assert(!_cache.ContainsKey(key));
            var node = _lru.AddFirst(new KeyValuePair<TKey, TValue>(key, value));
            _cache.Add(key, node);
            if (_cache.Count > _size)
            {
                node = _lru.Last;
                _lru.RemoveLast();
                _cache.Remove(node.Value.Key);
            }
        }
    }
}
