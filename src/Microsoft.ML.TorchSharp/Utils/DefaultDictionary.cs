// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.TorchSharp.Utils
{
    /// <summary>
    /// Dictionary with a default value for unseen keys.
    /// </summary>
    /// <typeparam name="TKey"></typeparam>
    /// <typeparam name="TValue"></typeparam>
    [Serializable]
    public class DefaultDictionary<TKey, TValue> : IDictionary<TKey, TValue>
    {
        private readonly Func<TValue> _init;
        private readonly Dictionary<TKey, TValue> _dictionary;

        public DefaultDictionary(Func<TValue> init)
        {
            _init = init;
            _dictionary = new Dictionary<TKey, TValue>();
        }

        public TValue this[TKey key]
        {
            get
            {
                if (!ContainsKey(key)) Add(key, _init());
                return _dictionary[key];
            }
            set
            {
                _dictionary[key] = value;
            }
        }

        /*** Below are auto-implemented methods ***/

        public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
        {
            return _dictionary.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)_dictionary).GetEnumerator();
        }

        public void Add(KeyValuePair<TKey, TValue> item)
        {
            _dictionary.Add(item.Key, item.Value);
        }

        public void Clear()
        {
            _dictionary.Clear();
        }

        public bool Contains(KeyValuePair<TKey, TValue> item)
        {
            return _dictionary.Contains(item);
        }

        public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
        {
            ((ICollection<KeyValuePair<TKey, TValue>>)_dictionary).CopyTo(array, arrayIndex);
        }

        public bool Remove(KeyValuePair<TKey, TValue> item)
        {
            return _dictionary.Remove(item.Key);
        }

        public int Count => _dictionary.Count;

        public bool IsReadOnly => false;

        public void Add(TKey key, TValue value)
        {
            _dictionary.Add(key, value);
        }

        public bool ContainsKey(TKey key)
        {
            return _dictionary.ContainsKey(key);
        }

        public bool Remove(TKey key)
        {
            return _dictionary.Remove(key);
        }

        public bool TryGetValue(TKey key, out TValue value)
        {
            return _dictionary.TryGetValue(key, out value);
        }

        public ICollection<TKey> Keys => _dictionary.Keys;

        public ICollection<TValue> Values => _dictionary.Values;
    }
}
