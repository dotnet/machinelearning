// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    /// <summary>
    /// The tree structure is simultaneously a tree, and a node in a tree. The interface to
    /// get children occurs through the tree itself implementing itself as a dictionary. This
    /// tree is not terribly efficient, and is meant to be accomodate additions, deletions,
    /// and change of values. Because it is implemented as a dictionary, there is an unfortunate
    /// collision in naming between the dictionary type of "values" (which in this case are
    /// child trees) and the tree type of values, called "node values."
    /// </summary>
    /// <typeparam name="TKey">Children are keyed with values of this type</typeparam>
    /// <typeparam name="TValue">The type of the node value</typeparam>
    public sealed class Tree<TKey, TValue> : IDictionary<TKey, Tree<TKey, TValue>>
    {
        // The key of this node in the parent, assuming this is a child node at all.
        // This back reference is necessary to complete any "remove" operations.
        private TKey _key;
        private bool _hasNodeValue;
        private TValue _nodeValue;
        // The dictionary implementation of tree is mostly a wrapping of this member, except
        // for the additional tracking necessary to maintain the parent and child key values.
        private readonly Dictionary<TKey, Tree<TKey, TValue>> _children;
        private Tree<TKey, TValue> _parent;

        public bool HasNodeValue { get { return _hasNodeValue; } }

        /// <summary>
        /// Either the node value, or the default of the value type,
        /// if <see cref="HasNodeValue"/> is false.
        /// </summary>
        public TValue NodeValue
        {
            get { return _nodeValue; }
            set
            {
                _nodeValue = value;
                _hasNodeValue = true;
            }
        }

        public Tree<TKey, TValue> this[TKey key]
        {
            get { return _children[key]; }
            set { Add(key, value); }
        }

        /// <summary>
        /// This is the key for this child node in its parent, if any. If this is not
        /// a child of any parent, that is, it is the root of its own tree, then 
        /// </summary>
        public TKey Key { get { return _key; } }

        /// <summary>
        /// The parent for this tree, or <c>null</c> if it has no parent.
        /// </summary>
        public Tree<TKey, TValue> Parent { get { return _parent; } }

        /// <summary>
        /// All child keys for this node.
        /// </summary>
        public ICollection<TKey> Keys { get { return _children.Keys; } }
        /// <summary>
        /// All children for this node.
        /// </summary>
        public ICollection<Tree<TKey, TValue>> Values { get { return _children.Values; } }

        /// <summary>
        /// The number of children with this node as a parent.
        /// </summary>
        public int Count { get { return _children.Count; } }

        public bool IsReadOnly { get { return false; } }

        /// <summary>
        /// Initializes a tree with no node value, and no children.
        /// </summary>
        public Tree()
        {
            _children = new Dictionary<TKey, Tree<TKey, TValue>>();
        }

        /// <summary>
        /// Tries to get the subtree for a child key.
        /// </summary>
        /// <param name="key">The key of the child to find</param>
        /// <param name="value">The child, if any, or <c>null</c> if no child
        /// with this key is present</param>
        /// <returns>Whether a child with this key was present</returns>
        public bool TryGetValue(TKey key, out Tree<TKey, TValue> value)
        {
            return _children.TryGetValue(key, out value);
        }

        public void ClearNodeValue()
        {
            _nodeValue = default(TValue);
            _hasNodeValue = false;
        }

        /// <summary>
        /// Sees whether a child with a given key is present.
        /// </summary>
        /// <param name="key">The key of the child to find</param>
        /// <returns></returns>
        public bool ContainsKey(TKey key)
        {
            return _children.ContainsKey(key);
        }

        public bool Contains(KeyValuePair<TKey, Tree<TKey, TValue>> item)
        {
            return item.Value != null && item.Value._parent == this && _children.Comparer.Equals(item.Key, item.Value.Key);
        }

        /// <summary>
        /// Adds a new child to this dictionary.
        /// </summary>
        /// <param name="item">The key / child node pair to add</param>
        public void Add(KeyValuePair<TKey, Tree<TKey, TValue>> item)
        {
            Add(item.Key, item.Value);
        }

        /// <summary>
        /// Adds a node as a child of this node. This will disconnect the 
        /// </summary>
        /// <param name="key"></param>
        /// <param name="newChild"></param>
        public void Add(TKey key, Tree<TKey, TValue> newChild)
        {
            Contracts.CheckValue(newChild, nameof(newChild));

            Tree<TKey, TValue> child;
            // Remove the old child, if any.
            if (_children.TryGetValue(key, out child))
                child.Detach();
            Contracts.Assert(!ContainsKey(key));
            // Remove the new child from any structure it may be part of, if any.
            newChild.Detach();
            newChild._key = key;
            newChild._parent = this;
            _children.Add(key, newChild);
        }

        /// <summary>
        /// Removes this node and all its descendants from a tree, leading it to
        /// be the root of its own tree. Following this, <see cref="Parent"/> will
        /// be <c>null</c>, and the previous parent (if any) will no longer have
        /// this node present as a child.
        /// </summary>
        public void Detach()
        {
            if (_parent == null)
                return;
            Contracts.Assert(_parent._children.ContainsKey(Key));
            _parent._children.Remove(Key);
            _parent = null;
            _key = default(TKey);
        }

        /// <summary>
        /// Remove a child with a specified key.
        /// </summary>
        /// <param name="key">The key of the child to remove</param>
        /// <returns></returns>
        public bool Remove(TKey key)
        {
            Tree<TKey, TValue> child;
            if (!_children.TryGetValue(key, out child))
                return false;
            child.Detach();
            return true;
        }

        public bool Remove(KeyValuePair<TKey, Tree<TKey, TValue>> item)
        {
            Tree<TKey, TValue> child;
            // We want to not remove if item.Value is not actually our child keyed on item.Key.
            if (!TryGetValue(item.Key, out child) || item.Value != child)
                return false;
            child.Detach();
            return true;
        }

        public void Clear()
        {
            // We do not rely on the Remove() method, since that will also
            // simultaneously change _children during the enumeration.
            foreach (var child in _children.Values)
            {
                child._parent = null;
                child._key = default(TKey);
            }
            _children.Clear();
        }

        public IEnumerator<KeyValuePair<TKey, Tree<TKey, TValue>>> GetEnumerator()
        {
            return _children.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public void CopyTo(KeyValuePair<TKey, Tree<TKey, TValue>>[] array, int arrayIndex)
        {
            IDictionary<TKey, Tree<TKey, TValue>> d = _children;
            d.CopyTo(array, arrayIndex);
        }
    }
}
