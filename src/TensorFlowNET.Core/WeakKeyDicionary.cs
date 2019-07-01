using System; 
using System.Collections; 
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis; 
using System.Text;
using System.Runtime.InteropServices;

namespace Tensorflow
{
    public class WeakKeyDictionary<TKey, TValue> : IDictionary<TKey, TValue> 
    {
  
        private Dictionary<WeakKey, TValue> _internalDictionary; 
        private object _internalObject = new object();
        private bool _finalized; 
 
        public WeakKeyDictionary()
        {
            _internalDictionary = new Dictionary<WeakKey, TValue>(new WeakComparer()); 
        }
  
        public WeakKeyDictionary(int capacity) 
        {
            _internalDictionary = new Dictionary<WeakKey, TValue>(capacity, new WeakComparer()); 
        }
 
        public WeakKeyDictionary(IEqualityComparer<TKey> comparer)
        { 
            _internalDictionary = new Dictionary<WeakKey, TValue>(new WeakComparer(comparer));
        } 
  
        public WeakKeyDictionary(int capacity, IEqualityComparer<TKey> comparer)
        { 
            _internalDictionary = new Dictionary<WeakKey, TValue>(capacity, new WeakComparer(comparer));
        }
 
        // FXCop: this is not empty; we need to mark this so we know if a key 
        // still has an active dictionary at its finalization.
        [SuppressMessage("Microsoft.Performance", "CA1821:RemoveEmptyFinalizers")] 
        ~WeakKeyDictionary() 
        {
            _finalized = true; 
        }
 
        public ICollection<TKey> Keys
        { 
            get
            { 
                List<TKey> list = new List<TKey>(); 
                lock (_internalObject)
                { 
                    foreach (WeakKey key in _internalDictionary.Keys)
                    {
                        object TKey = key.Target;
                        if (TKey != null) 
                        {
                            list.Add((TKey)TKey); 
                        } 
                    }
                } 
                return list;
            }
        }
  
        public ICollection<TValue> Values
        { 
            get { 
                lock (_internalObject) {
                    return _internalDictionary.Values; 
                }
            }
        }
  
        public int Count
        { 
            get
            {
                // Ensure a fairly accurate count. 
                ScavangeLostKeys();
                lock (_internalObject)
                {
                    return _internalDictionary.Count; 
                }
            } 
        } 
 
        public bool IsReadOnly 
        {
            get {
                return false;
            } 
        }
  
        [SuppressMessage("Microsoft.Usage", "CA1806:DoNotIgnoreMethodResults", Justification = "LostKeyFinder's purpose is to get garbage collected as soon as posible")] 
        public TValue this[TKey key]
        { 
            get {
                lock (_internalObject) {
                    return _internalDictionary[new WeakKey(key)];
                } 
            }
            set
            { 
                WeakKey Tkey = new WeakKey(key);
                lock (_internalObject) 
                {
                    //_internalDictionary[Tkey] = value;
                    _internalDictionary.Add(Tkey, value);
                }
                // This looks a bit weird but the purpose of the lost key finder is to execute 
                // code in some future garbage collection phase so we immediately create some garbage.
                new LostKeyFinder(this, Tkey);
            } 
        }
  
 
 
 
  
        public bool TryGetValue(TKey key, out TValue value)
        { 
            WeakKey tkey = new WeakKey(key); 
            lock (_internalObject)
            { 
                return _internalDictionary.TryGetValue(tkey, out value);
            }
        }
  
 
        [SuppressMessage("Microsoft.Usage", "CA1806:DoNotIgnoreMethodResults", Justification = "LostKeyFinder's purpose is to get garbage collected as soon as posible")] 
        public void Add(TKey key, TValue value) 
        {
            WeakKey tkey = new WeakKey(key); 
            lock (_internalObject)
            {
                _internalDictionary.Add(tkey, value);
            } 
            // This looks a bit weird but the purpose of the lost key finder is to execute
            // code in some future garbage collection phase so we immediately create some garbage. 
            new LostKeyFinder(this, tkey); 
 
        } 
 
        public bool ContainsKey(TKey key)
        {
            return _internalDictionary.ContainsKey(new WeakKey(key)); 
        }
  
        public bool Remove(TKey key) 
        {
            lock (_internalObject) 
            {
                return _internalDictionary.Remove(new WeakKey(key));
            }
        } 
 
        public void Add(KeyValuePair<TKey, TValue> item) 
        { 
            Add(item.Key, item.Value);
        } 
 
        public void Clear()
        {
            lock (_internalObject) 
            {
                _internalDictionary.Clear(); 
            } 
        }
  
        public bool Contains(KeyValuePair<TKey, TValue> item)
        {
            TValue value;
            bool result; 
            lock (_internalObject)
            { 
                result = _internalDictionary.TryGetValue(new WeakKey(item.Key), out value); 
            }
            if (result) 
            {
                return value.Equals(item.Value);
            }
            else
            {
                return false; 
            } 
        }
  
        public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
        {
            lock (_internalObject)
            { 
                foreach (KeyValuePair<WeakKey, TValue> item in _internalDictionary)
                { 
                    KeyValuePair<TKey, TValue> kv = new KeyValuePair<TKey, TValue>((TKey)item.Key.Target, item.Value); 
                    array[arrayIndex] = kv;
                    arrayIndex++; 
                }
            }
        }
  
        public bool Remove(KeyValuePair<TKey, TValue> item)
        { 
            WeakKey key = new WeakKey(item.Key); 
            lock (_internalObject)
            { 
                return _internalDictionary.Remove(key);
            }
        }
  
 
  
  
 
        public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator() 
        {
            List<WeakKey> lostKeys = null;
            lock (_internalObject)
            { 
                foreach (KeyValuePair<WeakKey, TValue> item in _internalDictionary)
                { 
                    object TKey = item.Key.Target; 
                    if (TKey != null)
                    { 
                        yield return new KeyValuePair<TKey, TValue>((TKey)TKey, item.Value);
                    }
                    else
                    { 
                        if (lostKeys == null)
                        { 
                            lostKeys = new List<WeakKey>(); 
                        }
                        lostKeys.Add(item.Key); 
                    }
                }
            }
            // Recover any lost keys. 
            if (lostKeys != null)
            { 
                lock (_internalObject) 
                {
                    foreach (WeakKey key in lostKeys) 
                    {
                        _internalDictionary.Remove(key);
                    }
                } 
            }
        } 
  
 
  
 
        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator(); 
        }
  
  
 
        private void ScavangeLostKeys() 
        {
            List<WeakKey> lostKeys = null;
            lock (_internalObject)
            { 
                foreach (WeakKey key in _internalDictionary.Keys)
                { 
                    if (!key.IsAlive) 
                    {
                        if (lostKeys == null) 
                        {
                            lostKeys = new List<WeakKey>();
                        }
                        lostKeys.Add(key); 
                    }
                } 
            } 
            if (lostKeys != null)
            { 
                lock (_internalObject)
                {
                    foreach (WeakKey key in lostKeys)
                    { 
                        _internalDictionary.Remove(key);
                    } 
                } 
            }
        }

        IEnumerator<KeyValuePair<TKey, TValue>> IEnumerable<KeyValuePair<TKey, TValue>>.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        private class WeakKey : WeakReference
        {
            private int _hashCode; 
            // private GCHandle _gcHandle;
  
            public WeakKey(TKey key) 
                : base(key, true)
            { 
                _hashCode = key.GetHashCode();
                // Keep the key alive until it is explicitly collected
                // _gcHandle = GCHandle.Alloc(this);
            } 
 
            internal void Release() 
            { 
                // _gcHandle.Free();
            } 
 
            public override int GetHashCode()
            {
                return _hashCode; 
            }
  
            public override bool Equals(object obj) 
            {
                if (obj == null) 
                {
                    return false;
                }
                if (obj.GetHashCode() != _hashCode) 
                {
                    return false; 
                } 
                if (obj != this && (!IsAlive || !obj.Equals(Target)))
                { 
                    return false;
                }
                return true;
            } 
        }
  
        private class WeakComparer : IEqualityComparer<WeakKey> 
        {
  
            private IEqualityComparer<TKey> _comparer;
            public WeakComparer()
            {
            } 
 
            public WeakComparer(IEqualityComparer<TKey> comparer) 
            { 
                _comparer = comparer;
            } 
 
            public bool Equals(WeakKey x, WeakKey y)
            {
                if (x.GetHashCode() != y.GetHashCode()) 
                {
                    return false; 
                } 
                if (object.ReferenceEquals(x, y))
                { 
                    return true;
                }
                object ref1 = x.Target;
                if (ref1 == null) 
                {
                    return false; 
                } 
                object ref2 = y.Target;
                if (ref2 == null) 
                {
                    return false;
                }
  
                if (_comparer != null)
                { 
                    return _comparer.Equals((TKey)ref1, (TKey)ref2); 
                }
                else
                {
                    return ref1.Equals(ref2);
                }
            } 
 
            public int GetHashCode(WeakKey obj) 
            { 
                return obj.GetHashCode();
            } 
        }
 
        private class LostKeyFinder
        { 
            WeakKeyDictionary<TKey, TValue> _dictionary;
            WeakKey _key; 
  
            public LostKeyFinder(WeakKeyDictionary<TKey, TValue> dictionary, WeakKey key)
            { 
                _dictionary = dictionary;
                _key = key;
            }
  
            ~LostKeyFinder()
            { 
                if (_dictionary._finalized || _key == null) 
                {
                    if (_key != null) 
                    {
                        _key.Release();
                        _key = null;
                    } 
                    return;
                } 
                // if (!_key.IsAlive) { 
                if (_key.Target == null)
                { 
                    lock (_dictionary._internalObject)
                    {
                        _dictionary._internalDictionary.Remove(_key);
                    } 
                    _key.Release();
                    _key = null; 
                } 
                else if (_dictionary._internalDictionary.ContainsKey(_key))
                { 
                    GC.ReRegisterForFinalize(this);
                }
            }
        } 
    }
} 
 