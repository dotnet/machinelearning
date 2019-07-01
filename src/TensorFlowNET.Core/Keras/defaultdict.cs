using System.Collections.Generic;

namespace System.Collections.Generic
{
    public class defaultdict<TKey, TValue> : Dictionary<TKey, TValue> where TValue : new()
    {
        public new TValue this[TKey key]
        {
            get
            {
                TValue val;
                if(!TryGetValue(key, out val))
                {
                    val = default(TValue);
                    Add(key, val);
                }
                return val;
            }
            set { base[key] = value; }
        }
    }
}