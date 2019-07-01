using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Feed dictionary item
    /// </summary>
    public class FeedItem
    {
        public object Key { get; }
        public object Value { get; }

        public FeedItem(object key, object val)
        {
            Key = key;
            Value = val;
        }
    }
}
