using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.Data.Analysis
{
    public class Grouping<TKey> : IGrouping<TKey, DataFrameRow>
    {
        private readonly TKey _key;
        private readonly ICollection<DataFrameRow> _rows;

        public Grouping(TKey key, ICollection<DataFrameRow> rows)
        {
            _key = key;
            _rows = rows;
        }

        public TKey Key => _key;

        public IEnumerator<DataFrameRow> GetEnumerator()
        {
            return _rows.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return _rows.GetEnumerator();
        }
    }
}
