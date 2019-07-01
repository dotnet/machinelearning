using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class _FetchMapper
    {
        protected List<ITensorOrOperation> _unique_fetches = new List<ITensorOrOperation>();
        protected List<int[]> _value_indices = new List<int[]>();
        public static _FetchMapper for_fetch(object fetch)
        {
            var fetches = fetch.GetType().IsArray ? (object[])fetch : new object[] { fetch };

            if(fetch is List<string> fetches1)
                return new _ListFetchMapper(fetches1.ToArray());
            if (fetch.GetType().IsArray)
                return new _ListFetchMapper(fetches);
            else
                return new _ElementFetchMapper(fetches, (List<NDArray> fetched_vals) => fetched_vals[0]);
        }

        public virtual NDArray build_results(List<NDArray> values)
        {
            var type = values[0].GetType();
            var nd = new NDArray(type, values.Count);
            nd.SetData(values.ToArray());
            return nd;
        }

        public virtual List<ITensorOrOperation> unique_fetches()
        {
            return _unique_fetches;
        }
    }
}
