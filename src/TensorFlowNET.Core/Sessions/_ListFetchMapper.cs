using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class _ListFetchMapper : _FetchMapper
    {
        private _FetchMapper[] _mappers;

        public _ListFetchMapper(object[] fetches)
        {
            _mappers = fetches.Select(fetch => _FetchMapper.for_fetch(fetch)).ToArray();
            (_unique_fetches, _value_indices) = _uniquify_fetches(_mappers);
        }

        private (List<ITensorOrOperation>, List<int[]>) _uniquify_fetches(_FetchMapper[] fetch_mappers)
        {
            var unique_fetches = new List<ITensorOrOperation>();
            var value_indices = new List<int[]>();
            var seen_fetches = new Dictionary<ITensorOrOperation, int>();

            foreach (var m in fetch_mappers)
            {
                var m_value_indices = new List<int>();
                foreach (var uf in m.unique_fetches())
                {
                    switch (uf)
                    {
                        case Tensor f:
                            if (!seen_fetches.ContainsKey(f))
                            {
                                seen_fetches[f] = seen_fetches.Count;
                                unique_fetches.Add(f);
                            }
                            m_value_indices.Add(seen_fetches.Count - 1);
                            break;
                        case Operation f:
                            if (!seen_fetches.ContainsKey(f))
                            {
                                seen_fetches[f] = seen_fetches.Count;
                                unique_fetches.Add(f);
                            }
                            m_value_indices.Add(seen_fetches.Count - 1);
                            break;
                        default:
                            throw new NotImplementedException("_uniquify_fetches");
                    }
                }
                value_indices.Add(m_value_indices.ToArray());
            }

            return (unique_fetches, value_indices);
        }
    }
}
