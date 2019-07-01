using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Fetch mapper for singleton tensors and ops.
    /// </summary>
    public class _ElementFetchMapper : _FetchMapper
    {
        private Func<List<NDArray>, object> _contraction_fn;

        public _ElementFetchMapper(object[] fetches, Func<List<NDArray>, object> contraction_fn)
        {
            var g = ops.get_default_graph();

            foreach(var fetch in fetches)
            {
                var el = g.as_graph_element(fetch, allow_tensor: true, allow_operation: true);
                _unique_fetches.Add(el);
            }
            
            _contraction_fn = contraction_fn;
        }

        /// <summary>
        /// Build results matching the original fetch shape.
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public override NDArray build_results(List<NDArray> values)
        {
            NDArray result = null;

            if (values.Count > 0)
            {
                var ret = _contraction_fn(values);
                switch (ret)
                {
                    case NDArray value:
                        result = value;
                        break;
                    case short value:
                        result = value;
                        break;
                    case int value:
                        result = value;
                        break;
                    case long value:
                        result = value;
                        break;
                    case float value:
                        result = value;
                        break;
                    case double value:
                        result = value;
                        break;
                    case string value:
                        result = value;
                        break;
                    default:
                        break;
                }
            }

            return result;
        }
    }
}
