using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Eager;

namespace Tensorflow
{
    public partial class ops
    {
        public static NameScope name_scope(string name,
            string default_name = "",
            object values = null) => new NameScope(name, default_name, values);

        /// <summary>
        /// Returns a context manager that creates hierarchical names for operations.
        /// </summary>
        public class NameScope : IPython
        {
            public string _name;
            public string _default_name;
            public object _values;
            public Context _ctx;
            public string _name_scope;
            public string old_stack = "";
            private object _g_manager;

            public NameScope(string name, string default_name = "", object values = null)
            {
                _name = name;
                _default_name = default_name;
                _values = values;
                // _ctx = new Context();
            }

            public void __enter__()
            {
                _name = _name == null ? _default_name : _name;

                Graph g = null;

                if (_values is List<Tensor> vList)
                    g = _get_graph_from_inputs(vList.ToArray());
                else if (_values is Tensor[] vArray)
                    g = _get_graph_from_inputs(vArray);

                if (g == null)
                    g = get_default_graph();

                old_stack = g._name_stack;
                _name_scope = g.name_scope(_name);
            }

            public void Dispose()
            {
                var g = get_default_graph();
                g._name_stack = old_stack;
                // Console.WriteLine($"name_scope: {g._name_stack} -> {old_stack}");
            }

            public void __exit__()
            {

            }

            /// <summary>
            /// __enter__()
            /// </summary>
            /// <param name="ns"></param>
            public static implicit operator string(NameScope ns)
            {
                return ns._name_scope;
            }
        }
    }
}
