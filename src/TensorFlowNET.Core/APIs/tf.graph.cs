using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static graph_util_impl graph_util => new graph_util_impl();
        public static Graph get_default_graph()
        {
            return ops.get_default_graph();
        }

        public static Graph Graph() => new Graph();

    }
}
