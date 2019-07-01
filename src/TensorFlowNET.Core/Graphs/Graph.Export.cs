using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class Graph
    {
        public Buffer ToGraphDef(Status s)
        {
            var buffer = new Buffer();
            c_api.TF_GraphToGraphDef(_handle, buffer, s);
            s.Check();
            // var def = GraphDef.Parser.ParseFrom(buffer);
            // buffer.Dispose();

            return buffer;
        }

        private GraphDef _as_graph_def(bool add_shapes = false)
        {
            var buffer = ToGraphDef(Status);
            Status.Check();
            var def = GraphDef.Parser.ParseFrom(buffer);
            buffer.Dispose();

            // Strip the experimental library field iff it's empty.
            // if(def.Library.Function.Count == 0)

            return def;
        }

        public GraphDef as_graph_def(bool add_shapes = false) 
            => _as_graph_def(add_shapes);
    }
}
