using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Tensorflow
{
    public class graph_io
    {
        public static string write_graph(Graph graph, string logdir, string name, bool as_text = true)
        {
            var graph_def = graph.as_graph_def();
            string path = Path.Combine(logdir, name);
            if (as_text)
                File.WriteAllText(path, graph_def.ToString());
            else
                File.WriteAllBytes(path, graph_def.ToByteArray());
            return path;
        }

        public static string write_graph(MetaGraphDef graph_def, string logdir, string name, bool as_text = true)
        {
            string path = Path.Combine(logdir, name);
            if (as_text)
                File.WriteAllText(path, graph_def.ToString());
            else
                File.WriteAllBytes(path, graph_def.ToByteArray());
            return path;
        }
    }
}
