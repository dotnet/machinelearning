using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow.Train;

namespace Tensorflow
{
    public static partial class tf
    {
        public static class train
        {
            public static Optimizer GradientDescentOptimizer(float learning_rate) => new GradientDescentOptimizer(learning_rate);

            public static Optimizer AdamOptimizer(float learning_rate) => new AdamOptimizer(learning_rate);

            public static Saver Saver(VariableV1[] var_list = null) => new Saver(var_list: var_list);

            public static string write_graph(Graph graph, string logdir, string name, bool as_text = true) 
                => graph_io.write_graph(graph, logdir, name, as_text);

            public static Saver import_meta_graph(string meta_graph_or_file,
                bool clear_devices = false,
                string import_scope = "") => saver._import_meta_graph_with_return_elements(meta_graph_or_file,
                    clear_devices,
                    import_scope).Item1;

            public static (MetaGraphDef, Dictionary<string, RefVariable>) export_meta_graph(string filename = "",
                bool as_text = false,
                bool clear_devices = false,
                bool clear_extraneous_savers = false,
                bool strip_default_attrs = false) => meta_graph.export_scoped_meta_graph(filename: filename,
                    as_text: as_text,
                    clear_devices: clear_devices,
                    clear_extraneous_savers: clear_extraneous_savers,
                    strip_default_attrs: strip_default_attrs);
        }
    }
}
