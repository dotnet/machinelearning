using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow.Summaries
{
    public class Summary
    {
        public FileWriter FileWriter(string logdir, Graph graph,
            int max_queue = 10, int flush_secs = 120, string filename_suffix = null,
            Session session = null)
            => new FileWriter(logdir, graph, max_queue: max_queue,
                flush_secs: flush_secs, filename_suffix: filename_suffix,
                session: session);

        public Tensor histogram(string name, Tensor tensor, string[] collections = null, string family = null)
        {
            var (tag, scope) = summary_scope(name, family: family, values: new Tensor[] { tensor }, default_name: "HistogramSummary");
            var val = gen_logging_ops.histogram_summary(tag: tag, values: tensor, name: scope);
            collect(val, collections?.ToList(), new List<string> { ops.GraphKeys.SUMMARIES });
            return val;
        }

        public Tensor merge_all(string key = ops.GraphKeys.SUMMARIES, string scope= null, string name= null)
        {
            var summary_ops = ops.get_collection(key, scope: scope);
            if (summary_ops == null)
                return null;
            else
                return merge((summary_ops as List<ITensorOrOperation>).Select(x => x as Tensor).ToArray(), name: name);
        }

        /// <summary>
        /// Merges summaries.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="collections"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor merge(Tensor[] inputs, string[] collections = null, string name = null)
        {
            return with(ops.name_scope(name, "Merge", inputs), delegate
            {
                var val = gen_logging_ops.merge_summary(inputs: inputs, name: name);
                collect(val, collections?.ToList(), new List<string>());
                return val;
            });
        }

        public Tensor scalar(string name, Tensor tensor, string[] collections = null, string family = null)
        {
            var (tag, scope) = summary_scope(name, family: family, values: new Tensor[] { tensor });
            var val = gen_logging_ops.scalar_summary(tags: tag, values: tensor, name: scope);
            collect(val, collections?.ToList(), new List<string> { ops.GraphKeys.SUMMARIES });
            return val;
        }

        /// <summary>
        /// Adds keys to a collection.
        /// </summary>
        /// <param name="val"The value to add per each key.></param>
        /// <param name="collections">A collection of keys to add.</param>
        /// <param name="default_collections">Used if collections is None.</param>
        public void collect(ITensorOrOperation val, List<string> collections, List<string> default_collections)
        {
            if (collections == null)
                collections = default_collections;
            foreach (var key in collections)
                ops.add_to_collection(key, val);
        }

        public (string, string) summary_scope(string name, string family = null, string default_name = null, Tensor[] values = null)
        {
            string scope_base_name = string.IsNullOrEmpty(family) ? name : $"{family}/{name}";
            return with(ops.name_scope(scope_base_name, default_name: default_name, values), scope =>
            {
                var tag = scope._name_scope;
                if (string.IsNullOrEmpty(family))
                    tag = tag.Remove(tag.Length - 1);
                else
                    tag = $"{family}/{tag.Remove(tag.Length - 1)}";

                return (tag, scope._name_scope);
            });
        }
    }
}
