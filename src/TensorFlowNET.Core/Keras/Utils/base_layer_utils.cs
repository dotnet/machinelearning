using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow.Keras.Utils
{
    public class base_layer_utils
    {
        /// <summary>
        /// Adds a new variable to the layer.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="shape"></param>
        /// <param name="dtype"></param>
        /// <param name="initializer"></param>
        /// <param name="trainable"></param>
        /// <returns></returns>
        public static RefVariable make_variable(string name,
            int[] shape,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            IInitializer initializer = null,
            bool trainable = true) => make_variable(name, shape, dtype, initializer, trainable, true);

        /// <summary>
        /// Adds a new variable to the layer.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="shape"></param>
        /// <param name="dtype"></param>
        /// <param name="initializer"></param>
        /// <param name="trainable"></param>
        /// <returns></returns>
        public static RefVariable make_variable(string name,
            int[] shape,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            IInitializer initializer = null,
            bool trainable = true,
            bool use_resource = true)
        {
            var initializing_from_value = false;

            ops.init_scope();

            Func<Tensor> init_val = () => initializer.call(new TensorShape(shape), dtype: dtype);

            var variable_dtype = dtype.as_base_dtype();
            var v = tf.Variable(init_val);

            return v;
        }

        /// <summary>
        /// Makes a layer name (or arbitrary string) unique within a TensorFlow graph.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public static string unique_layer_name(string name, Dictionary<(string, string), int> name_uid_map = null,
            string[] avoid_names = null, string @namespace = "", bool zero_based = false)
        {
            if (name_uid_map == null)
                name_uid_map = get_default_graph_uid_map();
            if (avoid_names == null)
                avoid_names = new string[0];

            string proposed_name = null;
            while (proposed_name == null || avoid_names.Contains(proposed_name))
            {
                var name_key = (@namespace, name);
                if (!name_uid_map.ContainsKey(name_key))
                    name_uid_map[name_key] = 0;

                if (zero_based)
                {
                    int number = name_uid_map[name_key];
                    if (number > 0)
                        proposed_name = $"{name}_{number}";
                    else
                        proposed_name = name;

                    name_uid_map[name_key] += 1;
                }
                else
                {
                    name_uid_map[name_key] += 1;
                    proposed_name = $"{name}_{name_uid_map[name_key]}";
                }
            }

            return proposed_name;
        }

        public static Dictionary<(string, string), int> get_default_graph_uid_map()
        {
            var graph = ops.get_default_graph();
            Dictionary<(string, string), int> name_uid_map = null;
            if (backend.PER_GRAPH_LAYER_NAME_UIDS.ContainsKey(graph))
            {
                name_uid_map = backend.PER_GRAPH_LAYER_NAME_UIDS[graph];
            }
            else
            {
                name_uid_map = new Dictionary<(string, string), int>();
                backend.PER_GRAPH_LAYER_NAME_UIDS[graph] = name_uid_map;
            }

            return name_uid_map;
        }
    }
}
