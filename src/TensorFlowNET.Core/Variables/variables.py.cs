using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class variables
    {
        /// <summary>
        /// Returns all variables created with `trainable=True`
        /// </summary>
        /// <returns></returns>
        public static object trainable_variables()
        {
            return ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES);
        }

        /// <summary>
        /// Returns all variables and `SaveableObject`s that must be checkpointed.
        /// </summary>
        /// <param name="scope"></param>
        /// <returns></returns>
        public static VariableV1[] _all_saveable_objects(string scope = "")
        {
            var all = new List<VariableV1>();

            var collection = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, scope);
            if(collection != null)
                all.AddRange(collection as List<VariableV1>);

            collection = ops.get_collection(ops.GraphKeys.SAVEABLE_OBJECTS, scope);
            if (collection != null)
                all.AddRange(collection as List<VariableV1>);

            return all.ToArray();
        }

        /// <summary>
        /// Returns global variables.
        /// </summary>
        /// <param name="scope">
        /// (Optional.) A string. If supplied, the resulting list is filtered
        /// to include only items whose `name` attribute matches `scope` using
        /// `re.match`. Items without a `name` attribute are never returned if a
        /// scope is supplied. The choice of `re.match` means that a `scope` without
        /// special tokens filters by prefix.
        /// </param>
        /// <returns>A list of `Variable` objects.</returns>
        public static List<VariableV1> global_variables(string scope = null)
        {
            var result = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, scope);

            return result == null ? new List<VariableV1>() : result as List<VariableV1>;
        }

        /// <summary>
        /// Returns an Op that initializes a list of variables.
        /// </summary>
        /// <param name="var_list">List of `Variable` objects to initialize.</param>
        /// <param name="name">Optional name for the returned operation.</param>
        /// <returns>An Op that run the initializers of all the specified variables.</returns>
        public static Operation variables_initializer(VariableV1[] var_list, string name = "init")
        {
            if (var_list.Length > 0)
                return control_flow_ops.group(var_list.Select(x => x.initializer).ToArray(), name);
            else
                return gen_control_flow_ops.no_op(name: name);
        }

        public static Tensor global_variables_initializer()
        {
            throw new NotImplementedException();
        }
    }
}
