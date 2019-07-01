using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static object get_collection(string key, string scope = "") => get_default_graph()
            .get_collection(key, scope: scope);

        /// <summary>
        /// Returns a context manager that creates hierarchical names for operations.
        /// </summary>
        /// <param name="name">The name argument that is passed to the op function.</param>
        /// <param name="default_name">The default name to use if the name argument is None.</param>
        /// <param name="values">The list of Tensor arguments that are passed to the op function.</param>
        /// <returns>The scope name.</returns>
        public static ops.NameScope name_scope(string name, 
            string default_name = "", 
            object values = null) => new ops.NameScope(name, default_name, values);
    }
}
