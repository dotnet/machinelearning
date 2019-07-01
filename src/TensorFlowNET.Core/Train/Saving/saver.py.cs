using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class saver
    {
        public static (Saver, object) _import_meta_graph_with_return_elements(string meta_graph_or_file,
            bool clear_devices = false,
            string import_scope = "",
            string[] return_elements = null)
        {
            var meta_graph_def = meta_graph.read_meta_graph_file(meta_graph_or_file);

            var meta = meta_graph.import_scoped_meta_graph_with_return_elements(
                        meta_graph_def,
                        clear_devices: clear_devices,
                        import_scope: import_scope,
                        return_elements: return_elements);

            var (imported_vars, imported_return_elements) = meta;

            var saver = _create_saver_from_imported_meta_graph(
                meta_graph_def, import_scope, imported_vars);

            return (saver, imported_return_elements);
        }

        /// <summary>
        /// Return a saver for restoring variable values to an imported MetaGraph.
        /// </summary>
        /// <param name="meta_graph_def"></param>
        /// <param name="import_scope"></param>
        /// <param name="imported_vars"></param>
        /// <returns></returns>
        public static Saver _create_saver_from_imported_meta_graph(MetaGraphDef meta_graph_def, 
            string import_scope, 
            Dictionary<string, VariableV1> imported_vars)
        {
            if(meta_graph_def.SaverDef != null)
            {
                // Infer the scope that is prepended by `import_scoped_meta_graph`.
                string scope = import_scope;
                var var_names = imported_vars.Keys.ToArray();
                if(var_names.Length > 0)
                {
                    var sample_key = var_names[0];
                    var sample_var = imported_vars[sample_key];
                    scope = string.Join("", sample_var.name.Skip(sample_key.Length));
                }
                return new Saver(saver_def: meta_graph_def.SaverDef, name: scope);
            }
            else
            {
                if(variables._all_saveable_objects(scope: import_scope).Length > 0)
                {
                    // Return the default saver instance for all graph variables.
                    return new Saver();
                }
                else
                {
                    // If no graph variables exist, then a Saver cannot be constructed.
                    Console.WriteLine("Saver not created because there are no variables in the" +
                        " graph to restore");
                    return null;
                }
            }
        }
    }
}
