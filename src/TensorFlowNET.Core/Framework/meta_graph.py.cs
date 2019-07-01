using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow.Operations;
using static Tensorflow.CollectionDef;
using static Tensorflow.MetaGraphDef.Types;

namespace Tensorflow
{
    public class meta_graph
    {
        public static MetaGraphDef read_meta_graph_file(string filename)
        {
            var bytes = File.ReadAllBytes(filename);
            var meta_graph_def = MetaGraphDef.Parser.ParseFrom(bytes);
            return meta_graph_def;
        }

        public static (Dictionary<string, VariableV1>, ITensorOrOperation[]) import_scoped_meta_graph_with_return_elements(MetaGraphDef meta_graph_or_file,
            bool clear_devices = false,
            string import_scope = "",
            Dictionary<string, Tensor> input_map = null,
            string unbound_inputs_col_name = "unbound_inputs",
            string[] return_elements = null)
        {
            var meta_graph_def = meta_graph_or_file;

            if (!string.IsNullOrEmpty(unbound_inputs_col_name))
            {
                foreach(var col in meta_graph_def.CollectionDef)
                {
                    if(col.Key == unbound_inputs_col_name)
                    {
                        throw new NotImplementedException("import_scoped_meta_graph_with_return_elements");
                    }
                }
            }

            // Sets graph to default graph if it's not passed in.
            var graph = ops.get_default_graph();

            // Gathers the list of nodes we are interested in.
            OpList producer_op_list = null;
            if (meta_graph_def.MetaInfoDef.StrippedOpList != null)
                producer_op_list = meta_graph_def.MetaInfoDef.StrippedOpList;
            var input_graph_def = meta_graph_def.GraphDef;
            // Remove all the explicit device specifications for this node. This helps to
            // make the graph more portable.
            if (clear_devices)
                foreach (var node in input_graph_def.Node)
                    node.Device = "";

            var scope_to_prepend_to_names = graph.unique_name("", mark_as_used: false);
            var imported_return_elements = importer.import_graph_def(input_graph_def,
                name: scope_to_prepend_to_names,
                input_map: input_map,
                producer_op_list: producer_op_list,
                return_elements: return_elements);

            // Restores all the other collections.
            var variable_objects = new Dictionary<ByteString, VariableV1>();
            foreach(var col in meta_graph_def.CollectionDef.OrderBy(x => x.Key))
            {
                // Don't add unbound_inputs to the new graph.
                if (col.Key == unbound_inputs_col_name)
                    continue;

                switch (col.Value.KindCase)
                {
                    case KindOneofCase.NodeList:
                        foreach(var value in col.Value.NodeList.Value)
                        {
                            var col_op = graph.as_graph_element(ops.prepend_name_scope(value, scope_to_prepend_to_names));
                            graph.add_to_collection(col.Key, col_op);
                        }
                        break;
                    case KindOneofCase.BytesList:
                        //var proto_type = ops.get_collection_proto_type(key)
                        if (ops.GraphKeys._VARIABLE_COLLECTIONS.Contains(col.Key))
                        {
                            foreach (var value in col.Value.BytesList.Value)
                            {
                                VariableV1 variable = null;
                                if (!variable_objects.ContainsKey(value))
                                {
                                    var proto = VariableDef.Parser.ParseFrom(value);
                                    if (proto.IsResource)
                                        variable = new ResourceVariable(variable_def: proto, import_scope: scope_to_prepend_to_names);
                                    else
                                        variable = new RefVariable(variable_def: proto, import_scope: scope_to_prepend_to_names);
                                    variable_objects[value] = variable;
                                }
                                variable = variable_objects[value];
                                graph.add_to_collection(col.Key, variable);
                            }
                        }
                        else
                        {
                            foreach(var value in col.Value.BytesList.Value)
                            {
                                switch (col.Key)
                                {
                                    case "cond_context":
                                        {
                                            var proto = CondContextDef.Parser.ParseFrom(value);
                                            var condContext = new CondContext().from_proto(proto, import_scope);
                                            graph.add_to_collection(col.Key, condContext);
                                        }
                                        break;
                                    case "while_context":
                                        {
                                            var proto = WhileContextDef.Parser.ParseFrom(value);
                                            var whileContext = new WhileContext().from_proto(proto, import_scope);
                                            graph.add_to_collection(col.Key, whileContext);
                                        }
                                        break;
                                    default:
                                        throw new NotImplementedException("import_scoped_meta_graph_with_return_elements");
                                }
                            }
                        }
                        
                        break;
                    default:
                        throw new NotImplementedException("import_scoped_meta_graph_with_return_elements");
                }
            }

            var variables = graph.get_collection<VariableV1>(ops.GraphKeys.GLOBAL_VARIABLES,
                                     scope: scope_to_prepend_to_names);
            var var_list = new Dictionary<string, VariableV1>();
            variables.ForEach(v => var_list[ops.strip_name_scope(v.name, scope_to_prepend_to_names)] = v);

            return (var_list, imported_return_elements);
        }

        /// <summary>
        /// Returns `MetaGraphDef` proto. Optionally writes it to filename.
        /// </summary>
        /// <param name="filename"></param>
        /// <param name="graph_def"></param>
        /// <param name="as_text"></param>
        /// <param name="unbound_inputs_col_name"></param>
        /// <param name="clear_devices"></param>
        /// <param name="saver_def"></param>
        /// <param name="clear_extraneous_savers"></param>
        /// <param name="strip_default_attrs"></param>
        /// <param name="meta_info_def"></param>
        /// <returns></returns>
        public static (MetaGraphDef, Dictionary<string, RefVariable>) export_scoped_meta_graph(string filename = "",
            GraphDef graph_def = null,
            bool as_text = false,
            string unbound_inputs_col_name = "unbound_inputs",
            bool clear_devices = false,
            SaverDef saver_def = null,
            bool clear_extraneous_savers= false,
            bool strip_default_attrs= false,
            byte[] meta_info_def = null)
        {
            var graph = ops.get_default_graph();

            var var_list = new Dictionary<string, RefVariable>();
            var variables = graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES) as List<RefVariable>;

            if (variables != null)
            {
                foreach (var v in variables)
                {
                    var_list[v.name] = v;
                }
            }

            var scoped_meta_graph_def = create_meta_graph_def(
                graph_def: graph_def,
                export_scope: "",
                exclude_nodes: "",
                clear_extraneous_savers: clear_extraneous_savers,
                saver_def: saver_def,
                strip_default_attrs: strip_default_attrs);

            if (!string.IsNullOrEmpty(filename))
                graph_io.write_graph(scoped_meta_graph_def, "", filename, as_text: as_text);

            return (scoped_meta_graph_def, var_list);
        }

        private static bool _should_include_node()
        {
            return true;
        }

        private static MetaGraphDef create_meta_graph_def(MetaInfoDef meta_info_def = null,
            GraphDef graph_def = null,
            string export_scope = "",
            string exclude_nodes = "",
            SaverDef saver_def = null,
            bool clear_extraneous_savers = false,
            bool strip_default_attrs = false)
        {
            // Sets graph to default graph if it's not passed in.
            var graph = ops.get_default_graph().as_default();
            // Creates a MetaGraphDef proto.
            var meta_graph_def = new MetaGraphDef();
            if (meta_info_def == null)
                meta_info_def = new MetaInfoDef();

            // Set the tf version strings to the current tf build.
            meta_info_def.TensorflowVersion = tf.VERSION;
            meta_info_def.TensorflowGitVersion = "unknown";
            meta_graph_def.MetaInfoDef = meta_info_def;

            // Adds graph_def or the default.
            if (graph_def == null)
                meta_graph_def.GraphDef = graph.as_graph_def(add_shapes: true);
            else
                meta_graph_def.GraphDef = graph_def;

            // Fills in meta_info_def.stripped_op_list using the ops from graph_def.
            if (meta_graph_def.MetaInfoDef.StrippedOpList == null || 
                meta_graph_def.MetaInfoDef.StrippedOpList.Op.Count == 0)
                meta_graph_def.MetaInfoDef.StrippedOpList = stripped_op_list_for_graph(meta_graph_def.GraphDef);

            var clist = graph.get_all_collection_keys();
            foreach(var ctype in clist)
            {
                if (clear_extraneous_savers)
                {
                    throw new NotImplementedException("create_meta_graph_def clear_extraneous_savers");
                }
                else
                {
                    add_collection_def(meta_graph_def, ctype, graph);
                }
            }

            return meta_graph_def;
        }

        private static void add_collection_def(MetaGraphDef meta_graph_def, 
            string key, 
            Graph graph = null,
            string export_scope = "")
        {
            if (!meta_graph_def.CollectionDef.ContainsKey(key))
                meta_graph_def.CollectionDef[key] = new CollectionDef();
            var col_def = meta_graph_def.CollectionDef[key];

            switch (graph.get_collection(key))
            {
                case List<RefVariable> collection_list:
                    col_def.BytesList = new Types.BytesList();
                    foreach (var x in collection_list)
                    {
                        var proto = x.to_proto(export_scope);
                        col_def.BytesList.Value.Add(proto.ToByteString());
                    }
                    
                    break;
                case List<object> collection_list:
                    col_def.NodeList = new Types.NodeList();
                    foreach (var x in collection_list)
                        if (x is ITensorOrOperation x2)
                            col_def.NodeList.Value.Add(ops.strip_name_scope(x2.name, export_scope));
                    break;
                case List<Operation> collection_list:
                    break;
            }
        }

        private static OpList stripped_op_list_for_graph(GraphDef graph_def)
        {
            var used_ops = ops_used_by_graph_def(graph_def);

            // Verify that all used ops are registered.
            // var registered_ops = op_def_registry.get_registered_ops();

            var op_list = new OpList();
            /*used_ops.OrderBy(x => x).Select(x => {

            }).ToArray();*/

            return op_list;
        }

        /// <summary>
        /// Collect the list of ops used by a graph.
        /// </summary>
        /// <param name="graph_def"></param>
        /// <returns></returns>
        private static string[] ops_used_by_graph_def(GraphDef graph_def)
        {
            var used_ops = new List<string>();

            Action<string> mark_op_as_used = (op) =>
            {
                if (!used_ops.Contains(op))
                {

                }

                used_ops.Add(op);
            };

            foreach (var node in graph_def.Node)
            {
                mark_op_as_used(node.Op);
            }

            return used_ops.ToArray();
        }
    }
}
