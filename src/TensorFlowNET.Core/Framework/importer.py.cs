using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.OpDef.Types;
using static Tensorflow.Python;

namespace Tensorflow
{
    public class importer
    {
        public static ITensorOrOperation[] import_graph_def(GraphDef graph_def,
            Dictionary<string, Tensor> input_map = null,
            string[] return_elements = null,
            string name = null,
            OpList producer_op_list = null)
        {
            var op_dict = op_def_registry.get_registered_ops();

            graph_def = _ProcessGraphDefParam(graph_def, op_dict);
            input_map = _ProcessInputMapParam(input_map);
            return_elements = _ProcessReturnElementsParam(return_elements);

            if (producer_op_list != null)
                _RemoveDefaultAttrs(op_dict, producer_op_list, graph_def);

            string prefix = "";
            var graph = ops.get_default_graph();
            with(ops.name_scope(name, "import", input_map.Values), scope =>
            {
                prefix = scope;
                /*if (!string.IsNullOrEmpty(prefix))
                    prefix = prefix.Substring(0, prefix.Length - 1);
                else
                    prefix = "";*/

                // Generate any input map tensors inside name scope
                input_map = _ConvertInputMapValues(name, input_map);
            });

            var scoped_options = c_api_util.ScopedTFImportGraphDefOptions();
            _PopulateTFImportGraphDefOptions(scoped_options, prefix, input_map, return_elements);

            var bytes = graph_def.ToByteString().ToArray();
            IntPtr buffer = c_api_util.tf_buffer(bytes);

            var status = new Status();
            // need to create a class ImportGraphDefWithResults with IDisposal
            var results = c_api.TF_GraphImportGraphDefWithResults(graph, buffer, scoped_options, status);
            status.Check(true);

            _ProcessNewOps(graph);

            if (return_elements == null)
                return null;
            else
                throw new NotImplementedException("import_graph_def return_elements");
        }

        private static void _ProcessNewOps(Graph graph)
        {
            foreach(var new_op in graph._add_new_tf_operations())
            {
                var original_device = new_op.Device;
            }
        }

        public static void _PopulateTFImportGraphDefOptions(ImportGraphDefOptions options, 
            string prefix, 
            Dictionary<string, Tensor> input_map,
            string[] return_elements)
        {
            c_api.TF_ImportGraphDefOptionsSetPrefix(options, prefix);
            c_api.TF_ImportGraphDefOptionsSetUniquifyNames(options, (char)1);

            foreach(var input in input_map)
            {
                throw new NotImplementedException("_PopulateTFImportGraphDefOptions");
            }

            if (return_elements == null)
                return_elements = new string[0];

            foreach (var name in return_elements)
            {
                throw new NotImplementedException("_PopulateTFImportGraphDefOptions");
            }
        }

        public static Dictionary<string, Tensor> _ConvertInputMapValues(string name, Dictionary<string, Tensor> input_map)
        {
            return input_map;
        }

        public static GraphDef _ProcessGraphDefParam(GraphDef graph_def, Dictionary<string, OpDef> op_dict)
        {
            foreach(var node in graph_def.Node)
            {
                if (!op_dict.ContainsKey(node.Op))
                    continue;

                var op_def = op_dict[node.Op];
                _SetDefaultAttrValues(node, op_def);
            }

            return graph_def;
        }

        private static void _SetDefaultAttrValues(NodeDef node_def, OpDef op_def)
        {
            foreach(var attr_def in op_def.Attr)
            {
                var key = attr_def.Name;
                if(attr_def.DefaultValue != null)
                {
                    if (node_def.Attr.ContainsKey(key))
                    {
                        var value = node_def.Attr[key];
                        if (value == null)
                            node_def.Attr[key] = attr_def.DefaultValue;
                    }
                    else
                    {
                        node_def.Attr[key] = attr_def.DefaultValue;
                    }
                }
            }
        }

        private static Dictionary<string, Tensor> _ProcessInputMapParam(Dictionary<string, Tensor> input_map)
        {
            if (input_map == null)
                return new Dictionary<string, Tensor>();

            return input_map;
        }

        private static string[] _ProcessReturnElementsParam(string[] return_elements)
        {
            if (return_elements == null)
                return null;

            return return_elements;
        }

        private static void _RemoveDefaultAttrs(Dictionary<string, OpDef> op_dict, OpList producer_op_list, GraphDef graph_def)
        {
            var producer_op_dict = new Dictionary<string, OpDef>();
            producer_op_list.Op.Select(op =>
            {
                producer_op_dict[op.Name] = op;
                return op;
            }).ToArray();

            foreach(var node in graph_def.Node)
            {
                // Remove any default attr values that aren't in op_def.
                if (producer_op_dict.ContainsKey(node.Op))
                {
                    var op_def = op_dict[node.Op];
                    var producer_op_def = producer_op_dict[node.Op];
                    foreach(var key in node.Attr)
                    {
                        if(_FindAttrInOpDef(key.Key, op_def) == null)
                        {
                            var attr_def = _FindAttrInOpDef(key.Key, producer_op_def);
                            if (attr_def != null && attr_def.DefaultValue != null &&
                                    node.Attr[key.Key] == attr_def.DefaultValue)
                                node.Attr[key.Key].ClearValue();
                        }
                    }
                }
            }
        }

        private static AttrDef _FindAttrInOpDef(string name, OpDef op_def)
        {
            return op_def.Attr.FirstOrDefault(x => x.Name == name);
        }
    }
}
