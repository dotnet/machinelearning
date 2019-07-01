using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow
{
    public class graph_util_impl
    {
        /// <summary>
        /// Replaces all the variables in a graph with constants of the same values.
        /// </summary>
        /// <param name="sess">Active TensorFlow session containing the variables.</param>
        /// <param name="input_graph_def">GraphDef object holding the network.</param>
        /// <param name="output_node_names">List of name strings for the result nodes of the graph.</param>
        /// <param name="variable_names_whitelist"></param>
        /// <param name="variable_names_blacklist"></param>
        /// <returns>GraphDef containing a simplified version of the original.</returns>
        public GraphDef convert_variables_to_constants(Session sess,
                                   GraphDef input_graph_def,
                                   string[] output_node_names,
                                   string[] variable_names_whitelist = null,
                                   string[] variable_names_blacklist = null)
        {
            // This graph only includes the nodes needed to evaluate the output nodes, and
            // removes unneeded nodes like those involved in saving and assignment.
            var inference_graph = extract_sub_graph(input_graph_def, output_node_names);

            // Identify the ops in the graph.
            var map_name_to_node = new Dictionary<string, NodeDef>();
            inference_graph.Node.Select(x => map_name_to_node[x.Name] = x).ToArray();

            // Get list of variables.
            var variable_names = new List<string>();
            var variable_dict_names = new List<string>();

            foreach (var node in inference_graph.Node)
            {
                if(new string[] { "Variable", "VariableV2", "VarHandleOp" }.Contains(node.Op))
                {
                    var variable_name = node.Name;

                    variable_dict_names.Add(variable_name);
                    if (node.Op == "VarHandleOp")
                        variable_names.Add(variable_name + "/Read/ReadVariableOp:0");
                    else
                        variable_names.Add(variable_name + ":0");
                }
                else if (new string[] { "ReadVariableOp", "ResourceGather" }.Contains(node.Op))
                {
                    // There can be one or more Identity ops in between the ReadVariableOp and
                    // VarHandleOp.  Store the Identity ops with the associated dtypes.
                    var source_op_name = get_input_name(node);
                    while(map_name_to_node[source_op_name].Op == "Identity")
                    {
                        throw new NotImplementedException("map_name_to_node[source_op_name].Op");
                        /*resource_identity_types[source_op_name] = node.attr["dtype"];
                        source_op_name = get_input_name(map_name_to_node[source_op_name]);*/
                    }
                }
            }

            // Gets map of variables and the associated data.
            NDArray returned_variables = null;
            if (variable_names != null)
                returned_variables = sess.run(variable_names);

            var variables_data_map = new Dictionary<string, NDArray>();
            foreach(var (i, name) in enumerate(variable_dict_names))
                variables_data_map[name] = returned_variables[i];
            print($"Froze {len(returned_variables)} variables.");

            // Reconstruct the graph with constants in place of variables.
            var output_graph_def = new GraphDef();
            int how_many_converted = 0;
            foreach(var input_node in inference_graph.Node)
            {
                var output_node = new NodeDef();
                if (variables_data_map.ContainsKey(input_node.Name))
                {
                    var data = variables_data_map[input_node.Name];
                    output_node = create_const_op(input_node.Name, input_node.Attr["dtype"],
                                    data, data.shape);
                    how_many_converted += 1;
                }
                // else if (resource_identity_types.ContainsKey(input_node.Name))
                else if(input_node.Op == "ReadVariableOp")
                {
                    output_node.Op = "Identity";
                    output_node.Name = input_node.Name;
                    output_node.Input.AddRange(new[] { input_node.Input[0] });
                    output_node.Attr["T"] = input_node.Attr["dtype"];
                }
                else if (input_node.Op == "ResourceGather")
                {

                }
                else
                {
                    output_node.MergeFrom(input_node);
                }

                output_graph_def.Node.AddRange(new[] { output_node });
            }

            output_graph_def.Library = inference_graph.Library;
            print($"Converted {how_many_converted} variables to const ops.");
            return output_graph_def;
        }

        private NodeDef create_const_op(string node_name, AttrValue dtype, NDArray data, int[] data_shape = null)
        {
            var output_node = new NodeDef
            {
                Op = "Const",
                Name = node_name
            };
            output_node.Attr["dtype"] = dtype;
            output_node.Attr["value"] = new AttrValue()
            {
                Tensor = tensor_util.make_tensor_proto(
                data, dtype: dtype.Type.as_tf_dtype(), shape: data_shape)
            };

            return output_node;
        }

        /// <summary>
        /// Gets the name of the first input. Errors if suffix is not :0.
        /// </summary>
        /// <param name="node"></param>
        /// <returns></returns>
        private string get_input_name(NodeDef node)
        {
            var details = node.Input[0].Split(':');
            if (details.Length == 1 || int.Parse(details[1]) == 0)
                return details[0];
            // While it is valid for input tensors to have a suffix that is not :0, this
            // method is used to find the associated ops, not tensors, and therefore it
            // is not valid.
            throw new ValueError($"Tensor name '{node.Input[0]}' is invalid.");
        }


        private GraphDef extract_sub_graph(GraphDef graph_def, string[] dest_nodes)
        {
            var (name_to_input_name, name_to_node, name_to_seq_num) = _extract_graph_summary(
                graph_def);

            var nodes_to_keep = _bfs_for_reachable_nodes(dest_nodes, name_to_input_name);
            var nodes_to_keep_list = nodes_to_keep.OrderBy(n => name_to_seq_num[n]).ToArray();
            // Now construct the output GraphDef
            var output = new GraphDef();
            foreach (var n in nodes_to_keep_list)
                output.Node.Add(name_to_node[n]); // need deep clone?
            output.Library = graph_def.Library;
            output.Versions = graph_def.Versions;

            return output;
        }

        private string[] _bfs_for_reachable_nodes(string[] target_nodes, Dictionary<string, string[]> name_to_input_name)
        {
            var nodes_to_keep = new List<string>();
            var next_to_visit = target_nodes.Select(x => x).ToList();
            while(next_to_visit.Count > 0)
            {
                var node = next_to_visit[0];
                next_to_visit.RemoveAt(0);
                if (nodes_to_keep.Contains(node))
                    continue;
                nodes_to_keep.Add(node);
                if (name_to_input_name.Keys.Contains(node))
                    next_to_visit.AddRange(name_to_input_name[node]);
            }

            return nodes_to_keep.ToArray();
        }

        private (Dictionary<string, string[]>, Dictionary<string, NodeDef>, Dictionary<string, int>) _extract_graph_summary(GraphDef graph_def)
        {
            var name_to_input_name = new Dictionary<string, string[]>();
            var name_to_node = new Dictionary<string, NodeDef>();
            var name_to_seq_num = new Dictionary<string, int>();

            int seq = 0;
            foreach (var node in graph_def.Node)
            {
                var n = _node_name(node.Name);
                name_to_node[n] = node;
                name_to_input_name[n] = node.Input.Select(x => _node_name(x)).ToArray();
                name_to_seq_num[n] = seq;
                seq++;
            }

            return (name_to_input_name, name_to_node, name_to_seq_num);
        }

        private string _node_name(string n)
        {
            return n.StartsWith("^") ? n.Substring(1) : n.Split(':')[0];
        }

        private string get_input_name(string node)
        {
            throw new NotImplementedException("");
        }
    }
}
