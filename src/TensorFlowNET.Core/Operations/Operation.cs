using Google.Protobuf.Collections;
#if GRAPH_SERIALIZE
using Newtonsoft.Json;
#endif
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{

    /// <summary>
    /// Represents a graph node that performs computation on tensors.
    /// 
    /// An `Operation` is a node in a TensorFlow `Graph` that takes zero or
    /// more `Tensor` objects as input, and produces zero or more `Tensor`
    /// objects as output. Objects of type `Operation` are created by
    /// calling an op constructor(such as `tf.matmul`)
    /// or `tf.Graph.create_op`.
    /// 
    /// For example `c = tf.matmul(a, b)` creates an `Operation` of type
    /// "MatMul" that takes tensors `a` and `b` as input, and produces `c`
    /// as output.
    /// 
    /// After the graph has been launched in a session, an `Operation` can
    /// be executed by passing it to
    /// `tf.Session.run`.
    /// `op.run()` is a shortcut for calling `tf.get_default_session().run(op)`.
    /// </summary>
    public partial class Operation : ITensorOrOperation
    {
        private readonly IntPtr _handle; // _c_op in python
        private readonly IntPtr _operDesc; 

        private Graph _graph;
        public string type => OpType;

#if GRAPH_SERIALIZE
        [JsonIgnore]
        public Graph graph => _graph;
        [JsonIgnore]
        public int _id => _id_value;
        [JsonIgnore]
        public int _id_value;
        [JsonIgnore]
        public Operation op => this;
#else
        public Graph graph => _graph;
        public int _id => _id_value;
        public int _id_value;
        public Operation op => this;
#endif
        public TF_DataType dtype => TF_DataType.DtInvalid;
        private Status status = new Status();

        public string name => _handle == IntPtr.Zero ? null : c_api.StringPiece(c_api.TF_OperationName(_handle));
        public string OpType => c_api.StringPiece(c_api.TF_OperationOpType(_handle));
        public string Device => c_api.StringPiece(c_api.TF_OperationDevice(_handle));

        private NodeDef _node_def;
#if GRAPH_SERIALIZE
        [JsonIgnore]
#endif
        public NodeDef node_def
        {
            get
            {
                if(_node_def == null)
                    _node_def = GetNodeDef();

                return _node_def;
            }
        }

        public Operation(IntPtr handle, Graph g=null)
        {
            if (handle == IntPtr.Zero)
                return;

            _handle = handle;
            _graph = g ?? ops.get_default_graph();
            _outputs = new Tensor[NumOutputs];
            for (int i = 0; i < NumOutputs; i++)
                _outputs[i] = new Tensor(this, i, OutputType(i));

            // Dict mapping op name to file and line information for op colocation
            // context managers.
            _control_flow_context = graph._get_control_flow_context();

            // Note: _control_flow_post_processing() must not be called here, the caller is responsible for calling it when using this constructor.
        }

        public Operation(Graph g, string opType, string oper_name)
        {
            _graph = g;

            _operDesc = c_api.TF_NewOperation(g, opType, oper_name);
            c_api.TF_SetAttrType(_operDesc, "dtype", TF_DataType.TF_INT32);
            _handle = c_api.TF_FinishOperation(_operDesc, status);

            // Dict mapping op name to file and line information for op colocation
            // context managers.
            _control_flow_context = graph._get_control_flow_context();
        }

        /// <summary>
        /// Creates an `Operation`.
        /// </summary>
        /// <param name="node_def">`node_def_pb2.NodeDef`.  `NodeDef` for the `Operation`.</param>
        /// <param name="g">`Graph`. The parent graph.</param>
        /// <param name="inputs">list of `Tensor` objects. The inputs to this `Operation`.</param>
        /// <param name="output_types">list of `DType` objects.</param>
        /// <param name="control_inputs">
        /// list of operations or tensors from which to have a
        /// control dependency.
        /// </param>
        /// <param name="input_types">
        /// List of `DType` objects representing the
        /// types of the tensors accepted by the `Operation`. By default
        /// uses `[x.dtype.base_dtype for x in inputs]`.  Operations that expect
        /// reference-typed inputs must specify these explicitly.
        /// </param>
        /// <param name="original_op"></param>
        /// <param name="op_def"></param>
        public Operation(NodeDef node_def, Graph g, Tensor[] inputs = null, TF_DataType[] output_types = null, ITensorOrOperation[] control_inputs = null, TF_DataType[] input_types = null, string original_op = "", OpDef op_def = null)
        {
            _graph = g;

            // Build the list of control inputs.
            var control_input_ops = new List<Operation>();
            if(control_inputs != null)
            {
                foreach(var c in control_inputs)
                {
                    switch (c)
                    {
                        case Operation c1:
                            control_input_ops.Add(c1);
                            break;
                        case Tensor tensor:
                            control_input_ops.Add(tensor.op);
                            break;
                        // TODO: IndexedSlices don't yet exist, but once they do, this needs to be uncommented
                        //case IndexedSlices islices:
                        //    control_input_ops.Add(islices.op);
                        //    break;
                        default:
                            throw new NotImplementedException($"Control input must be an Operation, a Tensor, or IndexedSlices: {c}");
                    }
                }
            }

            // Dict mapping op name to file and line information for op colocation
            // context managers.
            _control_flow_context = graph._get_control_flow_context();

            // This will be set by self.inputs.
            if (op_def == null)
                op_def = g.GetOpDef(node_def.Op);

            var grouped_inputs = _reconstruct_sequence_inputs(op_def, inputs, node_def.Attr);
            (_handle, _operDesc) = ops._create_c_op(g, node_def, grouped_inputs, control_input_ops.ToArray());

            // Initialize self._outputs.
            output_types = new TF_DataType[NumOutputs];
            for (int i = 0; i < NumOutputs; i++)
                output_types[i] = OutputType(i);

            _outputs = new Tensor[NumOutputs];
            for (int i = 0; i < NumOutputs; i++)
                _outputs[i] = new Tensor(this, i, OutputType(i));

            graph._add_op(this);

            if (_handle != IntPtr.Zero)
                _control_flow_post_processing();
        }

        public void run(FeedItem[] feed_dict = null, Session session = null)
        {
            ops._run_using_default_session(this, feed_dict, graph, session);
        }

        private object[] _reconstruct_sequence_inputs(OpDef op_def, Tensor[] inputs, MapField<string, AttrValue> attrs)
        {
            var grouped_inputs = new List<object>();
            int i = 0;
            int input_len = 0;
            bool is_sequence = false;
            foreach (var input_arg in op_def.InputArg)
            {
                if (!string.IsNullOrEmpty(input_arg.NumberAttr))
                {
                    input_len = (int)attrs[input_arg.NumberAttr].I;
                    is_sequence = true;
                }
                else if (!string.IsNullOrEmpty(input_arg.TypeListAttr))
                {
                    input_len = attrs[input_arg.TypeListAttr].List.Type.Count;
                    is_sequence = true;
                }
                else
                {
                    input_len = 1;
                    is_sequence = false;
                }

                if (is_sequence)
                    grouped_inputs.Add(inputs.Skip(i).Take(input_len).ToArray());
                else
                    grouped_inputs.Add(inputs[i]);

                i += input_len;
            }

            return grouped_inputs.ToArray();
        }

        public object get_attr(string name)
        {
            AttrValue x = null;

            using (var buf = new Buffer())
            {
                c_api.TF_OperationGetAttrValueProto(_handle, name, buf, status);
                status.Check(true);
                x = AttrValue.Parser.ParseFrom(buf);
            }

            string oneof_value = x.ValueCase.ToString();
            if (string.IsNullOrEmpty(oneof_value))
                return null;

            if(oneof_value == "list")
                throw new NotImplementedException($"Unsupported field type in {x.ToString()}");

            if (oneof_value == "type")
                return x.Type;

            object result = x.GetType().GetProperty(oneof_value).GetValue(x);
            if (result is Google.Protobuf.ByteString byteString)
                return byteString.ToStringUtf8();
            return result;
        }

        public TF_AttrMetadata GetAttributeMetadata(string attr_name, Status s)
        {
            return c_api.TF_OperationGetAttrMetadata(_handle, attr_name, s);
        }

        private NodeDef GetNodeDef()
        {
            using (var s = new Status())
            using (var buffer = new Buffer())
            {
                c_api.TF_OperationToNodeDef(_handle, buffer, s);
                s.Check();
                return NodeDef.Parser.ParseFrom(buffer);
            }
        }

        /// <summary>
        /// Update the input to this operation at the given index.
        /// 
        /// NOTE: This is for TF internal use only.Please don't use it.
        /// </summary>
        /// <param name="index">the index of the input to update.</param>
        /// <param name="tensor"> the Tensor to be used as the input at the given index.</param>
        public void _update_input(int index, Tensor tensor)
        {
            _assert_same_graph(tensor);

            var input = _tf_input(index);
            var output = tensor._as_tf_output();

            // Reset cached inputs.
            _inputs = null; 
            // after the c_api call next time _inputs is accessed 
            // the updated inputs are reloaded from the c_api
            c_api.UpdateEdge(_graph, output, input, status);
            //var updated_inputs = inputs;
            status.Check();
        }

        private void _assert_same_graph(Tensor tensor)
        {
            //TODO: implement
        }

        /// <summary>
        /// Create and return a new TF_Output for output_idx'th output of this op.
        /// </summary>
        public TF_Output _tf_output(int output_idx)
        {
            var tf_output =  new TF_Output(op, output_idx);
            return tf_output;
        }

        /// <summary>
        /// Create and return a new TF_Input for input_idx'th input of this op.
        /// </summary>
        public TF_Input _tf_input(int input_idx)
        {
            var tf_input = new TF_Input(op, input_idx);
            return tf_input;
        }

        public TensorShape GetTensorShape()
        {
            
            // This should not be handled this way but Status cstatus = new Status(); Status.IntPtr(cstatus); resulted in Status.IntPtr not found
            TF_Output tf_output = _tf_output(0);
            IntPtr cstatus = c_api.TF_NewStatus();


            var n = c_api.TF_GraphGetTensorNumDims(_graph, tf_output, cstatus);
            // Does Status check because I was unable to call the Status.IntPtr()
            if (c_api.TF_GetCode(cstatus) != TF_Code.TF_OK)
            {
                Console.WriteLine("Status of TF_GetCode is not TF_OK");
                throw new Exception("Status of TF_GetCode is not TF_OK");
            }
            var dims = new long[n];
            c_api.TF_GraphGetTensorShape(_graph, tf_output, dims, dims.Length, cstatus);
            // Does Status check because I was unable to call the Status.IntPtr()
            if (c_api.TF_GetCode(cstatus) != TF_Code.TF_OK)
            {
                Console.WriteLine("Status of TF_GetCode is not TF_OK");
                throw new Exception("Status of TF_GetCode is not TF_OK");
            }
            return new TensorShape(dims.Select(item => (int)item).ToArray());

        }
    }
}
