using Google.Protobuf;
using Google.Protobuf.Collections;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow
{
    public partial class RefVariable : VariableV1, IProtoBuf<VariableDef, RefVariable>
    {
        public bool _in_graph_mode = true;
        public Tensor _initial_value;
        public string _graph_key;
        public bool _trainable;
        public Tensor _variable;
        public Tensor _snapshot;
        public bool _save_slice_info;

        private Operation _initializer_op;
        public override Operation initializer => _initializer_op;
        public override Operation op => _variable.op;
        public Graph graph => _variable.graph;
        public TF_DataType dtype => _variable.dtype;
        public TensorShape shape => tensor_util.to_shape(_variable.shape);

        public override string name => _variable.name;

        public RefVariable(object initial_value = null,
            bool trainable = true,
            List<string> collections = null,
            bool validate_shape = true,
            string caching_device = "",
            string name = null,
            VariableDef variable_def = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            string import_scope = "") : base(initial_value,
                    trainable,
                    collections,
                    validate_shape,
                    caching_device,
                    name,
                    dtype)
        {
            _in_graph_mode = true;

            if (variable_def != null)
            {
                if (initial_value != null)
                    throw new ValueError("variable_def and initial_value are mutually exclusive.");
                _init_from_proto(variable_def, import_scope: import_scope);
            }
            else
            {
                _init_from_args(initial_value, trainable, collections, validate_shape, caching_device, name, dtype);
            }
        }

        private void _init_from_proto(VariableDef variable_def, string import_scope = "")
        {
            var g = ops.get_default_graph();

            _variable = g.as_graph_element(
                ops.prepend_name_scope(variable_def.VariableName,
                                import_scope: import_scope)) as Tensor;

            _initializer_op = g.as_graph_element(
                ops.prepend_name_scope(variable_def.InitializerName,
                               import_scope: import_scope)) as Operation;

            // Tests whether initial_value_name exists first for backwards compatibility.
            if (!string.IsNullOrEmpty(variable_def.InitialValueName))
                _initial_value = g.as_graph_element(
                    ops.prepend_name_scope(variable_def.InitialValueName,
                                 import_scope: import_scope)) as Tensor;
            else
                _initial_value = null;

            _trainable = variable_def.Trainable;
            _snapshot = g.as_graph_element(
                ops.prepend_name_scope(variable_def.SnapshotName,
                               import_scope: import_scope)) as Tensor;

            if (variable_def.SaveSliceInfoDef != null)
                throw new NotImplementedException("save_slice_info_def");
            else
                ;// _save_slice_info = null;

            //_caching_device = null;
            //_constraint = null;
        }

        private void _init_from_args(object initial_value,
            bool trainable = true,
            List<string> collections = null,
            bool validate_shape = true,
            string caching_device = "",
            string name = null,
            TF_DataType dtype = TF_DataType.DtInvalid)
        {
            if (initial_value is null)
                throw new ValueError("initial_value must be specified.");

            var init_from_fn = initial_value.GetType().Name == "Func`1";

            if(collections == null)
            {
                collections = new List<string> { ops.GraphKeys.GLOBAL_VARIABLES };
            }

            // Store the graph key so optimizers know how to only retrieve variables from
            // this graph.
            _graph_key = ops.get_default_graph().graph_key;

            _trainable = trainable;
            if (trainable && !collections.Contains(ops.GraphKeys.TRAINABLE_VARIABLES))
                collections.Add(ops.GraphKeys.TRAINABLE_VARIABLES);

            ops.init_scope();
            var values = init_from_fn ? new object[0] : new object[] { initial_value };
            with(ops.name_scope(name, "Variable", values), scope =>
            {
                name = scope;
                if (init_from_fn)
                {
                    // Use attr_scope and device(None) to simulate the behavior of
                    // colocate_with when the variable we want to colocate with doesn't
                    // yet exist.
                    string true_name = ops._name_from_scope_name(name);
                    var attr = new AttrValue
                    {
                        List = new AttrValue.Types.ListValue()
                    };
                    attr.List.S.Add(ByteString.CopyFromUtf8($"loc:{true_name}"));
                    with(ops.name_scope("Initializer"), scope2 =>
                    {
                        _initial_value = (initial_value as Func<Tensor>)();
                        _initial_value = ops.convert_to_tensor(_initial_value, name: "initial_value", dtype: dtype);
                    });
                    _variable = state_ops.variable_op_v2(_initial_value.shape, _initial_value.dtype.as_base_dtype(), name: name);
                }
                // Or get the initial value from a Tensor or Python object.
                else
                {
                    _initial_value = ops.convert_to_tensor(initial_value, name: "initial_value");

                    var shape = _initial_value.shape;
                    dtype = _initial_value.dtype;
                    _variable = gen_state_ops.variable_v2(shape, dtype.as_base_dtype(), scope);
                }

                // Manually overrides the variable's shape with the initial value's.
                if (validate_shape)
                {
                    var initial_value_shape = _initial_value.GetShape();
                    if (!initial_value_shape.is_fully_defined())
                        throw new ValueError($"initial_value must have a shape specified: {_initial_value}");
                }

                // If 'initial_value' makes use of other variables, make sure we don't
                // have an issue if these other variables aren't initialized first by
                // using their initialized_value() method.
                var _initial_value2 = _try_guard_against_uninitialized_dependencies(_initial_value);

                _initializer_op = gen_state_ops.assign(_variable, _initial_value2, validate_shape).op;

                if (!String.IsNullOrEmpty(caching_device))
                {

                }
                else
                {
                    ops.colocate_with(_initializer_op);

                    _snapshot = gen_array_ops.identity(_variable, name = "read");
                }

                ops.add_to_collections(collections, this as VariableV1);
            });
        }

        public Tensor _ref() => _variable;

        public Tensor value() => _snapshot;

        public Tensor _AsTensor() => _snapshot;

        public Tensor _as_graph_element() => _variable;

        public Tensor _TensorConversionFunction(TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool as_ref = false)
        {
            if (as_ref)
                return _ref();
            else
                return value();
        }

        /// <summary>
        /// Attempt to guard against dependencies on uninitialized variables.
        /// </summary>
        /// <param name="initial_value"></param>
        private Tensor _try_guard_against_uninitialized_dependencies(Tensor initial_value)
        {
            return _safe_initial_value_from_tensor(initial_value, new Dictionary<string, Operation>());
        }

        /// <summary>
        /// Replace dependencies on variables with their initialized values.
        /// </summary>
        /// <param name="tensor">A `Tensor`. The tensor to replace.</param>
        /// <param name="op_cache">A dict mapping operation names to `Operation`s.</param>
        /// <returns>A `Tensor` compatible with `tensor`.</returns>
        private Tensor _safe_initial_value_from_tensor(Tensor tensor, Dictionary<string, Operation> op_cache)
        {
            var op = tensor.op;
            var new_op = op_cache.ContainsKey(op.name) ? op_cache[op.name] : null;
            if(new_op == null)
            {
                new_op = _safe_initial_value_from_op(op, op_cache);
                op_cache[op.name] = new_op;
            }
            return new_op.outputs[tensor.value_index];
        }

        private Operation _safe_initial_value_from_op(Operation op, Dictionary<string, Operation> op_cache)
        {
            var op_type = op.node_def.Op;
            switch (op_type)
            {
                case "IsVariableInitialized":
                case "VarIsInitializedOp":
                case "ReadVariableOp":
                    return op;
                case "Variable":
                case "VariableV2":
                case "VarHandleOp":
                    break;
            }

            // Recursively build initializer expressions for inputs.
            return op;
        }

        /// <summary>
        /// Assigns a new value to the variable.
        /// </summary>
        /// <param name="value">The new value for this variable.</param>
        /// <param name="use_locking">If `True`, use locking during the assignment.</param>
        /// <param name="name">The name of the operation to be created</param>
        /// <param name="read_value">
        /// if True, will return something which evaluates to the
        /// new value of the variable; if False will return the assign op.
        /// </param>
        /// <returns>
        /// A `Tensor` that will hold the new value of this variable after
        /// the assignment has completed.
        /// </returns>
        public ITensorOrOperation assign(object value, bool use_locking = false, string name = null, bool read_value = true)
        {
            var assign = gen_state_ops.assign(_variable, value, use_locking: use_locking, name: name);
            if (read_value)
                return assign;
            return assign.op;
        }

        public override string ToString()
        {
            return $"tf.RefVariable '{name}' shape={shape} dtype={dtype}";
        }

        public VariableDef to_proto(string export_scope)
        {
            if(string.IsNullOrEmpty(export_scope) || _variable.name.StartsWith(export_scope))
            {
                var var_def = new VariableDef();
                var_def.VariableName = ops.strip_name_scope(_variable.name, export_scope);
                if (_initial_value != null)
                    var_def.InitialValueName = ops.strip_name_scope(_initial_value.name, export_scope);
                var_def.Trainable = _trainable;
                var_def.InitializerName = ops.strip_name_scope(initializer.name, export_scope);
                var_def.SnapshotName = ops.strip_name_scope(_snapshot.name, export_scope);
                if (_save_slice_info)
                    throw new NotImplementedException("to_proto _save_slice_info");

                return var_def;
            }

            throw new NotImplementedException("to_proto RefVariable");
        }

        public RefVariable from_proto(VariableDef proto, string import_scope)
        {
            throw new NotImplementedException();
        }
    }
}
