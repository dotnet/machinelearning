using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Variable based on resource handles.
    /// </summary>
    public class ResourceVariable : VariableV1
    {
        bool _in_graph_mode;
        Tensor _handle;
        TensorShape _shape;
        public TensorShape shape => _shape;
        string _handle_name;
        string _unique_id;
        Operation _initializer_op;
        public override Operation initializer => _initializer_op;
        Tensor _initial_value;
        bool _trainable;
        public bool tranable => _trainable;
        Tensor _cached_value;
        Tensor _graph_element;
        public override Tensor graph_element => _graph_element;
        TF_DataType _dtype;
        public TF_DataType dtype => _dtype;
        public override string name => _handle.name;
        public string Device => _handle.Device;
        public Graph Graph => _handle.graph;
        public override Operation op => _handle.op;

        public ResourceVariable(object initial_value = null,
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
            if (variable_def != null)
            {
                if (initial_value != null)
                    throw new ValueError("variable_def and initial_value are mutually exclusive.");
                _init_from_proto(variable_def, import_scope: import_scope);
            }
            else
            {
                throw new NotImplementedException("ResourceVariable _init_from_args");
                //_init_from_args(initial_value, trainable, collections, validate_shape, caching_device, name, dtype);
            }
        }

        private void _init_from_proto(VariableDef variable_def, string import_scope = null)
        {
            _in_graph_mode = true;
            if (!variable_def.IsResource)
                throw new ValueError("Trying to restore Variable as ResourceVariable.");

            // Create from variable_def.
            var g = ops.get_default_graph();
            var prepend_name_scope = ops.prepend_name_scope(variable_def.VariableName, import_scope: import_scope);
            _handle = g.as_graph_element(prepend_name_scope) as Tensor;
            _shape = new TensorShape(_handle.op.get_attr("shape") as TensorShapeProto);
            _handle_name = _handle.name;
            _unique_id = _handle_name;
            prepend_name_scope = ops.prepend_name_scope(variable_def.InitializerName, import_scope: import_scope);
            _initializer_op = g.as_graph_element(prepend_name_scope) as Operation;
            if (!string.IsNullOrEmpty(variable_def.InitialValueName))
            {
                prepend_name_scope = ops.prepend_name_scope(variable_def.InitialValueName, import_scope: import_scope);
                _initial_value = g.as_graph_element(prepend_name_scope) as Tensor;
            }

            _trainable = variable_def.Trainable;
            /*var (synchronization, aggregation, trainable) =
        variables.validate_synchronization_aggregation_trainable(
            variable_def.Synchronization,
            variable_def.Aggregation,
            variable_def.Trainable,
            variable_def.VariableName);*/
            if (!string.IsNullOrEmpty(variable_def.SnapshotName))
            {
                prepend_name_scope = ops.prepend_name_scope(variable_def.SnapshotName, import_scope: import_scope);
                var snapshot = g.as_graph_element(prepend_name_scope) as Tensor;
                if (snapshot.op.type != "ReadVariableOp")
                    _cached_value = snapshot;
                while (snapshot.op.type != "ReadVariableOp")
                    snapshot = snapshot.op.inputs[0];
                _graph_element = snapshot;
            }
            else
            {
                throw new NotImplementedException("SnapshotName _init_from_proto");
            }

            if (variable_def.SaveSliceInfoDef != null)
            {
                throw new NotImplementedException("SaveSliceInfoDef _init_from_proto");
            }

            _dtype = dtypes.as_tf_dtype((DataType)_handle.op.get_attr("dtype"));
        }

        public override string ToString()
        {
            return $"tf.ResourceVariable '{name}' shape={shape} dtype={dtype}";
        }
    }
}
