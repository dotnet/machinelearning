using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Variable store that carries a number of named Variables.
    /// </summary>
    public class _VariableStore
    {
        private Dictionary<string, object> _vars;
        private Dictionary<string, object> _partitioned_vars;
        private bool _store_eager_variables;

        public _VariableStore()
        {
            _vars = new Dictionary<string, object>();
            _partitioned_vars = new Dictionary<string, object>();
            _store_eager_variables = false;
        }

        public RefVariable get_variable(string name,
            TensorShape shape = null,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            object initializer = null, // IInitializer or Tensor
            bool? reuse = null,
            bool? trainable = null,
            bool validate_shape = true,
            VariableSynchronization synchronization = VariableSynchronization.Auto,
            VariableAggregation aggregation = VariableAggregation.None)
        {
            dtype = dtype.as_base_dtype();
            trainable = variable_scope._get_trainable_value(synchronization, trainable);

            return _true_getter(name, 
                shape: shape, 
                dtype: dtype, 
                initializer: initializer,
                trainable: trainable,
                validate_shape: validate_shape,
                synchronization: synchronization,
                aggregation: aggregation);
        }

        private RefVariable _true_getter(string name,
            TensorShape shape = null,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            object initializer = null,
            bool? trainable = null,
            bool validate_shape = true,
            VariableSynchronization synchronization = VariableSynchronization.Auto,
            VariableAggregation aggregation = VariableAggregation.None)
        {
            bool is_scalar = !(shape is null) && shape.NDim == 0;

            if (initializer is IInitializer init)
            {
                return _get_single_variable(name: name,
                    shape: shape,
                    dtype: dtype,
                    initializer: init,
                    trainable: trainable,
                    validate_shape: validate_shape,
                    synchronization: synchronization,
                    aggregation: aggregation);
            }
            else if (initializer is Tensor tensor)
            {
                return _get_single_variable(name: name,
                    shape: shape,
                    dtype: dtype,
                    initializer: tensor,
                    trainable: trainable,
                    validate_shape: validate_shape,
                    synchronization: synchronization,
                    aggregation: aggregation);
            }
            else
            {
                IInitializer init1 = null;
                return _get_single_variable(name: name,
                    shape: shape,
                    dtype: dtype,
                    initializer: init1,
                    trainable: trainable,
                    validate_shape: validate_shape,
                    synchronization: synchronization,
                    aggregation: aggregation);
            }
        }

        private RefVariable _get_single_variable(string name,
            TensorShape shape = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            IInitializer initializer = null,
            bool reuse = false,
            bool? trainable = null,
            bool validate_shape = false,
            bool? use_resource = null,
            VariableSynchronization synchronization = VariableSynchronization.Auto,
            VariableAggregation aggregation = VariableAggregation.None)
        {
            bool initializing_from_value = false;
            if (use_resource == null)
                use_resource = false;

            if (_vars.ContainsKey(name))
            {
                if (!reuse)
                {
                    var var = _vars[name];

                }
                throw new NotImplementedException("_get_single_variable");
            }

            RefVariable v = null;
            // Create the tensor to initialize the variable with default value.
            if (initializer == null)
            {
                if (dtype.is_floating())
                {
                    initializer = tf.glorot_uniform_initializer;
                    initializing_from_value = false;
                }
            }

            // Create the variable.
            ops.init_scope();
            {
                if (initializing_from_value)
                {

                }
                else
                {
                    Func<Tensor> init_val = () => initializer.call(shape, dtype);
                    var variable_dtype = dtype.as_base_dtype();

                    v = variable_scope.default_variable_creator(init_val,
                        name: name,
                        trainable: trainable,
                        dtype: variable_dtype,
                        validate_shape: validate_shape,
                        synchronization: synchronization,
                        aggregation: aggregation);
                }
            }

            _vars[name] = v;

            return v;
        }

        private RefVariable _get_single_variable(string name,
            TensorShape shape = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            Tensor initializer = null,
            bool reuse = false,
            bool? trainable = null,
            bool validate_shape = false,
            bool? use_resource = null,
            VariableSynchronization synchronization = VariableSynchronization.Auto,
            VariableAggregation aggregation = VariableAggregation.None)
        {
            if (use_resource == null)
                use_resource = false;

            if (_vars.ContainsKey(name))
            {
                if (!reuse)
                {
                    var var = _vars[name];

                }
                throw new NotImplementedException("_get_single_variable");
            }

            RefVariable v = null;
            // Create the variable.
            ops.init_scope();
            {
                var init_val = initializer;
                v = new RefVariable(init_val,
                    name: name,
                    validate_shape: validate_shape,
                    trainable: trainable.Value);
            }

            _vars[name] = v;

            return v;
        }
    }
}
