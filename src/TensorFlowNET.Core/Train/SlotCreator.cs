using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations.Initializers;
using static Tensorflow.Python;

namespace Tensorflow.Train
{
    public class SlotCreator
    {
        /// <summary>
        /// Create a slot initialized to 0 with same shape as the primary object.
        /// </summary>
        /// <param name="primary"></param>
        /// <param name="name"></param>
        /// <param name="dtype"></param>
        /// <param name="colocate_with_primary"></param>
        /// <returns></returns>
        public RefVariable create_zeros_slot(RefVariable primary, string name, TF_DataType dtype = TF_DataType.DtInvalid, bool colocate_with_primary = true)
        {
            if (dtype == TF_DataType.DtInvalid)
                dtype = primary.dtype;
            var slot_shape = primary.shape;
            if (slot_shape.is_fully_defined())
            {
                var initializer = new Zeros();
                return create_slot_with_initializer(
                    primary, initializer, slot_shape, dtype, name,
                    colocate_with_primary: colocate_with_primary);
            }
            else
            {
                throw new NotImplementedException("create_zeros_slot is not fully defined.");
            }
        }

        /// <summary>
        /// Creates a slot initialized using an `Initializer`.
        /// </summary>
        /// <returns></returns>
        public RefVariable create_slot_with_initializer(RefVariable primary, IInitializer initializer, TensorShape shape, 
            TF_DataType dtype, string name, bool colocate_with_primary = true)
        {
            var validate_shape = shape.is_fully_defined();
            var prefix = primary.op.name;
            return with(new variable_scope(string.Empty, prefix + "/" + name), delegate
            {
                return _create_slot_var(primary, initializer, "", validate_shape, shape, dtype);
            });
        }

        /// <summary>
        /// Helper function for creating a slot variable.
        /// </summary>
        /// <param name="primary"></param>
        /// <param name="val"></param>
        /// <param name="scope"></param>
        /// <param name="validate_shape"></param>
        /// <param name="shape"></param>
        /// <param name="dtype"></param>
        /// <returns></returns>
        private RefVariable _create_slot_var(VariableV1 primary, IInitializer val, string scope, bool validate_shape, 
            TensorShape shape, TF_DataType dtype)
        {
            bool use_resource = primary is ResourceVariable;
            if (resource_variable_ops.is_resource_variable(primary))
                use_resource = true;

            var slot = tf.get_variable(
              scope,
              initializer: val,
              trainable: false,
              use_resource: use_resource,
              shape: shape,
              dtype: dtype,
              validate_shape: validate_shape);

            return slot;
        }
    }
}
