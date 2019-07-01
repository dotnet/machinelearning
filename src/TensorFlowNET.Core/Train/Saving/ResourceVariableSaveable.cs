using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class ResourceVariableSaveable : SaveableObject
    {
        string _var_device;
        int[] _var_shape;
        Tensor handle_op;

        public ResourceVariableSaveable(Tensor var, string slice_spec, string name)
        {
            _var_device = var.Device;
            _var_shape = var.shape;
            handle_op = var.op.inputs[0];
            var tensor = var;
            var spec = new SaveSpec(tensor, slice_spec, name, dtype: var.dtype);

            op = var;
            specs = new SaveSpec[] { spec };
            this.name = name;
        }

        public override ITensorOrOperation restore(Tensor[] restored_tensors, TensorShape[] restored_shapes = null)
        {
            var restored_tensor = restored_tensors[0];
            restored_tensor = array_ops.identity(restored_tensor);
            return resource_variable_ops.shape_safe_assign_variable_handle(
                handle_op, _var_shape, restored_tensor);
        }
    }
}
