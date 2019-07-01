using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class SaveableObject
    {
        public Tensor op;
        public SaveSpec[] specs;
        public string name;
        public string device;

        public SaveableObject()
        {

        }

        public SaveableObject(Tensor var, string slice_spec, string name)
        {

        }

        public SaveableObject(Tensor op, SaveSpec[] specs, string name)
        {
            this.op = op;
            this.specs = specs;
            this.name = name;
        }

        public virtual ITensorOrOperation restore(Tensor[] restored_tensors, TensorShape[] restored_shapes = null)
        {
            var restored_tensor = restored_tensors[0];
            return gen_state_ops.assign(op,
                restored_tensor,
                validate_shape: restored_shapes == null && tensor_util.to_shape(op.shape).is_fully_defined());
        }
    }
}
