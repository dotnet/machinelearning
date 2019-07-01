using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor reshape(Tensor tensor,
                Tensor shape,
                string name = null) => gen_array_ops.reshape(tensor, shape, name);

        public static Tensor reshape(Tensor tensor,
                int[] shape,
                string name = null) => gen_array_ops.reshape(tensor, shape, name);
    }
}
