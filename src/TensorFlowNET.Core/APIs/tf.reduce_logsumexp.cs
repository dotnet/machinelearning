using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor reduce_logsumexp(Tensor input_tensor,
                int[] axis = null,
                bool keepdims = false,
                string name = null) => math_ops.reduce_logsumexp(input_tensor, axis, keepdims, name);

    }
}
