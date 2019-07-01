using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor diag(Tensor diagonal, string name = null)
            => gen_array_ops.diag(diagonal, name: name);

        public static Tensor matmul(Tensor a, Tensor b) 
            => gen_math_ops.mat_mul(a, b);
    }
}
