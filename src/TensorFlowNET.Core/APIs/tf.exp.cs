using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor exp(Tensor x,
                string name = null) => gen_math_ops.exp(x, name);

    }
}
