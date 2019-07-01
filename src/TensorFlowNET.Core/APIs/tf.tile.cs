using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor tile(Tensor input,
                Tensor multiples,
                string name = null) => gen_array_ops.tile(input, multiples, name);
        public static Tensor tile(NDArray input,
                int[] multiples,
                string name = null) => gen_array_ops.tile(input, multiples, name);

    }
}
