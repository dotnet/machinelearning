using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor convert_to_tensor(object value,
            string name = null) => ops.convert_to_tensor(value, name: name);
    }
}
