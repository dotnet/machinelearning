using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public interface IPoolFunction
    {
        Tensor Apply(Tensor value,
            int[] ksize,
            int[] strides,
            string padding,
            string data_format = "NHWC",
            string name = null);
    }
}
