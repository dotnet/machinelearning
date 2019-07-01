using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.tf;

namespace Tensorflow.Keras.Layers
{
    public class MaxPooling2D : Pooling2D
    {
        public MaxPooling2D(
            int[] pool_size,
            int[] strides,
            string padding = "valid",
            string data_format = null,
            string name = null) : base(nn.max_pool, pool_size,
                strides,
                padding: padding,
                data_format: data_format,
                name: name)
        {

        }
    }
}
