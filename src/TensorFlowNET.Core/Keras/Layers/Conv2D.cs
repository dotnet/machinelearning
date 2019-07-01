using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations.Activation;

namespace Tensorflow.Keras.Layers
{
    public class Conv2D : Conv
    {
        public Conv2D(int filters,
            int[] kernel_size,
            int[] strides = null,
            string padding = "valid",
            string data_format = "channels_last",
            int[] dilation_rate = null,
            IActivation activation = null,
            bool use_bias = true,
            IInitializer kernel_initializer = null,
            IInitializer bias_initializer = null,
            bool trainable = true,
            string name = null) : base(2, 
                filters,
                kernel_size,
                strides: strides,
                padding: padding,
                data_format: data_format,
                dilation_rate: dilation_rate,
                activation: activation,
                use_bias: use_bias,
                kernel_initializer: kernel_initializer,
                bias_initializer: bias_initializer,
                trainable: trainable, 
                name: name)
        {

        }
    }
}
