using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations.Activation;

namespace Tensorflow.Layers
{
    public class Dense : Keras.Layers.Dense
    {
        public Dense(int units,
            IActivation activation,
            bool use_bias = true,
            bool trainable = false,
            IInitializer kernel_initializer = null) : base(units,
                activation,
                use_bias: use_bias,
                trainable: trainable,
                kernel_initializer: kernel_initializer)
        {

        }
    }
}
