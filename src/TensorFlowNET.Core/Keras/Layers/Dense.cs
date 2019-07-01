using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Keras.Engine;
using Tensorflow.Operations.Activation;
using static Tensorflow.tf;

namespace Tensorflow.Keras.Layers
{
    public class Dense : Tensorflow.Layers.Layer
    {
        protected int units;
        protected IActivation activation;
        protected bool use_bias;
        protected IInitializer kernel_initializer;
        protected IInitializer bias_initializer;
        protected RefVariable kernel;
        protected RefVariable bias;

        public Dense(int units,
            IActivation activation,
            bool use_bias = true,
            bool trainable = false,
            IInitializer kernel_initializer = null,
            IInitializer bias_initializer = null) : base(trainable: trainable)
        {
            this.units = units;
            this.activation = activation;
            this.use_bias = use_bias;
            this.kernel_initializer = kernel_initializer;
            this.bias_initializer = bias_initializer;
            this.supports_masking = true;
            this.input_spec = new InputSpec(min_ndim: 2);
        }

        protected override void build(TensorShape input_shape)
        {
            var last_dim = input_shape.Dimensions.Last();
            var axes = new Dictionary<int, int>();
            axes[-1] = last_dim;
            input_spec = new InputSpec(min_ndim: 2, axes: axes);
            kernel = add_weight(
                "kernel",
                shape: new int[] { last_dim, units },
                initializer: kernel_initializer,
                dtype: _dtype,
                trainable: true);
            if (use_bias)
                bias = add_weight(
                  "bias",
                  shape: new int[] { units },
                  initializer: bias_initializer,
                  dtype: _dtype,
                  trainable: true);

            built = true;
        }

        protected override Tensor call(Tensor inputs, Tensor training = null)
        {
            Tensor outputs = null;
            var rank = inputs.rank;
            if(rank > 2)
            {
                throw new NotImplementedException("call rank > 2");
            }
            else
            {
                outputs = gen_math_ops.mat_mul(inputs, kernel);
            }

            if (use_bias)
                outputs = nn.bias_add(outputs, bias);
            if (activation != null)
                return activation.Activate(outputs);

            return outputs;
        }
    }
}
