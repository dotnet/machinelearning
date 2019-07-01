using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using Tensorflow.Operations;
using Tensorflow.Operations.Activation;

namespace Tensorflow.Keras.Layers
{
    public class Conv : Tensorflow.Layers.Layer
    {
        protected int rank;
        protected int filters;
        protected int[] kernel_size;
        protected int[] strides;
        protected string padding;
        protected string data_format;
        protected int[] dilation_rate;
        protected IActivation activation;
        protected bool use_bias;
        protected IInitializer kernel_initializer;
        protected IInitializer bias_initializer;
        protected RefVariable kernel;
        protected RefVariable bias;
        protected Convolution _convolution_op;

        public Conv(int rank, 
            int filters,
            int[] kernel_size,
            int[] strides = null,
            string padding = "valid",
            string data_format = null,
            int[] dilation_rate = null,
            IActivation activation = null,
            bool use_bias = true,
            IInitializer kernel_initializer = null,
            IInitializer bias_initializer = null,
            bool trainable = true, 
            string name = null) : base(trainable: trainable, name: name)
        {
            this.rank = rank;
            this.filters = filters;
            this.kernel_size = kernel_size;
            this.strides = strides;
            this.padding = padding;
            this.data_format = data_format;
            this.dilation_rate = dilation_rate;
            this.activation = activation;
            this.use_bias = use_bias;
            this.kernel_initializer = kernel_initializer;
            this.bias_initializer = bias_initializer;
            input_spec = new InputSpec(ndim: rank + 2);
        }

        protected override void build(TensorShape input_shape)
        {
            int channel_axis = data_format == "channels_first" ? 1 : -1;
            int input_dim = channel_axis < 0 ? 
                input_shape.Dimensions[input_shape.NDim + channel_axis] : 
                input_shape.Dimensions[channel_axis];
            var kernel_shape = new int[] { kernel_size[0], kernel_size[1], input_dim, filters };
            kernel = add_weight(name: "kernel",
                shape: kernel_shape,
                initializer: kernel_initializer,
                trainable: true,
                dtype: _dtype);
            if (use_bias)
                bias = add_weight(name: "bias",
                    shape: new int[] { filters },
                    initializer: bias_initializer,
                    trainable: true,
                    dtype: _dtype);

            var axes = new Dictionary<int, int>();
            axes.Add(-1, input_dim);
            input_spec = new InputSpec(ndim: rank + 2, axes: axes);

            string op_padding;
            if (padding == "causal")
                op_padding = "valid";
            else
                op_padding = padding;

            var df = conv_utils.convert_data_format(data_format, rank + 2);
            _convolution_op = nn_ops.Convolution(input_shape,
                kernel.shape,
                op_padding.ToUpper(),
                strides,
                dilation_rate,
                data_format: df);

            built = true;
        }

        protected override Tensor call(Tensor inputs, Tensor training = null)
        {
            var outputs = _convolution_op.__call__(inputs, kernel);
            if (use_bias)
            {
                if (data_format == "channels_first")
                {
                    throw new NotImplementedException("call channels_first");
                }
                else
                {                    
                    outputs = nn_ops.bias_add(outputs, bias, data_format: "NHWC");
                }
            }

            if (activation != null)
                return activation.Activate(outputs);

            return outputs;
        }
    }
}
