using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Layers;
using Tensorflow.Operations.Activation;

namespace Tensorflow
{
    public static partial class tf
    {
        public static class layers
        {
            public static Tensor conv2d(Tensor inputs,
                int filters,
                int[] kernel_size,
                int[] strides = null,
                string padding = "valid",
                string data_format= "channels_last",
                int[] dilation_rate = null,
                bool use_bias = true,
                IActivation activation = null,
                IInitializer kernel_initializer = null,
                IInitializer bias_initializer = null,
                bool trainable = true,
                string name = null)
            {
                if (strides == null)
                    strides = new int[] { 1, 1 };
                if (dilation_rate == null)
                    dilation_rate = new int[] { 1, 1 };
                if (bias_initializer == null)
                    bias_initializer = tf.zeros_initializer;

                var layer = new Conv2D(filters,
                    kernel_size: kernel_size,
                    strides: strides,
                    padding: padding,
                    data_format: data_format,
                    dilation_rate: dilation_rate,
                    activation: activation,
                    use_bias: use_bias,
                    kernel_initializer: kernel_initializer,
                    bias_initializer: bias_initializer,
                    trainable: trainable,
                    name: name);

                return layer.apply(inputs);
            }

            /// <summary>
            /// Functional interface for the batch normalization layer.
            /// http://arxiv.org/abs/1502.03167
            /// </summary>
            /// <param name="inputs"></param>
            /// <param name="axis"></param>
            /// <param name="momentum"></param>
            /// <param name="epsilon"></param>
            /// <param name="center"></param>
            /// <param name="scale"></param>
            /// <param name="beta_initializer"></param>
            /// <param name="gamma_initializer"></param>
            /// <param name="moving_mean_initializer"></param>
            /// <param name="moving_variance_initializer"></param>
            /// <param name="training"></param>
            /// <param name="trainable"></param>
            /// <param name="name"></param>
            /// <param name="renorm"></param>
            /// <param name="renorm_momentum"></param>
            /// <returns></returns>
            public static Tensor batch_normalization(Tensor inputs,
                int axis = -1,
                float momentum = 0.99f,
                float epsilon = 0.001f,
                bool center = true,
                bool scale = true,
                IInitializer beta_initializer = null,
                IInitializer gamma_initializer = null,
                IInitializer moving_mean_initializer = null,
                IInitializer moving_variance_initializer = null,
                Tensor training = null,
                bool trainable = true,
                string name = null,
                bool renorm = false,
                float renorm_momentum = 0.99f)
            {
                var layer = new BatchNormalization(
                    axis: axis,
                    momentum: momentum,
                    epsilon: epsilon,
                    center: center,
                    scale: scale,
                    beta_initializer: beta_initializer,
                    gamma_initializer: gamma_initializer,
                    moving_mean_initializer: moving_mean_initializer,
                    moving_variance_initializer: moving_variance_initializer,
                    renorm: renorm,
                    renorm_momentum: renorm_momentum,
                    trainable: trainable,
                    name: name);

                return layer.apply(inputs, training: training);
            }

            /// <summary>
            /// Max pooling layer for 2D inputs (e.g. images).
            /// </summary>
            /// <param name="inputs">The tensor over which to pool. Must have rank 4.</param>
            /// <param name="pool_size"></param>
            /// <param name="strides"></param>
            /// <param name="padding"></param>
            /// <param name="data_format"></param>
            /// <param name="name"></param>
            /// <returns></returns>
            public static Tensor max_pooling2d(Tensor inputs,
                int[] pool_size,
                int[] strides,
                string padding = "valid",
                string data_format = "channels_last",
                string name = null)
            {
                var layer = new MaxPooling2D(pool_size: pool_size,
                    strides: strides,
                    padding: padding,
                    data_format: data_format,
                    name: name);

                return layer.apply(inputs);
            }

            public static Tensor dense(Tensor inputs,
                int units,
                IActivation activation = null,
                bool use_bias = true,
                IInitializer kernel_initializer = null,
                IInitializer bias_initializer = null,
                bool trainable = true,
                string name = null,
                bool? reuse = null)
            {
                if (bias_initializer == null)
                    bias_initializer = tf.zeros_initializer;

                var layer = new Dense(units, activation, 
                    use_bias: use_bias,
                    bias_initializer: bias_initializer,
                    kernel_initializer: kernel_initializer);

                return layer.apply(inputs);
            }
        }
    }
}
