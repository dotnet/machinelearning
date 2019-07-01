using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;
using static Tensorflow.Python;

namespace Tensorflow
{
    public class nn_ops
    {
        public static Convolution Convolution(TensorShape input_shape,
            TensorShape filter_shape,
            string padding,
            int[] strides,
            int[] dilation_rate,
            string name = null,
            string data_format = null) => new Convolution(input_shape,
                filter_shape,
                padding,
                strides,
                dilation_rate,
                name: name,
                data_format: data_format);

        /// <summary>
        /// Adds `bias` to `value`.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="bias"></param>
        /// <param name="data_format"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor bias_add(Tensor value, 
            Tensor bias, 
            string data_format = null, 
            string name = null)
        {
            return Python.with(ops.name_scope(name, "BiasAdd", new { value, bias }), scope =>
            {
                name = scope;
                value = ops.convert_to_tensor(value, name: "input");
                var bias_tensor = ops.convert_to_tensor(bias, dtype: value.dtype, name: "bias");
                return gen_nn_ops.bias_add(value, bias_tensor, data_format: data_format, name: name);
            });
        }

        /// <summary>
        /// Computes dropout.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="rate"></param>
        /// <param name="noise_shape"></param>
        /// <param name="seed"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor dropout_v2(Tensor x, Tensor rate, Tensor noise_shape = null, int? seed = null, string name = null)
        {
            return with(ops.name_scope(name, "dropout", x), scope =>
            {
                name = scope;
                x = ops.convert_to_tensor(x, name: "x");
                if (!x.dtype.is_floating())
                    throw new NotImplementedException($"x has to be a floating point tensor since it's going to" +
                        $" be scaled. Got a {x.dtype} tensor instead.");

                rate = ops.convert_to_tensor(rate, dtype: x.dtype, name: "rate");
                // Do nothing if we know rate == 0
                var val = tensor_util.constant_value(rate);
                if (!(val is null) && val.Data<float>(0) == 0)
                    return x;

                noise_shape = _get_noise_shape(x, noise_shape);

                // Sample a uniform distribution on [0.0, 1.0) and select values larger than
                // rate.
                //
                // NOTE: Random uniform actually can only generate 2^23 floats on [1.0, 2.0)
                // and subtract 1.0.
                var random_tensor = random_ops.random_uniform(noise_shape, seed: seed, dtype: x.dtype);
                var keep_prob = 1.0f - rate;
                var scale = 1.0f / keep_prob;
                // NOTE: if (1.0 + rate) - 1 is equal to rate, then we want to consider that
                // float to be selected, hence we use a >= comparison.
                var keep_mask = random_tensor >= rate;
                var ret = x * scale * math_ops.cast(keep_mask, x.dtype);
                ret.SetShape(x.GetShape());
                return ret;
            });
        }

        private static Tensor _get_noise_shape(Tensor x, Tensor noise_shape)
        {
            if (noise_shape == null)
                return array_ops.shape(x);
            else
                return noise_shape;
        }

        public static Tensor log_softmax(Tensor logits, int axis = -1, string name = null)
        {
            return _softmax(logits, gen_nn_ops.log_softmax, axis, name);
        }

        public static Tensor _softmax(Tensor logits, Func<Tensor, string, Tensor> compute_op, int dim = -1, string name = null)
        {
            logits = ops.convert_to_tensor(logits);

            var shape = logits.shape;
            bool is_last_dim = dim == -1 || dim == shape.Length - 1;
            if (is_last_dim)
                return compute_op(logits, name);

            throw new NotImplementedException("_softmax helper");
        }

        /// <summary>
        /// Computes sparse softmax cross entropy between `logits` and `labels`.
        /// </summary>
        /// <param name="labels"></param>
        /// <param name="logits"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor sparse_softmax_cross_entropy_with_logits(Tensor labels = null,
            Tensor logits = null, string name = null)
        {
            // Reshape logits and labels to rank 2.
            return with(ops.name_scope(name, default_name: "SparseSoftmaxCrossEntropyWithLogits", (labels, logits)), delegate
            {
                labels = ops.convert_to_tensor(labels);
                logits = ops.convert_to_tensor(logits);
                var precise_logits = logits.dtype == TF_DataType.TF_HALF ? math_ops.cast(logits, dtypes.float32) : logits;

                // Store label shape for result later.
                var labels_static_shape = labels.GetShape();
                var labels_shape = array_ops.shape(labels);
                /*bool static_shapes_fully_defined = (
                    labels_static_shape.is_fully_defined() &&
                        logits.get_shape()[:-1].is_fully_defined());*/

                // Check if no reshapes are required.
                if(logits.GetShape().NDim == 2)
                {
                    var (cost, _) = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(
                        precise_logits, labels, name: name);
                    if (logits.dtype == dtypes.float16)
                        return math_ops.cast(cost, dtypes.float32);
                    else
                        return cost;
                }

                // Perform a check of the dynamic shapes if the static shapes are not fully
                // defined.
                throw new NotImplementedException("sparse_softmax_cross_entropy_with_logits");
            });
        }

        public static Tensor softmax_cross_entropy_with_logits_v2_helper(Tensor labels,
            Tensor logits,
            int axis = -1,
            string name = null)
        {
            return Python.with(ops.name_scope(name, "softmax_cross_entropy_with_logits", new { }), scope =>
            {
                var precise_logits = logits;
                var input_rank = array_ops.rank(precise_logits);
                var shape = logits.GetShape();

                if (axis != -1)
                    throw new NotImplementedException("softmax_cross_entropy_with_logits_v2_helper axis != -1");

                var input_shape = array_ops.shape(precise_logits);

                // Do the actual op computation.
                // The second output tensor contains the gradients.  We use it in
                // _CrossEntropyGrad() in nn_grad but not here.

                var (cost, unused_backprop) = gen_nn_ops.softmax_cross_entropy_with_logits(precise_logits, labels, name: name);

                // The output cost shape should be the input minus axis.
                var output_shape = array_ops.slice(input_shape, 
                    new int[] { 0 },
                    new Tensor[] { math_ops.subtract(input_rank, 1) });

                cost = array_ops.reshape(cost, output_shape);

                return cost;
            });
        }
    }
}
