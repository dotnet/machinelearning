using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;
using static Tensorflow.Python;

namespace Tensorflow
{
    public class nn_impl
    {
        /// <summary>
        /// Normalizes along dimension `axis` using an L2 norm.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="axis"></param>
        /// <param name="epsilon"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor l2_normalize(Tensor x, 
            int axis = 0,
            float epsilon = 1e-12f,
            string name = null)
        {
            return with(ops.name_scope(name, "l2_normalize", new { x }), scope =>
            {
                x = ops.convert_to_tensor(x, name: "x");
                var sq = math_ops.square(x);
                var square_sum = math_ops.reduce_sum(sq, axis, keepdims: true);
                var x_inv_norm = math_ops.rsqrt(math_ops.maximum(square_sum, epsilon));
                return math_ops.multiply(x, x_inv_norm, name: name);
            });
        }

        /// <summary>
        /// Calculate the mean and variance of `x`
        /// </summary>
        /// <param name="x"> A `Tensor`.</param>
        /// <param name="axes"> Array of ints.  Axes along which to compute mean and variance.</param>
        /// <param name="name"> Name used to scope the operations that compute the moments.</param>
        /// <param name="keep_dims"> Produce moments with the same dimensionality as the input.</param>
        /// <returns> Two `Tensor` objects: `mean` and `variance`.</returns>
        public static (Tensor, Tensor) moments(Tensor x, 
            int[] axes,
            string name = null,
            bool keep_dims = false)
        {
            return with(ops.name_scope(name, "moments", new { x, axes }), scope =>
            {
                // The dynamic range of fp16 is too limited to support the collection of
                // sufficient statistics. As a workaround we simply perform the operations
                // on 32-bit floats before converting the mean and variance back to fp16
                var y = math_ops.cast(x, TF_DataType.TF_FLOAT);
                // Compute true mean while keeping the dims for proper broadcasting.
                var mean = math_ops.reduce_mean(y, axes, true, name = "mean");
                // Sample variance, not unbiased variance
                // Note: stop_gradient does not change the gradient that gets
                // backpropagated to the mean from the variance calculation,
                // because that gradient is zero
                var variance = math_ops.reduce_mean(math_ops.square_difference(y, array_ops.stop_gradient(mean)), axes, true, name = "Variance");
                if (!keep_dims)
                {
                    mean = array_ops.squeeze(mean, axes);
                    variance = array_ops.squeeze(variance, axes);
                }
                // TODO: if x.dtype == dtypes.float16:
                if (x.dtype == TF_DataType.TF_HALF)
                    return (math_ops.cast(mean, x.dtype), math_ops.cast(variance, x.dtype));
                else
                    return (mean, variance);
            });
        }

        public static Tensor[] fused_batch_norm(Tensor x, 
            RefVariable scale,
            RefVariable offset,
            Tensor mean,
            Tensor variance,
            float epsilon = 0.001f,
            string data_format = "NHWC",
            bool is_training = true,
            string name = null)
        {
            x = ops.convert_to_tensor(x, name: "input");
            var scale_tensor = ops.convert_to_tensor(scale, name: "scale");
            var offset_tensor = ops.convert_to_tensor(offset, name: "offset");
            if (mean == null)
                mean = constant_op.constant(new float[0]);
            if(variance == null)
                variance = constant_op.constant(new float[0]);
            var min_epsilon = 1.001e-5f;
            epsilon = epsilon > min_epsilon ? epsilon : min_epsilon;

            return gen_nn_ops._fused_batch_norm(x,
                scale_tensor,
                offset_tensor,
                mean,
                variance,
                epsilon,
                data_format,
                is_training,
                name);
        }

        /// <summary>
        /// Same as math_ops.count_nonzero.
        /// The reduction is done in dtype, which can be faster for 32-bit dtypes.
        /// </summary>
        /// <param name="input_tensor">The numeric tensor.</param>
        /// <param name="dtype">The reduction dtype.</param>
        /// <returns>number of nonzero values with type dtype</returns>
        private static Tensor _count_nonzero(Tensor input_tensor, TF_DataType dtype = TF_DataType.TF_INT64)
        {
            return with(ops.name_scope("count_nonzero", "count_nonzero", new { input_tensor }), scope =>
            {
                var zero = array_ops.zeros(new NumSharp.Shape(), dtype: input_tensor.dtype);
                var nonzero_count = math_ops.reduce_sum(
                math_ops.cast(gen_math_ops.not_equal(input_tensor, zero), dtype: dtype), name: "nonzero_count");
                return nonzero_count;
            });
        }

        /// <summary>
        /// Returns the fraction of zeros in value.
        /// </summary>
        /// <param name="value">A tensor of numeric type.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>The fraction of zeros in value, with type float32.</returns>
        public static Tensor zero_fraction(Tensor value, string name = null)
        {
            return with(ops.name_scope(name, "zero_fraction", new { value }), scope =>
            {

                value = ops.convert_to_tensor(value, name: "value");
                Tensor size = array_ops.size(value, out_type: dtypes.int64);
                Func<ITensorOrOperation> fu_true = () => math_ops.cast(_count_nonzero(value, dtype: dtypes.int32));
                Tensor zero_fraction_float32 = null;

                size = gen_math_ops.less_equal(size, dtypes.int32.max());
                Tensor num_nonzero = control_flow_ops.cond(
                        size,
                        () => math_ops.cast(_count_nonzero(value, dtype: dtypes.int32)),
                        () => _count_nonzero(value, dtype: dtypes.int64)
                        );

                with(ops.name_scope("counts_to_fraction"), count_scope =>
                {
                    var num_zero = size - num_nonzero;
                    var num_zero_float32 = math_ops.cast(num_zero, dtype: dtypes.float32);
                    var size_float32 = math_ops.cast(size, dtype: dtypes.float32);
                    zero_fraction_float32 = num_zero_float32 / size_float32;
                });

                return array_ops.identity(zero_fraction_float32, "fraction");
            });
        }
    }
}
