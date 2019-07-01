using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor abs(Tensor x, string name = null)
            => math_ops.abs(x, name);

        /// <summary>
        /// Computes acos of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor acos(Tensor x, string name = null)
            => gen_math_ops.acos(x, name);

        /// <summary>
        /// Computes asin of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor asin(Tensor x, string name = null)
            => gen_math_ops.asin(x, name);

        public static Tensor add<Tx, Ty>(Tx a, Ty b) 
            => gen_math_ops.add(a, b);

        /// <summary>
        /// Computes atan of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor atan(Tensor x, string name = null)
            => gen_math_ops.atan(x, name);

        public static Tensor arg_max(Tensor input, int dimension, TF_DataType output_type = TF_DataType.TF_INT64, string name = null)
            => gen_math_ops.arg_max(input, dimension, output_type: output_type, name: name);

        public static Tensor arg_min(Tensor input, int dimension, TF_DataType output_type = TF_DataType.TF_INT64, string name = null)
            => gen_math_ops.arg_min(input, dimension, output_type: output_type, name: name);

        /// <summary>
        /// Returns element-wise smallest integer not less than x.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor ceil(Tensor x, string name = null)
            => gen_math_ops.ceil(x, name);

        /// <summary>
        /// Computes sin of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor sin(Tensor x, string name = null)
            => gen_math_ops.sin(x, name);

        /// <summary>
        /// Computes hyperbolic sine of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor sinh(Tensor x, string name = null)
            => gen_math_ops.sinh(x, name);

        /// <summary>
        /// Computes cos of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor cos(Tensor x, string name = null)
            => gen_math_ops.cos(x, name);

        /// <summary>
        /// Computes hyperbolic cosine of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor cosh(Tensor x, string name = null)
            => gen_math_ops.cosh(x, name);

        public static Tensor tan(Tensor x, string name = null)
            => gen_math_ops.tan(x, name);

        public static Tensor tanh(Tensor x, string name = null)
            => gen_math_ops.tanh(x, name);

        /// <summary>
        /// Returns element-wise largest integer not greater than x.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor floor(Tensor x, string name = null)
            => gen_math_ops.floor(x, name);

        /// <summary>
        /// Returns the truth value of (x > y) element-wise.
        /// </summary>
        /// <typeparam name="Tx"></typeparam>
        /// <typeparam name="Ty"></typeparam>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor greater<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.greater(x, y, name);

        /// <summary>
        /// Returns the truth value of (x >= y) element-wise.
        /// </summary>
        /// <typeparam name="Tx"></typeparam>
        /// <typeparam name="Ty"></typeparam>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor greater_equal<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.greater_equal(x, y, name);

        /// <summary>
        /// </summary>
        /// <typeparam name="Tx"></typeparam>
        /// <typeparam name="Ty"></typeparam>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor less<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.less(x, y, name);

        /// <summary>
        /// </summary>
        /// <typeparam name="Tx"></typeparam>
        /// <typeparam name="Ty"></typeparam>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor less_equal<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.less_equal(x, y, name);

        /// <summary>
        /// Computes natural logarithm of (1 + x) element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor log1p(Tensor x, string name = null)
            => gen_math_ops.log1p(x, name);

        /// <summary>
        /// Clips tensor values to a specified min and max.
        /// </summary>
        /// <param name="t"></param>
        /// <param name="clip_value_min"></param>
        /// <param name="clip_value_max"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor _clip_by_value(Tensor t, Tensor clip_value_min, Tensor clip_value_max, string name = null)
            => gen_math_ops._clip_by_value(t, clip_value_min, clip_value_max);

        public static Tensor sub(Tensor a, Tensor b) 
            => gen_math_ops.sub(a, b);

        public static Tensor divide(Tensor a, Tensor b)
            => gen_math_ops.real_div(a, b);

        public static Tensor sqrt(Tensor a, string name = null) 
            => gen_math_ops.sqrt(a, name);

        public static Tensor subtract<T>(Tensor x, T[] y, string name = null) where T : struct
            => gen_math_ops.sub(x, ops.convert_to_tensor(y, dtype: x.dtype.as_base_dtype(), name: "y"), name);

        public static Tensor log(Tensor x, string name = null)
            => gen_math_ops.log(x, name);

        public static Tensor equal(Tensor x, Tensor y, string name = null)
            => gen_math_ops.equal(x, y, name);

        /// <summary>
        /// Computes arctangent of `y/x` element-wise, respecting signs of the arguments.
        /// </summary>
        /// <param name="y"></param>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor atan2(Tensor y, Tensor x, string name = null)
            => gen_math_ops.atan2(y, x, name);

        /// <summary>
        /// Computes the maximum of elements across dimensions of a tensor.
        /// </summary>
        /// <typeparam name="Tx"></typeparam>
        /// <typeparam name="Ty"></typeparam>
        /// <param name="input"></param>
        /// <param name="axis"></param>
        /// <param name="keep_dims"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor max<Tx, Ty>(Tx input, Ty axis, bool keep_dims = false, string name = null)
            => gen_math_ops._max(input, axis, keep_dims: keep_dims, name: name);

        /// <summary>
        /// Computes the minimum of elements across dimensions of a tensor.
        /// </summary>
        /// <typeparam name="Tx"></typeparam>
        /// <typeparam name="Ty"></typeparam>
        /// <param name="input"></param>
        /// <param name="axis"></param>
        /// <param name="keep_dims"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor min<Tx, Ty>(Tx input, Ty axis, bool keep_dims = false, string name = null)
            => gen_math_ops._min(input, axis, keep_dims: keep_dims, name: name);

        /// <summary>
        /// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
        /// </summary>
        /// <typeparam name="T1"></typeparam>
        /// <typeparam name="T2"></typeparam>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor maximum<T1, T2>(T1 x, T2 y, string name = null)
            => gen_math_ops.maximum(x, y, name: name);

        /// <summary>
        /// </summary>
        /// <typeparam name="T1"></typeparam>
        /// <typeparam name="T2"></typeparam>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor minimum<T1, T2>(T1 x, T2 y, string name = null)
            => gen_math_ops.minimum(x, y, name: name);

        public static Tensor multiply<Tx, Ty>(Tx x, Ty y) 
            => gen_math_ops.mul(x, y);

        public static Tensor negative(Tensor x, string name = null)
            => gen_math_ops.neg(x, name);

        /// <summary>
        /// Divides x / y elementwise (using Python 2 division operator semantics).
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor div(Tensor x, Tensor y, string name = null)
            => math_ops.div(x, y, name: name);

        public static Tensor divide<T>(Tensor x, T[] y, string name = null) where T : struct
            => x / ops.convert_to_tensor(y, dtype: x.dtype.as_base_dtype(), name: "y");

        public static Tensor pow<T1, T2>(T1 x, T2 y) 
            => gen_math_ops.pow(x, y);

        /// <summary>
        /// Computes the sum of elements across dimensions of a tensor.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="axis"></param>
        /// <param name="reduction_indices"></param>
        /// <returns></returns>
        public static Tensor reduce_sum(Tensor input, int? axis = null, int? reduction_indices = null)
        {
            if(!axis.HasValue && reduction_indices.HasValue)
                return math_ops.reduce_sum(input, reduction_indices.Value);
            return math_ops.reduce_sum(input);
        }

        public static Tensor reduce_sum(Tensor input, int axis, int? reduction_indices = null)
            => math_ops.reduce_sum(input, axis);

        /// <summary>
        /// Computes the maximum of elements across dimensions of a tensor.
        /// </summary>
        /// <param name="input_tensor"></param>
        /// <param name="axis"></param>
        /// <param name="keepdims"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor reduce_max(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null)
            => math_ops.reduce_max(input_tensor, axis, keepdims, name);

        public static Tensor reduce_min(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null)
            => math_ops.reduce_min(input_tensor, axis, keepdims, name);

        public static Tensor sigmoid<T>(T x, string name = null)
            => math_ops.sigmoid(x, name: name);

        public static Tensor sum(Tensor input, int axis, bool keep_dims = false, string name = null)
            => gen_math_ops._sum(input, axis, keep_dims: keep_dims, name: name);

        public static Tensor reduce_mean(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null, int? reduction_indices = null)
            => math_ops.reduce_mean(input_tensor, axis: axis, keepdims: keepdims, name: name, reduction_indices: reduction_indices);

        public static Tensor round(Tensor x, string name = null)
            => gen_math_ops.round(x, name: name);

        public static Tensor cast(Tensor x, TF_DataType dtype = TF_DataType.DtInvalid, string name = null) 
            => math_ops.cast(x, dtype, name);

        public static Tensor argmax(Tensor input, int axis = -1, string name = null, int? dimension = null, TF_DataType output_type = TF_DataType.TF_INT64)
            => gen_math_ops.arg_max(input, axis, name: name, output_type: output_type);

        public static Tensor square(Tensor x, string name = null)
            => gen_math_ops.square(x, name: name);
    }
}
