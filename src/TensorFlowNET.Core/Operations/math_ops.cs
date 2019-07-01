using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Framework;
using static Tensorflow.Python;

namespace Tensorflow
{
    /// <summary>
    /// python\ops\math_ops.py
    /// </summary>
    public class math_ops
    {
        public static Tensor abs(Tensor x, string name = null)
        {
            return with(ops.name_scope(name, "Abs", new { x }), scope =>
            {
                x = ops.convert_to_tensor(x, name: "x");
                if (x.dtype.is_complex())
                    throw new NotImplementedException("math_ops.abs for dtype.is_complex");
                    //return gen_math_ops.complex_abs(x, Tout: x.dtype.real_dtype, name: name);
                return gen_math_ops._abs(x, name: name);
            });
        }

        public static Tensor add<Tx, Ty>(Tx x, Ty y, string name = null) 
            => gen_math_ops.add(x, y, name);

        /// <summary>
        /// Adds all input tensors element-wise.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor add_n(Tensor[] inputs, string name = null)
        {
            inputs = ops.convert_n_to_tensor_or_indexed_slices(inputs);

            if(inputs.Length == 1)
            {
                var values = inputs[0];
                if (name != null)
                    return array_ops.identity(values, name: name);
                return values;
            }
            
            return gen_math_ops.add_n(inputs, name: name);
        }

        public static Tensor cast(Tensor x, TF_DataType dtype = TF_DataType.DtInvalid, string name = null)
        {
            var base_type = dtype.as_base_dtype();
            if(base_type == x.dtype)
                return x;

            return with(ops.name_scope(name, "Cast", new { x }), scope =>
            {
                name = scope;
                x = ops.convert_to_tensor(x, name: "x");
                if (x.dtype.as_base_dtype() != base_type)
                    x = gen_math_ops.cast(x, base_type, name: name);

                return x;
            });
        }

        /// <summary>
        /// Divide two values using Python 2 semantics. Used for Tensor.__div__.
        /// </summary>
        /// <param name="x">`Tensor` numerator of real numeric type.</param>
        /// <param name="y">`Tensor` denominator of real numeric type.</param>
        /// <param name="name">A name for the operation</param>
        /// <returns>`x / y` returns the quotient of x and y.</returns>
        public static Tensor div(Tensor x, Tensor y, string name = null)
        {
            return with(ops.name_scope(name, "div", (x, y)), name_scope =>
            {
                name = name_scope;
                x = ops.convert_to_tensor(x, name: "x");
                y = ops.convert_to_tensor(y, dtype: x.dtype.as_base_dtype(), name = "y");
                var x_dtype = x.dtype.as_base_dtype();
                var y_dtype = y.dtype.as_base_dtype();
                if (x_dtype != y_dtype)
                    throw new TypeError($"x and y must have the same dtype, got {x_dtype} != {y_dtype}");
                if (x_dtype.is_floating() || x_dtype.is_complex())
                    return gen_math_ops.real_div(x, y, name: name);
                else
                    return gen_math_ops.floor_div(x, y, name: name);
            });
        }

        /// <summary>
        ///    Returns 0 if the denominator is zero.
        /// </summary>
        /// <param name="x">
        /// </param>
        /// <param name="y">
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'DivNoNan'.
        /// </param>
        /// <returns>
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    
        ///    *NOTE*: <c>DivNoNan</c> supports broadcasting. More about broadcasting
        ///    [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
        /// </remarks>
        public static Tensor div_no_nan(Tensor x, Tensor y, string name = null)
        {
            return with(ops.name_scope(name, "div_no_nan", (x, y)), name_scope =>
            {
                name = name_scope;
                x = ops.convert_to_tensor(x, name: "x");
                y = ops.convert_to_tensor(y, name: "y", dtype: x.dtype.as_base_dtype());
                var x_dtype = x.dtype.as_base_dtype();
                var y_dtype = y.dtype.as_base_dtype();
                if (x_dtype != y_dtype)
                    throw new TypeError($"x and y must have the same dtype, got {x_dtype} != {y_dtype}");
                return gen_math_ops.div_no_nan(x, y, name: name);
            });
        }

        public static Tensor equal<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.equal(x, y, name: name);

        public static Tensor sqrt(Tensor x, string name = null)
            => gen_math_ops.sqrt(x, name: name);

        public static Tensor multiply<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.mul(x, y, name: name);

        public static Tensor mul_no_nan<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.mul_no_nan(x, y, name: name);

        /// <summary>
        /// Computes the mean of elements across dimensions of a tensor.
        /// Reduces `input_tensor` along the dimensions given in `axis`.
        /// Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
        /// entry in `axis`. If `keepdims` is true, the reduced dimensionsare retained with length 1.
        /// If `axis` is None, all dimensions are reduced, and a tensor with a single element is returned.
        /// </summary>
        /// <param name="input_tensor"> The tensor to reduce. Should have numeric type.</param>
        /// <param name="axis">The dimensions to reduce. If `None` (the default), reduces all
        /// dimensions.Must be in the range `[-rank(input_tensor), rank(input_tensor))`.</param>
        /// <param name="keepdims"> If true, retains reduced dimensions with length 1.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        public static Tensor reduce_mean(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null, int? reduction_indices = null)
        {
            var r = _ReductionDims(input_tensor, axis);
            if (axis == null)
            {
                var m = gen_math_ops.mean(input_tensor, r, keepdims, name);
                return _may_reduce_to_scalar(keepdims, axis, m);
            }
            else
            {
                var m = gen_math_ops.mean(input_tensor, axis, keepdims, name);
                return _may_reduce_to_scalar(keepdims, axis, m);
            }
        }

        /// <summary>
        /// Computes the product of elements across dimensions of a tensor.
        /// </summary>
        /// <param name="input_tensor"></param>
        /// <param name="axis"></param>
        /// <param name="keepdims"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor reduce_prod(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null)
        {
            var r = _ReductionDims(input_tensor, axis);
            if (axis == null)
            {
                var m = gen_math_ops.prod(input_tensor, r, keepdims, name);
                return _may_reduce_to_scalar(keepdims, axis, m);
            }
            else
            {
                var m = gen_math_ops.prod(input_tensor, axis, keepdims, name);
                return _may_reduce_to_scalar(keepdims, axis, m);
            }
        }

        public static Tensor sigmoid<T>(T x, string name = null)
        {
            var x_tensor = ops.convert_to_tensor(x, name: "x");
            return gen_math_ops.sigmoid(x_tensor, name: name);
        }

        /// <summary>
        /// Returns (x - y)(x - y) element-wise.
        /// </summary>
        /// <param name="x"> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.</param>
        /// <param name="y"> A `Tensor`. Must have the same type as `x`.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns>A `Tensor`. Has the same type as `x`.</returns>
        public static Tensor square_difference(Tensor x, Tensor y, string name = null)
        {
            var m = gen_math_ops.squared_difference(x, y);
            return m;
        }

        public static Tensor square(Tensor x, string name = null)
        {
            return gen_math_ops.square(x, name);
        }

        public static Tensor subtract<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            return gen_math_ops.sub(x, y, name);
        }

        public static Tensor log(Tensor x, string name = null)
        {
            return gen_math_ops.log(x, name);
        }

        /// <summary>
        /// Helper function for reduction ops.
        /// </summary>
        /// <param name="input_shape">1-D Tensor, the shape of the Tensor being reduced.</param>
        /// <param name="axes">1-D Tensor, the reduction axes.</param>
        /// <returns>A 1-D Tensor, the output shape as if keepdims were set to True.</returns>
        public static Tensor reduced_shape(Tensor input_shape, Tensor axes)
        {
            input_shape = to_int32(input_shape);
            axes = to_int32(axes);

            var input_rank = array_ops.size(input_shape);
            axes = (axes + input_rank) % input_rank;
            var axes_shape = array_ops.shape(axes);
            var rng = math_ops.range(input_rank);
            var a1 = new Tensor[] { rng, axes };
            var fill = gen_array_ops.fill(axes_shape, 1);
            var a2 = new Tensor[] { input_shape, fill };

            return gen_data_flow_ops.dynamic_stitch(a1, a2);
        }

        /// <summary>
        /// Computes the reciprocal of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor reciprocal(Tensor x, string name = null)
            => gen_math_ops.reciprocal(x, name: name);

        /// <summary>
        /// Computes the "logical and" of elements across dimensions of a tensor.
        /// </summary>
        /// <param name="input_tensor"></param>
        /// <param name="axis"></param>
        /// <param name="keepdims"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor reduce_all(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null)
        {
            var all = gen_math_ops._all(input_tensor,
                    _ReductionDims(input_tensor, axis),
                    keepdims,
                    name: name);

            return _may_reduce_to_scalar(keepdims, axis, all);
        }

        /// <summary>
        /// Computes log(sum(exp(elements across dimensions of a tensor))).
        /// Reduces `input_tensor` along the dimensions given in `axis`.
        /// Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
        /// entry in `axis`. If `keepdims` is true, the reduced dimensions
        /// are retained with length 1.

        /// If `axis` has no entries, all dimensions are reduced, and a
        /// tensor with a single element is returned.

        /// This function is more numerically stable than log(sum(exp(input))). It avoids
        /// overflows caused by taking the exp of large inputs and underflows caused by
        /// taking the log of small inputs.
        /// </summary>
        /// <param name="input_tensor"> The tensor to reduce. Should have numeric type.</param>
        /// <param name="axis"> The dimensions to reduce. If `None` (the default), reduces all 
        /// dimensions.Must be in the range `[-rank(input_tensor), rank(input_tensor))`.</param>
        /// <param name="keepdims"></param>
        /// <returns> The reduced tensor.</returns>
        public static Tensor reduce_logsumexp(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null)
        {
            return with(ops.name_scope(name, "ReduceLogSumExp", new { input_tensor }), scope =>
            {
                var raw_max = reduce_max(input_tensor, axis, true);
                var my_max = array_ops.stop_gradient(array_ops.where(gen_math_ops.is_finite(raw_max), raw_max, array_ops.zeros_like(raw_max)));
                var result = gen_math_ops.log(
                reduce_sum(
                    gen_math_ops.exp(gen_math_ops.sub(input_tensor, my_max)),
                    axis[0],
                    keepdims));
                if (!keepdims)
                {
                    my_max = array_ops.reshape(my_max, array_ops.shape(result));
                }
                result = gen_math_ops.add(result, my_max);
                return _may_reduce_to_scalar(keepdims, axis, result);
            });
        }

        public static Tensor reduce_max(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null)
        {
            var r = _ReductionDims(input_tensor, axis);
            var max = (axis != null) ? gen_math_ops._max(input_tensor, axis, keepdims, name) :
                gen_math_ops._max(input_tensor, r, keepdims, name);
            return _may_reduce_to_scalar(keepdims, axis, max);
        }

        public static Tensor reduce_min(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null)
        {
            var r = _ReductionDims(input_tensor, axis);
            var min = gen_math_ops._min(input_tensor, r, keepdims, name);
            return _may_reduce_to_scalar(keepdims, axis, min);
        }

        /// <summary>
        /// Computes the sum along segments of a tensor.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="segment_ids"></param>
        /// <param name="num_segments"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor unsorted_segment_sum(Tensor data, Tensor segment_ids, Tensor num_segments, string name = null)
            => gen_math_ops.unsorted_segment_sum(data, segment_ids, num_segments, name: name);
        
        /// <summary>
        /// Casts a tensor to type `int32`.
        /// </summary>
        /// <param name="x">A `Tensor` or `SparseTensor` or `IndexedSlices`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with type `int32`.</returns>
        private static Tensor to_int32(Tensor x, string name = "ToInt32")
        {
            return __case__(x, TF_DataType.TF_INT32, name: name);
        }

        /// <summary>
        /// Casts a tensor to a new type.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        /// <returns>A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` and same type as `dtype`.</returns>
        public static Tensor __case__(Tensor x, TF_DataType dtype, string name = null)
        {
            var base_type = dtype.as_base_dtype();
            if (x is Tensor && base_type == x.dtype)
                return x;

            // math_ops.py cast
            throw new NotImplementedException();
        }

        public static Tensor reduce_sum(Tensor input_tensor, Tensor axis = null, bool keepdims = false, string name = null)
        {
            var r = _ReductionDims(input_tensor, axis);
            var m = gen_math_ops._sum(input_tensor, r, keep_dims: keepdims, name: name);
            return _may_reduce_to_scalar(keepdims, axis, m);
        }

        public static Tensor reduce_sum(Tensor input_tensor, int axis, bool keepdims = false, string name = null)
        {
            var m = gen_math_ops._sum(input_tensor, axis, keep_dims: keepdims, name: name);
            return _may_reduce_to_scalar(keepdims, new int[] { axis }, m);
        }

        private static Tensor _may_reduce_to_scalar(bool keepdims, Tensor axis, Tensor output)
        {
            if (!common_shapes.has_fully_defined_shape(output) &&
                !keepdims &&
                axis == null)
                // We want set_shape to be reflected in the C API graph for when we run it.
                output.shape = new int[0];
            return output;
        }

        private static Tensor _may_reduce_to_scalar(bool keepdims, int[] axis, Tensor output)
        {
            if (!common_shapes.has_fully_defined_shape(output) &&
                !keepdims &&
                axis == null)
                output.shape = new int[0];
            return output;
        }

        private static Tensor _ReductionDims(Tensor x, Tensor axis)
        {
            if (axis != null)
            {
                return axis;
            }
            else
            {
                var rank = array_ops.rank(x);
                return range(0, rank, 1);
            }
        }

        private static Tensor _ReductionDims(Tensor x, int[] axis)
        {
            if (axis != null)
            {
                // should return axis. or check before.
                return ops.convert_to_tensor(axis, TF_DataType.TF_INT32);
            }
            else
            {
                var rank = common_shapes.rank(x);

                // we rely on Range and Rank to do the right thing at run-time.
                if (rank == -1) return range(0, array_ops.rank(x));

                if (rank.HasValue && rank.Value > -1)
                {
                   return constant_op.constant(np.arange(rank.Value), TF_DataType.TF_INT32);
                }

                return range(0, rank, 1);
            }
        }

        /// <summary>
        /// Computes reciprocal of square root of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor rsqrt(Tensor x, string name = null)
            => gen_math_ops.rsqrt(x, name: name);

        public static Tensor range(object start, object limit = null, object delta = null, TF_DataType dtype = TF_DataType.DtInvalid, string name = "range" )
        {
            if(limit == null)
            {
                limit = start;
                start = 0;
            }

            if (delta == null)
                delta = 1;

            return with(ops.name_scope(name, "Range", new { start, limit, delta }), scope =>
            {
                name = scope;
                var start1 = ops.convert_to_tensor(start, name: "start");
                var limit1 = ops.convert_to_tensor(limit, name: "limit");
                var delta1 = ops.convert_to_tensor(delta, name: "delta");

                return gen_math_ops.range(start1, limit1, delta1, name);
            });
        }

        public static Tensor floordiv(Tensor x, Tensor y, string name = null)
        {
            return with(ops.name_scope(name, "floordiv", new { x, y }), scope =>
            {
                return gen_math_ops.floor_div(x, y, scope);
            });
        }

        public static Tensor maximum<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.maximum(x, y, name: name);

        public static Tensor matmul(Tensor a, Tensor b,
            bool transpose_a = false, bool transpose_b = false,
            bool adjoint_a = false, bool adjoint_b = false,
            bool a_is_sparse = false, bool b_is_sparse = false,
            string name = null)
        {
            Tensor result = null;

            with(ops.name_scope(name, "MatMul", new Tensor[] { a, b }), scope =>
            {
                name = scope;

                if (transpose_a && adjoint_a)
                    throw new ValueError("Only one of transpose_a and adjoint_a can be True.");
                if (transpose_b && adjoint_b)
                    throw new ValueError("Only one of transpose_b and adjoint_b can be True.");

                a = ops.convert_to_tensor(a, name: "a");
                b = ops.convert_to_tensor(b, name: "b");

                result = gen_math_ops.mat_mul(a, b, transpose_a, transpose_b, name);
            });

            return result;
        }

        /// <summary>
        /// Returns the complex conjugate of a complex number.
        /// </summary>
        /// <param name="x">`Tensor` to conjugate.  Must have numeric or variant type.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor` that is the conjugate of `x` (with the same type).</returns>
        public static Tensor conj(Tensor x, string name = null)
        {
            var dt = x.dtype;
            if (dt.is_floating() || dt.is_integer())
                return x;

            return with(ops.name_scope(name, "Conj", new List<Tensor> { x }), scope =>
            {

                return x;
            });
        }

        public static Tensor truediv(Tensor x, Tensor y, string name = null)
            => _truediv_python3(x, y, name);

        public static Tensor _truediv_python3(Tensor x, Tensor y, string name = null)
        {
            return with(ops.name_scope(name, "truediv", new { x, y }), scope =>
            {
                name = scope;
                var x_dtype = x.dtype.as_base_dtype();
                var y_dtype = y.dtype.as_base_dtype();

                return gen_math_ops.real_div(x, y, name: name);
            });
        }
    }
}
