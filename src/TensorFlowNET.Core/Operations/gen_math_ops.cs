using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Tensorflow
{
    public static class gen_math_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Tensor _all(Tensor input, Tensor axis, bool keep_dims = false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("All", name, args: new { input, reduction_indices = axis, keep_dims = keep_dims });

            return _op.outputs[0];
        }

        /// <summary>
        /// Add all input tensors element wise.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor add_n(Tensor[] inputs, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("AddN", name, args: new { inputs });

            return _op.outputs[0];
        }

        /// <summary>
        /// Returns the index with the largest value across dimensions of a tensor.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="dimension"></param>
        /// <param name="output_type"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor arg_max(Tensor input, int dimension, TF_DataType output_type = TF_DataType.TF_INT64, string name = null)
            => _op_def_lib._apply_op_helper("ArgMax", name, args: new { input, dimension, output_type }).outputs[0];

        /// <summary>
        /// Returns the index with the smallest value across dimensions of a tensor.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="dimension"></param>
        /// <param name="output_type"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor arg_min(Tensor input, int dimension, TF_DataType output_type= TF_DataType.TF_INT64, string name= null)
            =>_op_def_lib._apply_op_helper("ArgMin", name, args: new { input, dimension, output_type }).outputs[0];


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
            var op = _op_def_lib._apply_op_helper("DivNoNan", name: name, args: new { x, y });
            return op.output;
        }

        /// <summary>
        /// Computes the mean of elements across dimensions of a tensor.
        /// Reduces `input` along the dimensions given in `axis`. Unless
        /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
        /// `axis`. If `keep_dims` is true, the reduced dimensions are retained with length 1.
        /// </summary>
        /// <param name="input">A `Tensor`. Must be one of the following types: 
        /// `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`. 
        /// The tensor to reduce.</param>
        /// <param name="axis">A `Tensor`. Must be one of the following types: `int32`, `int64`. The dimensions to reduce.</param>
        /// <param name="keep_dims"> An optional `bool`. Defaults to `False`. If true, retain reduced dimensions with length 1.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns> A `Tensor`. Has the same type as `input`.</returns>
        public static Tensor mean<T1, T2>(T1 input, T2 axis, bool keep_dims= false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Mean", name, args: new { input, reduction_indices = axis, keep_dims = keep_dims });

            return _op.outputs[0];
        }

        public static Tensor prod<T1, T2>(T1 input, T2 axis, bool keep_dims = false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Prod", name, args: new { input, reduction_indices = axis, keep_dims });

            return _op.outputs[0];
        }

        public static Tensor acos(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Acos", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor asin(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Asin", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor add<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Add", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor atan(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Atan", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor ceil(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Ceil", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor sin(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Sin", name, args: new { x });

            return _op.outputs[0];
        }

        /// <summary>
        ///    Computes sigmoid of <c>x</c> element-wise.
        /// </summary>
        /// <param name="x">
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Sigmoid'.
        /// </param>
        /// <returns>
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    Specifically, <c>y = 1 / (1 + exp(-x))</c>.
        /// </remarks>
        public static Tensor sigmoid(Tensor x, string name = "Sigmoid")
        {
            var op = _op_def_lib._apply_op_helper("Sigmoid", name: name, new { x });

            return op.output;
        }

        /// <summary>
        ///    Computes the gradient of the sigmoid of <c>x</c> wrt its input.
        /// </summary>
        /// <param name="y">
        /// </param>
        /// <param name="dy">
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'SigmoidGrad'.
        /// </param>
        /// <returns>
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    Specifically, <c>grad = dy * y * (1 - y)</c>, where <c>y = sigmoid(x)</c>, and
        ///    <c>dy</c> is the corresponding input gradient.
        /// </remarks>
        public static Tensor sigmoid_grad(Tensor y, Tensor dy, string name = "SigmoidGrad")
        {
            var op = _op_def_lib._apply_op_helper("SigmoidGrad", name: name, args: new { y, dy });

            return op.outputs[0];
        }

        public static Tensor sinh(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Sinh", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor cos(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Cos", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor cosh(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Cosh", name, args: new { x });

            return _op.outputs[0];
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
        {
            var _op = _op_def_lib._apply_op_helper("UnsortedSegmentSum", name, new { data, segment_ids, num_segments });
            return _op.outputs[0];
        }

        public static Tensor tan(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Tan", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor tanh(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Tanh", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor floor(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Floor", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor _clip_by_value(Tensor t, Tensor clip_value_min, Tensor clip_value_max, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("ClipByValue", name, args: new { t, clip_value_min, clip_value_max });

            return _op.outputs[0];
        }

        public static Tensor greater<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Greater", name: name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor greater_equal<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("GreaterEqual", name: name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor less<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Less", name: name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor less_equal<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("LessEqual", name: name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor log1p(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Log1p", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor squared_difference(Tensor x, Tensor y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("SquaredDifference", name, args: new { x, y, name });

            return _op.outputs[0];
        }

        /// <summary>
        /// Computes square of x element-wise.
        /// </summary>
        /// <param name="x"> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns> A `Tensor`. Has the same type as `x`.</returns>
        public static Tensor square(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Square", name, args: new { x });

            return _op.outputs[0];
        }

        /// <summary>
        /// Returns which elements of x are finite.
        /// </summary>
        /// <param name="x"> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns> A `Tensor` of type `bool`.</returns>
        public static Tensor is_finite(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("IsFinite", name, args: new { x });

            return _op.outputs[0];
        }

        /// <summary>
        /// Computes exponential of x element-wise.  \\(y = e^x\\).
        /// </summary>
        /// <param name="x"> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns> A `Tensor`. Has the same type as `x`.</returns>
        public static Tensor exp(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Exp", name, args: new { x });

            return _op.outputs[0];
        }

        /// <summary>
        /// Computes natural logarithm of x element-wise.
        /// </summary>
        /// <param name="x"> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.</param>
        /// <param name="name"> name: A name for the operation (optional).</param>
        /// <returns> A `Tensor`. Has the same type as `x`.</returns>
        public static Tensor log(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Log", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor cast(Tensor x, TF_DataType DstT, bool Truncate= false, string name= "")
        {
            var _op = _op_def_lib._apply_op_helper("Cast", name, args: new { x, DstT, Truncate });

            return _op.outputs[0];
        }

        public static Tensor neg(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Neg", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor sqrt(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Sqrt", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor sub<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Sub", name, args: new { x, y });

            return _op.outputs[0];
        }

        /// <summary>
        /// Returns the truth value of (x == y) element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor equal<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Equal", name, args: new { x, y });

            return _op.outputs[0];
        }

        /// <summary>
        /// Returns the truth value of (x != y) element-wise.
        /// </summary>
        /// <typeparam name="Tx">The type of the x.</typeparam>
        /// <typeparam name="Ty">The type of the y.</typeparam>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="name">The name.</param>
        /// <returns></returns>
        public static Tensor not_equal<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("NotEqual", name, args: new { x, y });

            return _op.outputs[0];
        }


        public static Tensor atan2(Tensor y, Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Atan2", name, args: new { y, x });

            return _op.outputs[0];
        }

        public static Tensor mul<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Mul", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor mul_no_nan<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("MulNoNan", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor real_div(Tensor x, Tensor y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("RealDiv", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor reciprocal(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Reciprocal", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor floor_mod(Tensor x, Tensor y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("FloorMod", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor floor_div(Tensor x, Tensor y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("FloorDiv", name, args: new { x, y });

            return _op.outputs[0];
        }

        /// <summary>
        /// Multiply the matrix "a" by the matrix "b".
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="transpose_a"></param>
        /// <param name="transpose_b"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor mat_mul(Tensor a, Tensor b, bool transpose_a = false, bool transpose_b = false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("MatMul", name, args: new { a, b, transpose_a, transpose_b });

            return _op.outputs[0];
        }

        /// <summary>
        /// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor maximum<T1, T2>(T1 x, T2 y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Maximum", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor minimum<T1, T2>(T1 x, T2 y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Minimum", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor _abs(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Abs", name, new { x });

            return _op.outputs[0];
        }

        public static Tensor _max<Tx, Ty>(Tx input, Ty axis, bool keep_dims=false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Max", name, new { input, reduction_indices = axis, keep_dims });

            return _op.outputs[0];
        }

        public static Tensor _min<Tx, Ty>(Tx input, Ty axis, bool keep_dims = false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Min", name, new { input, reduction_indices = axis, keep_dims });

            return _op.outputs[0];
        }

        public static Tensor pow<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Pow", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor _sum(Tensor input, Tensor axis = null, bool keep_dims = false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Sum", name, args: new { input, reduction_indices = axis, keep_dims });

            return _op.outputs[0];
        }

        public static Tensor _sum(Tensor input, int axis, bool keep_dims = false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Sum", name, args: new { input, reduction_indices = axis, keep_dims });

            return _op.outputs[0];
        }

        /// <summary>
        /// Creates a sequence of numbers.
        /// </summary>
        /// <param name="start"></param>
        /// <param name="limit"></param>
        /// <param name="delta"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor range(Tensor start, Tensor limit, Tensor delta, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Range", name, new { start, limit, delta });

            return _op.outputs[0];
        }

        /// <summary>
        ///    Rounds the values of a tensor to the nearest integer, element-wise.
        /// </summary>
        /// <param name="x">
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Round'.
        /// </param>
        /// <returns>
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    Rounds half to even.  Also known as bankers rounding. If you want to round
        ///    according to the current system rounding mode use std::cint.
        /// </remarks>
        public static Tensor round(Tensor x, string name = "Round")
        {
            var op = _op_def_lib._apply_op_helper("Round", name: name, new { x });

            return op.output;
        }

        /// <summary>
        /// Computes reciprocal of square root of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor rsqrt(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Rsqrt", name, new { x });

            return _op.outputs[0];
        }

        /// <summary>
        /// Returns the fraction of zeros in value.
        /// </summary>
        /// <param name="value">A tensor of numeric type.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>The fraction of zeros in value, with type float32.</returns>
        public static Tensor zero_fraction(Tensor value, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("zero_fraction", name, new { value, name });

            return _op.outputs[0];
        }
    }
}
