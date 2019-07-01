using NumSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow.Eager;

namespace Tensorflow
{
    public static class gen_array_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();
        public static Execute _execute = new Execute();

        /// <summary>
        /// Concatenates tensors along one dimension.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="axis"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor concat_v2(Tensor[] values, int axis, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("ConcatV2", name: name, args: new { values, axis });

            return _op.outputs[0];
        }

        public static Tensor[] concat_offset(Tensor concat_dim, Tensor[] shape, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("ConcatOffset", name: name, args: new { concat_dim, shape });

            return _op.outputs;
        }

        /// <summary>
        ///    Returns a diagonal tensor with a given diagonal values.
        /// </summary>
        /// <param name="diagonal">
        ///    Rank k tensor where k is at most 1.
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Diag'.
        /// </param>
        /// <returns>
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    Given a <c>diagonal</c>, this operation returns a tensor with the <c>diagonal</c> and
        ///    everything else padded with zeros. The diagonal is computed as follows:
        ///    
        ///    Assume <c>diagonal</c> has dimensions [D1,..., Dk], then the output is a tensor of
        ///    rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:
        ///    
        ///    <c>output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]</c> and 0 everywhere else.
        ///    
        ///    For example:
        ///    
        ///   <code>
        ///    # 'diagonal' is [1, 2, 3, 4]
        ///    tf.diag(diagonal) ==&amp;gt; [[1, 0, 0, 0]
        ///    [0, 2, 0, 0]
        ///    [0, 0, 3, 0]
        ///    [0, 0, 0, 4]]
        ///   </code>
        /// </remarks>
        public static Tensor diag(Tensor diagonal, string name = null)
        {
            var op = _op_def_lib._apply_op_helper("Diag", name: name, args: new { diagonal });

            return op.output;
        }

        public static Tensor expand_dims(Tensor input, int axis, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("ExpandDims", name: name, args: new { input, dim = axis });

            return _op.outputs[0];
        }

        public static Tensor gather_v2(Tensor @params, Tensor indices, int axis, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("GatherV2", name: name, new { @params, indices, axis });

            return _op.outputs[0];
        }

        public static Tensor pack(Tensor[] values, int axis = 0, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Pack", name: name, args: new { values, axis });

            return _op.outputs[0];
        }

        public static Tensor placeholder(TF_DataType dtype, TensorShape shape = null, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Placeholder", name: name, args: new { dtype, shape });
            var _result = _op.outputs;
            var _inputs_flat = _op.inputs;

            var _attrs = new Dictionary<string, object>();
            _attrs["dtype"] = _op.get_attr("dtype");
            _attrs["shape"] = _op.get_attr("shape");

            _execute.record_gradient("Placeholder", _inputs_flat, _attrs, _result, name);

            return new Tensor(_op, 0, dtype);
        }

        /// <summary>
        ///    An identity op that triggers an error if a gradient is requested.
        /// </summary>
        /// <param name="input">
        ///    any tensor.
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'PreventGradient'.
        /// </param>
        /// <param name="message">
        ///    Will be printed in the error when anyone tries to differentiate
        ///    this operation.
        /// </param>
        /// <returns>
        ///    the same input tensor.
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    When executed in a graph, this op outputs its input tensor as-is.
        ///    
        ///    When building ops to compute gradients, the TensorFlow gradient system
        ///    will return an error when trying to lookup the gradient of this op,
        ///    because no gradient must ever be registered for this function.  This
        ///    op exists to prevent subtle bugs from silently returning unimplemented
        ///    gradients in some corner cases.
        /// </remarks>
        public static Tensor prevent_gradient(Tensor input, string message = "", string name = null)
        {
            var op = _op_def_lib._apply_op_helper("PreventGradient", name: name, args: new { input, message });
            return op.output;
        }

        /// <summary>
        /// Return a tensor with the same shape and contents as the input tensor or value.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="name"></param>
        public static Tensor identity(Tensor input, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Identity", name, new { input });

            return _op.outputs[0];
        }

        public static Tensor invert_permutation(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("InvertPermutation", name, new { x });

            return _op.outputs[0];
        }

        public static Tensor log(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Log", name: name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor rank(Tensor input, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Rank", name: name, args: new { input });

            return _op.outputs[0];
        }

        /// <summary>
        /// Creates a tensor filled with a scalar value.
        /// </summary>
        /// <param name="dims">A `Tensor`.</param>
        /// <param name="value">A `Tensor`. 0-D (scalar). Value to fill the returned tensor.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor`. Has the same type as `value`.</returns>
        public static Tensor fill<T>(Tensor dims, T value, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Fill", name, new { dims, value });

            return _op.outputs[0];
        }

        /// <summary>
        /// Return the reduction indices for computing gradients of s0 op s1 with broadcast.
        /// </summary>
        /// <param name="s0">A `Tensor`. Must be one of the following types: `int32`, `int64`.</param>
        /// <param name="s1">A `Tensor`. Must have the same type as `s0`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A tuple of `Tensor` objects (r0, r1).</returns>
        public static (Tensor, Tensor) broadcast_gradient_args(Tensor s0, Tensor s1, string name = "")
        {
            var _op = _op_def_lib._apply_op_helper("BroadcastGradientArgs", name, new { s0, s1 });

            return (_op.outputs[0], _op.outputs[1]);
        }

        public static Tensor reshape<T1, T2>(T1 tensor, T2 shape, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Reshape", name, new { tensor, shape });
            return _op.outputs[0];
        }

        public static Tensor reshape(Tensor tensor, int[] shape, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Reshape", name, new { tensor, shape });
            return _op.outputs[0];
        }

        /// <summary>
        /// Finds unique elements in a 1-D tensor.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="out_idx"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static (Tensor, Tensor) unique(Tensor x, TF_DataType out_idx = TF_DataType.TF_INT32, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Unique", name, new { x, out_idx });
            // TODO
            //var _result = _UniqueOutput._make(_op.outputs);
            return (_op.outputs[0], _op.outputs[1]);
        }

        public static Tensor where()
        {
            throw new NotImplementedException("where");
        }

        public static Tensor one_hot(Tensor indices, int depth,
            Tensor on_value = null,
            Tensor off_value = null,
            TF_DataType dtype = TF_DataType.DtInvalid, 
            int axis = -1,
            string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("OneHot", name, new { indices, depth, on_value, off_value, axis });
            return _op.outputs[0];
        }

        /// <summary>
        /// A placeholder op that passes through `input` when its output is not fed.
        /// </summary>
        /// <param name="input">The default value to produce when output is not fed.</param>
        /// <param name="shape"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor placeholder_with_default<T>(T input, int[] shape, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("PlaceholderWithDefault", name, new { input, shape, name });
            return _op.outputs[0];
        }

        public static Tensor select<Tx, Ty>(Tensor condition, Tx t, Ty e, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Select", name, new { condition, t, e });
            return _op.outputs[0];
        }

        public static Tensor scatter_nd(Tensor indices, Tensor updates, Tensor[] shape, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("ScatterNd", name, new { indices, updates, shape });
            return _op.outputs[0];
        }

        public static Tensor shape(Tensor input, TF_DataType out_type = TF_DataType.TF_INT32, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Shape", name, new { input, out_type });
            return _op.outputs[0];
        }

        /// <summary>
        /// Returns shape of tensors.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="out_type"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor[] shape_n(Tensor[] input, TF_DataType out_type = TF_DataType.TF_INT32, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("ShapeN", name, new { input, out_type });
            return _op.outputs;
        }

        public static Tensor size(Tensor input, TF_DataType out_type = TF_DataType.TF_INT32, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Size", name, new { input, out_type });
            return _op.outputs[0];
        }

        /// <summary>
        /// Return a slice from 'input'
        /// </summary>
        /// <param name="input"></param>
        /// <param name="begin"></param>
        /// <param name="size"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor slice(Tensor input, Tensor begin, Tensor size, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Slice", name, new { input, begin, size });
            return _op.outputs[0];
        }

        public static Tensor[] split(Tensor axis, Tensor value, int num_split, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Split", name, new { split_dim = axis, value, num_split });
            return _op.outputs;
        }

        public static Tensor tile(Tensor input, Tensor multiples, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Tile", name, new { input, multiples });
            return _op.outputs[0];
        }
        public static Tensor tile(NDArray input, int[] multiples, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Tile", name, new { input, multiples });
            return _op.outputs[0];
        }

        public static Tensor transpose<T1, T2>(T1 x, T2 perm, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Transpose", name, new { x, perm });
            return _op.outputs[0];
        }

        public static Tensor zeros_like(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("ZerosLike", name, new { x });
            return _op.outputs[0];
        }

        public static Tensor stop_gradient(Tensor x, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("StopGradient", name, args: new { input = x, name });

            return _op.outputs[0];
        }

        public static Tensor strided_slice(Tensor input, Tensor begin, Tensor end, Tensor strides,
            int begin_mask = 0,
            int end_mask = 0,
            int ellipsis_mask = 0,
            int new_axis_mask = 0,
            int shrink_axis_mask = 0,
            string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("StridedSlice", name, new
            {
                input,
                begin,
                end,
                strides,
                begin_mask,
                end_mask,
                ellipsis_mask,
                new_axis_mask,
                shrink_axis_mask
            });

            return _op.outputs[0];
        }

        public static Tensor slice<Tb, Ts>(Tensor input, Tb[] begin, Ts[] size, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("Slice", name, new { input, begin, size });
            return _op.outputs[0];
        }

        /// <summary>
        /// Removes dimensions of size 1 from the shape of a tensor.
        /// Given a tensor `input`, this operation returns a tensor of the same type with
        /// all dimensions of size 1 removed.If you don't want to remove all size 1
        /// dimensions, you can remove specific size 1 dimensions by specifying
        /// `axis`.
        /// </summary>
        /// <param name="input"> A `Tensor`. The `input` to squeeze.</param>
        /// <param name="axis"> An optional list of `ints`. Defaults to `[]`. If specified, only squeezes the dimensions listed.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns> A `Tensor`. Has the same type as `input`.</returns>
        public static Tensor squeeze(Tensor input, int[] axis = null, string name = null)
        {
            if (axis == null) axis = new int[0];
            var _op = _op_def_lib._apply_op_helper("Squeeze", name, args: new { input, squeeze_dims = axis });

            return _op.outputs[0];
        }

        /// <summary>
        /// Return the shape of s0 op s1 with broadcast.
        /// Given `s0` and `s1`, tensors that represent shapes, compute `r0`, the
        /// broadcasted shape. `s0`, `s1` and `r0` are all integer vectors.
        /// </summary>
        /// <param name="s0"> A `Tensor`. Must be one of the following types: `int32`, `int64`.</param>
        /// <param name="s1"> A `Tensor`. Must have the same type as `s0`.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns> `Tensor`. Has the same type as `s0`.</returns>
        public static Tensor broadcast_args(Tensor s0, Tensor s1, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("BroadcastArgs", name, args: new { s0, s1, name });

            return _op.outputs[0];
        }
    }
}
