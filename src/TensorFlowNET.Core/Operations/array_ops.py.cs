using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow
{
    public class array_ops
    {
        public static Tensor placeholder_with_default<T>(T input, int[] shape, string name = null) 
            => gen_array_ops.placeholder_with_default(input, shape, name);

        public static Tensor prevent_gradient(Tensor input, string message = "", string name = null)
            => gen_array_ops.prevent_gradient(input, message: message, name: name);

        public static Tensor zeros(Shape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
        {
            dtype = dtype.as_base_dtype();
            return with(ops.name_scope(name, "zeros", shape), scope =>
            {
                name = scope;
                switch (dtype)
                {
                    case TF_DataType.TF_BOOL:
                        return _constant_if_small(false, shape, dtype, name);
                    case TF_DataType.TF_DOUBLE:
                        return _constant_if_small(0.0D, shape, dtype, name);
                    case TF_DataType.TF_FLOAT:
                        return _constant_if_small(0.0F, shape, dtype, name);
                    case TF_DataType.TF_INT32:
                        return _constant_if_small(0, shape, dtype, name);
                    default:
                        throw new TypeError("can't find type for zeros");
                }
            });
        }

        public static Tensor zeros(Tensor shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
        {
            dtype = dtype.as_base_dtype();
            return with(ops.name_scope(name, "zeros", shape), scope =>
            {
                name = scope;
                switch (dtype)
                {
                    case TF_DataType.TF_BOOL:
                        return gen_array_ops.fill(shape, tf.constant(false, dtype: dtype), name: name);
                    case TF_DataType.TF_DOUBLE:
                        return gen_array_ops.fill(shape, tf.constant(0.0D, dtype: dtype), name: name);
                    case TF_DataType.TF_FLOAT:
                        return gen_array_ops.fill(shape, tf.constant(0.0F, dtype: dtype), name: name);
                    case TF_DataType.TF_INT32:
                        return gen_array_ops.fill(shape, tf.constant(0, dtype: dtype), name: name);
                    default:
                        throw new TypeError("can't find type for zeros");
                }
                
            });
        }

        private static Tensor _constant_if_small(int value, Tensor shape)
        {
            return shape < 1000;
        }

        private static Tensor _constant_if_small<T>(T value, Shape shape, TF_DataType dtype, string name)
        {
            Tensor tShape = null;
            if (shape.Size < 1000)
            {
                return constant_op.constant(value, shape: shape, dtype: dtype, name: name);
            }
            else
            {
                tShape = constant_op._tensor_shape_tensor_conversion_function(shape.as_shape());
                var c = constant_op.constant(0, dtype: dtype);
                return gen_array_ops.fill(tShape, c, name: name);
            }
        }

        public static Tensor _autopacking_conversion_function(object[] v, TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool as_ref = false)
        {
            var inferred_dtype = _get_dtype_from_nested_lists(v);
            if (dtype == TF_DataType.DtInvalid)
                dtype = inferred_dtype;

            return _autopacking_helper(v, dtype, name == null ? "packed" : name);
        }

        private static TF_DataType _get_dtype_from_nested_lists(object[] list_or_tuple)
        {
            TF_DataType dtype = TF_DataType.DtInvalid;

            foreach(var obj in list_or_tuple)
            {
                switch (obj)
                {
                    case Tensor t:
                        dtype = t.dtype.as_base_dtype();
                        break;
                }

                if (dtype != TF_DataType.DtInvalid)
                    break;
            }

            return dtype;
        }

        public static Tensor _autopacking_helper(object[] list_or_tuple, TF_DataType dtype, string name)
        {
            var must_pack = false;
            var converted_elems = new List<object>();
            return with(ops.name_scope(name), scope =>
            {
                foreach (var (i, elem) in enumerate(list_or_tuple))
                {
                    converted_elems.Add(elem);
                    must_pack = true;
                }

                if(must_pack)
                {
                    var elems_as_tensors = new List<Tensor>();
                    foreach (var (i, elem) in enumerate(converted_elems))
                    {
                        if (elem is Tensor tensor)
                            elems_as_tensors.Add(tensor);
                        else
                        {
                            var elem_tensor = constant_op.constant(elem, dtype: dtype, name: i.ToString());
                            elems_as_tensors.Add(elem_tensor);
                        }
                    }

                    return gen_array_ops.pack(elems_as_tensors.ToArray(), name: scope);
                }
                else
                {
                    // return converted_elems.ToArray();
                    throw new NotImplementedException("_autopacking_helper.converted_elems");
                }
            });
        }

        public static Tensor expand_dims(Tensor input, int axis = -1, string name = null, int dim = -1) 
            => expand_dims_v2(input, axis, name);

        private static Tensor expand_dims_v2(Tensor input, int axis, string name = null) 
            => gen_array_ops.expand_dims(input, axis, name);

        /// <summary>
        /// Returns the rank of a tensor.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor rank(Tensor input, string name = null)
            => rank_internal(input, name, optimize: true);

        public static Tensor rank_internal(Tensor input, string name = null, bool optimize = true)
        {
            return with(ops.name_scope(name, "Rank", new List<Tensor> { input }), scope =>
            {
                name = scope;
                var input_tensor = ops.convert_to_tensor(input);
                var input_shape = tensor_util.to_shape(input_tensor.shape);
                if (optimize && input_shape.NDim > 0)
                    return constant_op.constant(input_shape.NDim, dtype: tf.int32, name: name);
                else
                    return gen_array_ops.rank(input, name);
            });
        }

        /// <summary>
        /// Creates a tensor with all elements set to 1.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        /// <param name="optimize"></param>
        /// <returns></returns>
        public static Tensor ones_like<T>(T tensor, TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool optimize = true)
            => ones_like_impl(tensor, dtype, name, optimize);

        public static Tensor reshape<T1, T2>(T1 tensor, T2 shape, string name = null)
            => gen_array_ops.reshape(tensor, shape, null);

        private static Tensor ones_like_impl<T>(T tensor, TF_DataType dtype, string name, bool optimize = true)
        {
            return with(ops.name_scope(name, "ones_like", new { tensor }), scope =>
            {
                name = scope;
                var tensor1 = ops.convert_to_tensor(tensor, name: "tensor");
                var ones_shape = shape_internal(tensor1, optimize: optimize);
                if (dtype == TF_DataType.DtInvalid)
                    dtype = tensor1.dtype;
                var ret = ones(ones_shape, dtype: dtype, name: name);
                ret.shape = tensor1.shape;
                return ret;
            });
        }

        public static Tensor ones(Tensor shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
        {
            dtype = dtype.as_base_dtype();
            return with(ops.name_scope(name, "ones", new { shape }), scope =>
            {
                name = scope;
                var output = gen_array_ops.fill(shape, constant_op.constant(1.0f, dtype: dtype), name: name);
                return output;
            });
        }

        public static Tensor ones(Tensor[] shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
        {
            dtype = dtype.as_base_dtype();
            return with(ops.name_scope(name, "ones", new { shape }), scope =>
            {
                name = scope;
                var output = _constant_if_small(1, shape[0]);
                var shape1 = ops.convert_to_tensor(shape, dtype: TF_DataType.TF_INT32);
                output = gen_array_ops.fill(shape1, constant_op.constant(1, dtype: dtype), name: name);
                return output;
            });
        }

        public static Tensor ones(int[] dims, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
        {
            dtype = dtype.as_base_dtype();
            return with(ops.name_scope(name, "ones", new { dims }), scope =>
            {
                name = scope;
                var shape = ops.convert_to_tensor(dims, dtype: TF_DataType.TF_INT32);
                var output = gen_array_ops.fill(shape, constant_op.constant(1.0f, dtype: dtype), name: name);
                return output;
            });
        }

        public static Tensor one_hot(Tensor indices, int depth, 
            Tensor on_value = null,
            Tensor off_value = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            int axis = -1,
            string name = null)
        {
            return with(ops.name_scope(name, "one_hot", new { indices, depth, dtype }), scope =>
            {
                name = scope;
                var on_exists = false;
                var off_exists = false;
                var on_dtype = TF_DataType.DtInvalid;
                var off_dtype = TF_DataType.DtInvalid;

                if (dtype == TF_DataType.DtInvalid)
                    dtype = TF_DataType.TF_FLOAT;

                if(!on_exists)
                {
                    on_value = ops.convert_to_tensor(1, dtype, name: "on_value");
                    on_dtype = dtype;
                }

                if (!off_exists)
                {
                    off_value = ops.convert_to_tensor(0, dtype, name = "off_value");
                    off_dtype = dtype;
                }

                return gen_array_ops.one_hot(indices, depth,
                    on_value: on_value,
                    off_value: off_value,
                    axis: axis,
                    name: name);
            });
        }

        public static (Tensor, Tensor) unique(Tensor x, TF_DataType out_idx = TF_DataType.TF_INT32, string name = null)
            => gen_array_ops.unique(x, out_idx: out_idx, name: name);

        public static Tensor where(Tensor condition, object x = null, object y = null, string name = null)
        {
            if( x == null && y == null)
            {
                throw new NotImplementedException("where");
            }
            else if(x != null && y != null)
            {
                return gen_array_ops.select(condition, x, y, name);
            }
            else
            {
                throw new ValueError("x and y must both be non-None or both be None.");
            }
        }

        /// <summary>
        /// Returns the shape of a tensor.
        /// </summary>
        /// <param name="input">A `Tensor` or `SparseTensor`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <param name="out_type">
        /// (Optional) The specified output type of the operation
        /// (`int32` or `int64`). Defaults to `tf.int32`.
        /// </param>
        /// <returns>A `Tensor` of type `out_type`.</returns>
        public static Tensor shape(Tensor input, string name = null, TF_DataType out_type = TF_DataType.TF_INT32)
            => shape_internal(input, name, optimize: true, out_type: out_type);

        public static Tensor size(Tensor input, string name = null, bool optimize = true, TF_DataType out_type = TF_DataType.TF_INT32)
            => size_internal(input, name, optimize: optimize, out_type: out_type);

        private static Tensor shape_internal(Tensor input, string name = null, bool optimize = true, TF_DataType out_type = TF_DataType.TF_INT32)
        {
            return with(ops.name_scope(name, "Shape", new { input }), scope =>
            {
                name = scope;

                if (!tf.context.executing_eagerly())
                {
                    var input_tensor = ops.convert_to_tensor(input);
                    var input_shape = tensor_util.to_shape(input_tensor.shape);
                    if (optimize && input_tensor.NDims > -1 && input_shape.is_fully_defined())
                    {
                        var nd = np.array(input_tensor.shape).astype(out_type.as_numpy_datatype());
                        return constant_op.constant(nd, name: name);
                    }
                }

                return gen_array_ops.shape(input, name: name, out_type: out_type);
            });
        }

        private static Tensor size_internal(Tensor input, string name = null, bool optimize = true, TF_DataType out_type = TF_DataType.TF_INT32)
        {
            return with(ops.name_scope(name, "Size", new { input }), scope =>
            {
                name = scope;

                var input_tensor = ops.convert_to_tensor(input);
                var input_shape = tensor_util.to_shape(input_tensor.shape);
                if (optimize)
                {
                    if (input_shape.is_fully_defined())
                    {
                        return constant_op.constant(input_shape.Size, dtype: out_type, name: name);
                    }
                }

                return gen_array_ops.size(input, name: name, out_type: out_type);
            });
        }

        public static Tensor zeros_like(Tensor tensor, TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool optimize = true)
        {
            return with(ops.name_scope(name, "zeros_like", new Tensor[] { tensor }), scope =>
            {
                name = scope;
                tensor = ops.convert_to_tensor(tensor, name: "tensor");

                // is_fully_defined return unexpected value.
                if (optimize && tensor_util.to_shape(tensor.shape).is_fully_defined() && dtype != TF_DataType.TF_VARIANT)
                {

                }

                if(dtype != TF_DataType.DtInvalid && dtype != tensor.dtype && dtype != TF_DataType.TF_VARIANT)
                {
                    throw new NotImplementedException("zeros_like");
                    // return zeros(shape_internal(tensor, optimize: optimize), dtype: dtype, name: name);
                }
                else
                {
                    return gen_array_ops.zeros_like(tensor, name: name);
                }
            });
        }

        /// <summary>
        ///   When building ops to compute gradients, this op prevents the contribution of
        ///   its inputs to be taken into account.Normally, the gradient generator adds ops
        ///   to a graph to compute the derivatives of a specified 'loss' by recursively
        ///   finding out inputs that contributed to its computation.If you insert this op
        ///   in the graph it inputs are masked from the gradient generator.  They are not
        ///   taken into account for computing gradients.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor stop_gradient(Tensor input, string name = null)
            => gen_array_ops.stop_gradient(input,  name);

        /// <summary>
        /// Extracts a strided slice of a tensor (generalized python array indexing).
        /// </summary>
        /// <param name="input_"></param>
        /// <param name="begin"></param>
        /// <param name="end"></param>
        /// <param name="strides"></param>
        /// <param name="begin_mask"></param>
        /// <param name="end_mask"></param>
        /// <param name="ellipsis_mask"></param>
        /// <param name="new_axis_mask"></param>
        /// <param name="shrink_axis_mask"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor strided_slice(Tensor input_, Tensor begin, Tensor end, 
            Tensor strides = null,
            int begin_mask = 0,
            int end_mask = 0,
            int ellipsis_mask = 0,
            int new_axis_mask = 0,
            int shrink_axis_mask = 0,
            string name = null)
        {
            var op = gen_array_ops.strided_slice(
                input: input_,
                begin: begin,
                end: end,
                strides: strides,
                begin_mask: begin_mask,
                end_mask: end_mask,
                ellipsis_mask: ellipsis_mask,
                new_axis_mask: new_axis_mask,
                shrink_axis_mask: shrink_axis_mask,
                name: name);

            string parent_name = name;

            return op;
        }

        /// <summary>
        /// Removes dimensions of size 1 from the shape of a tensor.
        /// Given a tensor `input`, this operation returns a tensor of the same type with
        /// all dimensions of size 1 removed.If you don't want to remove all size 1
        /// dimensions, you can remove specific size 1 dimensions by specifying
        /// `axis`.
        /// </summary>
        /// <param name="input"> A `Tensor`. The `input` to squeeze.</param>
        /// <param name="axis"> An optional list of `ints`. Defaults to `[]`.
        /// If specified, only squeezes the dimensions listed.The dimension
        /// index starts at 0. It is an error to squeeze a dimension that is not 1.
        /// Must be in the range `[-rank(input), rank(input))`.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <param name="squeeze_dims" >Deprecated keyword argument that is now axis.</param>
        /// <returns>A `Tensor`. Has the same type as `input`.
        /// Contains the same data as `input`, but has one or more dimensions of
        /// size 1 removed.</returns>
        public static Tensor squeeze(Tensor input, int[] axis = null, string name = null, int[] squeeze_dims = null)
            => gen_array_ops.squeeze(input, axis, name);

        public static Tensor identity(Tensor input, string name = null)
            => gen_array_ops.identity(input, name);

        public static Tensor invert_permutation(Tensor x, string name = null)
            => gen_array_ops.invert_permutation(x, name: name);

        /// <summary>
        /// Computes the shape of a broadcast given symbolic shapes.
        /// When shape_x and shape_y are Tensors representing shapes(i.e.the result of
        /// calling tf.shape on another Tensor) this computes a Tensor which is the shape
        /// of the result of a broadcasting op applied in tensors of shapes shape_x and
        /// shape_y.
        /// For example, if shape_x is [1, 2, 3] and shape_y is [5, 1, 3], the result is a
        /// Tensor whose value is [5, 2, 3].
        /// This is useful when validating the result of a broadcasting operation when the
        /// tensors do not have statically known shapes.
        /// </summary>
        /// <param name="shape_x"> A rank 1 integer `Tensor`, representing the shape of x.</param>
        /// <param name="shape_y"> A rank 1 integer `Tensor`, representing the shape of y.</param>
        /// <returns> A rank 1 integer `Tensor` representing the broadcasted shape.</returns>
        public static Tensor broadcast_dynamic_shape(Tensor shape_x, Tensor shape_y)
            => gen_array_ops.broadcast_args(shape_x, shape_y);

        public static Tensor broadcast_static_shape(Tensor shape_x, Tensor shape_y)
            => Framework.common_shapes.broadcast_shape(shape_x, shape_y);

        /// <summary>
        /// Concatenates tensors along one dimension.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="axis"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor concat(Tensor[] values, int axis, string name = "concat")
        {
            if(values.Length == 1) // Degenerate case of one tensor.
            {
                return with(ops.name_scope(name), scope => {
                    var t = ops.convert_to_tensor(axis, name: "concat_dim", dtype: TF_DataType.TF_INT32);
                    return identity(values[0], name = scope);
                });
            }

            return gen_array_ops.concat_v2(values, axis, name: name);
        }

        public static Tensor gather(Tensor @params, Tensor indices, string name = null, int axis = 0)
            => gen_array_ops.gather_v2(@params, indices, axis, name: name);

        public static Tensor transpose<T1, T2>(T1 a, T2 perm, string name = "transpose", bool conjugate = false)
        {
            return with(ops.name_scope(name, "transpose", new { a }), scope =>
            {
                return gen_array_ops.transpose(a, perm, name: scope);
            });
        }

        public static Tensor slice<Tb, Ts>(Tensor input, Tb[] begin, Ts[] size, string name = null)
            => gen_array_ops.slice(input, begin, size, name: name);

        public static Tensor stack(object values, int axis = 0, string name = "stack")
        {
            if (axis == 0)
                // If the input is a constant list, it can be converted to a constant op
                return ops.convert_to_tensor(values, name: name);

            throw new NotImplementedException("array_ops.stack");
        }

        public static Tensor placeholder(TF_DataType dtype)
        {
            throw new NotImplementedException("array_ops.placeholder");
        }
    }
}
