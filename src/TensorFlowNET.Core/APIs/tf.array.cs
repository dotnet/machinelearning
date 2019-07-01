using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        /// <summary>
        /// Concatenates tensors along one dimension.
        /// </summary>
        /// <param name="values">A list of `Tensor` objects or a single `Tensor`.</param>
        /// <param name="axis"></param>
        /// <param name="name"></param>
        /// <returns>A `Tensor` resulting from concatenation of the input tensors.</returns>
        public static Tensor concat(IList<Tensor> values, int axis, string name = "concat")
        {
            if (values.Count == 1)
                throw new NotImplementedException("tf.concat length is 1");

            return gen_array_ops.concat_v2(values.ToArray(), axis, name: name);
        }

        /// <summary>
        /// Inserts a dimension of 1 into a tensor's shape.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="axis"></param>
        /// <param name="name"></param>
        /// <param name="dim"></param>
        /// <returns>
        /// A `Tensor` with the same data as `input`, but its shape has an additional
        /// dimension of size 1 added.
        /// </returns>
        public static Tensor expand_dims(Tensor input, int axis = -1, string name = null, int dim = -1)
            => array_ops.expand_dims(input, axis, name, dim);

        /// <summary>
        /// Creates a tensor filled with a scalar value.
        /// </summary>
        /// <param name="dims"></param>
        /// <param name="value"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor fill<T>(Tensor dims, T value, string name = null)
            => gen_array_ops.fill(dims, value, name: name);

        /// <summary>
        /// Return the elements, either from `x` or `y`, depending on the `condition`.
        /// </summary>
        /// <returns></returns>
        public static Tensor where<Tx, Ty>(Tensor condition, Tx x, Ty y, string name = null)
            => array_ops.where(condition, x, y, name);

        /// <summary>
        /// Transposes `a`. Permutes the dimensions according to `perm`.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="perm"></param>
        /// <param name="name"></param>
        /// <param name="conjugate"></param>
        /// <returns></returns>
        public static Tensor transpose<T1>(T1 a, int[] perm = null, string name = "transpose", bool conjugate = false)
            => array_ops.transpose(a, perm, name, conjugate);

        public static Tensor squeeze(Tensor input, int[] axis = null, string name = null, int squeeze_dims = -1)
            => gen_array_ops.squeeze(input, axis, name);

        /// <summary>
        /// Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="axis"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor stack(object values, int axis = 0, string name = "stack")
            => array_ops.stack(values, axis, name: name);

        public static Tensor one_hot(Tensor indices, int depth,
            Tensor on_value = null,
            Tensor off_value = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            int axis = -1,
            string name = null) => array_ops.one_hot(indices, depth, dtype: dtype, axis: axis, name: name);

        /// <summary>
        /// A placeholder op that passes through `input` when its output is not fed.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="input">A `Tensor`. The default value to produce when output is not fed.</param>
        /// <param name="shape">
        /// A `tf.TensorShape` or list of `int`s. The (possibly partial) shape of
        /// the tensor.
        /// </param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor`. Has the same type as `input`.</returns>
        public static Tensor placeholder_with_default<T>(T input, int[] shape, string name = null)
            => gen_array_ops.placeholder_with_default(input, shape, name: name);
    }
}
