using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Framework;
using Tensorflow.Operations;
using static Tensorflow.Python;

namespace Tensorflow.Gradients
{
    /// <summary>
    /// tensorflow\python\ops\array_grad.py
    /// </summary>
    [RegisterGradient("array_grad")]
    public class array_grad
    {
        [RegisterGradient("ConcatV2")]
        public static Tensor[] _ConcatGradV2(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            return _ConcatGradHelper(op, grad, start_value_index: 0, end_value_index: -1, dim_index: -1);
        }

        /// <summary>
        /// Gradient for concat op.
        /// </summary>
        /// <param name="op">An operation.</param>
        /// <param name="grad">
        /// `Tensor` or `IndexedSlices` representing the gradients with respect
        /// to each output of the op.
        /// </param>
        /// <param name="start_value_index">An integer index of the first value in the op.inputs.</param>
        /// <param name="end_value_index">An integer index of the last value in the op.inputs.</param>
        /// <param name="dim_index">An interger index of concat_dim or axis parameter in op.inputs.</param>
        /// <returns>
        /// Tensors representing the partial gradients with respect to each input
        /// of the op.
        /// </returns>
        private static Tensor[] _ConcatGradHelper(Operation op, Tensor grad, int start_value_index, int end_value_index, int dim_index)
        {
            // Degenerate concatenation, just return grad.
            if (len(op.inputs) == 2)
                return end_value_index <= dim_index ? new Tensor[] { grad, null } : new Tensor[] { null, grad };

            var concat_dim = op.inputs[dim_index];
            var input_values = op.inputs._inputs.Skip(start_value_index)
                .Take(end_value_index == -1 ? op.inputs.Length - 1 : end_value_index - start_value_index)
                .ToArray();

            var out_grads = new List<Tensor>();
            if (constant_op.is_constant(concat_dim))
            {
                /*If concat_dim is a constant defined in a different context,
                then we duplicate it in the current context to avoid passing it
                through an Enter node.
                This is a small optimization in general, but it is required when
                compiling with XLA, as XLA needs the concat input to be folded into a
                constant.*/
                var grad_context = control_flow_util.GetOutputContext(grad.op);
                var dim_context = control_flow_util.GetOutputContext(concat_dim.op);
                if (dim_context != grad_context)
                {
                    var value = tensor_util.constant_value(concat_dim);
                    concat_dim = constant_op.constant(value: value, dtype: concat_dim.dtype);
                }
            }

            // Using mod here for convenience since concat_dim is already verified
            // in concat implementation to be within the allowed [-rank, rank) range.
            var non_neg_concat_dim = concat_dim % array_ops.rank(input_values[0]);

            // Get the inputs' tensor shapes
            var sizes = _ExtractInputShapes(input_values);

            /* The magic number of 16 was found through benchmarking a range of sizes
             on CPUs and a Maxwell TitanX.  A speedup was seen in a large majority of
             cases when switching implementations at N=16, but it is possible that
             there will be a small number of performance regressions.*/
            if (len(sizes) > 16)
            {
                // extract the size of each input along the concat dimension
                var slice = array_ops.slice(array_ops.stack(sizes, axis: 1),
                        new Tensor[] { non_neg_concat_dim, tf.constant(0) },
                        new Tensor[] { tf.constant(1), tf.constant(-1) });
                var squeeze_sizes = array_ops.squeeze(slice);
                out_grads = gen_array_ops.split(grad, squeeze_sizes, non_neg_concat_dim).ToList();
            }
            else
            {
                var offset = gen_array_ops.concat_offset(non_neg_concat_dim, sizes);
                foreach (var (begin, size) in zip(offset, sizes))
                    out_grads.Add(gen_array_ops.slice(grad, begin, size));
            }

            return (end_value_index <= dim_index ? 
                out_grads.ToArray().Concat(new Tensor[] { null }) : 
                new Tensor[] { null }.Concat(out_grads)).ToArray();
        }

        [RegisterGradient("ExpandDims")]
        public static Tensor[] _ExpandDimsGrad(Operation op, Tensor[] grads)
        {
            return new Tensor[] { _ReshapeToInput(op, grads[0]), null };
        }

        /// <summary>
        /// Extract the shapes of a set of input tensors.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        private static Tensor[] _ExtractInputShapes(Tensor[] inputs)
        {
            var sizes = new Tensor[inputs.Length];
            bool fully_known = true;
            for(int i = 0; i < inputs.Length; i++)
            {
                var x = inputs[i];

                var input_shape = array_ops.shape(x);
                if (!(input_shape is Tensor) || input_shape.op.type != "Const")
                {
                    fully_known = false;
                    break;
                }

              sizes[i] = input_shape;
            }

            if (fully_known)
                return sizes;
            else
                return gen_array_ops.shape_n(inputs);
        }

        /// <summary>
        /// Gradient for GatherV2 op.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="grads"></param>
        /// <returns></returns>
        [RegisterGradient("GatherV2")]
        public static Tensor[] _GatherV2Grad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var @params = op.inputs[0];
            ops.colocate_with(@params);

            var params_shape = array_ops.shape(@params, out_type: tf.int64);
            params_shape = math_ops.cast(params_shape, tf.int32);

            var indices = op.inputs[1];
            var indices_size = array_ops.expand_dims(array_ops.size(indices), 0);
            var axis = op.inputs[2];
            var axis_static = tensor_util.constant_value(axis);

            // For axis 0 gathers, build an appropriately shaped IndexedSlices.
            if((int)axis_static == 0)
            {
                var params_tail_shape = params_shape.slice(new NumSharp.Slice(start:1));
                var values_shape = array_ops.concat(new[] { indices_size, params_tail_shape }, 0);
                var values = array_ops.reshape(grad, values_shape);
                indices = array_ops.reshape(indices, indices_size);
                return new Tensor[]
                {
                    new IndexedSlices(values, indices, params_shape),
                    null,
                    null
                };
            }

            return new Tensor[] { null, null };
        }

        [RegisterGradient("Reshape")]
        public static Tensor[] _ReshapeGrad(Operation op, Tensor[] grads)
        {
            return new Tensor[] { array_ops.reshape(grads[0], array_ops.shape(op.inputs[0])), null };
        }

        [RegisterGradient("Squeeze")]
        public static Tensor[] _SqueezeGrad(Operation op, Tensor[] grads)
        {
            return new Tensor[] { _ReshapeToInput(op, grads[0]) };
        }

        private static Tensor _ReshapeToInput(Operation op, Tensor grad)
        {
            return array_ops.reshape(grad, array_ops.shape(op.inputs[0]));
        }

        [RegisterGradient("Transpose")]
        public static Tensor[] _TransposeGrad(Operation op, Tensor[] grads)
        {
            var p = op.inputs[1];
            return new Tensor[] { array_ops.transpose(grads[0], array_ops.invert_permutation(p)), null };
        }
    }
}
