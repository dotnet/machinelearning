using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;
using Tensorflow.Operations.Activation;

namespace Tensorflow
{
    public static partial class tf
    {
        public static class nn
        {
            /// <summary>
            /// Computes dropout.
            /// </summary>
            /// <param name="x">A floating point tensor.</param>
            /// <param name="keep_prob">(deprecated) A deprecated alias for `(1-rate)`.</param>
            /// <param name="noise_shape"></param>
            /// <param name="seed">Used to create random seeds.</param>
            /// <param name="name"></param>
            /// <param name="rate">A scalar `Tensor` with the same type as `x`.</param>
            /// <returns>A Tensor of the same shape of `x`.</returns>
            public static Tensor dropout(Tensor x, Tensor keep_prob = null, Tensor noise_shape = null, int? seed = null, string name = null,
                float? rate = null)
            {
                Tensor keep = null;
                if (keep_prob != null)
                    keep = 1.0f - keep_prob;
                var rate_tensor = keep;

                return nn_ops.dropout_v2(x, rate: rate_tensor, noise_shape: noise_shape, seed: seed, name: name);
            }

            public static (Tensor, Tensor) moments(Tensor x,
                int[] axes,
                string name = null,
                bool keep_dims = false) => nn_impl.moments(x, 
                    axes, 
                    name: name, 
                    keep_dims: keep_dims);

            public static Tensor embedding_lookup(RefVariable @params,
                Tensor ids,
                string partition_strategy = "mod",
                string name = null) => embedding_ops._embedding_lookup_and_transform(@params,
                    ids,
                    partition_strategy: partition_strategy,
                    name: name);

            public static Tensor embedding_lookup(Tensor @params,
                Tensor ids,
                string partition_strategy = "mod",
                string name = null) => embedding_ops._embedding_lookup_and_transform(new Tensor[] { @params },
                    ids,
                    partition_strategy: partition_strategy,
                    name: name);

            public static IActivation relu() => new relu();

            public static Tensor relu(Tensor features, string name = null) => gen_nn_ops.relu(features, name);

            public static Tensor[] fused_batch_norm(Tensor x,
                RefVariable scale,
                RefVariable offset,
                Tensor mean = null,
                Tensor variance = null,
                float epsilon = 0.001f,
                string data_format = "NHWC",
                bool is_training = true,
                string name = null) => nn_impl.fused_batch_norm(x, scale, offset, mean, variance,
                    epsilon: epsilon,
                    data_format: data_format,
                    is_training: is_training,
                    name: name);

            public static IPoolFunction max_pool => new MaxPoolFunction();

            public static Tensor[] top_k(Tensor input, int k = 1, bool sorted = true, string name = null)
                => gen_nn_ops.top_kv2(input, k: k, sorted: sorted, name: name);

            public static Tensor bias_add(Tensor value, RefVariable bias, string data_format = null, string name = null)
            {
                return Python.with(ops.name_scope(name, "BiasAdd", new { value, bias }), scope =>
                {
                    name = scope;
                    return gen_nn_ops.bias_add(value, bias, data_format: data_format, name: name);
                });
            }

            public static Tensor softmax(Tensor logits, int axis = -1, string name = null)
                => gen_nn_ops.softmax(logits, name);

            /// <summary>
            /// Computes sparse softmax cross entropy between `logits` and `labels`.
            /// </summary>
            /// <param name="labels"></param>
            /// <param name="logits"></param>
            /// <param name="name"></param>
            /// <returns></returns>
            public static Tensor sparse_softmax_cross_entropy_with_logits(Tensor labels = null,
            Tensor logits = null, string name = null)
                => nn_ops.sparse_softmax_cross_entropy_with_logits(labels: labels, logits: logits, name: name);

            public static Tensor softmax_cross_entropy_with_logits_v2(Tensor labels, Tensor logits, int axis = -1, string name = null)
                => nn_ops.softmax_cross_entropy_with_logits_v2_helper(labels, logits, axis: axis, name: name);
        }
    }
}
