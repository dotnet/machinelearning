using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow
{
    public class LossesImpl
    {
        public Tensor compute_weighted_loss(Tensor losses, Tensor weights = null, string scope = null,
            string loss_collection = ops.GraphKeys.LOSSES, string reduction = Reduction.SUM_BY_NONZERO_WEIGHTS)
        {
            return with(ops.name_scope(scope, default_name: "weighted_loss", (losses, weights)), delegate
            {
                // Save the `reduction` argument for loss normalization when distributing
                // to multiple replicas. Used only for estimator + v1 optimizer flow.
                ops.get_default_graph()._last_loss_reduction = reduction;

                /*var dp = weights_broadcast_ops.assert_broadcastable(weights, losses);
                with(ops.control_dependencies(dp), delegate
                {

                });*/

                losses = ops.convert_to_tensor(losses);
                var input_dtype = losses.dtype;
                losses = math_ops.cast(losses, dtype: dtypes.float32);
                weights = math_ops.cast(weights, dtype: dtypes.float32);
                var weighted_losses = math_ops.multiply(losses, weights);
                Tensor loss = null;
                if (reduction == Reduction.NONE)
                    loss = weighted_losses;
                else
                {
                    loss = math_ops.reduce_sum(weighted_losses);
                    if (reduction == Reduction.MEAN)
                        loss = _safe_mean(
                            loss, math_ops.reduce_sum(array_ops.ones_like(losses) * weights));
                    else if (reduction == Reduction.SUM_BY_NONZERO_WEIGHTS ||
                          reduction == Reduction.SUM_OVER_NONZERO_WEIGHTS)
                        loss = _safe_mean(loss, _num_present(losses, weights));
                    else if (reduction == Reduction.SUM_OVER_BATCH_SIZE)
                        loss = _safe_mean(loss, _num_elements(losses));
                }

                // Convert the result back to the input type.
                loss = math_ops.cast(loss, input_dtype);
                Operations.Losses.Util.add_loss(loss, loss_collection);
                return loss;
            });
        }

        public Tensor _safe_mean(Tensor losses, Tensor num_present)
        {
            var total_loss = math_ops.reduce_sum(losses);
            return math_ops.div_no_nan(total_loss, num_present, name: "value");
        }

        public Tensor _num_elements(Tensor losses)
        {
            throw new NotImplementedException("LossesImpl._num_elements");
        }

        public Tensor _num_present(Tensor losses, Tensor weights, bool per_batch = false)
        {
            return with(ops.name_scope(null, default_name: "num_present", (losses, weights)), name_scope =>
            {
                string scope = name_scope;
                weights = math_ops.cast(weights, dtype: dtypes.float32);
                var present = array_ops.where(
                    math_ops.equal(weights, 0.0),
                    array_ops.zeros_like(weights),
                    array_ops.ones_like(weights));
                present = weights_broadcast_ops.broadcast_weights(present, losses);

                if (per_batch)
                    return math_ops.reduce_sum(
                        present,
                        axis: math_ops.range(1, array_ops.rank(present)),
                        keepdims: true,
                        name: scope);
                return math_ops.reduce_sum(present, name: scope);
            });
        }

        public Tensor sparse_softmax_cross_entropy(Tensor labels, 
            Tensor logits,
            float weights = 1.0f,
            string scope = null,
            string loss_collection= ops.GraphKeys.LOSSES,
            string reduction = Reduction.SUM_BY_NONZERO_WEIGHTS)
        {
            return with(ops.name_scope(scope,
                "sparse_softmax_cross_entropy_loss",
                (logits, labels, weights)),
                name_scope =>
                {
                    scope = name_scope;
                    Tensor weights_tensor = null;
                    (labels, logits, weights_tensor) = _remove_squeezable_dimensions(
                        labels, logits, weights, expected_rank_diff: 1);

                    var losses = nn_ops.sparse_softmax_cross_entropy_with_logits(labels: labels,
                                                         logits: logits,
                                                         name: "xentropy");
                    return compute_weighted_loss(losses, weights_tensor, scope, loss_collection, reduction: reduction);
                });
        }

        public (Tensor, Tensor, Tensor) _remove_squeezable_dimensions(Tensor labels,
            Tensor predictions,
            float weights = 0,
            int expected_rank_diff = 0)
        {
            (labels, predictions) = confusion_matrix.remove_squeezable_dimensions(
                labels, predictions, expected_rank_diff: expected_rank_diff);

            if(weights > 0)
            {
                var weights_tensor = ops.convert_to_tensor(weights);
                var labels_rank = labels.GetShape().NDim;
                var weights_shape = weights_tensor.GetShape();
                var weights_rank = weights_shape.NDim;

                if (labels_rank > -1 && weights_rank > -1)
                {
                    // Use static rank.
                    var rank_diff = weights_rank - labels_rank;
                    if (rank_diff == 1)
                        weights = array_ops.squeeze(weights_tensor, new int[] { -1 });
                    return (labels, predictions, weights_tensor);
                }

                // Use dynamic rank.
                throw new NotImplementedException("_remove_squeezable_dimensions dynamic rank");
            }

            throw new NotImplementedException("_remove_squeezable_dimensions");
        }
    }
}
