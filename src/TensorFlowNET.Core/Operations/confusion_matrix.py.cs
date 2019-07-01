using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow
{
    public class confusion_matrix
    {
        /// <summary>
        /// Squeeze last dim if ranks differ from expected by exactly 1.
        /// </summary>
        /// <param name="labels"></param>
        /// <param name="predictions"></param>
        /// <param name="expected_rank_diff"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static (Tensor, Tensor) remove_squeezable_dimensions(Tensor labels,
            Tensor predictions,
            int expected_rank_diff = 0,
            string name = null)
        {
            return with(ops.name_scope(name, default_name: "remove_squeezable_dimensions", (labels, predictions)), delegate
            {
                predictions = ops.convert_to_tensor(predictions);
                labels = ops.convert_to_tensor(labels);
                var predictions_shape = predictions.GetShape();
                var predictions_rank = predictions_shape.NDim;
                var labels_shape = labels.GetShape();
                var labels_rank = labels_shape.NDim;
                if(labels_rank > -1 && predictions_rank > -1)
                {
                    // Use static rank.
                    var rank_diff = predictions_rank - labels_rank;
                    if (rank_diff == expected_rank_diff + 1)
                        predictions = array_ops.squeeze(predictions, new int[] { -1 });
                    else if (rank_diff == expected_rank_diff - 1)
                        labels = array_ops.squeeze(labels, new int[] { -1 });
                    return (labels, predictions);
                }

                // Use dynamic rank.
                throw new NotImplementedException("remove_squeezable_dimensions dynamic rank");
            });
        }
    }
}
