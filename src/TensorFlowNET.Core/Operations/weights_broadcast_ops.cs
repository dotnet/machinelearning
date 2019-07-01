using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow
{
    public class weights_broadcast_ops
    {
        public static Tensor broadcast_weights(Tensor weights, Tensor values)
        {
            return with(ops.name_scope(null, "broadcast_weights", (weights, values)), scope =>
            {
                values = ops.convert_to_tensor(values, name: "values");
                weights = ops.convert_to_tensor(
                    weights, dtype: values.dtype.as_base_dtype(), name: "weights");

                // Try static check for exact match.
                var weights_shape = weights.GetShape();
                var values_shape = values.GetShape();
                if (weights_shape.is_fully_defined() &&
                    values_shape.is_fully_defined())
                    return weights;

                return math_ops.multiply(
                    weights, array_ops.ones_like(values), name: scope);
            });
        }
    }
}
