using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations.Losses
{
    public class Util
    {
        public static void add_loss(Tensor loss, string loss_collection = ops.GraphKeys.LOSSES)
        {
            if (!string.IsNullOrEmpty(loss_collection))
                ops.add_to_collection(loss_collection, loss);
        }
    }
}
