using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static class Distribute
    {
        public static VariableAggregationType get_loss_reduction()
        {
            return VariableAggregationType.MEAN;
        }
    }
}
