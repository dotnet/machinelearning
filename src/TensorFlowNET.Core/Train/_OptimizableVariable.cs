using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Framework;

namespace Tensorflow
{
    public interface _OptimizableVariable
    {
        Tensor target();
        Operation update_op(Optimizer optimizer, Tensor g);
    }
}
