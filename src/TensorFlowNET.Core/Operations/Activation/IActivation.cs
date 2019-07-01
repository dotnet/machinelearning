using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations.Activation
{
    public interface IActivation
    {
        Tensor Activate(Tensor features, string name = null);
    }
}
