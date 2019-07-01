using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations.Activation
{
    public class relu : IActivation
    {
        public Tensor Activate(Tensor features, string name = null)
        {
            OpDefLibrary _op_def_lib = new OpDefLibrary();

            var _op = _op_def_lib._apply_op_helper("Relu", name: name, args: new
            {
                features
            });

            return _op.outputs[0];
        }
    }
}
