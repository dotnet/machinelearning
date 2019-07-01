using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class gen_training_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Tensor apply_adam(RefVariable var, RefVariable m, RefVariable v, Tensor beta1_power, Tensor beta2_power, 
            Tensor lr, Tensor beta1, Tensor beta2, Tensor epsilon, Tensor grad, 
            bool use_locking = false, bool use_nesterov = false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("ApplyAdam", name, new
            {
                var,
                m,
                v,
                beta1_power,
                beta2_power,
                lr,
                beta1,
                beta2,
                epsilon,
                grad,
                use_locking,
                use_nesterov
            });

            return _op.outputs[0];
        }

        public static Tensor apply_gradient_descent(RefVariable var, Tensor alpha, Tensor delta, bool use_locking = false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("ApplyGradientDescent", name, new
            {
                var,
                alpha,
                delta,
                use_locking
            });

            return _op.outputs[0];
        }
    }
}
