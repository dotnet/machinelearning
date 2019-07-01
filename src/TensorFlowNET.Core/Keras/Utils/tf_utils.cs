using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Framework;

namespace Tensorflow.Keras.Utils
{
    public class tf_utils
    {
        public static bool are_all_symbolic_tensors(Tensor[] tensors)
        {
            return tensors.Select(x => is_symbolic_tensor(x)).Count() == tensors.Length;
        }

        public static bool? constant_value(Tensor pred)
        {
            return smart_module.smart_constant_value(pred);
        }

        public static bool is_symbolic_tensor(Tensor tensor)
        {
            return true;
        }

        public static Tensor[] smart_cond<T>(Tensor pred, 
            Func<T[]> true_fn = null,
            Func<T[]> false_fn = null, 
            string name = null)
        {
            return smart_module.smart_cond(pred,
                true_fn: true_fn,
                false_fn: false_fn,
                name: name);
        }
    }
}
