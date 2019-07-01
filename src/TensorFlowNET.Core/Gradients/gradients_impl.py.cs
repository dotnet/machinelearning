using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using static Tensorflow.Python;

namespace Tensorflow
{
    public class gradients_impl
    {
        public static Tensor[] gradients(Tensor[] ys,
            Tensor[] xs, 
            Tensor[] grad_ys = null,
            string name = "gradients",
            bool colocate_gradients_with_ops = false,
            bool gate_gradients = false,
            int? aggregation_method = null)
        {
            return gradients_util._GradientsHelper(ys, xs, grad_ys, name, colocate_gradients_with_ops, gate_gradients);
        }

        private static List<Tensor> _AsList(object ys)
        {
            List<Tensor> ret = null;

            switch (ys)
            {
                case Tensor value:
                    ret = new List<Tensor> { value };
                    break;
                case List<Tensor> value:
                    ret = value;
                    break;
            }

            return ret;
        }
    }
}
