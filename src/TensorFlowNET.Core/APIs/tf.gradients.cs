using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        public static Tensor[] gradients(Tensor[] ys,
            Tensor[] xs,
            Tensor[] grad_ys = null,
            string name = "gradients",
            bool colocate_gradients_with_ops = false,
            bool gate_gradients = false,
            int? aggregation_method = null,
            Tensor[] stop_gradients = null)
        {
            return gradients_util._GradientsHelper(ys, 
                xs, 
                grad_ys, 
                name, 
                colocate_gradients_with_ops, 
                gate_gradients,
                stop_gradients: stop_gradients);
        }

        public static Tensor[] gradients(Tensor ys,
            Tensor[] xs,
            Tensor[] grad_ys = null,
            string name = "gradients",
            bool colocate_gradients_with_ops = false,
            bool gate_gradients = false,
            int? aggregation_method = null,
            Tensor[] stop_gradients = null)
        {
            return gradients_util._GradientsHelper(new Tensor[] { ys },
                xs,
                grad_ys,
                name,
                colocate_gradients_with_ops,
                gate_gradients,
                stop_gradients: stop_gradients);
        }

        public static Tensor[] gradients(Tensor ys,
            Tensor xs,
            Tensor[] grad_ys = null,
            string name = "gradients",
            bool colocate_gradients_with_ops = false,
            bool gate_gradients = false,
            int? aggregation_method = null,
            Tensor[] stop_gradients = null)
        {
            return gradients_util._GradientsHelper(new Tensor[] { ys },
                new Tensor[] { xs },
                grad_ys,
                name,
                colocate_gradients_with_ops,
                gate_gradients,
                stop_gradients: stop_gradients);
        }
    }
}
