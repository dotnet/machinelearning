using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public static partial class tf
    {
        // public static Tensor constant(NDArray nd, string name = "Const") => constant_op.constant(nd, name: name);

        public static Tensor constant(object value,
            TF_DataType dtype = TF_DataType.DtInvalid,
            int[] shape = null,
            string name = "Const",
            bool verify_shape = false) => constant_op._constant_impl(value,
                dtype,
                shape,
                name,
                verify_shape: verify_shape,
                allow_broadcast: false);

        public static Tensor zeros(Shape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null) => array_ops.zeros(shape, dtype, name);

        public static Tensor size(Tensor input,
            string name = null,
            TF_DataType out_type = TF_DataType.TF_INT32) => array_ops.size(input,
                name,
                optimize: true,
                out_type: out_type);
    }
}
