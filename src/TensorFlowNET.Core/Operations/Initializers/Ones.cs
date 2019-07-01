using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations.Initializers
{
    public class Ones : IInitializer
    {
        private TF_DataType dtype;

        public Ones(TF_DataType dtype = TF_DataType.TF_FLOAT)
        {
            this.dtype = dtype;
        }

        public Tensor call(TensorShape shape, TF_DataType dtype = TF_DataType.DtInvalid)
        {
            if (dtype == TF_DataType.DtInvalid)
                dtype = this.dtype;

            return array_ops.ones(shape.Dimensions, dtype);
        }

        public object get_config()
        {
            return new { dtype = dtype.name() };
        }
    }
}
