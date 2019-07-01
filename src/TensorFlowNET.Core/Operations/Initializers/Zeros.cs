using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations.Initializers
{
    public class Zeros : IInitializer
    {
        private TF_DataType dtype;

        public Zeros(TF_DataType dtype = TF_DataType.TF_FLOAT)
        {
            this.dtype = dtype;
        }

        public Tensor call(TensorShape shape, TF_DataType dtype = TF_DataType.DtInvalid)
        {
            if (dtype == TF_DataType.DtInvalid)
                dtype = this.dtype;

            return array_ops.zeros(shape, dtype);
        }

        public object get_config()
        {
            return new { dtype = dtype.name() };
        }
    }
}
