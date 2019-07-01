using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public interface IInitializer
    {
        Tensor call(TensorShape shape, TF_DataType dtype = TF_DataType.DtInvalid);
        object get_config();
    }
}
