using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public enum TF_AttrType
    {
        TF_ATTR_STRING = 0,
        TF_ATTR_INT = 1,
        TF_ATTR_FLOAT = 2,
        TF_ATTR_BOOL = 3,
        TF_ATTR_TYPE = 4,
        TF_ATTR_SHAPE = 5,
        TF_ATTR_TENSOR = 6,
        TF_ATTR_PLACEHOLDER = 7,
        TF_ATTR_FUNC = 8
    }
}
