using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class ReferenceVariableSaveable : SaveableObject
    {
        private SaveSpec _spec;

        public ReferenceVariableSaveable(Tensor var, string slice_spec, string name)
        {
            _spec = new SaveSpec(var, slice_spec, name, dtype: var.dtype);
            op = var;
            specs = new SaveSpec[] { _spec };
            this.name = name;
        }
    }
}
