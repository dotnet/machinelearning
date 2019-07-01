using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public partial class RefVariable
    {
        public static implicit operator _VariableScopeStore(RefVariable variable)
        {
            return null;
        }

        public static implicit operator RefVariable(_VariableScopeStore store)
        {
            return null;
        }

        public static implicit operator Tensor(RefVariable var)
        {
            return var._AsTensor();
        }

        public static implicit operator RefVariable(Tensor var)
        {
            return null;
        }
    }
}
