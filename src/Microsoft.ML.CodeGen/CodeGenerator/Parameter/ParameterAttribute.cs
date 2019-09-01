using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.Parameter
{
    internal class ParameterAttribute : System.Attribute
    {
        public int Position;

        public ParameterAttribute(int pos)
        {
            Position = pos;
        }
    }
}
