using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.Parameter
{
    internal class OptionalParameter : IParameter
    {
        public string ParameterName;
        public string ParameterValue;

        public string ToParameter()
        {
            // ParameterName:ParameterValue
            return $"{ParameterName}:{ParameterValue}";
        }
    }
}
