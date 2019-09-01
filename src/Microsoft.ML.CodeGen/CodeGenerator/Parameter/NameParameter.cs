using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.Parameter
{
    internal class NameParameter : IParameter
    {
        public string ParameterName;
        public string ParameterValue;

        public string ToParameter()
        {
            // "parameterValue"
            return $"{ParameterValue}";
        }
    }

    internal class NameArrayParameter : IParameter
    {
        public string ParameterName;
        public string[] ArrayParameterValue;
        public string ToParameter()
        {
            return $"new []{{{string.Join(",", ArrayParameterValue)}}}";
        }
    }
}
