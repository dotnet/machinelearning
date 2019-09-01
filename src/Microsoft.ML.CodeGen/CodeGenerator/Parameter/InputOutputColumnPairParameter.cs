using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.Parameter
{
    internal class InputOutputColumnPairParameter : IParameter
    {
        public string[] InputColumns;
        public string[] OutputColumns;
        private const string _argumentsName = "InputOutputColumnPair";
        public string ToParameter()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("new []{");
            for (int i = 0; i < InputColumns.Length; i++)
            {
                sb.Append("new ");
                sb.Append(_argumentsName);
                sb.Append("(");
                sb.Append(OutputColumns[i]);
                sb.Append(",");
                sb.Append(InputColumns[i]);
                sb.Append(")");
                sb.Append(",");
            }
            sb.Remove(sb.Length - 1, 1); // remove extra ,

            sb.Append("}");
            return sb.ToString();
        }
    }
}
