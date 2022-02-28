// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace Microsoft.ML.AutoML.SourceGenerator
{
    internal class Utils
    {
        public static EstimatorsContract GetEstimatorsFromJson(string json)
        {
            return JsonSerializer.Deserialize<EstimatorsContract>(json);
        }

        public static string CreateEstimatorName(string functionName, string estimatorType)
        {
            if (estimatorType == "BinaryClassification")
            {
                return functionName + "Binary";
            }
            if (estimatorType == "MultiClassification")
            {
                return functionName + "Multi";
            }
            if (estimatorType == "Regression")
            {
                return functionName + "Regression";
            }
            if (estimatorType == "Ranking")
            {
                return functionName + "Ranking";
            }
            if (estimatorType == "OneVersusAll")
            {
                return functionName + "Ova";
            }

            return functionName;
        }

        public static string CapitalFirstLetter(string str)
        {
            if (str == null)
                return null;

            if (str.Length > 1)
                return char.ToUpper(str[0]) + str.Substring(1);

            return str.ToUpper();
        }

        public static string PrettyPrintListOfString(IEnumerable<string> strs)
        {
            // ["str1", "str2", "str3"] => "\"str1\", \"str2\", \"str3\""
            var sb = new StringBuilder();
            foreach (var str in strs)
            {
                sb.Append($"\"{str}\"");
                sb.Append(", ");
            }

            return sb.ToString();
        }

        public static string ToTitleCase(string str)
        {
            return string.Join(string.Empty, str.Split('_', ' ', '-').Select(x => CapitalFirstLetter(x)));
        }
    }
}
