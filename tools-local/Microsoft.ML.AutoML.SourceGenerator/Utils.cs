// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.AutoML.SourceGenerator
{
    internal class Utils
    {
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
