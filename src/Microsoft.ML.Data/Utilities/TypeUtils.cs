// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Internal.Internallearn
{
    public static class TypeUtils
    {
        /// <summary>
        /// Returns a pretty representation of the type.
        /// </summary>
        public static string PrettyName(Type type)
        {
            Contracts.AssertValue(type, "type");
            StringBuilder sb = new StringBuilder();
            BuildPrettyName(type, sb);
            return sb.ToString();
        }

        private static void BuildPrettyName(Type type, StringBuilder sb)
        {
            // Arrays
            if (type.IsArray)
            {
                // Store the original and walk to get the non-array element type.
                Type origType = type;
                while (type.IsArray)
                    type = type.GetElementType();
                BuildPrettyName(type, sb);
                // Rewalk to get the [] items in indexing order.
                type = origType;
                while (type.IsArray)
                {
                    sb.Append('[');
                    for (int i = 1; i < type.GetArrayRank(); ++i)
                        sb.Append(',');
                    sb.Append(']');
                    type = type.GetElementType();
                }
                return;
            }
            // Output the names of generic parameters.
            if (type.IsGenericParameter)
            {
                Contracts.Assert(type.FullName == null);
                sb.Append(type.Name);
                return;
            }
            // Get excluding the namespace and any possible following type-array.
            string name = type.FullName ?? type.Name;
            Match m = Regex.Match(type.FullName, @"^(?:\w+\.)*([^\[]+)");
            Contracts.Assert(m.Success);
            Contracts.Assert(m.Groups.Count == 2);
            string[] subNames = m.Groups[1].Value.Split('+');
            // Get the generic type arguments, if there are any.
            Type[] genTypes = type.IsGenericType ? type.GetGenericArguments() : null;
            int iGenTypes = 0;

            for (int i = 0; i < subNames.Length; ++i)
            {
                if (i > 0)
                    sb.Append('.');
                string subName = subNames[i];
                if (!subName.Contains('`'))
                {
                    sb.Append(subName);
                    continue;
                }
                string[] subparts = subName.Split('`');
                Contracts.Assert(subparts.Length == 2);
                Contracts.Assert(type.IsGenericType);
                sb.Append(subparts[0]);
                sb.Append('<');
                int numGenerics = int.Parse(subparts[1]);
                Contracts.Assert(iGenTypes + numGenerics <= Utils.Size(genTypes));
                while (numGenerics-- > 0)
                {
                    Type parameter = genTypes[iGenTypes++];
                    // Leave generic parameters as blank.
                    if (!parameter.IsGenericParameter)
                        BuildPrettyName(parameter, sb);
                    if (numGenerics > 0)
                        sb.Append(',');
                }
                sb.Append('>');
            }
            Contracts.Assert(iGenTypes == Utils.Size(genTypes));
        }
    }
}
