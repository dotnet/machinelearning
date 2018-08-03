// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;
using System.CodeDom;
using Microsoft.ML.Runtime.Data;
using Microsoft.CSharp;
using System.IO;

namespace Microsoft.ML.Runtime.Api
{
    /// <summary>
    /// Utility methods for code generation.
    /// </summary>
    internal static class CodeGenerationUtils
    {
        /// <summary>
        /// Replace placeholders with provided values. Assert that every placeholder is found.
        /// </summary>
        public static string MultiReplace(string text, Dictionary<string, string> replacementMap)
        {
            Contracts.AssertValue(text);
            Contracts.AssertValue(replacementMap);
            var pattern = @"\/\*#(.*)#\*\/.*\/\*#\/\1#\*\/[\n\r]{0,2}";

            int seenTags = 0;
            var result = Regex.Replace(text, pattern,
                match =>
                {
                    var tag = match.Groups[1].Value;
                    string replacement;
                    bool found = replacementMap.TryGetValue(tag, out replacement);
                    Contracts.Assert(found);
                    seenTags++;
                    return replacement;
                }, RegexOptions.Singleline);

            Contracts.Assert(seenTags == replacementMap.Count);
            return result;
        }

        /// <summary>
        /// Append a field declaration to the provided <see cref="StringBuilder"/>.
        /// </summary>
        public static void AppendFieldDeclaration(CSharpCodeProvider codeProvider, StringBuilder target, int columnIndex,
            string fieldName, ColumnType colType, bool appendInitializer, bool useVBuffer)
        {
            Contracts.AssertValueOrNull(codeProvider);
            Contracts.AssertValue(target);
            Contracts.Assert(columnIndex >= 0);
            Contracts.AssertNonEmpty(fieldName);
            Contracts.AssertValue(colType);

            var attributes = new List<string>();
            string generatedCsTypeName = GetBackingTypeName(colType, useVBuffer, attributes);

            if (codeProvider != null && !codeProvider.IsValidIdentifier(fieldName))
            {
                attributes.Add(string.Format("[ColumnName({0})]", GetCSharpString(codeProvider, fieldName)));
                fieldName = string.Format("Column{0}", columnIndex);
            }

            const string indent = "            ";
            if (attributes.Count > 0)
            {
                foreach (var attr in attributes)
                {
                    target.Append(indent);
                    target.AppendLine(attr);
                }
            }
            target.Append(indent);
            target.AppendFormat("public {0} {1}", generatedCsTypeName, fieldName);

            if (appendInitializer && colType.IsKnownSizeVector && !useVBuffer)
            {
                Contracts.Assert(generatedCsTypeName.EndsWith("[]"));
                var csItemType = generatedCsTypeName.Substring(0, generatedCsTypeName.Length - 2);
                target.AppendFormat(" = new {0}[{1}]", csItemType, colType.VectorSize);
            }
            target.AppendLine(";");
        }

        /// <summary>
        /// Generates a C# string for a given input (with proper escaping).
        /// </summary>
        public static string GetCSharpString(CSharpCodeProvider codeProvider, string value)
        {
            using (var writer = new StringWriter())
            {
                codeProvider.GenerateCodeFromExpression(new CodePrimitiveExpression(value), writer, null);
                return writer.ToString();
            }
        }

        /// <summary>
        /// Gets the C# strings representing the type name for a variable corresponding to
        /// the <see cref="IDataView"/> column type.
        ///
        /// If the type is a vector, then <paramref name="useVBuffer"/> controls whether the array field is
        /// generated or <see cref="VBuffer{T}"/>.
        ///
        /// If additional attributes are required, they are appended to the <paramref name="attributes"/> list.
        /// </summary>
        private static string GetBackingTypeName(ColumnType colType, bool useVBuffer, List<string> attributes)
        {
            Contracts.AssertValue(colType);
            Contracts.AssertValue(attributes);
            if (colType.IsVector)
            {
                if (colType.IsKnownSizeVector)
                {
                    // By default, arrays are assumed variable length, unless a [VectorType(dim1, dim2, ...)]
                    // attribute is applied to the fields.
                    var vectorType = colType.AsVector;
                    var dimensions = new int[vectorType.DimCount];
                    for (int i = 0; i < dimensions.Length; i++)
                        dimensions[i] = vectorType.GetDim(i);
                    attributes.Add(string.Format("[VectorType({0})]", string.Join(", ", dimensions)));
                }

                var itemType = GetBackingTypeName(colType.ItemType, false, attributes);
                return useVBuffer ? string.Format("VBuffer<{0}>", itemType) : string.Format("{0}[]", itemType);
            }

            if (colType.IsText)
                return "string";

            if (colType.IsKey)
            {
                // The way to define a key type in C# is by specifying the [KeyType] attribute and
                // making the field type equal to the underlying raw type.
                var key = colType.AsKey;
                attributes.Add(string.Format("[KeyType(Count={0}, Min={1}, Contiguous={2})]",
                    key.Count,
                    key.Min,
                    key.Contiguous ? "true" : "false"));
            }

            return colType.AsPrimitive.RawType.Name;
        }
    }
}
