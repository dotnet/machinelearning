// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    public static class DataExtensions
    {
        // Delimiter, header, column datatype inference
        public static (TextLoader.Arguments TextLoaderArgs, IEnumerable<(string Name, ColumnPurpose Purpose)> ColumnPurpopses) InferColumns(this DataOperationsCatalog catalog, string path, string label,
            char? separatorChar = null, bool? allowQuotedStrings = null, bool? supportSparse = null, bool trimWhitespace = false, bool groupColumns = true)
        {
            UserInputValidationUtil.ValidateInferColumnsArgs(path, label);
            var mlContext = new MLContext();
            return ColumnInferenceApi.InferColumns(mlContext, path, label, separatorChar, allowQuotedStrings, supportSparse, trimWhitespace, groupColumns);
        }

        public static (TextLoader.Arguments TextLoaderArgs, IEnumerable<(string Name, ColumnPurpose Purpose)> ColumnPurpopses) InferColumns(this DataOperationsCatalog catalog, string path, int labelColumnIndex,
            bool hasHeader = false, char? separatorChar = null, bool? allowQuotedStrings = null, bool? supportSparse = null, 
            bool trimWhitespace = false, bool groupColumns = true)
        {
            UserInputValidationUtil.ValidateInferColumnsArgs(path, labelColumnIndex);
            var mlContext = new MLContext();
            return ColumnInferenceApi.InferColumns(mlContext, path, labelColumnIndex, hasHeader, separatorChar, allowQuotedStrings, supportSparse, trimWhitespace, groupColumns);
        }
    }
}
