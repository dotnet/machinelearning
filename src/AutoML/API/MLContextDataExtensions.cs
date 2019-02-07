// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    public static class DataExtensions
    {
        // Delimiter, header, column datatype inference
        public static ColumnInferenceResult InferColumns(this DataOperationsCatalog catalog, string path, string label,
            bool hasHeader = false, char? separatorChar = null, bool? allowQuotedStrings = null, bool? supportSparse = null, bool trimWhitespace = false, bool groupColumns = true)
        {
            UserInputValidationUtil.ValidateInferColumnsArgs(path, label);
            var mlContext = new MLContext();
            return ColumnInferenceApi.InferColumns(mlContext, path, label, hasHeader, separatorChar, allowQuotedStrings, supportSparse, trimWhitespace, groupColumns);
        }

        public static IDataView AutoRead(this DataOperationsCatalog catalog, string path, string label,
            bool hasHeader = false, char? separatorChar = null, bool? allowQuotedStrings = null, bool? supportSparse = null, bool trimWhitespace = false, bool groupColumns = true)
        {
            UserInputValidationUtil.ValidateAutoReadArgs(path, label);
            var mlContext = new MLContext();
            var columnInferenceResult = ColumnInferenceApi.InferColumns(mlContext, path, label, hasHeader, separatorChar, allowQuotedStrings, supportSparse, trimWhitespace, groupColumns);
            var textLoader = columnInferenceResult.BuildTextLoader();
            return textLoader.Read(path);
        }

        public static TextLoader CreateTextLoader(this DataOperationsCatalog catalog, ColumnInferenceResult columnInferenceResult)
        {
            UserInputValidationUtil.ValidateCreateTextReaderArgs(columnInferenceResult);
            return columnInferenceResult.BuildTextLoader();
        }

        // Task inference
        public static MachineLearningTaskType InferTask(this DataOperationsCatalog catalog, IDataView dataView)
        {
            throw new NotImplementedException();
        }

        public enum MachineLearningTaskType
        {
            Regression,
            BinaryClassification,
            MultiClassClassification
        }
    }

    public class ColumnInferenceResult
    {
        public readonly IEnumerable<(TextLoader.Column, ColumnPurpose)> Columns;
        public readonly bool AllowQuotedStrings;
        public readonly bool SupportSparse;
        public readonly char[] Separators;
        public readonly bool HasHeader;
        public readonly bool TrimWhitespace;

        public ColumnInferenceResult(IEnumerable<(TextLoader.Column, ColumnPurpose)> columns,
            bool allowQuotedStrings, bool supportSparse, char[] separators, bool hasHeader, bool trimWhitespace)
        {
            Columns = columns;
            AllowQuotedStrings = allowQuotedStrings;
            SupportSparse = supportSparse;
            Separators = separators;
            HasHeader = hasHeader;
            TrimWhitespace = trimWhitespace;
        }

        internal TextLoader BuildTextLoader()
        {
            var context = new MLContext();
            return new TextLoader(context, new TextLoader.Arguments()
            {
                AllowQuoting = AllowQuotedStrings,
                AllowSparse = SupportSparse,
                Column = Columns.Select(c => c.Item1).ToArray(),
                Separators = Separators,
                HasHeader = HasHeader,
                TrimWhitespace = TrimWhitespace
            });
        }
    }
}
