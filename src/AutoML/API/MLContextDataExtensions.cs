using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    public static class DataExtensions
    {
        // Delimiter, header, column datatype inference
        public static ColumnInferenceResult InferColumns(this DataOperations catalog, string path, string label,
            bool hasHeader = false, char? separatorChar = null, bool? allowQuotedStrings = null, bool? supportSparse = null, bool trimWhitespace = false)
        {
            UserInputValidationUtil.ValidateInferColumnsArgs(path, label);
            var mlContext = new MLContext();
            return ColumnInferenceApi.InferColumns(mlContext, path, label, hasHeader, separatorChar, allowQuotedStrings, supportSparse, trimWhitespace);
        }

        public static IDataView AutoRead(this DataOperations catalog, string path, string label,
            bool hasHeader = false, char? separatorChar = null, bool? allowQuotedStrings = null, bool? supportSparse = null, bool trimWhitespace = false)
        {
            UserInputValidationUtil.ValidateAutoReadArgs(path, label);
            var mlContext = new MLContext();
            var columnInferenceResult = ColumnInferenceApi.InferColumns(mlContext, path, label, hasHeader, separatorChar, allowQuotedStrings, supportSparse, trimWhitespace);
            var textLoader = columnInferenceResult.BuildTextLoader();
            return textLoader.Read(path);
        }

        public static TextLoader CreateTextReader(this DataOperations catalog, ColumnInferenceResult columnInferenceResult)
        {
            UserInputValidationUtil.ValidateCreateTextReaderArgs(columnInferenceResult);
            return columnInferenceResult.BuildTextLoader();
        }

        // Task inference
        public static MachineLearningTaskType InferTask(this DataOperations catalog, IDataView dataView)
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
        public readonly string Separator;
        public readonly bool HasHeader;
        public readonly bool TrimWhitespace;

        public ColumnInferenceResult(IEnumerable<(TextLoader.Column, ColumnPurpose)> columns,
            bool allowQuotedStrings, bool supportSparse, string separator, bool hasHeader, bool trimWhitespace)
        {
            Columns = columns;
            AllowQuotedStrings = allowQuotedStrings;
            SupportSparse = supportSparse;
            Separator = separator;
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
                Separator = Separator,
                HasHeader = HasHeader,
                TrimWhitespace = TrimWhitespace
            });
        }
    }
}
