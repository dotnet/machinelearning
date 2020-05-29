// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal static class ColumnInferenceApi
    {
        public static ColumnInferenceResults InferColumns(MLContext context, string path, uint labelColumnIndex,
            bool hasHeader, char? separatorChar, bool? allowQuotedStrings, bool? supportSparse, bool trimWhitespace, bool groupColumns)
        {
            var sample = TextFileSample.CreateFromFullFile(path);
            var splitInference = InferSplit(context, sample, separatorChar, allowQuotedStrings, supportSparse);
            var typeInference = InferColumnTypes(context, sample, splitInference, hasHeader, labelColumnIndex, null);

            // If no headers, suggest label column name as 'Label'
            if (!hasHeader)
            {
                typeInference.Columns[labelColumnIndex].SuggestedName = DefaultColumnNames.Label;
            }

            var columnInfo = new ColumnInformation() { LabelColumnName = typeInference.Columns[labelColumnIndex].SuggestedName };

            return InferColumns(context, path, columnInfo, hasHeader, splitInference, typeInference, trimWhitespace, groupColumns);
        }

        public static ColumnInferenceResults InferColumns(MLContext context, string path, string labelColumn,
            char? separatorChar, bool? allowQuotedStrings, bool? supportSparse, bool trimWhitespace, bool groupColumns)
        {
            var columnInfo = new ColumnInformation() { LabelColumnName = labelColumn };
            return InferColumns(context, path, columnInfo, separatorChar, allowQuotedStrings, supportSparse, trimWhitespace, groupColumns);
        }

        public static ColumnInferenceResults InferColumns(MLContext context, string path, ColumnInformation columnInfo,
            char? separatorChar, bool? allowQuotedStrings, bool? supportSparse, bool trimWhitespace, bool groupColumns, bool hasHeader = true)
        {
            var sample = TextFileSample.CreateFromFullFile(path);
            var splitInference = InferSplit(context, sample, separatorChar, allowQuotedStrings, supportSparse);
            var typeInference = InferColumnTypes(context, sample, splitInference, hasHeader, null, columnInfo.LabelColumnName);
            return InferColumns(context, path, columnInfo, hasHeader, splitInference, typeInference, trimWhitespace, groupColumns);
        }

        public static ColumnInferenceResults InferColumns(MLContext context, string path, ColumnInformation columnInfo, bool hasHeader,
            TextFileContents.ColumnSplitResult splitInference, ColumnTypeInference.InferenceResult typeInference,
            bool trimWhitespace, bool groupColumns)
        {
            var loaderColumns = ColumnTypeInference.GenerateLoaderColumns(typeInference.Columns);
            var typedLoaderOptions = new TextLoader.Options
            {
                Columns = loaderColumns,
                Separators = new[] { splitInference.Separator.Value },
                AllowSparse = splitInference.AllowSparse,
                AllowQuoting = splitInference.AllowQuote,
                ReadMultilines = splitInference.ReadMultilines,
                HasHeader = hasHeader,
                TrimWhitespace = trimWhitespace
            };
            var textLoader = context.Data.CreateTextLoader(typedLoaderOptions);
            var dataView = textLoader.Load(path);

            // Validate all columns specified in column info exist in inferred data view
            ColumnInferenceValidationUtil.ValidateSpecifiedColumnsExist(columnInfo, dataView);

            var purposeInferenceResult = PurposeInference.InferPurposes(context, dataView, columnInfo);

            // start building result objects
            IEnumerable<TextLoader.Column> columnResults = null;
            IEnumerable<(string, ColumnPurpose)> purposeResults = null;

            // infer column grouping and generate column names
            if (groupColumns)
            {
                var groupingResult = ColumnGroupingInference.InferGroupingAndNames(context, hasHeader,
                    typeInference.Columns, purposeInferenceResult);

                columnResults = groupingResult.Select(c => c.GenerateTextLoaderColumn());
                purposeResults = groupingResult.Select(c => (c.SuggestedName, c.Purpose));
            }
            else
            {
                columnResults = loaderColumns;
                purposeResults = purposeInferenceResult.Select(p => (dataView.Schema[p.ColumnIndex].Name, p.Purpose));
            }

            var textLoaderOptions = new TextLoader.Options()
            {
                Columns = columnResults.ToArray(),
                AllowQuoting = splitInference.AllowQuote,
                AllowSparse = splitInference.AllowSparse,
                Separators = new char[] { splitInference.Separator.Value },
                ReadMultilines = splitInference.ReadMultilines,
                HasHeader = hasHeader,
                TrimWhitespace = trimWhitespace
            };

            return new ColumnInferenceResults()
            {
                TextLoaderOptions = textLoaderOptions,
                ColumnInformation = ColumnInformationUtil.BuildColumnInfo(purposeResults)
            };
        }

        private static TextFileContents.ColumnSplitResult InferSplit(MLContext context, TextFileSample sample, char? separatorChar, bool? allowQuotedStrings, bool? supportSparse)
        {
            var separatorCandidates = separatorChar == null ? TextFileContents.DefaultSeparators : new char[] { separatorChar.Value };
            var splitInference = TextFileContents.TrySplitColumns(context, sample, separatorCandidates);

            // respect passed-in overrides
            if (allowQuotedStrings != null)
            {
                splitInference.AllowQuote = allowQuotedStrings.Value;
            }
            if (supportSparse != null)
            {
                splitInference.AllowSparse = supportSparse.Value;
            }

            if (!splitInference.IsSuccess)
            {
                throw new InferenceException(InferenceExceptionType.ColumnSplit,
                    "Unable to split the file provided into multiple, consistent columns. " +
                    "Readable formats include delimited files such as CSV/TSV. " +
                    "Check for a consistent number of columns and proper escaping and quoting.");
            }

            return splitInference;
        }

        private static ColumnTypeInference.InferenceResult InferColumnTypes(MLContext context, TextFileSample sample,
            TextFileContents.ColumnSplitResult splitInference, bool hasHeader, uint? labelColumnIndex, string label)
        {
            // infer column types
            var typeInferenceResult = ColumnTypeInference.InferTextFileColumnTypes(context, sample,
                new ColumnTypeInference.Arguments
                {
                    ColumnCount = splitInference.ColumnCount,
                    Separator = splitInference.Separator.Value,
                    AllowSparse = splitInference.AllowSparse,
                    AllowQuote = splitInference.AllowQuote,
                    ReadMultilines = splitInference.ReadMultilines,
                    HasHeader = hasHeader,
                    LabelColumnIndex = labelColumnIndex,
                    Label = label
                });

            if (!typeInferenceResult.IsSuccess)
            {
                throw new InferenceException(InferenceExceptionType.ColumnDataType, "Unable to infer column types of the file provided.");
            }

            return typeInferenceResult;
        }
    }
}
