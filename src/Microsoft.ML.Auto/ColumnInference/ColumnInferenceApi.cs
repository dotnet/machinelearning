// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal static class ColumnInferenceApi
    {
        public static (TextLoader.Arguments, IEnumerable<(string, ColumnPurpose)>) InferColumns(MLContext context, string path, uint labelColumnIndex,
            bool hasHeader, char? separatorChar, bool? allowQuotedStrings, bool? supportSparse, bool trimWhitespace, bool groupColumns)
        {
            var sample = TextFileSample.CreateFromFullFile(path);
            var splitInference = InferSplit(sample, separatorChar, allowQuotedStrings, supportSparse);
            var typeInference = InferColumnTypes(context, sample, splitInference, hasHeader);

            // If label column index > inferred # of columns, throw error
            if (labelColumnIndex >= typeInference.Columns.Count())
            {
                throw new ArgumentOutOfRangeException(nameof(labelColumnIndex), $"Label column index ({labelColumnIndex}) is >= than # of inferred columns ({typeInference.Columns.Count()}).");
            }

            // if no column is named label,
            // rename label column to default ML.NET label column name
            if (!typeInference.Columns.Any(c => c.SuggestedName == DefaultColumnNames.Label))
            {
                typeInference.Columns[labelColumnIndex].SuggestedName = DefaultColumnNames.Label;
            }

            return InferColumns(context, path, typeInference.Columns[labelColumnIndex].SuggestedName,
                hasHeader, splitInference, typeInference, trimWhitespace, groupColumns);
        }

        public static (TextLoader.Arguments, IEnumerable<(string, ColumnPurpose)>) InferColumns(MLContext context, string path, string label,
            char? separatorChar, bool? allowQuotedStrings, bool? supportSparse, bool trimWhitespace, bool groupColumns)
        {
            var sample = TextFileSample.CreateFromFullFile(path);
            var splitInference = InferSplit(sample, separatorChar, allowQuotedStrings, supportSparse);
            var typeInference = InferColumnTypes(context, sample, splitInference, true);
            return InferColumns(context, path, label, true, splitInference, typeInference, trimWhitespace, groupColumns);
        }

        public static (TextLoader.Arguments, IEnumerable<(string, ColumnPurpose)>) InferColumns(MLContext context, string path, string label, bool hasHeader,
            TextFileContents.ColumnSplitResult splitInference, ColumnTypeInference.InferenceResult typeInference,
            bool trimWhitespace, bool groupColumns)
        {
            var loaderColumns = ColumnTypeInference.GenerateLoaderColumns(typeInference.Columns);
            if (!loaderColumns.Any(t => label.Equals(t.Name)))
            {
                throw new InferenceException(InferenceType.Label, $"Specified Label Column '{label}' was not found.");
            }
            var typedLoaderArgs = new TextLoader.Arguments
            {
                Column = loaderColumns,
                Separators = new[] { splitInference.Separator.Value },
                AllowSparse = splitInference.AllowSparse,
                AllowQuoting = splitInference.AllowQuote,
                HasHeader = hasHeader,
                TrimWhitespace = trimWhitespace
            };
            var textLoader = context.Data.CreateTextLoader(typedLoaderArgs);
            var dataView = textLoader.Read(path);

            var purposeInferenceResult = PurposeInference.InferPurposes(context, dataView, label);

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

            return (new TextLoader.Arguments()
            {
                Column = columnResults.ToArray(),
                AllowQuoting = splitInference.AllowQuote,
                AllowSparse = splitInference.AllowSparse,
                Separators = new char[] { splitInference.Separator.Value },
                HasHeader = hasHeader,
                TrimWhitespace = trimWhitespace
            }, purposeResults);
        }

        private static TextFileContents.ColumnSplitResult InferSplit(TextFileSample sample, char? separatorChar, bool? allowQuotedStrings, bool? supportSparse)
        {
            var separatorCandidates = separatorChar == null ? TextFileContents.DefaultSeparators : new char[] { separatorChar.Value };
            var splitInference = TextFileContents.TrySplitColumns(sample, separatorCandidates);

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
                throw new InferenceException(InferenceType.ColumnSplit, "Unable to split the file provided into multiple, consistent columns.");
            }

            return splitInference;
        }

        private static ColumnTypeInference.InferenceResult InferColumnTypes(MLContext context, TextFileSample sample,
            TextFileContents.ColumnSplitResult splitInference, bool hasHeader)
        {
            // infer column types
            var typeInferenceResult = ColumnTypeInference.InferTextFileColumnTypes(context, sample,
                new ColumnTypeInference.Arguments
                {
                    ColumnCount = splitInference.ColumnCount,
                    Separator = splitInference.Separator.Value,
                    AllowSparse = splitInference.AllowSparse,
                    AllowQuote = splitInference.AllowQuote,
                    HasHeader = hasHeader
                });

            if (!typeInferenceResult.IsSuccess)
            {
                throw new InferenceException(InferenceType.ColumnDataKind, "Unable to infer column types of the file provided.");
            }

            return typeInferenceResult;
        }
    }
}
