// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal static class ColumnInferenceApi
    {
        public static ColumnInferenceResult InferColumns(MLContext context, string path, string label,
            bool hasHeader, char? separatorChar, bool? allowQuotedStrings, bool? supportSparse, bool trimWhitespace, bool groupColumns)
        {
            var sample = TextFileSample.CreateFromFullFile(path);
            var splitInference = InferSplit(sample, separatorChar, allowQuotedStrings, supportSparse);
            var typeInference = InferColumnTypes(context, sample, splitInference, hasHeader);
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

            (TextLoader.Column, ColumnPurpose Purpose)[] inferredColumns = null;
            // infer column grouping and generate column names
            if (groupColumns)
            {
                var groupingResult = ColumnGroupingInference.InferGroupingAndNames(context, hasHeader,
                    typeInference.Columns, purposeInferenceResult);

                // build result objects & return
                inferredColumns = groupingResult.Select(c => (c.GenerateTextLoaderColumn(), c.Purpose)).ToArray();
            }
            else
            {
                inferredColumns = new (TextLoader.Column, ColumnPurpose Purpose)[loaderColumns.Length];
                for (int i = 0; i < loaderColumns.Length; i++)
                {
                    inferredColumns[i] = (loaderColumns[i], purposeInferenceResult[i].Purpose);
                }
            }
            return new ColumnInferenceResult(inferredColumns, splitInference.AllowQuote, splitInference.AllowSparse, new char[] { splitInference.Separator.Value }, hasHeader, trimWhitespace);
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
