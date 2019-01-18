using System;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal static class ColumnInferenceApi
    {
        public static ColumnInferenceResult InferColumns(MLContext context, string path, string label, 
            bool hasHeader, char? separatorChar, bool? allowQuotedStrings, bool? supportSparse, bool trimWhitespace)
        {
            var sample = TextFileSample.CreateFromFullFile(path);
            var splitInference = InferSplit(sample, separatorChar, allowQuotedStrings, supportSparse);
            var typeInference = InferColumnTypes(context, sample, splitInference, hasHeader);
            var typedLoaderArgs = new TextLoader.Arguments
            {
                Column = ColumnTypeInference.GenerateLoaderColumns(typeInference.Columns),
                Separator = splitInference.Separator,
                AllowSparse = splitInference.AllowSparse,
                AllowQuoting = splitInference.AllowQuote,
                HasHeader = hasHeader,
                TrimWhitespace = trimWhitespace
            };
            var textLoader = context.Data.CreateTextReader(typedLoaderArgs);
            var dataView = textLoader.Read(path);

            var purposeInferenceResult = PurposeInference.InferPurposes(context, dataView, label);

            // infer column grouping and generate column names
            var groupingResult = ColumnGroupingInference.InferGroupingAndNames(context, hasHeader,
                typeInference.Columns, purposeInferenceResult);

            // build result objects & return
            var inferredColumns = groupingResult.Select(c => (c.GenerateTextLoaderColumn(), c.Purpose)).ToArray();
            return new ColumnInferenceResult(inferredColumns, splitInference.AllowQuote, splitInference.AllowSparse, splitInference.Separator, hasHeader, trimWhitespace);
        }

        private static TextFileContents.ColumnSplitResult InferSplit(TextFileSample sample, char? separatorChar, bool? allowQuotedStrings, bool? supportSparse)
        {
            var separatorCandidates = separatorChar == null ? TextFileContents.DefaultSeparators : new char[] { separatorChar.Value };
            var splitInference = TextFileContents.TrySplitColumns(sample, separatorCandidates);

            // respect passed-in overrides
            if(allowQuotedStrings != null)
            {
                splitInference.AllowQuote = allowQuotedStrings.Value;
            }
            if(supportSparse != null)
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
                    Separator = splitInference.Separator,
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
