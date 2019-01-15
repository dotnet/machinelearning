using System;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal static class ColumnInferenceApi
    {
        public static ColumnInferenceResult InferColumns(MLContext context, string path, string label, 
            bool hasHeader, string separator, bool? isQuoted, bool? isSparse)
        {
            var sample = TextFileSample.CreateFromFullFile(path);
            Func<TextLoader, IDataView> createDataView = (textLoader) => 
            {
                return textLoader.Read(path); 
            };
            return InferColumns(context, sample, createDataView, label, hasHeader, separator, isQuoted, isSparse);
        }

        public static ColumnInferenceResult InferColumns(MLContext context, IMultiStreamSource multiStreamSource, 
            string label, bool hasHeader, string separator, bool? isQuoted, bool? isSparse)
        {
            // heuristic: use first stream in multi-stream source to infer column types & split
            var stream = multiStreamSource.Open(0);
            var sample = TextFileSample.CreateFromFullStream(stream);

            Func<TextLoader, IDataView> createDataView = (textLoader) =>
            {
                return textLoader.Read(multiStreamSource);
            };

            return InferColumns(context, sample, createDataView, label, hasHeader, separator, isQuoted, isSparse);
        }

        private static TextFileContents.ColumnSplitResult InferSplit(TextFileSample sample, string separator, bool? isQuoted, bool? isSparse)
        {
            var separatorCandidates = separator == null ? TextFileContents.DefaultSeparators : new string[] { separator };
            var splitInference = TextFileContents.TrySplitColumns(sample, separatorCandidates);

            // respect passed-in overrides
            if(isQuoted != null)
            {
                splitInference.AllowQuote = isQuoted.Value;
            }
            if(isSparse != null)
            {
                splitInference.AllowSparse = isSparse.Value;
            }
            
            if (!splitInference.IsSuccess)
            {
                throw new InferenceException(InferenceType.ColumnSplit, "Unable to split the file provided into multiple, consistent columns.");
            }

            return splitInference;
        }

        private static ColumnTypeInference.InferenceResult InferColumnTypes(MLContext context, TextFileSample sample,
            TextFileContents.ColumnSplitResult splitInference)
        {
            // infer column types
            var typeInferenceResult = ColumnTypeInference.InferTextFileColumnTypes(context, sample,
                new ColumnTypeInference.Arguments
                {
                    ColumnCount = splitInference.ColumnCount,
                    Separator = splitInference.Separator,
                    AllowSparse = splitInference.AllowSparse,
                    AllowQuote = splitInference.AllowQuote,
                });

            if (!typeInferenceResult.IsSuccess)
            {
                throw new InferenceException(InferenceType.ColumnDataKind, "Unable to infer column types of the file provided.");
            }

            return typeInferenceResult;
        }

        private static ColumnInferenceResult InferColumns(MLContext context,
            TextFileSample sample, Func<TextLoader, IDataView> createDataView, string label, 
            bool hasHeader, string separator, bool? isQuoted, bool? isSparse)
        {
            var splitInference = InferSplit(sample, separator, isQuoted, isSparse);
            var typeInference = InferColumnTypes(context, sample, splitInference);
            var typedLoaderArgs = new TextLoader.Arguments
            {
                Column = ColumnTypeInference.GenerateLoaderColumns(typeInference.Columns),
                Separator = splitInference.Separator,
                AllowSparse = splitInference.AllowSparse,
                AllowQuoting = splitInference.AllowQuote,
                HasHeader = hasHeader
            };
            var textLoader = context.Data.CreateTextReader(typedLoaderArgs);
            var dataView = createDataView(textLoader);

            var purposeInferenceResult = PurposeInference.InferPurposes(context, dataView, label);

            // infer column grouping and generate column names
            var groupingResult = ColumnGroupingInference.InferGroupingAndNames(context, hasHeader,
                typeInference.Columns, purposeInferenceResult);

            // build result objects & return
            var inferredColumns = groupingResult.Select(c => (c.GenerateTextLoaderColumn(), c.Purpose)).ToArray();
            return new ColumnInferenceResult(inferredColumns, splitInference.AllowQuote, splitInference.AllowSparse, splitInference.Separator, hasHeader);
        }
    }
}
