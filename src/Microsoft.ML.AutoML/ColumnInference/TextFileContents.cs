// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Utilities for various heuristics against text files.
    /// Currently, separator inference and column count detection.
    /// </summary>
    internal static class TextFileContents
    {
        public class ColumnSplitResult
        {
            public readonly bool IsSuccess;
            public readonly char? Separator;
            public readonly int ColumnCount;

            public bool AllowQuote { get; set; }
            public bool AllowSparse { get; set; }
            public bool ReadMultilines { get; set; }

            public ColumnSplitResult(bool isSuccess, char? separator, bool allowQuote, bool readMultilines, bool allowSparse, int columnCount)
            {
                IsSuccess = isSuccess;
                Separator = separator;
                AllowQuote = allowQuote;
                AllowSparse = allowSparse;
                ColumnCount = columnCount;
                ReadMultilines = readMultilines;
            }
        }

        // If the fraction of lines having the same number of columns exceeds this, we consider the column count to be known.
        private const Double UniformColumnCountThreshold = 0.98;

        public static readonly char[] DefaultSeparators = { '\t', ',', ' ', ';' };

        /// <summary>
        /// Attempt to detect text loader arguments.
        /// The algorithm selects the first 'acceptable' set: the one that recognizes the same number of columns in at
        /// least <see cref="UniformColumnCountThreshold"/> of the sample's lines,
        /// and this number of columns is more than 1.
        /// We sweep on separator, allow sparse and allow quote parameter.
        /// </summary>
        public static ColumnSplitResult TrySplitColumns(MLContext context, IMultiStreamSource source, char[] separatorCandidates)
        {
            var sparse = new[] { false, true };
            var quote = new[] { true, false };
            var tryMultiline = new[] { false, true };
            var foundAny = false;
            var result = default(ColumnSplitResult);
            foreach (var perm in (from _allowSparse in sparse
                                  from _allowQuote in quote
                                  from _sep in separatorCandidates
                                  from _tryMultiline in tryMultiline
                                  select new { _allowSparse, _allowQuote, _sep, _tryMultiline }))
            {
                var options = new TextLoader.Options
                {
                    Columns = new[] { new TextLoader.Column() {
                        Name = "C",
                        DataKind = DataKind.String,
                        Source = new[] { new TextLoader.Range(0, null) }
                    } },
                    Separators = new[] { perm._sep },
                    AllowQuoting = perm._allowQuote,
                    AllowSparse = perm._allowSparse,
                    ReadMultilines = perm._tryMultiline,
                };

                if (TryParseFile(context, options, source, out result))
                {
                    foundAny = true;
                    break;
                }
            }
            return foundAny ? result : new ColumnSplitResult(false, null, true, true, true, 0);
        }

        private static bool TryParseFile(MLContext context, TextLoader.Options options, IMultiStreamSource source,
            out ColumnSplitResult result)
        {
            result = null;
            // try to instantiate data view with swept arguments
            try
            {
                var textLoader = context.Data.CreateTextLoader(options, source);
                var idv = context.Data.TakeRows(textLoader.Load(source), 1000);
                var columnCounts = new List<int>();
                var column = idv.Schema["C"];

                using (var cursor = idv.GetRowCursor(new[] { column }))
                {
                    var getter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(column);

                    VBuffer<ReadOnlyMemory<char>> line = default;
                    while (cursor.MoveNext())
                    {
                        getter(ref line);
                        columnCounts.Add(line.Length);
                    }
                }

                var mostCommon = columnCounts.GroupBy(x => x).OrderByDescending(x => x.Count()).First();
                if (mostCommon.Count() < UniformColumnCountThreshold * columnCounts.Count)
                {
                    return false;
                }

                // disallow single-column case
                if (mostCommon.Key <= 1) { return false; }

                result = new ColumnSplitResult(true, options.Separators.First(), options.AllowQuoting, options.ReadMultilines, options.AllowSparse, mostCommon.Key);
                return true;
            }
            // fail gracefully if unable to instantiate data view with swept arguments
            catch (Exception)
            {
                return false;
            }
        }
    }
}
