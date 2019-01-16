// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
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
            public readonly string Separator;
            public readonly int ColumnCount;

            public bool AllowQuote { get; set; }
            public bool AllowSparse { get; set; }

            public ColumnSplitResult(bool isSuccess, string separator, bool allowQuote, bool allowSparse, int columnCount)
            {
                IsSuccess = isSuccess;
                Separator = separator;
                AllowQuote = allowQuote;
                AllowSparse = allowSparse;
                ColumnCount = columnCount;
            }
        }

        // If the fraction of lines having the same number of columns exceeds this, we consider the column count to be known.
        private const Double UniformColumnCountThreshold = 0.98;

        public static char[] DefaultSeparators = new[] { '\t', ',', ';', ' ' };

        /// <summary>
        /// Attempt to detect text loader arguments.
        /// The algorithm selects the first 'acceptable' set: the one that recognizes the same number of columns in at
        /// least <see cref="UniformColumnCountThreshold"/> of the sample's lines,
        /// and this number of columns is more than 1.
        /// We sweep on separator, allow sparse and allow quote parameter.
        /// </summary>
        public static ColumnSplitResult TrySplitColumns(IMultiStreamSource source, char[] separatorCandidates)
        {
            var sparse = new[] { true, false };
            var quote = new[] { true, false };
            var foundAny = false;
            var result = default(ColumnSplitResult);
            foreach (var perm in (from _allowSparse in sparse
                                    from _allowQuote in quote
                                    from _sep in separatorCandidates
                                    select new { _allowSparse, _allowQuote, _sep }))
            {
                var args = new TextLoader.Arguments
                {
                    Column = new[] { TextLoader.Column.Parse("C:TX:0-**") },
                    Separator = perm._sep.ToString(),
                    AllowQuoting = perm._allowQuote,
                    AllowSparse = perm._allowSparse
                };

                if (TryParseFile(args, source, out result))
                {
                    foundAny = true;
                    break;
                }
            }
            return foundAny ? result : new ColumnSplitResult(false, null, true, true, 0);
        }

        private static bool TryParseFile(TextLoader.Arguments args, IMultiStreamSource source, out ColumnSplitResult result)
        {
            result = null;
            var textLoader = new TextLoader(new MLContext(), args);
            var idv = textLoader.Read(source).Take(1000);
            var columnCounts = new List<int>();
            var column = idv.Schema["C"];
            var columnIndex = column.Index;

            using (var cursor = idv.GetRowCursor(x => x == columnIndex))
            {
                var getter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(columnIndex);

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

            result = new ColumnSplitResult(true, args.Separator, args.AllowQuoting, args.AllowSparse, mostCommon.Key);
            return true;
        }
    }
}