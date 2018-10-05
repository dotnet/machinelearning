// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;
using System.Collections.Concurrent;
using Microsoft.ML.Runtime.Data.IO;

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// Utilities for various heuristics against text files.
    /// Currently, separator inference and column count detection.
    /// </summary>
    public static class TextFileContents
    {
        public struct ColumnSplitResult
        {
            public readonly bool IsSuccess;
            public readonly string Separator;
            public readonly bool AllowQuote;
            public readonly bool AllowSparse;
            public readonly int ColumnCount;

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

        public static string[] DefaultSeparators = new[] { "tab", ",", ";", " " };

        /// <summary>
        /// Attempt to detect text loader arguments.
        /// The algorithm selects the first 'acceptable' set: the one that recognizes the same number of columns in at
        /// least <see cref="UniformColumnCountThreshold"/> of the sample's lines,
        /// and this number of columns is more than 1.
        /// We sweep on separator, allow sparse and allow quote parameter.
        /// </summary>
        public static ColumnSplitResult TrySplitColumns(IHostEnvironment env, IMultiStreamSource source,
            string[] separatorCandidates, bool? allowSparse = null, bool? allowQuote = null, bool skipStrictValidation = false)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("CandidateLoader");
            h.CheckValue(source, nameof(source));
            h.CheckNonEmpty(separatorCandidates, nameof(separatorCandidates));
            // Default value for sparse and quote is true.
            bool[] sparse = new[] { true, false };
            bool[] quote = new[] { true, false };
            if (allowSparse.HasValue)
                sparse = new[] { allowSparse.Value };
            if (allowQuote.HasValue)
                quote = new[] { allowQuote.Value };
            bool foundAny = false;
            var result = default(ColumnSplitResult);
            using (var ch = env.Register("SplitColumns").Start("SplitColumns"))
            {
                foreach (var perm in (from _allowSparse in sparse
                                      from _allowQuote in quote
                                      from _sep in separatorCandidates
                                      select new { _allowSparse, _allowQuote, _sep }))
                {
                    var args = new TextLoader.Arguments
                    {
                        Column = new[] { TextLoader.Column.Parse("C:TX:0-**") },
                        Separator = perm._sep,
                        AllowQuoting = perm._allowQuote,
                        AllowSparse = perm._allowSparse
                    };

                    if (TryParseFile(ch, args, source, skipStrictValidation, out result))
                    {
                        foundAny = true;
                        break;
                    }
                }

                if (foundAny)
                    ch.Info("Discovered {0} columns using separator '{1}'.", result.ColumnCount, result.Separator);
                else
                {
                    // REVIEW: May need separate messages for GUI-specific and non-specific. This component can be used
                    // by itself outside the GUI.
                    ch.Info("Couldn't determine columns in the file using separators {0}. Does the input file consist of only a single column? "
                        + "If so, in TLC GUI, please close the import wizard, and then, in the loader settings to the right, manually add a column, "
                        + "choose a name, and set source index to 0.",
                        string.Join(",", separatorCandidates.Select(c => string.Format("'{0}'", GetSeparatorString(c)))));
                }
            }
            return foundAny ? result : new ColumnSplitResult(false, null, true, true, 0);
        }

        private static string GetSeparatorString(string sep)
        {
            Contracts.AssertValue(sep);
            if (sep.Length == 1)
                return TextSaver.SeparatorCharToString(sep[0]);
            return sep;
        }

        private static bool TryParseFile(IChannel ch, TextLoader.Arguments args, IMultiStreamSource source, bool skipStrictValidation, out ColumnSplitResult result)
        {
            result = default(ColumnSplitResult);
            try
            {
                // No need to provide information from unsuccessful loader, so we create temporary environment and get information from it in case of success
                using (var loaderEnv = new ConsoleEnvironment(0, true))
                {
                    var messages = new ConcurrentBag<ChannelMessage>();
                    loaderEnv.AddListener<ChannelMessage>(
                        (src, msg) =>
                        {
                            messages.Add(msg);
                        });
                    var idv = TextLoader.ReadFile(loaderEnv, args, source).Take(1000);
                    var columnCounts = new List<int>();
                    int columnIndex;
                    bool found = idv.Schema.TryGetColumnIndex("C", out columnIndex);
                    ch.Assert(found);

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

                    Contracts.Check(columnCounts.Count > 0);
                    var mostCommon = columnCounts.GroupBy(x => x).OrderByDescending(x => x.Count()).First();
                    if (!skipStrictValidation && mostCommon.Count() < UniformColumnCountThreshold * columnCounts.Count)
                        return false;

                    // If user explicitly specified separator we're allowing "single" column case;
                    // Otherwise user will see message informing that we were not able to detect any columns.
                    if (!skipStrictValidation && mostCommon.Key <= 1)
                        return false;

                    result = new ColumnSplitResult(true, args.Separator, args.AllowQuoting, args.AllowSparse, mostCommon.Key);
                    ch.Trace("Discovered {0} columns using separator '{1}'", mostCommon.Key, args.Separator);
                    foreach (var msg in messages)
                        ch.Send(msg);
                    return true;
                }
            }
            catch (Exception ex)
            {
                if (!ex.IsMarked())
                    throw;
                // For known exceptions, we just continue to the next separator candidate.
            }
            return false;
        }
    }
}
