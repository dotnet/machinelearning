// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// This class incapsulates logic for automatic inference of column types for the text file.
    /// It also attempts to guess whether there is a header row.
    /// </summary>
    public static class ColumnTypeInference
    {
        // Maximum number of columns to invoke type inference.
        // REVIEW: revisit this requirement. Either work for arbitrary number of columns,
        // or have a 'dumb' inference that would quickly figure everything out.
        private const int SmartColumnsLim = 10000;

        public sealed class Arguments
        {
            public string Separator;
            public bool AllowSparse;
            public bool AllowQuote;
            public int ColumnCount;
            public int MaxRowsToRead;

            public Arguments()
            {
                MaxRowsToRead = 100;
            }
        }

        private class IntermediateColumn
        {
            private readonly ReadOnlyMemory<char>[] _data;
            private readonly int _columnId;
            private PrimitiveType _suggestedType;
            private bool? _hasHeader;

            public int ColumnId
            {
                get { return _columnId; }
            }

            public PrimitiveType SuggestedType
            {
                get { return _suggestedType; }
                set { _suggestedType = value; }
            }

            public bool? HasHeader
            {
                get { return _hasHeader; }
                set { _hasHeader = value; }
            }

            public IntermediateColumn(ReadOnlyMemory<char>[] data, int columnId)
            {
                _data = data;
                _columnId = columnId;
            }

            public ReadOnlyMemory<char>[] RawData { get { return _data; } }
        }

        public struct Column
        {
            public readonly int ColumnIndex;
            public readonly string SuggestedName;
            public readonly PrimitiveType ItemType;

            public Column(int columnIndex, string suggestedName, PrimitiveType itemType)
            {
                ColumnIndex = columnIndex;
                SuggestedName = suggestedName;
                ItemType = itemType;
            }
        }

        public struct InferenceResult
        {
            public readonly Column[] Columns;
            public readonly bool HasHeader;
            public readonly bool IsSuccess;
            public readonly ReadOnlyMemory<char>[][] Data;

            private InferenceResult(bool isSuccess, Column[] columns, bool hasHeader, ReadOnlyMemory<char>[][] data)
            {
                IsSuccess = isSuccess;
                Columns = columns;
                HasHeader = hasHeader;
                Data = data;
            }

            public static InferenceResult Success(Column[] columns, bool hasHeader, ReadOnlyMemory<char>[][] data)
            {
                return new InferenceResult(true, columns, hasHeader, data);
            }

            public static InferenceResult Fail()
            {
                return new InferenceResult(false, null, false, null);
            }
        }

        private interface ITypeInferenceExpert
        {
            void Apply(IntermediateColumn[] columns);
        }

        /// <summary>
        /// Current design is as follows: there's a sequence of 'experts' that each look at all the columns.
        /// Every expert may or may not assign the 'answer' (suggested type) to a column. If the expert needs
        /// some information about the column (for example, the column values), this information is lazily calculated
        /// by the column object, not the expert itself, to allow the reuse of the same information by another
        /// expert.
        /// </summary>
        private static class Experts
        {
            public sealed class BooleanValues : ITypeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    foreach (var col in columns)
                    {
                        if (!col.RawData.Skip(1)
                            .All(x =>
                                {
                                    bool value;
                                    return Conversions.Instance.TryParse(ref x, out value);
                                })
                            )
                        {
                            continue;
                        }

                        col.SuggestedType = BoolType.Instance;
                        bool first;

                        col.HasHeader = !Conversions.Instance.TryParse(ref col.RawData[0], out first);
                    }
                }
            }

            public sealed class AllNumericValues : ITypeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    foreach (var col in columns)
                    {
                        if (!col.RawData.Skip(1)
                            .All(x =>
                                {
                                    Single value;
                                    return Conversions.Instance.TryParse(ref x, out value);
                                })
                            )
                        {
                            continue;
                        }

                        col.SuggestedType = NumberType.R4;
                        Single first;

                        col.HasHeader = !DoubleParser.TryParse(col.RawData[0].Span, out first);
                    }
                }
            }

            public sealed class EverythingText : ITypeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    foreach (var col in columns)
                    {
                        if (col.SuggestedType != null)
                            continue;

                        col.SuggestedType = TextType.Instance;
                        col.HasHeader = IsLookLikeHeader(col.RawData[0]);
                    }
                }

                private bool? IsLookLikeHeader(ReadOnlyMemory<char> value)
                {
                    var v = value.ToString();
                    if (v.Length > 100)
                        return false;
                    var headerCandidates = new[] { "^Label", "^Feature", "^Market", "^m_", "^Weight" };
                    foreach (var candidate in headerCandidates)
                    {
                        if (Regex.IsMatch(v, candidate, RegexOptions.IgnoreCase))
                            return true;
                    }

                    return null;
                }
            }
        }

        private static IEnumerable<ITypeInferenceExpert> GetExperts()
        {
            // Current logic is pretty primitive: if every value (except the first) of a column
            // parses as a boolean it's boolean, if it parses as numeric then it's numeric. Otherwise, it is text.
            yield return new Experts.BooleanValues();
            yield return new Experts.AllNumericValues();
            yield return new Experts.EverythingText();
        }

        /// <summary>
        /// Auto-detect column types of the file.
        /// </summary>
        public static InferenceResult InferTextFileColumnTypes(IHostEnvironment env, IMultiStreamSource fileSource, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(fileSource, nameof(fileSource));
            env.CheckValue(args, nameof(args));
            env.CheckNonEmpty(args.Separator, nameof(args.Separator));
            env.Check(args.MaxRowsToRead > 0);

            using (var ch = env.Register("InferTextFileColumnTypes").Start("TypeInference"))
            {
                var result = InferTextFileColumnTypesCore(env, fileSource, args, ch);
                ch.Done();
                return result;
            }
        }

        private static InferenceResult InferTextFileColumnTypesCore(IHostEnvironment env, IMultiStreamSource fileSource, Arguments args, IChannel ch)
        {
            Contracts.AssertValue(ch);
            ch.AssertValue(env);
            ch.AssertValue(fileSource);
            ch.AssertValue(args);

            if (args.ColumnCount==0)
            {
                ch.Error("Too many empty columns for automatic inference.");
                return InferenceResult.Fail();
            }

            if (args.ColumnCount >= SmartColumnsLim)
            {
                ch.Error("Too many columns for automatic inference.");
                return InferenceResult.Fail();
            }

            // Read the file as the specified number of text columns.
            var textLoaderArgs = new TextLoader.Arguments
            {
                Column = new[] { TextLoader.Column.Parse(string.Format("C:TX:0-{0}", args.ColumnCount - 1)) },
                Separator = args.Separator,
                AllowSparse = args.AllowSparse,
                AllowQuoting = args.AllowQuote,
            };
            var idv = TextLoader.ReadFile(env, textLoaderArgs, fileSource);
            idv = idv.Take(args.MaxRowsToRead);

            // Read all the data into memory.
            // List items are rows of the dataset.
            var data = new List<ReadOnlyMemory<char>[]>();
            using (var cursor = idv.GetRowCursor(col => true))
            {
                int columnIndex;
                bool found = cursor.Schema.TryGetColumnIndex("C", out columnIndex);
                Contracts.Assert(found);
                var colType = cursor.Schema.GetColumnType(columnIndex);
                Contracts.Assert(colType.ItemType.IsText);
                ValueGetter<VBuffer<ReadOnlyMemory<char>>> vecGetter = null;
                ValueGetter<ReadOnlyMemory<char>> oneGetter = null;
                bool isVector = colType.IsVector;
                if (isVector)
                    vecGetter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(columnIndex);
                else
                {
                    Contracts.Assert(args.ColumnCount == 1);
                    oneGetter = cursor.GetGetter<ReadOnlyMemory<char>>(columnIndex);
                }

                VBuffer<ReadOnlyMemory<char>> line = default;
                ReadOnlyMemory<char> tsValue = default;
                while (cursor.MoveNext())
                {
                    if (isVector)
                    {
                        vecGetter(ref line);
                        Contracts.Assert(line.Length == args.ColumnCount);
                        var values = new ReadOnlyMemory<char>[args.ColumnCount];
                        line.CopyTo(values);
                        data.Add(values);
                    }
                    else
                    {
                        oneGetter(ref tsValue);
                        var values = new[] { tsValue };
                        data.Add(values);
                    }
                }
            }

            if (data.Count < 2)
            {
                ch.Error("Too few rows ({0}) for automatic inference.", data.Count);
                return InferenceResult.Fail();
            }

            var cols = new IntermediateColumn[args.ColumnCount];
            for (int i = 0; i < args.ColumnCount; i++)
                cols[i] = new IntermediateColumn(data.Select(x => x[i]).ToArray(), i);

            foreach (var expert in GetExperts())
                expert.Apply(cols);

            Contracts.Check(cols.All(x => x.SuggestedType != null), "Column type inference must be conclusive");

            // Aggregating header signals.
            int suspect = 0;
            var usedNames = new HashSet<string>();
            for (int i = 0; i < args.ColumnCount; i++)
            {
                if (cols[i].HasHeader == true)
                {
                    if (usedNames.Add(cols[i].RawData[0].ToString()))
                        suspect++;
                    else
                    {
                        // duplicate value in the first column is a strong signal that this is not a header
                        suspect -= args.ColumnCount;
                    }
                }
                else if (cols[i].HasHeader == false)
                    suspect--;
            }

            // REVIEW: Why not use this for column names as well?
            TextLoader.Arguments fileArgs;
            bool hasHeader;
            if (TextLoader.FileContainsValidSchema(env, fileSource, out fileArgs))
                hasHeader = fileArgs.HasHeader;
            else
                hasHeader = suspect > 0;

            // suggest names
            var names = new List<string>();
            usedNames.Clear();
            foreach (var col in cols)
            {
                string name0;
                string name;
                name0 = name = SuggestName(col, hasHeader);
                int i = 0;
                while (!usedNames.Add(name))
                    name = string.Format("{0}_{1:00}", name0, i++);
                names.Add(name);
            }
            var outCols =
                cols.Select((x, i) => new Column(x.ColumnId, names[i], x.SuggestedType)).ToArray();

            var numerics = outCols.Count(x => x.ItemType.IsNumber);

            ch.Info("Detected {0} numeric and {1} text columns.", numerics, outCols.Length - numerics);
            if (hasHeader)
                ch.Info("Generated column names from the file header.");

            return InferenceResult.Success(outCols, hasHeader, cols.Select(col => col.RawData).ToArray());
        }

        private static string SuggestName(IntermediateColumn column, bool hasHeader)
        {
            var header = column.RawData[0].ToString();
            return (hasHeader && !string.IsNullOrWhiteSpace(header)) ? Sanitize(header) : string.Format("col{0}", column.ColumnId);
        }

        private static string Sanitize(string header)
        {
            // replace all non-letters and non-digits with '_'.
            return string.Join("", header.Select(x => Char.IsLetterOrDigit(x) ? x : '_'));
        }

        public static TextLoader.Column[] GenerateLoaderColumns(Column[] columns)
        {
            var loaderColumns = new List<TextLoader.Column>();
            foreach (var col in columns)
            {
                var loaderColumn = TextLoader.Column.Parse(string.Format("{0}:{1}:{2}", col.SuggestedName, col.ItemType, col.ColumnIndex));
                if (loaderColumn != null && loaderColumn.IsValid())
                    loaderColumns.Add(loaderColumn);
            }
            return loaderColumns.ToArray();
        }
    }

}
