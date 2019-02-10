// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    /// <summary>
    /// This class incapsulates logic for automatic inference of column types for the text file.
    /// It also attempts to guess whether there is a header row.
    /// </summary>
    internal static class ColumnTypeInference
    {
        // Maximum number of columns to invoke type inference.
        // REVIEW: revisit this requirement. Either work for arbitrary number of columns,
        // or have a 'dumb' inference that would quickly figure everything out.
        private const int SmartColumnsLim = 10000;

        internal sealed class Arguments
        {
            public char Separator;
            public bool AllowSparse;
            public bool AllowQuote;
            public int ColumnCount;
            public bool HasHeader;
            public int MaxRowsToRead;

            public Arguments()
            {
                MaxRowsToRead = 10000;
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
            public readonly PrimitiveType ItemType;

            public string SuggestedName;

            public Column(int columnIndex, string suggestedName, PrimitiveType itemType)
            {
                ColumnIndex = columnIndex;
                SuggestedName = suggestedName;
                ItemType = itemType;
            }
        }

        public readonly struct InferenceResult
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
            internal sealed class BooleanValues : ITypeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    foreach (var col in columns)
                    {
                        if (!col.RawData.Skip(1)
                            .All(x =>
                            {
                                bool value;
                                return Conversions.TryParse(in x, out value);
                            })
                            )
                        {
                            continue;
                        }

                        col.SuggestedType = BoolType.Instance;
                        bool first;

                        col.HasHeader = !Conversions.TryParse(in col.RawData[0], out first);
                    }
                }
            }

            internal sealed class AllNumericValues : ITypeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    foreach (var col in columns)
                    {
                        // skip columns that already have a suggested type
                        if(col.SuggestedType != null)
                        {
                            continue;
                        }

                        if (!col.RawData.Skip(1)
                            .All(x =>
                            {
                                Single value;
                                return Conversions.TryParse(in x, out value);
                            })
                            )
                        {
                            continue;
                        }

                        col.SuggestedType = NumberType.R4;

                        var headerStr = col.RawData[0].ToString();
                        col.HasHeader = !double.TryParse(headerStr, out var doubleVal);
                    }
                }
            }

            internal sealed class EverythingText : ITypeInferenceExpert
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
        public static InferenceResult InferTextFileColumnTypes(MLContext env, IMultiStreamSource fileSource, Arguments args)
        {
            return InferTextFileColumnTypesCore(env, fileSource, args);
        }

        private static InferenceResult InferTextFileColumnTypesCore(MLContext env, IMultiStreamSource fileSource, Arguments args)
        {
            if (args.ColumnCount == 0)
            {
                // too many empty columns for automatic inference
                return InferenceResult.Fail();
            }

            if (args.ColumnCount >= SmartColumnsLim)
            {
                // too many columns for automatic inference
                return InferenceResult.Fail();
            }

            // read the file as the specified number of text columns
            var textLoaderArgs = new TextLoader.Arguments
            {
                Column = new[] { new TextLoader.Column("C", DataKind.TX, 0, args.ColumnCount - 1) },
                Separators = new[] { args.Separator },
                AllowSparse = args.AllowSparse,
                AllowQuoting = args.AllowQuote,
            };
            var textLoader = new TextLoader(env, textLoaderArgs);
            var idv = textLoader.Read(fileSource);
            idv = idv.Take(args.MaxRowsToRead);

            // read all the data into memory.
            // list items are rows of the dataset.
            var data = new List<ReadOnlyMemory<char>[]>();
            using (var cursor = idv.GetRowCursor(idv.Schema))
            {
                var column = cursor.Schema.GetColumnOrNull("C");
                int columnIndex = column.Value.Index;
                var colType = column.Value.Type;
                ValueGetter<VBuffer<ReadOnlyMemory<char>>> vecGetter = null;
                ValueGetter<ReadOnlyMemory<char>> oneGetter = null;
                bool isVector = colType.IsVector();
                if (isVector) { vecGetter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(columnIndex); }
                else
                {
                    oneGetter = cursor.GetGetter<ReadOnlyMemory<char>>(columnIndex);
                }

                VBuffer<ReadOnlyMemory<char>> line = default;
                ReadOnlyMemory<char> tsValue = default;
                while (cursor.MoveNext())
                {
                    if (isVector)
                    {
                        vecGetter(ref line);
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
                // too few rows for automatic inference
                return InferenceResult.Fail();
            }

            var cols = new IntermediateColumn[args.ColumnCount];
            for (int i = 0; i < args.ColumnCount; i++)
            {
                cols[i] = new IntermediateColumn(data.Select(x => x[i]).ToArray(), i);
            }

            foreach (var expert in GetExperts())
            {
                expert.Apply(cols);
            }

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

            // suggest names
            var names = new List<string>();
            usedNames.Clear();
            foreach (var col in cols)
            {
                string name0;
                string name;
                name0 = name = SuggestName(col, args.HasHeader);
                int i = 0;
                while (!usedNames.Add(name))
                    name = string.Format("{0}_{1:00}", name0, i++);
                names.Add(name);
            }
            var outCols =
                cols.Select((x, i) => new Column(x.ColumnId, names[i], x.SuggestedType)).ToArray();

            var numerics = outCols.Count(x => x.ItemType.IsNumber());
            
            return InferenceResult.Success(outCols, args.HasHeader, cols.Select(col => col.RawData).ToArray());
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
                var loaderColumn = new TextLoader.Column(col.SuggestedName, col.ItemType.GetRawKind(), col.ColumnIndex);
                loaderColumns.Add(loaderColumn);
            }
            return loaderColumns.ToArray();
        }
    }

}
