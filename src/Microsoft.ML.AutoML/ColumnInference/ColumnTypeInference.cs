// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.ML.Data;
using Microsoft.ML.Data.Conversion;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// This class encapsulates logic for automatic inference of column types for the text file.
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
            public uint? LabelColumnIndex;
            public string Label;
            public bool ReadMultilines;

            public Arguments()
            {
                MaxRowsToRead = 10000;
            }
        }

        private class IntermediateColumn
        {
            private readonly ReadOnlyMemory<char>[] _data;
            private readonly int _columnId;
            private PrimitiveDataViewType _suggestedType;
            private bool? _hasHeader;

            public int ColumnId
            {
                get { return _columnId; }
            }

            public PrimitiveDataViewType SuggestedType
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

            public string Name { get; set; }

            public bool HasAllBooleanValues()
            {
                if (RawData.Skip(1)
                    .All(x =>
                    {
                        bool value;
                        // (note: Conversions.Instance.TryParse parses an empty string as a Boolean)
                        return !string.IsNullOrEmpty(x.ToString()) &&
                            Conversions.DefaultInstance.TryParse(in x, out value);
                    }))
                {
                    return true;
                }

                return false;
            }
        }

        public class Column
        {
            public readonly int ColumnIndex;

            public PrimitiveDataViewType ItemType;
            public string SuggestedName;

            public Column(int columnIndex, string suggestedName, PrimitiveDataViewType itemType)
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
                        // skip columns that already have a suggested type,
                        // or that don't have all Boolean values
                        if (col.SuggestedType != null ||
                            !col.HasAllBooleanValues())
                        {
                            continue;
                        }

                        col.SuggestedType = BooleanDataViewType.Instance;
                        bool first;

                        col.HasHeader = !Conversions.DefaultInstance.TryParse(in col.RawData[0], out first);
                    }
                }
            }

            internal sealed class AllNumericValues : ITypeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    foreach (var col in columns)
                    {
                        if (!col.RawData.Skip(1)
                            .All(x =>
                            {
                                float value;
                                return Conversions.DefaultInstance.TryParse(in x, out value);
                            })
                            )
                        {
                            continue;
                        }

                        col.SuggestedType = NumberDataViewType.Single;

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

                        col.SuggestedType = TextDataViewType.Instance;
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
            // parses as numeric then it's numeric. Else if it parses as a Boolean, it's Boolean. Otherwise, it is text.
            yield return new Experts.AllNumericValues();
            yield return new Experts.BooleanValues();
            yield return new Experts.EverythingText();
        }

        /// <summary>
        /// Auto-detect column types of the file.
        /// </summary>
        public static InferenceResult InferTextFileColumnTypes(MLContext context, IMultiStreamSource fileSource, Arguments args)
        {
            return InferTextFileColumnTypesCore(context, fileSource, args);
        }

        private static InferenceResult InferTextFileColumnTypesCore(MLContext context, IMultiStreamSource fileSource, Arguments args)
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
            var textLoaderOptions = new TextLoader.Options
            {
                Columns = new[] { new TextLoader.Column("C", DataKind.String, 0, args.ColumnCount - 1) },
                Separators = new[] { args.Separator },
                AllowSparse = args.AllowSparse,
                AllowQuoting = args.AllowQuote,
                ReadMultilines = args.ReadMultilines,
            };
            var textLoader = context.Data.CreateTextLoader(textLoaderOptions);
            var idv = textLoader.Load(fileSource);
            idv = context.Data.TakeRows(idv, args.MaxRowsToRead);

            // read all the data into memory.
            // list items are rows of the dataset.
            var data = new List<ReadOnlyMemory<char>[]>();
            using (var cursor = idv.GetRowCursor(idv.Schema))
            {
                var column = cursor.Schema.GetColumnOrNull("C").Value;
                var colType = column.Type;
                ValueGetter<VBuffer<ReadOnlyMemory<char>>> vecGetter = null;
                ValueGetter<ReadOnlyMemory<char>> oneGetter = null;
                bool isVector = colType.IsVector();
                if (isVector) { vecGetter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(column); }
                else
                {
                    oneGetter = cursor.GetGetter<ReadOnlyMemory<char>>(column);
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
            usedNames.Clear();
            foreach (var col in cols)
            {
                string name0;
                string name;
                name0 = name = SuggestName(col, args.HasHeader);
                int i = 0;
                while (!usedNames.Add(name))
                {
                    name = string.Format("{0}_{1:00}", name0, i++);
                }
                col.Name = name;
            }

            // validate & retrieve label column
            var labelColumn = GetAndValidateLabelColumn(args, cols);

            // if label column has all Boolean values, set its type as Boolean
            if (labelColumn.HasAllBooleanValues())
            {
                labelColumn.SuggestedType = BooleanDataViewType.Instance;
            }

            var outCols = cols.Select(x => new Column(x.ColumnId, x.Name, x.SuggestedType)).ToArray();

            return InferenceResult.Success(outCols, args.HasHeader, cols.Select(col => col.RawData).ToArray());
        }

        private static string SuggestName(IntermediateColumn column, bool hasHeader)
        {
            var header = column.RawData[0].ToString();
            return (hasHeader && !string.IsNullOrWhiteSpace(header)) ? header : string.Format("col{0}", column.ColumnId);
        }

        private static IntermediateColumn GetAndValidateLabelColumn(Arguments args, IntermediateColumn[] cols)
        {
            IntermediateColumn labelColumn = null;
            if (args.LabelColumnIndex != null)
            {
                // if label column index > inferred # of columns, throw error
                if (args.LabelColumnIndex >= cols.Count())
                {
                    throw new ArgumentOutOfRangeException(nameof(args.LabelColumnIndex), $"Label column index ({args.LabelColumnIndex}) is >= than # of inferred columns ({cols.Count()}).");
                }

                labelColumn = cols[args.LabelColumnIndex.Value];
            }
            else
            {
                labelColumn = cols.FirstOrDefault(c => c.Name == args.Label);
                if (labelColumn == null)
                {
                    throw new ArgumentException($"Specified label column '{args.Label}' was not found.");
                }
            }

            return labelColumn;
        }

        public static TextLoader.Column[] GenerateLoaderColumns(Column[] columns)
        {
            var loaderColumns = new List<TextLoader.Column>();
            foreach (var col in columns)
            {
                var loaderColumn = new TextLoader.Column(col.SuggestedName, col.ItemType.GetRawKind().ToDataKind(), col.ColumnIndex);
                loaderColumns.Add(loaderColumn);
            }
            return loaderColumns.ToArray();
        }
    }

}
