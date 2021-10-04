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
    /// Automatic inference of column purposes for the data view.
    /// This is used in the context of text import wizard, but can be used outside as well.
    /// </summary>
    internal static class PurposeInference
    {
        public const int MaxRowsToRead = 1000;

        public class Column
        {
            public readonly int ColumnIndex;
            public readonly ColumnPurpose Purpose;

            public Column(int columnIndex, ColumnPurpose purpose)
            {
                ColumnIndex = columnIndex;
                Purpose = purpose;
            }
        }

        /// <summary>
        /// The design is the same as for <see cref="ColumnTypeInference"/>: there's a sequence of 'experts'
        /// that each look at all the columns. Every expert may or may not assign the 'answer' (suggested purpose)
        /// to a column. If the expert needs some information about the column (for example, the column values),
        /// this information is lazily calculated by the column object, not the expert itself, to allow the reuse
        /// of the same information by another expert.
        /// </summary>
        private interface IPurposeInferenceExpert
        {
            void Apply(IntermediateColumn[] columns);
        }

        private class IntermediateColumn
        {
            private readonly IDataView _data;
            private readonly int _columnId;
            private ColumnPurpose _suggestedPurpose;
            private readonly Lazy<DataViewType> _type;
            private readonly Lazy<string> _columnName;
            private IReadOnlyList<ReadOnlyMemory<char>> _cachedData;

            public bool IsPurposeSuggested { get; private set; }

            public ColumnPurpose SuggestedPurpose
            {
                get { return _suggestedPurpose; }
                set
                {
                    _suggestedPurpose = value;
                    IsPurposeSuggested = true;
                }
            }

            public DataViewType Type { get { return _type.Value; } }

            public string ColumnName { get { return _columnName.Value; } }

            public IntermediateColumn(IDataView data, int columnId, ColumnPurpose suggestedPurpose = ColumnPurpose.Ignore)
            {
                _data = data;
                _columnId = columnId;
                _type = new Lazy<DataViewType>(() => _data.Schema[_columnId].Type);
                _columnName = new Lazy<string>(() => _data.Schema[_columnId].Name);
                _suggestedPurpose = suggestedPurpose;
            }

            public Column GetColumn()
            {
                return new Column(_columnId, _suggestedPurpose);
            }

            public IReadOnlyList<ReadOnlyMemory<char>> GetColumnData()
            {
                if (_cachedData != null)
                    return _cachedData;

                var results = new List<ReadOnlyMemory<char>>();
                var column = _data.Schema[_columnId];

                using (var cursor = _data.GetRowCursor(new[] { column }))
                {
                    var getter = cursor.GetGetter<ReadOnlyMemory<char>>(column);
                    while (cursor.MoveNext())
                    {
                        var value = default(ReadOnlyMemory<char>);
                        getter(ref value);

                        var copy = new ReadOnlyMemory<char>(value.ToArray());

                        results.Add(copy);
                    }
                }

                _cachedData = results;

                return results;
            }
        }

        private static class Experts
        {
            internal sealed class TextClassification : IPurposeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    string[] commonImageExtensions = { ".bmp", ".dib", ".rle", ".jpg", ".jpeg", ".jpe", ".jfif", ".gif", ".tif", ".tiff", ".png" };
                    foreach (var column in columns)
                    {
                        if (column.IsPurposeSuggested || !column.Type.IsText())
                            continue;

                        var data = column.GetColumnData();

                        long sumLength = 0;
                        int sumSpaces = 0;
                        var seen = new HashSet<string>();
                        int imagePathCount = 0;
                        foreach (var span in data)
                        {
                            sumLength += span.Length;
                            seen.Add(span.ToString());
                            string spanStr = span.ToString();
                            sumSpaces += spanStr.Count(x => x == ' ');

                            foreach (var ext in commonImageExtensions)
                            {
                                if (spanStr.EndsWith(ext, StringComparison.OrdinalIgnoreCase))
                                {
                                    imagePathCount++;
                                    break;
                                }
                            }
                        }

                        if (imagePathCount < data.Count - 1)
                        {
                            Double avgLength = 1.0 * sumLength / data.Count;
                            Double cardinalityRatio = 1.0 * seen.Count / data.Count;
                            Double avgSpaces = 1.0 * sumSpaces / data.Count;
                            if (cardinalityRatio < 0.7)
                                column.SuggestedPurpose = ColumnPurpose.CategoricalFeature;
                            // (note: the columns.Count() == 1 condition below, in case a dataset has only
                            // a 'name' and a 'label' column, forces what would be an 'ignore' column to become a text feature)
                            else if (cardinalityRatio >= 0.85 && (avgLength > 30 || avgSpaces >= 1 || columns.Count() == 1))
                                column.SuggestedPurpose = ColumnPurpose.TextFeature;
                            else if (cardinalityRatio >= 0.9)
                                column.SuggestedPurpose = ColumnPurpose.Ignore;
                        }
                        else
                            column.SuggestedPurpose = ColumnPurpose.ImagePath;
                    }
                }
            }

            internal sealed class NumericAreFeatures : IPurposeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    foreach (var column in columns)
                    {
                        if (column.IsPurposeSuggested)
                            continue;
                        if (column.Type.GetItemType().IsNumber())
                            column.SuggestedPurpose = ColumnPurpose.NumericFeature;
                    }
                }
            }

            internal sealed class BooleanProcessing : IPurposeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    foreach (var column in columns)
                    {
                        if (column.IsPurposeSuggested)
                            continue;
                        if (column.Type.GetItemType().IsBool())
                            column.SuggestedPurpose = ColumnPurpose.NumericFeature;
                    }
                }
            }

            internal sealed class TextArraysAreText : IPurposeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    foreach (var column in columns)
                    {
                        if (column.IsPurposeSuggested)
                            continue;
                        if (column.Type.IsVector() && column.Type.GetItemType().IsText())
                            column.SuggestedPurpose = ColumnPurpose.TextFeature;
                    }
                }
            }

            internal sealed class IgnoreEverythingElse : IPurposeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    foreach (var column in columns)
                    {
                        if (!column.IsPurposeSuggested)
                            column.SuggestedPurpose = ColumnPurpose.Ignore;
                    }
                }
            }
        }

        private static IEnumerable<IPurposeInferenceExpert> GetExperts()
        {
            // Each of the experts respects the decisions of all the experts above.

            // Single-value text columns may be category, name, text or ignore.
            yield return new Experts.TextClassification();
            // Vector-value text columns are always treated as text.
            // REVIEW: could be improved.
            yield return new Experts.TextArraysAreText();
            // Check column on boolean only values.
            yield return new Experts.BooleanProcessing();
            // All numeric columns are features.
            yield return new Experts.NumericAreFeatures();
            // Everything else is ignored.
            yield return new Experts.IgnoreEverythingElse();
        }

        /// <summary>
        /// Auto-detect purpose for the data view columns.
        /// </summary>
        public static PurposeInference.Column[] InferPurposes(MLContext context, IDataView data,
            ColumnInformation columnInfo)
        {
            data = context.Data.TakeRows(data, MaxRowsToRead);
            var allColumns = new List<IntermediateColumn>();
            var columnsToInfer = new List<IntermediateColumn>();

            for (var i = 0; i < data.Schema.Count; i++)
            {
                var column = data.Schema[i];
                IntermediateColumn intermediateCol;

                if (column.IsHidden)
                {
                    intermediateCol = new IntermediateColumn(data, i, ColumnPurpose.Ignore);
                    allColumns.Add(intermediateCol);
                    continue;
                }

                var columnPurpose = columnInfo.GetColumnPurpose(column.Name);
                if (columnPurpose == null)
                {
                    intermediateCol = new IntermediateColumn(data, i);
                    columnsToInfer.Add(intermediateCol);
                }
                else
                {
                    intermediateCol = new IntermediateColumn(data, i, columnPurpose.Value);
                }

                allColumns.Add(intermediateCol);
            }

            foreach (var expert in GetExperts())
            {
                expert.Apply(columnsToInfer.ToArray());
            }

            return allColumns.Select(c => c.GetColumn()).ToArray();
        }
    }
}
