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
            private bool _isPurposeSuggested;
            private ColumnPurpose _suggestedPurpose;
            private readonly Lazy<ColumnType> _type;
            private readonly Lazy<string> _columnName;
            private object _cachedData;

            public bool IsPurposeSuggested { get { return _isPurposeSuggested; } }

            public ColumnPurpose SuggestedPurpose
            {
                get { return _suggestedPurpose; }
                set
                {
                    _suggestedPurpose = value;
                    _isPurposeSuggested = true;
                }
            }

            public ColumnType Type { get { return _type.Value; } }

            public string ColumnName { get { return _columnName.Value; } }

            public IntermediateColumn(IDataView data, int columnId, ColumnPurpose suggestedPurpose = ColumnPurpose.Ignore)
            {
                _data = data;
                _columnId = columnId;
                _type = new Lazy<ColumnType>(() => _data.Schema[_columnId].Type);
                _columnName = new Lazy<string>(() => _data.Schema[_columnId].Name);
                _suggestedPurpose = suggestedPurpose;
            }

            public Column GetColumn()
            {
                return new Column(_columnId, _suggestedPurpose);
            }

            public T[] GetData<T>()
            {
                if (_cachedData is T[])
                    return _cachedData as T[];

                var results = new List<T>();
                using (var cursor = _data.GetRowCursor(new[] { _data.Schema[_columnId] }))
                {
                    var getter = cursor.GetGetter<T>(_columnId);
                    while (cursor.MoveNext())
                    {
                        T value = default(T);
                        getter(ref value);
                        results.Add(value);
                    }
                }

                T[] resultArray;
                _cachedData = resultArray = results.ToArray();
                return resultArray;
            }
        }

        private static class Experts
        {
            internal sealed class HeaderComprehension : IPurposeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    foreach (var column in columns)
                    {
                        if (column.IsPurposeSuggested)
                            continue;
                        else if (Regex.IsMatch(column.ColumnName, @"^m_queryid$", RegexOptions.IgnoreCase))
                            column.SuggestedPurpose = ColumnPurpose.Group;
                        else if (Regex.IsMatch(column.ColumnName, @"group", RegexOptions.IgnoreCase))
                            column.SuggestedPurpose = ColumnPurpose.Group;
                        else if (Regex.IsMatch(column.ColumnName, @"^m_\w+id$", RegexOptions.IgnoreCase))
                            column.SuggestedPurpose = ColumnPurpose.Name;
                        else if (Regex.IsMatch(column.ColumnName, @"^id$", RegexOptions.IgnoreCase))
                            column.SuggestedPurpose = ColumnPurpose.Name;
                        else if (Regex.IsMatch(column.ColumnName, @"^m_", RegexOptions.IgnoreCase))
                            column.SuggestedPurpose = ColumnPurpose.Ignore;
                        else
                            continue;
                    }
                }
            }

            internal sealed class TextClassification : IPurposeInferenceExpert
            {
                public void Apply(IntermediateColumn[] columns)
                {
                    string[] commonImageExtensions = { ".bmp", ".dib", ".rle", ".jpg", ".jpeg", ".jpe", ".jfif", ".gif", ".tif", ".tiff", ".png" };
                    foreach (var column in columns)
                    {
                        if (column.IsPurposeSuggested || !column.Type.IsText())
                            continue;
                        var data = column.GetData<ReadOnlyMemory<char>>();

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

                        if (imagePathCount < data.Length - 1)
                        {
                            Double avgLength = 1.0 * sumLength / data.Length;
                            Double cardinalityRatio = 1.0 * seen.Count / data.Length;
                            Double avgSpaces = 1.0 * sumSpaces / data.Length;
                            if (cardinalityRatio < 0.7 || seen.Count < 100)
                                column.SuggestedPurpose = ColumnPurpose.CategoricalFeature;
                            // (note: the columns.Count() == 1 condition below, in case a dataset has only
                            // a 'name' and a 'label' column, forces what would be a 'name' column to become a text feature)
                            else if (cardinalityRatio >= 0.85 && (avgLength > 30 || avgSpaces >= 1 || columns.Count() == 1))
                                column.SuggestedPurpose = ColumnPurpose.TextFeature;
                            else if (cardinalityRatio >= 0.9)
                                column.SuggestedPurpose = ColumnPurpose.Name;
                            else
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

            // Use column names to suggest purpose.
            yield return new Experts.HeaderComprehension();
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
        public static PurposeInference.Column[] InferPurposes(MLContext context, IDataView data, string label,
            IDictionary<string, ColumnPurpose> columnOverrides = null)
        {
            data = data.Take(MaxRowsToRead);

            var allColumns = new List<IntermediateColumn>();
            var columnsToInfer = new List<IntermediateColumn>();

            for (var i = 0; i < data.Schema.Count; i++)
            {
                var column = data.Schema[i];
                IntermediateColumn intermediateCol;

                if(column.Name == label)
                {
                    intermediateCol = new IntermediateColumn(data, i, ColumnPurpose.Label);
                }
                else if (column.IsHidden)
                {
                    intermediateCol = new IntermediateColumn(data, i, ColumnPurpose.Ignore);
                }
                else if(columnOverrides != null && columnOverrides.TryGetValue(column.Name, out var columnPurpose))
                {
                    intermediateCol = new IntermediateColumn(data, i, columnPurpose);
                }
                else
                {
                    intermediateCol = new IntermediateColumn(data, i);
                    columnsToInfer.Add(intermediateCol);
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
