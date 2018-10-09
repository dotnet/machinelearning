// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// Automatic inference of column purposes for the data view.
    /// This is used in the context of text import wizard, but can be used outside as well.
    /// </summary>
    public static class PurposeInference
    {
        public sealed class Arguments
        {
            public int MaxRowsToRead;

            public Arguments()
            {
                MaxRowsToRead = 1000;
            }
        }

        public struct Column
        {
            public readonly int ColumnIndex;
            public readonly ColumnPurpose Purpose;
            public readonly DataKind ItemKind;

            public Column(int columnIndex, ColumnPurpose purpose, DataKind itemKind)
            {
                ColumnIndex = columnIndex;
                Purpose = purpose;
                ItemKind = itemKind;
            }
        }

        public struct InferenceResult
        {
            public readonly Column[] Columns;

            public InferenceResult(Column[] columns)
            {
                Columns = columns;
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
            void Apply(IChannel ch, IntermediateColumn[] columns);
        }

        private class IntermediateColumn
        {
            private readonly IDataView _data;
            private readonly int _columnId;
            private bool _isPurposeSuggested;
            private ColumnPurpose _suggestedPurpose;
            private readonly Lazy<Data.ColumnType> _type;
            private readonly Lazy<string> _columnName;
            private object _cachedData;

            public int ColumnId
            {
                get { return _columnId; }
            }

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

            public Data.ColumnType Type { get { return _type.Value; } }

            public string ColumnName { get { return _columnName.Value; } }

            public IntermediateColumn(IDataView data, int columnId)
            {
                _data = data;
                _columnId = columnId;
                _type = new Lazy<Data.ColumnType>(() => _data.Schema.GetColumnType(_columnId));
                _columnName = new Lazy<string>(() => _data.Schema.GetColumnName(_columnId));
            }

            public Column GetColumn()
            {
                Contracts.Assert(_isPurposeSuggested);
                return new Column(_columnId, _suggestedPurpose, _type.Value.RawKind);
            }

            public T[] GetData<T>()
            {
                if (_cachedData is T[])
                    return _cachedData as T[];

                var results = new List<T>();
                using (var cursor = _data.GetRowCursor(id => id == _columnId))
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
            public sealed class HeaderComprehension : IPurposeInferenceExpert
            {
                public void Apply(IChannel ch, IntermediateColumn[] columns)
                {
                    foreach (var column in columns)
                    {
                        if (column.IsPurposeSuggested)
                            continue;
                        if (Regex.IsMatch(column.ColumnName, @"label", RegexOptions.IgnoreCase))
                            column.SuggestedPurpose = ColumnPurpose.Label;
                        if (Regex.IsMatch(column.ColumnName, @"^target$", RegexOptions.IgnoreCase))
                            column.SuggestedPurpose = ColumnPurpose.Label;
                        else if (Regex.IsMatch(column.ColumnName, @"^m_rating$", RegexOptions.IgnoreCase))
                            column.SuggestedPurpose = ColumnPurpose.Label;
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
                        ch.Info("Column '{0}' is {1} based on header.", column.ColumnName, column.SuggestedPurpose);
                    }
                }
            }

            public sealed class TextClassification : IPurposeInferenceExpert
            {
                public void Apply(IChannel ch, IntermediateColumn[] columns)
                {
                    string[] commonImageExtensions = { ".bmp", ".dib", ".rle", ".jpg", ".jpeg", ".jpe", ".jfif", ".gif", ".tif", ".tiff", ".png" };
                    foreach (var column in columns)
                    {
                        if (column.IsPurposeSuggested || !column.Type.IsText)
                            continue;
                        var data = column.GetData<ReadOnlyMemory<char>>();

                        long sumLength = 0;
                        int sumSpaces = 0;
                        var seen = new HashSet<uint>();
                        int imagePathCount = 0;
                        foreach (var span in data)
                        {
                            sumLength += span.Length;
                            seen.Add(Hashing.MurmurHash(0, span.Span));
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
                            else if (cardinalityRatio >= 0.85 && (avgLength > 30 || avgSpaces >= 1))
                                column.SuggestedPurpose = ColumnPurpose.TextFeature;
                            else if (cardinalityRatio >= 0.9)
                                column.SuggestedPurpose = ColumnPurpose.Name;
                            else
                                column.SuggestedPurpose = ColumnPurpose.Ignore;
                        }
                        else
                            column.SuggestedPurpose = ColumnPurpose.ImagePath;

                        ch.Info("Text column '{0}' purpose detected as '{1}' based on values.", column.ColumnName, column.SuggestedPurpose);
                    }
                }
            }

            public sealed class FirstNumericOrBooleanIsLabel : IPurposeInferenceExpert
            {
                public void Apply(IChannel ch, IntermediateColumn[] columns)
                {
                    if (columns.Any(x => x.IsPurposeSuggested && x.SuggestedPurpose == ColumnPurpose.Label))
                        return;

                    var firstNumeric =
                        columns.FirstOrDefault(x => !x.IsPurposeSuggested && !x.Type.IsVector && (x.Type.IsNumber || x.Type.IsBool));

                    if (firstNumeric != null)
                    {
                        firstNumeric.SuggestedPurpose = ColumnPurpose.Label;
                        ch.Info("Column '{0}' auto-designated as label.", firstNumeric.ColumnName);
                    }
                }
            }

            public sealed class NumericAreFeatures : IPurposeInferenceExpert
            {
                public void Apply(IChannel ch, IntermediateColumn[] columns)
                {
                    foreach (var column in columns)
                    {
                        if (column.IsPurposeSuggested)
                            continue;
                        if (column.Type.ItemType.IsNumber)
                            column.SuggestedPurpose = ColumnPurpose.NumericFeature;
                    }
                }
            }

            public sealed class BooleanProcessing : IPurposeInferenceExpert
            {
                public void Apply(IChannel ch, IntermediateColumn[] columns)
                {
                    foreach (var column in columns)
                    {
                        if (column.IsPurposeSuggested)
                            continue;
                        if (column.Type.ItemType.IsBool)
                            column.SuggestedPurpose = ColumnPurpose.NumericFeature;
                    }
                }
            }

            public sealed class TextArraysAreText : IPurposeInferenceExpert
            {
                public void Apply(IChannel ch, IntermediateColumn[] columns)
                {
                    foreach (var column in columns)
                    {
                        if (column.IsPurposeSuggested)
                            continue;
                        if (column.Type.IsVector && column.Type.ItemType.IsText)
                            column.SuggestedPurpose = ColumnPurpose.TextFeature;
                    }
                }
            }

            public sealed class IgnoreEverythingElse : IPurposeInferenceExpert
            {
                public void Apply(IChannel ch, IntermediateColumn[] columns)
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
            // If there's no label column yet, first single-value numeric column is a label.
            yield return new Experts.FirstNumericOrBooleanIsLabel();
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
        /// <param name="env">The host environment to use.</param>
        /// <param name="data">The data to use for inference.</param>
        /// <param name="columnIndices">Indices of columns that we're interested in.</param>
        /// <param name="args">Additional arguments to inference.</param>
        /// <param name="dataRoles">(Optional) User defined Role mappings for data.</param>
        /// <returns>The result includes the array of auto-detected column purposes.</returns>
        public static InferenceResult InferPurposes(IHostEnvironment env, IDataView data, IEnumerable<int> columnIndices, Arguments args,
            RoleMappedData dataRoles = null)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("InferPurposes");
            host.CheckValue(data, nameof(data));
            host.CheckValue(columnIndices, nameof(columnIndices));

            InferenceResult result;
            using (var ch = host.Start("InferPurposes"))
            {
                var takenData = data.Take(args.MaxRowsToRead);
                var cols = columnIndices.Select(x => new IntermediateColumn(takenData, x)).ToList();
                data = takenData;

                if (dataRoles != null)
                {
                    var items = dataRoles.Schema.GetColumnRoles();
                    foreach (var item in items)
                    {
                        Enum.TryParse(item.Key.Value, out ColumnPurpose purpose);
                        var col = cols.Find(x => x.ColumnName == item.Value.Name);
                        col.SuggestedPurpose = purpose;
                    }
                }

                foreach (var expert in GetExperts())
                {
                    using (var expertChannel = host.Start(expert.GetType().ToString()))
                    {
                        expert.Apply(expertChannel, cols.ToArray());
                    }
                }

                ch.Check(cols.All(x => x.IsPurposeSuggested), "Purpose inference must be conclusive");

                result = new InferenceResult(cols.Select(x => x.GetColumn()).ToArray());

                ch.Info("Automatic purpose inference complete");
            }
            return result;
        }
    }
}
