// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime.Data;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// This class incapsulates logic for grouping together the inferred columns of the text file based on their type
    /// and purpose, and generating column names.
    /// </summary>
    public static class ColumnGroupingInference
    {
        /// <summary>
        /// This is effectively a merger of <see cref="PurposeInference.Column"/> and a <see cref="ColumnTypeInference.Column"/>
        /// with support for vector-value columns.
        /// </summary>
        public struct GroupingColumn
        {
            public string SuggestedName;

            [JsonConverter(typeof(StringEnumConverter))]
            public DataKind ItemKind;

            [JsonConverter(typeof(StringEnumConverter))]
            public ColumnPurpose Purpose;
            public string ColumnRangeSelector;

            public GroupingColumn(string name, DataKind kind, ColumnPurpose purpose, string rangeSelector)
            {
                SuggestedName = name;
                ItemKind = kind;
                Purpose = purpose;
                ColumnRangeSelector = rangeSelector;
            }
        }

        public struct InferenceResult
        {
            public readonly GroupingColumn[] Columns;

            public InferenceResult(GroupingColumn[] columns)
            {
                Columns = columns;
            }
        }

        /// <summary>
        /// Group together the single-valued columns with the same type and purpose and generate column names.
        /// </summary>
        /// <param name="env">The host environment to use.</param>
        /// <param name="hasHeader">Whether the original file had a header. 
        /// If yes, the <see cref="ColumnTypeInference.Column.SuggestedName"/> fields are used to generate the column
        /// names, otherwise they are ignored.</param>
        /// <param name="types">The (detected) column types.</param>
        /// <param name="purposes">The (detected) column purposes. Must be parallel to <paramref name="types"/>.</param>
        /// <returns>The struct containing an array of grouped columns specifications.</returns>
        public static InferenceResult InferGroupingAndNames(IHostEnvironment env, bool hasHeader, ColumnTypeInference.Column[] types, PurposeInference.Column[] purposes)
        {
            Contracts.CheckValue(env, nameof(env));
            Contracts.CheckNonEmpty(types, nameof(types));
            Contracts.CheckNonEmpty(purposes, nameof(purposes));
            Contracts.Check(types.Length == purposes.Length);

            var result = new List<GroupingColumn>();
            using (var ch = env.Register("InferGrouping").Start("InferGrouping"))
            {
                var tuples = types.Zip(purposes, Tuple.Create).ToList();
                var grouped =
                    from t in tuples
                    group t by
                        new
                        {
                            t.Item1.ItemType,
                            t.Item2.Purpose,
                            purposeGroupId = GetPurposeGroupId(t.Item1.ColumnIndex, t.Item2.Purpose)
                        }
                        into g
                        select g;

                foreach (var g in grouped)
                {
                    string name = (hasHeader && g.Count() == 1)
                        ? g.First().Item1.SuggestedName
                        : GetName(g.Key.ItemType.RawKind, g.Key.Purpose, result);

                    string range = GetRange(g.Select(t => t.Item1.ColumnIndex).ToArray());
                    if (g.Count() > 1)
                        ch.Info("Grouped data from columns '{0}' into vector column '{1}'.", range, name);
                    result.Add(new GroupingColumn(name, g.Key.ItemType.RawKind, g.Key.Purpose, range));
                }

                ch.Done();
            }

            return new InferenceResult(result.ToArray());
        }

        private static int GetPurposeGroupId(int columnIndex, ColumnPurpose purpose)
        {
            if (purpose == ColumnPurpose.CategoricalFeature || purpose == ColumnPurpose.TextFeature)
                return columnIndex;
            return 0;
        }

        private static string GetName(DataKind itemKind, ColumnPurpose purpose, List<GroupingColumn> previousColumns)
        {
            string prefix = GetPurposeName(purpose, itemKind);
            int i = 0;
            string name = prefix;
            while (previousColumns.Any(x => x.SuggestedName == name))
            {
                i++;
                name = string.Format("{0}{1:00}", prefix, i);
            }

            return name;
        }

        private static string GetPurposeName(ColumnPurpose purpose, DataKind itemKind)
        {
            switch (purpose)
            {
            case ColumnPurpose.NumericFeature:
                if (itemKind == DataKind.Bool)
                    return "BooleanFeatures";
                else
                    return "Features";
            case ColumnPurpose.CategoricalFeature:
                return "Cat";
            default:
                return Enum.GetName(typeof(ColumnPurpose), purpose);
            }
        }

        /// <summary>
        /// Generates a range selector from the array of indices.
        /// </summary>
        private static string GetRange(int[] indices)
        {
            var sb = new StringBuilder();
            var sorted = indices.OrderBy(x => x).ToArray();

            sb.Append(indices[0]);
            var prev = sorted[0];
            var start = sorted[0];
            for (int i = 1; i < sorted.Length; i++)
            {
                if (sorted[i] > prev + 1)
                {
                    if (prev > start)
                        sb.AppendFormat("-{0}", prev);
                    start = sorted[i];
                    sb.AppendFormat(",{0}", start);
                }
                prev = sorted[i];
            }
            if (prev > start)
            {
                sb.AppendFormat("-{0}", prev);
            }

            return sb.ToString();
        }

        /// <summary>
        /// Generates individual indices from a string range selector.
        /// </summary>
        public static int[] GetRange(string indicies)
        {
            string[] rangesLocal = indicies.Split(',');
            var indices = new List<int>();
            foreach (string range in rangesLocal)
            {
                if (range.Contains('-'))
                {
                    string[] startEnd = range.Split('-');
                    Contracts.Assert(startEnd.Length == 2);
                    int start = int.Parse(startEnd[0]);
                    int end = int.Parse(startEnd[1]);
                    Contracts.Assert(start >= 0);
                    Contracts.Assert(start <= end);
                    for (int i = start; i <= end; i++)
                        indices.Add(i);
                }
                else
                {
                    int index = int.Parse(range);
                    Contracts.Assert(index >= 0);
                    indices.Add(index);
                }
            }

            indices.Sort((a, b) => a - b);
            return indices.ToArray();
        }

        public static TextLoader.Column[] GenerateLoaderColumns(GroupingColumn[] columns)
        {
            var loaderColumns = new List<TextLoader.Column>();
            foreach (var col in columns)
            {
                var loaderColumn = TextLoader.Column.Parse(string.Format("{0}:{1}:{2}", col.SuggestedName, col.ItemKind, col.ColumnRangeSelector));
                if (loaderColumn != null && loaderColumn.IsValid())
                    loaderColumns.Add(loaderColumn);
            }
            return loaderColumns.ToArray();
        }

    }
}
