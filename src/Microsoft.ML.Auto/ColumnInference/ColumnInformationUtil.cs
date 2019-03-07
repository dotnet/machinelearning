// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;

namespace Microsoft.ML.Auto
{
    internal static class ColumnInformationUtil
    {
        internal static ColumnPurpose? GetColumnPurpose(this ColumnInformation columnInfo, string columnName)
        {
            if (columnName == columnInfo.LabelColumn)
            {
                return ColumnPurpose.Label;
            }

            if (columnName == columnInfo.WeightColumn)
            {
                return ColumnPurpose.Weight;
            }

            if (columnName == columnInfo.SamplingKeyColumn)
            {
                return ColumnPurpose.SamplingKey;
            }

            if (columnInfo.CategoricalColumns.Contains(columnName))
            {
                return ColumnPurpose.CategoricalFeature;
            }

            if (columnInfo.NumericColumns.Contains(columnName))
            {
                return ColumnPurpose.NumericFeature;
            }

            if (columnInfo.TextColumns.Contains(columnName))
            {
                return ColumnPurpose.TextFeature;
            }

            if (columnInfo.IgnoredColumns.Contains(columnName))
            {
                return ColumnPurpose.Ignore;
            }

            return null;
        }

        internal static ColumnInformation BuildColumnInfo(IEnumerable<(string name, ColumnPurpose purpose)> columnPurposes)
        {
            var columnInfo = new ColumnInformation();

            foreach (var column in columnPurposes)
            {
                switch (column.purpose)
                {
                    case ColumnPurpose.Label:
                        columnInfo.LabelColumn = column.name;
                        break;
                    case ColumnPurpose.Weight:
                        columnInfo.WeightColumn = column.name;
                        break;
                    case ColumnPurpose.SamplingKey:
                        columnInfo.SamplingKeyColumn = column.name;
                        break;
                    case ColumnPurpose.CategoricalFeature:
                        columnInfo.CategoricalColumns.Add(column.name);
                        break;
                    case ColumnPurpose.Ignore:
                        columnInfo.IgnoredColumns.Add(column.name);
                        break;
                    case ColumnPurpose.NumericFeature:
                        columnInfo.NumericColumns.Add(column.name);
                        break;
                    case ColumnPurpose.TextFeature:
                        columnInfo.TextColumns.Add(column.name);
                        break;
                }
            }

            return columnInfo;
        }

        public static ColumnInformation BuildColumnInfo(IEnumerable<(string, DataViewType, ColumnPurpose, ColumnDimensions)> columns)
        {
            return BuildColumnInfo(columns.Select(c => (c.Item1, c.Item3)));
        }
    }
}
