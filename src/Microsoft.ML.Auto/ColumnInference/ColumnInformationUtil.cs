// Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

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

            if (columnName == columnInfo.NameColumn)
            {
                return ColumnPurpose.Name;
            }

            if (columnName == columnInfo.GroupIdColumn)
            {
                return ColumnPurpose.Group;
            }

            if (columnName == columnInfo.WeightColumn)
            {
                return ColumnPurpose.Weight;
            }

            if (columnInfo.CategoricalColumns?.Contains(columnName) == true)
            {
                return ColumnPurpose.CategoricalFeature;
            }

            if (columnInfo.NumericColumns?.Contains(columnName) == true)
            {
                return ColumnPurpose.NumericFeature;
            }

            if (columnInfo.TextColumns?.Contains(columnName) == true)
            {
                return ColumnPurpose.TextFeature;
            }

            return null;
        }

        internal static ColumnInformation BuildColumnInfo(IEnumerable<(string name, ColumnPurpose purpose)> columnPurposes)
        {
            var columnInfo = new ColumnInformation();

            var categoricalColumns = new List<string>();
            var numericColumns = new List<string>();
            var textColumns = new List<string>();
            var ignoredColumns = new List<string>();
            columnInfo.CategoricalColumns = categoricalColumns;
            columnInfo.NumericColumns = numericColumns;
            columnInfo.TextColumns = textColumns;
            columnInfo.IgnoredColumns = ignoredColumns;

            foreach (var column in columnPurposes)
            {
                switch (column.purpose)
                {
                    case ColumnPurpose.CategoricalFeature:
                        categoricalColumns.Add(column.name);
                        break;
                    case ColumnPurpose.Group:
                        columnInfo.GroupIdColumn = column.name;
                        break;
                    case ColumnPurpose.Ignore:
                        ignoredColumns.Add(column.name);
                        break;
                    case ColumnPurpose.Label:
                        columnInfo.LabelColumn = column.name;
                        break;
                    case ColumnPurpose.Name:
                        columnInfo.NameColumn = column.name;
                        break;
                    case ColumnPurpose.NumericFeature:
                        numericColumns.Add(column.name);
                        break;
                    case ColumnPurpose.TextFeature:
                        textColumns.Add(column.name);
                        break;
                    case ColumnPurpose.Weight:
                        columnInfo.WeightColumn = column.name;
                        break;
                }
            }

            return columnInfo;
        }
    }
}
