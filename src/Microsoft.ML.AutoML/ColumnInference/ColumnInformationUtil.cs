// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.AutoML
{
    internal static class ColumnInformationUtil
    {
        internal static ColumnPurpose? GetColumnPurpose(this ColumnInformation columnInfo, string columnName)
        {
            if (columnName == columnInfo.LabelColumnName)
            {
                return ColumnPurpose.Label;
            }

            if (columnName == columnInfo.ExampleWeightColumnName)
            {
                return ColumnPurpose.Weight;
            }

            if (columnName == columnInfo.GroupIdColumnName)
            {
                return ColumnPurpose.GroupId;
            }

            if (columnName == columnInfo.SamplingKeyColumnName)
            {
                return ColumnPurpose.SamplingKey;
            }

            if (columnInfo.CategoricalColumnNames.Contains(columnName))
            {
                return ColumnPurpose.CategoricalFeature;
            }

            if (columnInfo.NumericColumnNames.Contains(columnName))
            {
                return ColumnPurpose.NumericFeature;
            }

            if (columnInfo.TextColumnNames.Contains(columnName))
            {
                return ColumnPurpose.TextFeature;
            }

            if (columnInfo.IgnoredColumnNames.Contains(columnName))
            {
                return ColumnPurpose.Ignore;
            }

            if (columnName == columnInfo.UserIdColumnName)
            {
                return ColumnPurpose.UserId;
            }

            if (columnName == columnInfo.ItemIdColumnName)
            {
                return ColumnPurpose.ItemId;
            }

            if (columnInfo.ImagePathColumnNames.Contains(columnName))
            {
                return ColumnPurpose.ImagePath;
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
                        columnInfo.LabelColumnName = column.name;
                        break;
                    case ColumnPurpose.Weight:
                        columnInfo.ExampleWeightColumnName = column.name;
                        break;
                    case ColumnPurpose.SamplingKey:
                        columnInfo.SamplingKeyColumnName = column.name;
                        break;
                    case ColumnPurpose.CategoricalFeature:
                        columnInfo.CategoricalColumnNames.Add(column.name);
                        break;
                    case ColumnPurpose.Ignore:
                        columnInfo.IgnoredColumnNames.Add(column.name);
                        break;
                    case ColumnPurpose.NumericFeature:
                        columnInfo.NumericColumnNames.Add(column.name);
                        break;
                    case ColumnPurpose.UserId:
                        columnInfo.UserIdColumnName = column.name;
                        break;
                    case ColumnPurpose.ItemId:
                        columnInfo.ItemIdColumnName = column.name;
                        break;
                    case ColumnPurpose.GroupId:
                        columnInfo.GroupIdColumnName = column.name;
                        break;
                    case ColumnPurpose.TextFeature:
                        columnInfo.TextColumnNames.Add(column.name);
                        break;
                    case ColumnPurpose.ImagePath:
                        columnInfo.ImagePathColumnNames.Add(column.name);
                        break;
                }
            }

            return columnInfo;
        }

        public static ColumnInformation BuildColumnInfo(IEnumerable<DatasetColumnInfo> columns)
        {
            return BuildColumnInfo(columns.Select(c => (c.Name, c.Purpose)));
        }

        /// <summary>
        /// Get all column names that are in <paramref name="columnInformation"/>.
        /// </summary>
        /// <param name="columnInformation">Column information.</param>
        public static IEnumerable<string> GetColumnNames(ColumnInformation columnInformation)
        {
            var columnNames = new List<string>();
            AddStringToListIfNotNull(columnNames, columnInformation.LabelColumnName);
            AddStringToListIfNotNull(columnNames, columnInformation.UserIdColumnName);
            AddStringToListIfNotNull(columnNames, columnInformation.ItemIdColumnName);
            AddStringToListIfNotNull(columnNames, columnInformation.GroupIdColumnName);
            AddStringToListIfNotNull(columnNames, columnInformation.ExampleWeightColumnName);
            AddStringToListIfNotNull(columnNames, columnInformation.SamplingKeyColumnName);
            AddStringsToListIfNotNull(columnNames, columnInformation.CategoricalColumnNames);
            AddStringsToListIfNotNull(columnNames, columnInformation.IgnoredColumnNames);
            AddStringsToListIfNotNull(columnNames, columnInformation.NumericColumnNames);
            AddStringsToListIfNotNull(columnNames, columnInformation.TextColumnNames);
            AddStringsToListIfNotNull(columnNames, columnInformation.ImagePathColumnNames);
            return columnNames;
        }

        public static IDictionary<ColumnPurpose, int> CountColumnsByPurpose(ColumnInformation columnInformation)
        {
            var result = new Dictionary<ColumnPurpose, int>();
            var columnNames = GetColumnNames(columnInformation);
            foreach (var columnName in columnNames)
            {
                var purpose = columnInformation.GetColumnPurpose(columnName);
                if (purpose == null)
                {
                    continue;
                }

                result.TryGetValue(purpose.Value, out int count);
                result[purpose.Value] = ++count;
            }
            return result;
        }

        private static void AddStringsToListIfNotNull(List<string> list, IEnumerable<string> strings)
        {
            foreach (var str in strings)
            {
                AddStringToListIfNotNull(list, str);
            }
        }

        private static void AddStringToListIfNotNull(List<string> list, string str)
        {
            if (str != null)
            {
                list.Add(str);
            }
        }
    }
}
