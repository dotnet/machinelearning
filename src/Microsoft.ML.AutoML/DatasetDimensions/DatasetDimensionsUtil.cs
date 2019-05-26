// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal static class DatasetDimensionsUtil
    {
        public static ColumnDimensions CalcStringColumnDimensions(IDataView data, DataViewSchema.Column column)
        {
            var numMissingValues = 0;
            var summaryStats = new SummaryStatistics();

            var columnValues = GetStringColumnValues(data, column);
            foreach (var columnValue in columnValues)
            {
                // Count empty strings as missing values
                if (string.IsNullOrEmpty(columnValue))
                {
                    numMissingValues++;
                }

                // Calculate summary stats by # of spaces string contains
                var numSpaces = columnValue.Count(c => c == ' ');
                summaryStats.Add(numSpaces);
            }

            return new ColumnDimensions(columnValues.Distinct().Count(), numMissingValues, summaryStats);
        }

        public static ColumnDimensions CalcNumericColumnDimensions(IDataView data, DataViewSchema.Column column)
        {
            var numMissingValues = 0;
            var summaryStats = new SummaryStatistics();

            var columnValues = GetColumnValues<float>(data, column);
            foreach (var columnValue in columnValues)
            {
                if (float.IsNaN(columnValue))
                {
                    numMissingValues++;
                }
                else // Note: Do not add NaNs to summary stats -- makes column mean / other stats NaN as well
                {
                    summaryStats.Add(columnValue);
                }
            }

            return new ColumnDimensions(columnValues.Distinct().Count(), numMissingValues, summaryStats);
        }

        public static ColumnDimensions CalcNumericVectorColumnDimensions(IDataView data, DataViewSchema.Column column)
        {
            var numMissingValues = 0;

            var columnVectors = GetColumnValues<VBuffer<float>>(data, column);
            foreach (var columnVector in columnVectors)
            {
                if (VBufferUtils.HasNaNs(columnVector))
                {
                    numMissingValues++;
                }
            }

            // Note: In future, potentially calculate per-slot stats for vector columns.
            return new ColumnDimensions(null, numMissingValues, null);
        }

        public static ulong CountRows(IDataView data, ulong maxRows)
        {
            var cursor = data.GetRowCursor(new[] { data.Schema[0] });
            ulong rowCount = 0;
            while (cursor.MoveNext())
            {
                if (++rowCount == maxRows)
                {
                    break;
                }
            }
            return rowCount;
        }

        public static bool IsDataViewEmpty(IDataView data)
        {
            return CountRows(data, 1) == 0;
        }
        
        private static IEnumerable<T> GetColumnValues<T>(IDataView data, DataViewSchema.Column column)
        {
            var values = new List<T>();
            using (var cursor = data.GetRowCursor(new[] { column }))
            {
                var getter = cursor.GetGetter<T>(column);
                while (cursor.MoveNext())
                {
                    var value = default(T);
                    getter(ref value);
                    values.Add(value);
                }
            }
            return values;
        }
        
        private static IEnumerable<string> GetStringColumnValues(IDataView data, DataViewSchema.Column column)
        {
            return GetColumnValues<ReadOnlyMemory<char>>(data, column)
                .Select(x => x.ToString());
        }
    }
}
