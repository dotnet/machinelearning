// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal class DatasetColumnInfo
    {
        public readonly string Name;
        public readonly DataViewType Type;
        public readonly ColumnPurpose Purpose;
        public readonly ColumnDimensions Dimensions;

        public DatasetColumnInfo(string name, DataViewType type, ColumnPurpose purpose, ColumnDimensions dimensions)
        {
            Name = name;
            Type = type;
            Purpose = purpose;
            Dimensions = dimensions;
        }
    }

    internal static class DatasetColumnInfoUtil
    {
        public static DatasetColumnInfo[] GetDatasetColumnInfo(MLContext context, IDataView data, ColumnInformation columnInfo)
        {
            var purposes = PurposeInference.InferPurposes(context, data, columnInfo);
            var colDimensions = DatasetDimensionsApi.CalcColumnDimensions(context, data, purposes);
            var cols = new DatasetColumnInfo[data.Schema.Count];
            for (var i = 0; i < cols.Length; i++)
            {
                var schemaCol = data.Schema[i];
                var col = new DatasetColumnInfo(schemaCol.Name, schemaCol.Type, purposes[i].Purpose, colDimensions[i]);
                cols[i] = col;
            }
            return cols;
        }
    }
}
