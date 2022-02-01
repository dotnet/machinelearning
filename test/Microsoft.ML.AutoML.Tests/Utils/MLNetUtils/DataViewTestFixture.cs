// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Test
{
    static class DataViewTestFixture
    {
        public static IDataView BuildDummyDataView(IEnumerable<DatasetColumnInfo> columns, bool createDummyRow = true)
        {
            return BuildDummyDataView(columns.Select(c => (c.Name, c.Type)), createDummyRow);
        }

        public static IDataView BuildDummyDataView(DataViewSchema schema, bool createDummyRow = true)
        {
            return BuildDummyDataView(schema.Select(c => (c.Name, c.Type)), createDummyRow);
        }

        public static IDataView BuildDummyDataView(IEnumerable<(string name, DataViewType type)> columns, bool createDummyRow = true)
        {
            var dataBuilder = new ArrayDataViewBuilder(new MLContext(1));

            foreach (var column in columns)
            {
                if (column.type == NumberDataViewType.Single)
                {
                    dataBuilder.AddColumn(column.name, NumberDataViewType.Single, createDummyRow ? new float[] { 0 } : new float[] { });
                }
                if (column.type == NumberDataViewType.Double)
                {
                    dataBuilder.AddColumn(column.name, NumberDataViewType.Double, createDummyRow ? new double[] { 0 } : new double[] { });
                }
                if (column.type == NumberDataViewType.UInt64)
                {
                    dataBuilder.AddColumn(column.name, NumberDataViewType.UInt64, createDummyRow ? new System.UInt64[] { 0 } : new System.UInt64[] { });
                }
                else if (column.type == BooleanDataViewType.Instance)
                {
                    dataBuilder.AddColumn(column.name, BooleanDataViewType.Instance, createDummyRow ? new bool[] { false } : new bool[] { });
                }
                else if (column.type == TextDataViewType.Instance)
                {
                    dataBuilder.AddColumn(column.name, createDummyRow ? new string[] { "a" } : new string[] { });
                }
                else if (column.type.IsVector() && column.type.GetItemType() == NumberDataViewType.Single)
                {
                    dataBuilder.AddColumn(
                        column.name,
                        Util.GetKeyValueGetter(createDummyRow ? new string[] { "1", "2" } : new string[] { }),
                        NumberDataViewType.Single,
                        createDummyRow ? new float[] { 0, 0 } : new float[] { });
                }
            }

            return dataBuilder.GetDataView();
        }
    }
}
