// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Xunit;

namespace Microsoft.Data.Analysis.Tests
{
    public partial class DataFrameIDataViewTests
    {
        [Fact]
        public void TestIDataView()
        {
            IDataView dataView = DataFrameTests.MakeDataFrameWithAllColumnTypes(10, withNulls: false);

            DataDebuggerPreview preview = dataView.Preview();
            Assert.Equal(10, preview.RowView.Length);
            Assert.Equal(15, preview.ColumnView.Length);

            Assert.Equal("Byte", preview.ColumnView[0].Column.Name);
            Assert.Equal((byte)0, preview.ColumnView[0].Values[0]);
            Assert.Equal((byte)1, preview.ColumnView[0].Values[1]);

            Assert.Equal("Decimal", preview.ColumnView[1].Column.Name);
            Assert.Equal((double)0, preview.ColumnView[1].Values[0]);
            Assert.Equal((double)1, preview.ColumnView[1].Values[1]);

            Assert.Equal("Double", preview.ColumnView[2].Column.Name);
            Assert.Equal((double)0, preview.ColumnView[2].Values[0]);
            Assert.Equal((double)1, preview.ColumnView[2].Values[1]);

            Assert.Equal("Float", preview.ColumnView[3].Column.Name);
            Assert.Equal((float)0, preview.ColumnView[3].Values[0]);
            Assert.Equal((float)1, preview.ColumnView[3].Values[1]);

            Assert.Equal("Int", preview.ColumnView[4].Column.Name);
            Assert.Equal((int)0, preview.ColumnView[4].Values[0]);
            Assert.Equal((int)1, preview.ColumnView[4].Values[1]);

            Assert.Equal("Long", preview.ColumnView[5].Column.Name);
            Assert.Equal((long)0, preview.ColumnView[5].Values[0]);
            Assert.Equal((long)1, preview.ColumnView[5].Values[1]);

            Assert.Equal("Sbyte", preview.ColumnView[6].Column.Name);
            Assert.Equal((sbyte)0, preview.ColumnView[6].Values[0]);
            Assert.Equal((sbyte)1, preview.ColumnView[6].Values[1]);

            Assert.Equal("Short", preview.ColumnView[7].Column.Name);
            Assert.Equal((short)0, preview.ColumnView[7].Values[0]);
            Assert.Equal((short)1, preview.ColumnView[7].Values[1]);

            Assert.Equal("Uint", preview.ColumnView[8].Column.Name);
            Assert.Equal((uint)0, preview.ColumnView[8].Values[0]);
            Assert.Equal((uint)1, preview.ColumnView[8].Values[1]);

            Assert.Equal("Ulong", preview.ColumnView[9].Column.Name);
            Assert.Equal((ulong)0, preview.ColumnView[9].Values[0]);
            Assert.Equal((ulong)1, preview.ColumnView[9].Values[1]);

            Assert.Equal("Ushort", preview.ColumnView[10].Column.Name);
            Assert.Equal((ushort)0, preview.ColumnView[10].Values[0]);
            Assert.Equal((ushort)1, preview.ColumnView[10].Values[1]);

            Assert.Equal("String", preview.ColumnView[11].Column.Name);
            Assert.Equal("0".ToString(), preview.ColumnView[11].Values[0].ToString());
            Assert.Equal("1".ToString(), preview.ColumnView[11].Values[1].ToString());

            Assert.Equal("Char", preview.ColumnView[12].Column.Name);
            Assert.Equal((ushort)65, preview.ColumnView[12].Values[0]);
            Assert.Equal((ushort)66, preview.ColumnView[12].Values[1]);

            Assert.Equal("Bool", preview.ColumnView[13].Column.Name);
            Assert.Equal(true, preview.ColumnView[13].Values[0]);
            Assert.Equal(false, preview.ColumnView[13].Values[1]);

            Assert.Equal("ArrowString", preview.ColumnView[14].Column.Name);
            Assert.Equal("foo".ToString(), preview.ColumnView[14].Values[0].ToString());
            Assert.Equal("foo".ToString(), preview.ColumnView[14].Values[1].ToString());
        }

        [Fact]
        public void TestIDataViewSchemaInvalidate()
        {
            DataFrame df = DataFrameTests.MakeDataFrameWithAllMutableColumnTypes(10, withNulls: false);

            IDataView dataView = df;

            DataViewSchema schema = dataView.Schema;
            Assert.Equal(14, schema.Count);

            df.Columns.Remove("Bool");
            schema = dataView.Schema;
            Assert.Equal(13, schema.Count);

            DataFrameColumn boolColumn = new PrimitiveDataFrameColumn<bool>("Bool", Enumerable.Range(0, (int)df.Rows.Count).Select(x => x % 2 == 1));
            df.Columns.Insert(0, boolColumn);
            schema = dataView.Schema;
            Assert.Equal(14, schema.Count);
            Assert.Equal("Bool", schema[0].Name);

            DataFrameColumn boolClone = boolColumn.Clone();
            boolClone.SetName("BoolClone");
            df.Columns[1] = boolClone;
            schema = dataView.Schema;
            Assert.Equal("BoolClone", schema[1].Name);
        }

        [Fact]
        public void TestIDataViewWithNulls()
        {
            int length = 10;
            IDataView dataView = DataFrameTests.MakeDataFrameWithAllColumnTypes(length, withNulls: true);

            DataDebuggerPreview preview = dataView.Preview();
            Assert.Equal(length, preview.RowView.Length);
            Assert.Equal(15, preview.ColumnView.Length);

            Assert.Equal("Byte", preview.ColumnView[0].Column.Name);
            Assert.Equal((byte)0, preview.ColumnView[0].Values[0]);
            Assert.Equal((byte)1, preview.ColumnView[0].Values[1]);
            Assert.Equal((byte)4, preview.ColumnView[0].Values[4]);
            Assert.Equal((byte)0, preview.ColumnView[0].Values[5]); // null row
            Assert.Equal((byte)6, preview.ColumnView[0].Values[6]);

            Assert.Equal("Decimal", preview.ColumnView[1].Column.Name);
            Assert.Equal((double)0, preview.ColumnView[1].Values[0]);
            Assert.Equal((double)1, preview.ColumnView[1].Values[1]);
            Assert.Equal((double)4, preview.ColumnView[1].Values[4]);
            Assert.Equal(double.NaN, preview.ColumnView[1].Values[5]); // null row
            Assert.Equal((double)6, preview.ColumnView[1].Values[6]);

            Assert.Equal("Double", preview.ColumnView[2].Column.Name);
            Assert.Equal((double)0, preview.ColumnView[2].Values[0]);
            Assert.Equal((double)1, preview.ColumnView[2].Values[1]);
            Assert.Equal((double)4, preview.ColumnView[2].Values[4]);
            Assert.Equal(double.NaN, preview.ColumnView[2].Values[5]); // null row
            Assert.Equal((double)6, preview.ColumnView[2].Values[6]);

            Assert.Equal("Float", preview.ColumnView[3].Column.Name);
            Assert.Equal((float)0, preview.ColumnView[3].Values[0]);
            Assert.Equal((float)1, preview.ColumnView[3].Values[1]);
            Assert.Equal((float)4, preview.ColumnView[3].Values[4]);
            Assert.Equal(float.NaN, preview.ColumnView[3].Values[5]); // null row
            Assert.Equal((float)6, preview.ColumnView[3].Values[6]);

            Assert.Equal("Int", preview.ColumnView[4].Column.Name);
            Assert.Equal((int)0, preview.ColumnView[4].Values[0]);
            Assert.Equal((int)1, preview.ColumnView[4].Values[1]);
            Assert.Equal((int)4, preview.ColumnView[4].Values[4]);
            Assert.Equal((int)0, preview.ColumnView[4].Values[5]); // null row
            Assert.Equal((int)6, preview.ColumnView[4].Values[6]);

            Assert.Equal("Long", preview.ColumnView[5].Column.Name);
            Assert.Equal((long)0, preview.ColumnView[5].Values[0]);
            Assert.Equal((long)1, preview.ColumnView[5].Values[1]);
            Assert.Equal((long)4, preview.ColumnView[5].Values[4]);
            Assert.Equal((long)0, preview.ColumnView[5].Values[5]); // null row
            Assert.Equal((long)6, preview.ColumnView[5].Values[6]);

            Assert.Equal("Sbyte", preview.ColumnView[6].Column.Name);
            Assert.Equal((sbyte)0, preview.ColumnView[6].Values[0]);
            Assert.Equal((sbyte)1, preview.ColumnView[6].Values[1]);
            Assert.Equal((sbyte)4, preview.ColumnView[6].Values[4]);
            Assert.Equal((sbyte)0, preview.ColumnView[6].Values[5]); // null row
            Assert.Equal((sbyte)6, preview.ColumnView[6].Values[6]);

            Assert.Equal("Short", preview.ColumnView[7].Column.Name);
            Assert.Equal((short)0, preview.ColumnView[7].Values[0]);
            Assert.Equal((short)1, preview.ColumnView[7].Values[1]);
            Assert.Equal((short)4, preview.ColumnView[7].Values[4]);
            Assert.Equal((short)0, preview.ColumnView[7].Values[5]); // null row
            Assert.Equal((short)6, preview.ColumnView[7].Values[6]);

            Assert.Equal("Uint", preview.ColumnView[8].Column.Name);
            Assert.Equal((uint)0, preview.ColumnView[8].Values[0]);
            Assert.Equal((uint)1, preview.ColumnView[8].Values[1]);
            Assert.Equal((uint)4, preview.ColumnView[8].Values[4]);
            Assert.Equal((uint)0, preview.ColumnView[8].Values[5]); // null row
            Assert.Equal((uint)6, preview.ColumnView[8].Values[6]);

            Assert.Equal("Ulong", preview.ColumnView[9].Column.Name);
            Assert.Equal((ulong)0, preview.ColumnView[9].Values[0]);
            Assert.Equal((ulong)1, preview.ColumnView[9].Values[1]);
            Assert.Equal((ulong)4, preview.ColumnView[9].Values[4]);
            Assert.Equal((ulong)0, preview.ColumnView[9].Values[5]); // null row
            Assert.Equal((ulong)6, preview.ColumnView[9].Values[6]);

            Assert.Equal("Ushort", preview.ColumnView[10].Column.Name);
            Assert.Equal((ushort)0, preview.ColumnView[10].Values[0]);
            Assert.Equal((ushort)1, preview.ColumnView[10].Values[1]);
            Assert.Equal((ushort)4, preview.ColumnView[10].Values[4]);
            Assert.Equal((ushort)0, preview.ColumnView[10].Values[5]); // null row
            Assert.Equal((ushort)6, preview.ColumnView[10].Values[6]);

            Assert.Equal("String", preview.ColumnView[11].Column.Name);
            Assert.Equal("0", preview.ColumnView[11].Values[0].ToString());
            Assert.Equal("1", preview.ColumnView[11].Values[1].ToString());
            Assert.Equal("4", preview.ColumnView[11].Values[4].ToString());
            Assert.Equal("", preview.ColumnView[11].Values[5].ToString()); // null row
            Assert.Equal("6", preview.ColumnView[11].Values[6].ToString());

            Assert.Equal("Char", preview.ColumnView[12].Column.Name);
            Assert.Equal((ushort)65, preview.ColumnView[12].Values[0]);
            Assert.Equal((ushort)66, preview.ColumnView[12].Values[1]);
            Assert.Equal((ushort)69, preview.ColumnView[12].Values[4]);
            Assert.Equal((ushort)0, preview.ColumnView[12].Values[5]); // null row
            Assert.Equal((ushort)71, preview.ColumnView[12].Values[6]);

            Assert.Equal("Bool", preview.ColumnView[13].Column.Name);
            Assert.Equal(true, preview.ColumnView[13].Values[0]);
            Assert.Equal(false, preview.ColumnView[13].Values[1]);
            Assert.Equal(true, preview.ColumnView[13].Values[4]);
            Assert.Equal(false, preview.ColumnView[13].Values[5]); // null row
            Assert.Equal(true, preview.ColumnView[13].Values[6]);

            Assert.Equal("ArrowString", preview.ColumnView[14].Column.Name);
            Assert.Equal("foo", preview.ColumnView[14].Values[0].ToString());
            Assert.Equal("foo", preview.ColumnView[14].Values[1].ToString());
            Assert.Equal("foo", preview.ColumnView[14].Values[4].ToString());
            Assert.Equal("", preview.ColumnView[14].Values[5].ToString()); // null row
            Assert.Equal("foo", preview.ColumnView[14].Values[6].ToString());
        }

        [Fact]
        public void TestDataFrameFromIDataView()
        {
            DataFrame df = DataFrameTests.MakeDataFrameWithAllColumnTypes(10, withNulls: false);
            df.Columns.Remove("Char"); // Because chars are returned as uint16 by IDataView, so end up comparing CharDataFrameColumn to UInt16DataFrameColumn and fail asserts
            IDataView dfAsIDataView = df;
            DataFrame newDf = dfAsIDataView.ToDataFrame();
            Assert.Equal(dfAsIDataView.GetRowCount(), newDf.Rows.Count);
            Assert.Equal(dfAsIDataView.Schema.Count, newDf.Columns.Count);
            for (int i = 0; i < df.Columns.Count; i++)
            {
                Assert.True(df.Columns[i].ElementwiseEquals(newDf.Columns[i]).All());
            }
        }

        [Fact]
        public void TestDataFrameFromIDataView_SelectColumns()
        {
            DataFrame df = DataFrameTests.MakeDataFrameWithAllColumnTypes(10, withNulls: false);
            IDataView dfAsIDataView = df;
            DataFrame newDf = dfAsIDataView.ToDataFrame("Int", "Double");
            Assert.Equal(dfAsIDataView.GetRowCount(), newDf.Rows.Count);
            Assert.Equal(2, newDf.Columns.Count);
            Assert.True(df.Columns["Int"].ElementwiseEquals(newDf.Columns["Int"]).All());
            Assert.True(df.Columns["Double"].ElementwiseEquals(newDf.Columns["Double"]).All());
        }

        [Theory]
        [InlineData(10, 5)]
        [InlineData(110, 100)]
        [InlineData(110, -1)]
        public void TestDataFrameFromIDataView_SelectRows(int dataFrameSize, int rowSize)
        {
            DataFrame df = DataFrameTests.MakeDataFrameWithAllColumnTypes(dataFrameSize, withNulls: false);
            df.Columns.Remove("Char"); // Because chars are returned as uint16 by DataViewSchema, so end up comparing CharDataFrameColumn to UInt16DataFrameColumn and fail asserts
            df.Columns.Remove("Decimal"); // Because decimal is returned as double by DataViewSchema, so end up comparing DecimalDataFrameColumn to DoubleDataFrameColumn and fail asserts
            IDataView dfAsIDataView = df;
            DataFrame newDf;
            if (rowSize == 100)
            {
                // Test default
                newDf = dfAsIDataView.ToDataFrame();
            }
            else
            {
                newDf = dfAsIDataView.ToDataFrame(rowSize);
            }
            if (rowSize == -1)
            {
                rowSize = dataFrameSize;
            }
            Assert.Equal(rowSize, newDf.Rows.Count);
            Assert.Equal(df.Columns.Count, newDf.Columns.Count);
            for (int i = 0; i < newDf.Columns.Count; i++)
            {
                Assert.Equal(rowSize, newDf.Columns[i].Length);
                Assert.Equal(df.Columns[i].Name, newDf.Columns[i].Name);
            }
            Assert.Equal(dfAsIDataView.Schema.Count, newDf.Columns.Count);
            for (int c = 0; c < df.Columns.Count; c++)
            {
                for (int r = 0; r < rowSize; r++)
                {
                    Assert.Equal(df.Columns[c][r], newDf.Columns[c][r]);
                }
            }
        }

        [Fact]
        public void TestDataFrameFromIDataView_SelectColumnsAndRows()
        {
            DataFrame df = DataFrameTests.MakeDataFrameWithAllColumnTypes(10, withNulls: false);
            IDataView dfAsIDataView = df;
            DataFrame newDf = dfAsIDataView.ToDataFrame(5, "Int", "Double");
            Assert.Equal(5, newDf.Rows.Count);
            for (int i = 0; i < newDf.Columns.Count; i++)
            {
                Assert.Equal(5, newDf.Columns[i].Length);
            }
            Assert.Equal(2, newDf.Columns.Count);
            for (int r = 0; r < 5; r++)
            {
                Assert.Equal(df.Columns["Int"][r], newDf.Columns["Int"][r]);
                Assert.Equal(df.Columns["Double"][r], newDf.Columns["Double"][r]);
            }
        }

        private class InputData
        {
            public string Name { get; set; }
            public bool FilterNext { get; set; }
            public float Value { get; set; }
        }

        private IDataView GetASampleIDataView()
        {
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var enumerableOfData = new[]
            {
                new InputData() { Name = "Joey", FilterNext = false, Value = 1.0f },
                new InputData() { Name = "Chandler", FilterNext = false , Value = 2.0f},
                new InputData() { Name = "Ross", FilterNext = false , Value = 3.0f},
                new InputData() { Name = "Monica", FilterNext = true , Value = 4.0f},
                new InputData() { Name = "Rachel", FilterNext = true , Value = 5.0f},
                new InputData() { Name = "Phoebe", FilterNext = false , Value = 6.0f},
            };

            IDataView data = mlContext.Data.LoadFromEnumerable(enumerableOfData);
            return data;
        }

        private void VerifyDataFrameColumnAndDataViewColumnValues<T>(string columnName, IDataView data, DataFrame df, int maxRows = -1)
        {
            int cc = 0;
            var nameDataViewColumn = data.GetColumn<T>(columnName);
            foreach (var value in nameDataViewColumn)
            {
                if (maxRows != -1 && cc >= maxRows)
                {
                    return;
                }
                Assert.Equal(value, df.Columns[columnName][cc++]);
            }
        }

        [Fact]
        public void TestDataFrameFromIDataView_MLData()
        {
            IDataView data = GetASampleIDataView();
            DataFrame df = data.ToDataFrame();
            Assert.Equal(6, df.Rows.Count);
            Assert.Equal(3, df.Columns.Count);
            foreach (var column in df.Columns)
            {
                Assert.Equal(6, column.Length);
            }

            VerifyDataFrameColumnAndDataViewColumnValues<string>("Name", data, df);
            VerifyDataFrameColumnAndDataViewColumnValues<bool>("FilterNext", data, df);
            VerifyDataFrameColumnAndDataViewColumnValues<float>("Value", data, df);
        }

        [Fact]
        public void TestDataFrameFromIDataView_MLData_SelectColumns()
        {
            IDataView data = GetASampleIDataView();
            DataFrame df = data.ToDataFrame("Name", "Value");
            Assert.Equal(6, df.Rows.Count);
            Assert.Equal(2, df.Columns.Count);
            foreach (var column in df.Columns)
            {
                Assert.Equal(6, column.Length);
            }

            VerifyDataFrameColumnAndDataViewColumnValues<string>("Name", data, df);
            VerifyDataFrameColumnAndDataViewColumnValues<float>("Value", data, df);
        }

        [Theory]
        [InlineData(3)]
        [InlineData(0)]
        public void TestDataFrameFromIDataView_MLData_SelectRows(int maxRows)
        {
            IDataView data = GetASampleIDataView();
            DataFrame df = data.ToDataFrame(maxRows);
            Assert.Equal(maxRows, df.Rows.Count);
            Assert.Equal(3, df.Columns.Count);
            foreach (var column in df.Columns)
            {
                Assert.Equal(maxRows, column.Length);
            }

            VerifyDataFrameColumnAndDataViewColumnValues<string>("Name", data, df, maxRows);
            VerifyDataFrameColumnAndDataViewColumnValues<bool>("FilterNext", data, df, maxRows);
            VerifyDataFrameColumnAndDataViewColumnValues<float>("Value", data, df, maxRows);
        }

        [Fact]
        public void TestDataFrameFromIDataView_MLData_SelectColumnsAndRows()
        {
            IDataView data = GetASampleIDataView();
            DataFrame df = data.ToDataFrame(3, "Name", "Value");
            Assert.Equal(3, df.Rows.Count);
            Assert.Equal(2, df.Columns.Count);
            foreach (var column in df.Columns)
            {
                Assert.Equal(3, column.Length);
            }

            VerifyDataFrameColumnAndDataViewColumnValues<string>("Name", data, df, 3);
            VerifyDataFrameColumnAndDataViewColumnValues<float>("Value", data, df, 3);
        }
    }
}
