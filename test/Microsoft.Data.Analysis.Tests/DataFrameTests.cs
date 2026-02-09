// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.Data.Analysis.Tests
{
    public partial class DataFrameTests : BaseTestClass
    {
        [Fact]
        public void TestIndexer()
        {
            DataFrame dataFrame = MakeDataFrameWithTwoColumns(length: 10);
            var foo = dataFrame[0, 0];
            Assert.Equal(0, dataFrame[0, 0]);
            Assert.Equal(11, dataFrame[1, 1]);
            Assert.Equal(2, dataFrame.Columns.Count);
            Assert.Equal("Int1", dataFrame.Columns[0].Name);

            var headList = dataFrame.Head(5);
            Assert.Equal(14, (int)headList.Rows[4][1]);

            var tailList = dataFrame.Tail(5);
            Assert.Equal(19, (int)tailList.Rows[4][1]);

            dataFrame[2, 1] = 1000;
            Assert.Equal(1000, dataFrame[2, 1]);

            var row = dataFrame.Rows[4];
            Assert.Equal(14, (int)row[1]);

            var column = dataFrame["Int2"] as Int32DataFrameColumn;
            Assert.Equal(1000, (int)column[2]);

            Assert.Throws<ArgumentException>(() => dataFrame["Int5"]);
            Assert.Throws<ArgumentException>(() => dataFrame[(string)null]);
        }

        [Fact]
        public void ColumnInsertTest()
        {
            var df = DataFrame.LoadCsvFromString("a1,a2\n1,2\n3,4");

            var dc0 = DataFrameColumn.Create("a0", new int[] { 0, 0 });
            df.Columns.Insert(0, dc0);
            var dc = df.Columns["a1"];
            Assert.Equal("a1", dc.Name);
        }

        [Fact]
        public void ColumnAndTableCreationTest()
        {
            const int rowCount = 10;
            DataFrameColumn intColumn = new Int32DataFrameColumn("IntColumn", Enumerable.Range(0, rowCount).Select(x => x));
            DataFrameColumn floatColumn = new SingleDataFrameColumn("FloatColumn", Enumerable.Range(0, rowCount).Select(x => (float)x));
            DataFrame dataFrame = new DataFrame();
            dataFrame.Columns.Insert(0, intColumn);
            dataFrame.Columns.Insert(1, floatColumn);
            Assert.Equal(rowCount, dataFrame.Rows.Count);
            Assert.Equal(2, dataFrame.Columns.Count);
            Assert.Equal(2, dataFrame.Columns.LongCount());
            Assert.Equal(rowCount, dataFrame.Columns[0].Length);
            Assert.Equal("IntColumn", dataFrame.Columns[0].Name);
            Assert.Equal(rowCount, dataFrame.Columns[1].Length);
            Assert.Equal("FloatColumn", dataFrame.Columns[1].Name);

            //add column with bigger length than other columns in the dataframe
            DataFrameColumn bigColumn = new SingleDataFrameColumn("BigColumn", Enumerable.Range(0, rowCount + 1).Select(x => (float)x));
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.Insert(2, bigColumn));
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.Add(bigColumn));

            //add column smaller than other columns in the dataframe
            DataFrameColumn smallColumn = new SingleDataFrameColumn("SmallColumn", Enumerable.Range(0, rowCount - 1).Select(x => (float)x));
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.Insert(2, smallColumn));
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.Add(smallColumn));

            //add column with duplicate name
            DataFrameColumn repeatedName = new SingleDataFrameColumn("FloatColumn", Enumerable.Range(0, rowCount).Select(x => (float)x));
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.Insert(2, repeatedName));

            //Insert column at index out of range
            DataFrameColumn extraColumn = new SingleDataFrameColumn("OtherFloatColumn", Enumerable.Range(0, rowCount).Select(x => (float)x));
            var columnCount = dataFrame.Columns.Count;
            Assert.Throws<ArgumentOutOfRangeException>(() => dataFrame.Columns.Insert(columnCount + 1, repeatedName));

            Assert.Equal(2, dataFrame.Columns.Count);
            DataFrameColumn intColumnCopy = new Int32DataFrameColumn("IntColumn", Enumerable.Range(0, rowCount).Select(x => x));
            Assert.Throws<ArgumentException>(() => dataFrame.Columns[1] = intColumnCopy);

            DataFrameColumn differentIntColumn = new Int32DataFrameColumn("IntColumn1", Enumerable.Range(0, rowCount).Select(x => x));
            dataFrame.Columns[1] = differentIntColumn;
            Assert.True(object.ReferenceEquals(differentIntColumn, dataFrame.Columns[1]));

            dataFrame.Columns.RemoveAt(1);
            Assert.Single(dataFrame.Columns);
            Assert.True(ReferenceEquals(intColumn, dataFrame.Columns[0]));

            // Test the params constructor
            DataFrame dataFrame1 = new DataFrame(intColumn, floatColumn);
            Assert.Equal(2, dataFrame1.Columns.Count);
            Assert.Equal(intColumn, dataFrame1.Columns[0]);
            Assert.Equal(floatColumn, dataFrame1.Columns[1]);
        }

        [Fact]
        public void InsertAndRemoveColumnToTheEndOfNotEmptyDataFrameTests()
        {
            DataFrame dataFrame = MakeDataFrameWithAllMutableColumnTypes(10);
            DataFrameColumn intColumn = new Int32DataFrameColumn("NewIntColumn", Enumerable.Range(0, 10).Select(x => x));

            int columnCount = dataFrame.Columns.Count;
            DataFrameColumn originalLastColumn = dataFrame.Columns[columnCount - 1];

            //Insert new column at the end
            dataFrame.Columns.Insert(columnCount, intColumn);
            Assert.Equal(columnCount + 1, dataFrame.Columns.Count);

            //Remove first
            dataFrame.Columns.RemoveAt(0);
            Assert.Equal(columnCount, dataFrame.Columns.Count);

            //Check that int column was inserted
            DataFrameColumn intColumn_1 = dataFrame.Columns["NewIntColumn"];
            Assert.True(ReferenceEquals(intColumn, intColumn_1));

            //Check that last column of the original dataframe was not removed
            DataFrameColumn lastColumn_1 = dataFrame.Columns[originalLastColumn.Name];
            Assert.True(ReferenceEquals(originalLastColumn, lastColumn_1));

            //Check that new column is the last one
            int newIndex = dataFrame.Columns.IndexOf("NewIntColumn");
            Assert.Equal(columnCount - 1, newIndex);

            //Check that original last column now has correct index
            int newIndexForOriginalLastColumn = dataFrame.Columns.IndexOf(originalLastColumn.Name);
            Assert.Equal(columnCount - 2, newIndexForOriginalLastColumn);
        }

        [Fact]
        public void AddAndRemoveColumnToTheEmptyDataFrameTests()
        {
            DataFrame dataFrame = new DataFrame();
            DataFrameColumn intColumn = new Int32DataFrameColumn("NewIntColumn", Enumerable.Range(0, 10).Select(x => x));

            dataFrame.Columns.Add(intColumn);
            Assert.Single(dataFrame.Columns);
            Assert.Equal(10, dataFrame.Rows.Count);

            dataFrame.Columns.Remove(intColumn);
            Assert.Empty(dataFrame.Columns);
            Assert.Equal(0, dataFrame.Rows.Count);
        }

        [Fact]
        public void ClearColumnsTests()
        {
            //Arrange
            DataFrame dataFrame = MakeDataFrameWithAllMutableColumnTypes(10);

            //Act
            dataFrame.Columns.Clear();

            //Assert
            Assert.Empty(dataFrame.Columns);

            Assert.Equal(0, dataFrame.Rows.Count);
            Assert.Equal(0, dataFrame.Columns.LongCount());
        }

        [Fact]
        public void RenameColumnWithSetNameTests()
        {
            StringDataFrameColumn city = new StringDataFrameColumn("City", new string[] { "London", "Berlin" });
            PrimitiveDataFrameColumn<int> temp = new PrimitiveDataFrameColumn<int>("Temperature", new int[] { 12, 13 });

            DataFrame dataframe = new DataFrame(city, temp);

            // Change the name of the column:
            dataframe["City"].SetName("Town");
            var renamedColumn = dataframe["Town"];

            Assert.Throws<ArgumentException>(() => dataframe["City"]);

            Assert.NotNull(renamedColumn);
            Assert.Equal("Town", renamedColumn.Name);
            Assert.True(ReferenceEquals(city, renamedColumn));
        }

        [Fact]
        public void RenameColumnWithRenameColumnTests()
        {
            StringDataFrameColumn city = new StringDataFrameColumn("City", new string[] { "London", "Berlin" });
            PrimitiveDataFrameColumn<int> temp = new PrimitiveDataFrameColumn<int>("Temperature", new int[] { 12, 13 });

            DataFrame dataframe = new DataFrame(city, temp);

            // Change the name of the column:
            dataframe.Columns.RenameColumn("City", "Town");
            var renamedColumn = dataframe["Town"];

            Assert.Throws<ArgumentException>(() => dataframe["City"]);

            Assert.NotNull(renamedColumn);
            Assert.Equal("Town", renamedColumn.Name);
            Assert.True(ReferenceEquals(city, renamedColumn));
        }

        [Fact]
        public void TestColumnReverseOrderState()
        {
            var column = new Int32DataFrameColumn("Int", Enumerable.Range(0, 10));
            var newColumn = 1 - column;
            var checkOrderColumn = 1 - newColumn;
            Assert.True(checkOrderColumn.ElementwiseEquals(column).All());
        }

        [Fact]
        public void TestProjectionAndAppend()
        {
            DataFrame df = MakeDataFrameWithTwoColumns(10);

            df["Int3"] = df.Columns["Int1"] * 2 + df.Columns["Int2"];
            Assert.Equal(16, df.Columns["Int3"][2]);
        }

        [Fact]
        public void TestOrderBy()
        {
            DataFrame df = MakeDataFrameWithAllMutableColumnTypes(20);
            df.Columns["Int"][0] = 100;
            df.Columns["Int"][19] = -1;
            df.Columns["Int"][5] = 2000;

            // Sort by "Int" in ascending order and nulls last
            var sortedDf = df.OrderBy("Int");
            Assert.Null(sortedDf.Columns["Int"][19]);
            Assert.Equal(-1, sortedDf.Columns["Int"][0]);
            Assert.Equal(100, sortedDf.Columns["Int"][17]);
            Assert.Equal(2000, sortedDf.Columns["Int"][18]);

            // Sort by "Int" in descending order and nulls last
            sortedDf = df.OrderByDescending("Int");
            Assert.Null(sortedDf.Columns["Int"][19]);
            Assert.Equal(-1, sortedDf.Columns["Int"][18]);
            Assert.Equal(100, sortedDf.Columns["Int"][1]);
            Assert.Equal(2000, sortedDf.Columns["Int"][0]);

            // Sort by "Int" in ascending order and nulls first
            sortedDf = df.OrderBy("Int", putNullValuesLast: false);
            Assert.Null(sortedDf.Columns["Int"][0]);
            Assert.Equal(-1, sortedDf.Columns["Int"][1]);
            Assert.Equal(100, sortedDf.Columns["Int"][18]);
            Assert.Equal(2000, sortedDf.Columns["Int"][19]);

            // Sort by "Int" in descending order and nulls first
            sortedDf = df.OrderByDescending("Int", putNullValuesLast: false);
            Assert.Null(sortedDf.Columns["Int"][0]);
            Assert.Equal(-1, sortedDf.Columns["Int"][19]);
            Assert.Equal(100, sortedDf.Columns["Int"][2]);
            Assert.Equal(2000, sortedDf.Columns["Int"][1]);

            // Sort by "String" in ascending order and nulls last
            sortedDf = df.OrderBy("String");
            Assert.Null(sortedDf.Columns["Int"][19]);
            Assert.Equal(1, sortedDf.Columns["Int"][1]);
            Assert.Equal(8, sortedDf.Columns["Int"][17]);
            Assert.Equal(9, sortedDf.Columns["Int"][18]);

            // Sort by "String" in descending order and nulls last
            sortedDf = df.OrderByDescending("String");
            Assert.Null(sortedDf.Columns["Int"][19]);
            Assert.Equal(8, sortedDf.Columns["Int"][1]);
            Assert.Equal(9, sortedDf.Columns["Int"][0]);

            // Sort by "String" in ascending order and nulls first
            sortedDf = df.OrderBy("String", putNullValuesLast: false);
            Assert.Null(sortedDf.Columns["Int"][0]);
            Assert.Equal(1, sortedDf.Columns["Int"][2]);
            Assert.Equal(8, sortedDf.Columns["Int"][18]);
            Assert.Equal(9, sortedDf.Columns["Int"][19]);

            // Sort by "String" in descending order and nulls first
            sortedDf = df.OrderByDescending("String", putNullValuesLast: false);
            Assert.Null(sortedDf.Columns["Int"][0]);
            Assert.Equal(8, sortedDf.Columns["Int"][2]);
            Assert.Equal(9, sortedDf.Columns["Int"][1]);
        }

        [Fact]
        public void TestGroupBy()
        {
            DataFrame df = MakeDataFrameWithNumericAndBoolColumns(10);
            DataFrame count = df.GroupBy("Bool").Count();
            Assert.Equal(2, count.Rows.Count);
            Assert.Equal((long)5, count.Columns["Int"][0]);
            Assert.Equal((long)4, count.Columns["Decimal"][1]);
            for (int r = 0; r < count.Rows.Count; r++)
            {
                for (int c = 1; c < count.Columns.Count; c++)
                {
                    Assert.Equal((long)(r == 0 ? 5 : 4), count.Columns[c][r]);
                }
            }

            DataFrame first = df.GroupBy("Bool").First();
            Assert.Equal(2, first.Rows.Count);
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < count.Columns.Count; c++)
                {
                    DataFrameColumn originalColumn = df.Columns[c];
                    DataFrameColumn firstColumn = first.Columns[originalColumn.Name];
                    Assert.Equal(originalColumn[r], firstColumn[r]);
                }
            }

            DataFrame head = df.GroupBy("Bool").Head(3);
            List<int> verify = new List<int>() { 0, 3, 1, 4, 2, 5 };
            for (int r = 0; r < 5; r++)
            {
                for (int c = 0; c < count.Columns.Count; c++)
                {
                    DataFrameColumn originalColumn = df.Columns[c];
                    DataFrameColumn headColumn = head.Columns[originalColumn.Name];
                    Assert.Equal(originalColumn[r].ToString(), headColumn[verify[r]].ToString());
                }
            }
            for (int c = 0; c < count.Columns.Count; c++)
            {
                DataFrameColumn originalColumn = df.Columns[c];
                if (originalColumn.Name == "Bool")
                    continue;
                DataFrameColumn headColumn = head.Columns[originalColumn.Name];
                Assert.Equal(originalColumn[7], headColumn[verify[5]]);
            }
            Assert.Equal(6, head.Rows.Count);

            DataFrame tail = df.GroupBy("Bool").Tail(3);
            Assert.Equal(6, tail.Rows.Count);
            List<int> originalColumnVerify = new List<int>() { 6, 8, 7, 9 };
            List<int> tailColumnVerity = new List<int>() { 1, 2, 4, 5 };
            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < count.Columns.Count; c++)
                {
                    DataFrameColumn originalColumn = df.Columns[c];
                    DataFrameColumn tailColumn = tail.Columns[originalColumn.Name];
                    Assert.Equal(originalColumn[originalColumnVerify[r]].ToString(), tailColumn[tailColumnVerity[r]].ToString());
                }
            }

            DataFrame max = df.GroupBy("Bool").Max();
            Assert.Equal(2, max.Rows.Count);
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < count.Columns.Count; c++)
                {
                    DataFrameColumn originalColumn = df.Columns[c];
                    if (originalColumn.Name == "Bool" || originalColumn.Name == "Char")
                        continue;
                    DataFrameColumn maxColumn = max.Columns[originalColumn.Name];
                    Assert.Equal(((long)(r == 0 ? 8 : 9)).ToString(), maxColumn[r].ToString());
                }
            }

            DataFrame min = df.GroupBy("Bool").Min();
            Assert.Equal(2, min.Rows.Count);

            DataFrame product = df.GroupBy("Bool").Product();
            Assert.Equal(2, product.Rows.Count);

            DataFrame sum = df.GroupBy("Bool").Sum();
            Assert.Equal(2, sum.Rows.Count);

            DataFrame mean = df.GroupBy("Bool").Mean();
            Assert.Equal(2, mean.Rows.Count);
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < count.Columns.Count; c++)
                {
                    DataFrameColumn originalColumn = df.Columns[c];
                    if (originalColumn.Name == "Bool" || originalColumn.Name == "Char")
                        continue;
                    DataFrameColumn minColumn = min.Columns[originalColumn.Name];
                    Assert.Equal(r == 0 ? "0" : "1", minColumn[r].ToString());

                    DataFrameColumn productColumn = product.Columns[originalColumn.Name];
                    Assert.Equal("0", productColumn[r].ToString());

                    DataFrameColumn sumColumn = sum.Columns[originalColumn.Name];
                    Assert.Equal("20", sumColumn[r].ToString());
                }
            }

            DataFrame columnSum = df.GroupBy("Bool").Sum("Int");
            Assert.Equal(2, columnSum.Columns.Count);
            Assert.Equal(20, columnSum.Columns["Int"][0]);
            Assert.Equal(20, columnSum.Columns["Int"][1]);
            DataFrame columnMax = df.GroupBy("Bool").Max("Int");
            Assert.Equal(2, columnMax.Columns.Count);
            Assert.Equal(8, columnMax.Columns["Int"][0]);
            Assert.Equal(9, columnMax.Columns["Int"][1]);
            DataFrame columnProduct = df.GroupBy("Bool").Product("Int");
            Assert.Equal(2, columnProduct.Columns.Count);
            Assert.Equal(0, columnProduct.Columns["Int"][0]);
            Assert.Equal(0, columnProduct.Columns["Int"][1]);
            DataFrame columnMin = df.GroupBy("Bool").Min("Int");
            Assert.Equal(2, columnMin.Columns.Count);
            Assert.Equal(0, columnMin.Columns["Int"][0]);
            Assert.Equal(1, columnMin.Columns["Int"][1]);

            DataFrame countIntColumn = df.GroupBy("Bool").Count("Int");
            Assert.Equal(2, countIntColumn.Columns.Count);
            Assert.Equal(2, countIntColumn.Rows.Count);
            Assert.Equal((long)5, countIntColumn.Columns["Int"][0]);
            Assert.Equal((long)4, countIntColumn.Columns["Int"][1]);

            DataFrame firstDecimalColumn = df.GroupBy("Bool").First("Decimal");
            Assert.Equal(2, firstDecimalColumn.Columns.Count);
            Assert.Equal(2, firstDecimalColumn.Rows.Count);
            Assert.Equal((decimal)0, firstDecimalColumn.Columns["Decimal"][0]);
            Assert.Equal((decimal)1, firstDecimalColumn.Columns["Decimal"][1]);
        }

        [Fact]
        public void TestGroupByDifferentColumnTypes()
        {
            void GroupCountAndAssert(DataFrame frame)
            {
                DataFrame grouped = frame.GroupBy("Column1").Count();
                Assert.Equal(2, grouped.Rows.Count);
            }

            DataFrame df = MakeDataFrame<byte, bool>(10, false);
            GroupCountAndAssert(df);

            df = MakeDataFrame<char, bool>(10, false);
            GroupCountAndAssert(df);

            df = MakeDataFrame<decimal, bool>(10, false);
            GroupCountAndAssert(df);

            df = MakeDataFrame<double, bool>(10, false);
            GroupCountAndAssert(df);

            df = MakeDataFrame<float, bool>(10, false);
            GroupCountAndAssert(df);

            df = MakeDataFrame<int, bool>(10, false);
            GroupCountAndAssert(df);

            df = MakeDataFrame<long, bool>(10, false);
            GroupCountAndAssert(df);

            df = MakeDataFrame<sbyte, bool>(10, false);
            GroupCountAndAssert(df);

            df = MakeDataFrame<short, bool>(10, false);
            GroupCountAndAssert(df);

            df = MakeDataFrame<uint, bool>(10, false);
            GroupCountAndAssert(df);

            df = MakeDataFrame<ulong, bool>(10, false);
            GroupCountAndAssert(df);

            df = MakeDataFrame<ushort, bool>(10, false);
            GroupCountAndAssert(df);
        }

        [Fact]
        public void TestIEnumerable()
        {
            DataFrame df = MakeDataFrameWithAllColumnTypes(10);

            int totalValueCount = 0;
            for (int i = 0; i < df.Columns.Count; i++)
            {
                DataFrameColumn baseColumn = df.Columns[i];
                foreach (object value in baseColumn)
                {
                    totalValueCount++;
                }
            }
            Assert.Equal(10 * df.Columns.Count, totalValueCount);

            // spot check a few column types:

            StringDataFrameColumn stringColumn = (StringDataFrameColumn)df.Columns["String"];
            StringBuilder actualStrings = new StringBuilder();
            foreach (string value in stringColumn)
            {
                if (value == null)
                {
                    actualStrings.Append("<null>");
                }
                else
                {
                    actualStrings.Append(value);
                }
            }
            Assert.Equal("01234<null>6789", actualStrings.ToString());

            ArrowStringDataFrameColumn arrowStringColumn = (ArrowStringDataFrameColumn)df.Columns["ArrowString"];
            actualStrings.Clear();
            foreach (string value in arrowStringColumn)
            {
                if (value == null)
                {
                    actualStrings.Append("<null>");
                }
                else
                {
                    actualStrings.Append(value);
                }
            }
            Assert.Equal("foofoofoofoofoo<null>foofoofoofoo", actualStrings.ToString());

            SingleDataFrameColumn floatColumn = (SingleDataFrameColumn)df.Columns["Float"];
            actualStrings.Clear();
            foreach (float? value in floatColumn)
            {
                if (value == null)
                {
                    actualStrings.Append("<null>");
                }
                else
                {
                    actualStrings.Append(value);
                }
            }
            Assert.Equal("01234<null>6789", actualStrings.ToString());

            Int32DataFrameColumn intColumn = (Int32DataFrameColumn)df.Columns["Int"];
            actualStrings.Clear();
            foreach (int? value in intColumn)
            {
                if (value == null)
                {
                    actualStrings.Append("<null>");
                }
                else
                {
                    actualStrings.Append(value);
                }
            }
            Assert.Equal("01234<null>6789", actualStrings.ToString());
        }

        [Fact]
        public void TestColumnClamp()
        {
            DataFrame df = MakeDataFrameWithNumericColumns(10);
            // Out of place
            DataFrameColumn clamped = df.Columns["Int"].Clamp(3, 7);
            Assert.Equal(3, clamped[0]);
            Assert.Equal(0, df.Columns["Int"][0]);
            Assert.Equal(3, clamped[1]);
            Assert.Equal(1, df.Columns["Int"][1]);
            Assert.Equal(3, clamped[2]);
            Assert.Equal(2, df.Columns["Int"][2]);
            Assert.Equal(3, clamped[3]);
            Assert.Equal(3, df.Columns["Int"][3]);
            Assert.Equal(4, clamped[4]);
            Assert.Equal(4, df.Columns["Int"][4]);
            Assert.Null(clamped[5]);
            Assert.Null(df.Columns["Int"][5]);
            Assert.Equal(6, clamped[6]);
            Assert.Equal(6, df.Columns["Int"][6]);
            Assert.Equal(7, clamped[7]);
            Assert.Equal(7, df.Columns["Int"][7]);
            Assert.Equal(7, clamped[8]);
            Assert.Equal(8, df.Columns["Int"][8]);
            Assert.Equal(7, clamped[9]);
            Assert.Equal(9, df.Columns["Int"][9]);

            // In place
            df.Columns["Int"].Clamp(3, 7, true);
            Assert.Equal(3, df.Columns["Int"][0]);
            Assert.Equal(3, df.Columns["Int"][1]);
            Assert.Equal(3, df.Columns["Int"][2]);
            Assert.Equal(3, df.Columns["Int"][3]);
            Assert.Equal(4, df.Columns["Int"][4]);
            Assert.Null(df.Columns["Int"][5]);
            Assert.Equal(6, df.Columns["Int"][6]);
            Assert.Equal(7, df.Columns["Int"][7]);
            Assert.Equal(7, df.Columns["Int"][8]);
            Assert.Equal(7, df.Columns["Int"][9]);
        }

        [Fact]
        public void TestDataFrameClamp()
        {
            DataFrame df = MakeDataFrameWithAllColumnTypes(10);
            IEnumerable<DataViewSchema.Column> dfColumns = ((IDataView)df).Schema;

            void VerifyDataFrameClamp(DataFrame clampedColumn)
            {

                IEnumerable<DataViewSchema.Column> clampedColumns = ((IDataView)clampedColumn).Schema;
                Assert.Equal(df.Columns.Count, clampedColumn.Columns.Count);
                Assert.Equal(dfColumns, clampedColumns);
                for (int c = 0; c < df.Columns.Count; c++)
                {
                    DataFrameColumn column = clampedColumn.Columns[c];
                    if (column.IsNumericColumn())
                    {
                        for (int i = 0; i < 4; i++)
                        {
                            Assert.Equal("3", column[i].ToString());
                        }
                        Assert.Equal(4.ToString(), column[4].ToString());
                        Assert.Null(column[5]);
                        Assert.Equal(6.ToString(), column[6].ToString());
                        for (int i = 7; i < 10; i++)
                        {
                            Assert.Equal("7", column[i].ToString());
                        }
                    }
                    else
                    {
                        for (int i = 0; i < column.Length; i++)
                        {
                            var colD = df.Columns[c][i];
                            var ocD = column[i];
                            Assert.Equal(df.Columns[c][i], column[i]);
                        }
                    }
                }
            }

            // Out of place
            DataFrame clamped = df.Clamp(3, 7);
            VerifyDataFrameClamp(clamped);
            for (int i = 0; i < 10; i++)
            {
                if (i != 5)
                    Assert.Equal(i, df.Columns["Int"][i]);
                else
                    Assert.Null(df.Columns["Int"][5]);
            }

            // Inplace
            df.Clamp(3, 7, true);
            VerifyDataFrameClamp(df);

        }

        [Fact]
        public void TestPrefixAndSuffix()
        {
            DataFrame df = MakeDataFrameWithAllColumnTypes(10);
            IEnumerable<DataViewSchema.Column> columnNames = ((IDataView)df).Schema;

            DataFrame prefix = df.AddPrefix("Prefix_");
            IEnumerable<DataViewSchema.Column> prefixNames = ((IDataView)prefix).Schema;
            foreach ((DataViewSchema.Column First, DataViewSchema.Column Second) in columnNames.Zip(((IDataView)df).Schema, (e1, e2) => (e1, e2)))
            {
                Assert.Equal(First.Name, Second.Name);
            }
            foreach ((DataViewSchema.Column First, DataViewSchema.Column Second) in prefixNames.Zip(columnNames, (e1, e2) => (e1, e2)))
            {
                Assert.Equal(First.Name, "Prefix_" + Second.Name);
            }

            // Inplace
            df.AddPrefix("Prefix_", true);
            prefixNames = ((IDataView)df).Schema;
            foreach ((DataViewSchema.Column First, DataViewSchema.Column Second) in columnNames.Zip(prefixNames, (e1, e2) => (e1, e2)))
            {
                Assert.Equal("Prefix_" + First.Name, Second.Name);
            }

            DataFrame suffix = df.AddSuffix("_Suffix");
            IEnumerable<DataViewSchema.Column> suffixNames = ((IDataView)suffix).Schema;
            foreach ((DataViewSchema.Column First, DataViewSchema.Column Second) in ((IDataView)df).Schema.Zip(columnNames, (e1, e2) => (e1, e2)))
            {
                Assert.Equal(First.Name, "Prefix_" + Second.Name);
            }
            foreach ((DataViewSchema.Column First, DataViewSchema.Column Second) in columnNames.Zip(suffixNames, (e1, e2) => (e1, e2)))
            {
                Assert.Equal("Prefix_" + First.Name + "_Suffix", Second.Name);
            }

            // InPlace
            df.AddSuffix("_Suffix", true);
            suffixNames = ((IDataView)df).Schema;
            foreach ((DataViewSchema.Column First, DataViewSchema.Column Second) in columnNames.Zip(suffixNames, (e1, e2) => (e1, e2)))
            {
                Assert.Equal("Prefix_" + First.Name + "_Suffix", Second.Name);
            }
        }

        [Fact]
        public void TestSample()
        {
            DataFrame df = MakeDataFrameWithAllColumnTypes(10);
            DataFrame sampled = df.Sample(7);
            Assert.Equal(7, sampled.Rows.Count);
            Assert.Equal(df.Columns.Count, sampled.Columns.Count);

            // all sampled rows should be unique.
            HashSet<int?> uniqueRowValues = new HashSet<int?>();
            foreach (int? value in sampled.Columns["Int"])
            {
                uniqueRowValues.Add(value);
            }
            Assert.Equal(uniqueRowValues.Count, sampled.Rows.Count);

            // should throw exception as sample size is greater than dataframe rows
            Assert.Throws<ArgumentException>(() => df.Sample(13));
        }

        [Fact]
        public void TestDescription()
        {
            DataFrame df = MakeDataFrameWithAllMutableColumnTypes(10);

            DataFrame description = df.Description();
            DataFrameColumn descriptionColumn = description.Columns[0];
            Assert.Equal("Description", descriptionColumn.Name);
            Assert.Equal("Length (excluding null values)", descriptionColumn[0]);
            Assert.Equal("Max", descriptionColumn[1]);
            Assert.Equal("Min", descriptionColumn[2]);
            Assert.Equal("Mean", descriptionColumn[3]);
            for (int i = 1; i < description.Columns.Count - 1; i++)
            {
                DataFrameColumn column = description.Columns[i];
                Assert.Equal(df.Columns[i - 1].Name, column.Name);
                Assert.Equal(4, column.Length);
                Assert.Equal((float)9, column[0]);
                Assert.Equal((float)9, column[1]);
                Assert.Equal((float)0, column[2]);
                Assert.Equal((float)4, column[3]);
            }

            // Explicitly check the dateTimes column
            DataFrameColumn dateTimeColumn = description.Columns[description.Columns.Count - 1];
            Assert.Equal("DateTime", dateTimeColumn.Name);
            Assert.Equal(4, dateTimeColumn.Length);
            Assert.Equal((float)9, dateTimeColumn[0]);
            Assert.Null(dateTimeColumn[1]);
            Assert.Null(dateTimeColumn[2]);
            Assert.Null(dateTimeColumn[3]);
        }

        [Fact]
        public void TestInfo()
        {
            DataFrame df = MakeDataFrameWithAllMutableColumnTypes(10);

            // Add a column manually here until we fix https://github.com/dotnet/corefxlab/issues/2784
            PrimitiveDataFrameColumn<DateTime> dateTimes = new PrimitiveDataFrameColumn<DateTime>("DateTimes");
            for (int i = 0; i < 10; i++)
            {
                dateTimes.Append(DateTime.Parse("2019/01/01"));
            }
            df.Columns.Add(dateTimes);

            DataFrame Info = df.Info();
            DataFrameColumn infoColumn = Info.Columns[0];
            Assert.Equal("Info", infoColumn.Name);
            Assert.Equal("Length (excluding null values)", infoColumn[1]);
            Assert.Equal("DataType", infoColumn[0]);

            for (int i = 1; i < Info.Columns.Count; i++)
            {
                DataFrameColumn column = Info.Columns[i];
                Assert.Equal(df.Columns[i - 1].DataType.ToString(), column[0].ToString());
                Assert.Equal(2, column.Length);
            }
        }

        [Fact]
        public void TestDropNulls()
        {
            //Create dataframe with 20 rows, where 1 row has only 1 null value and 1 row has all null values
            DataFrame df = MakeDataFrameWithAllMutableColumnTypes(20);
            df[0, 0] = null;

            DataFrame anyNulls = df.DropNulls();
            Assert.Equal(18, anyNulls.Rows.Count);

            DataFrame allNulls = df.DropNulls(DropNullOptions.All);
            Assert.Equal(19, allNulls.Rows.Count);
        }

        [Fact]
        public void TestInsertMismatchedColumnToEmptyDataFrame()
        {
            DataFrame df = new DataFrame();
            DataFrameColumn dataFrameColumn1 = new Int32DataFrameColumn("Int1");
            df.Columns.Insert(0, dataFrameColumn1);

            // should throw exception as column sizes are mismatched.

            Assert.Throws<ArgumentException>(() => df.Columns.Insert(1, new Int32DataFrameColumn("Int2", Enumerable.Range(0, 5).Select(x => x))));
        }

        [Fact]
        public void TestFillNulls()
        {
            DataFrame df = MakeDataFrameWithTwoColumns(20);
            Assert.Null(df[10, 0]);
            DataFrame fillNulls = df.FillNulls(1000);
            Assert.Equal(1000, (int)fillNulls[10, 1]);
            Assert.Null(df[10, 0]);
            df.FillNulls(1000, true);
            Assert.Equal(1000, df[10, 1]);

            StringDataFrameColumn strColumn = new StringDataFrameColumn("String", 0);
            strColumn.Append(null);
            strColumn.Append(null);
            Assert.Equal(2, strColumn.Length);
            Assert.Equal(2, strColumn.NullCount);
            DataFrameColumn filled = strColumn.FillNulls("foo");
            Assert.Equal(2, strColumn.Length);
            Assert.Equal(2, strColumn.NullCount);
            Assert.Equal(2, filled.Length);
            Assert.Equal(0, filled.NullCount);
            Assert.Equal("foo", filled[0]);
            Assert.Equal("foo", filled[1]);
            Assert.Null(strColumn[0]);
            Assert.Null(strColumn[1]);

            // In place
            strColumn.FillNulls("foo", true);
            Assert.Equal(2, strColumn.Length);
            Assert.Equal(0, strColumn.NullCount);
            Assert.Equal("foo", strColumn[0]);
            Assert.Equal("foo", strColumn[1]);

            // ArrowStringColumn (not inplace)
            ArrowStringDataFrameColumn arrowColumn = CreateArrowStringColumn(3);
            Assert.Equal(3, arrowColumn.Length);
            Assert.Equal(1, arrowColumn.NullCount);
            Assert.Null(arrowColumn[1]);
            ArrowStringDataFrameColumn arrowColumnFilled = arrowColumn.FillNulls("foo");
            Assert.Equal(3, arrowColumn.Length);
            Assert.Equal(1, arrowColumn.NullCount);
            Assert.Equal(3, arrowColumnFilled.Length);
            Assert.Equal(0, arrowColumnFilled.NullCount);
            Assert.Equal("foo", arrowColumnFilled[1]);
            Assert.Equal(arrowColumn[0], arrowColumnFilled[0]);
            Assert.Equal(arrowColumn[2], arrowColumnFilled[2]);
        }

        [Fact]
        public void TestValueCounts()
        {
            DataFrame df = MakeDataFrameWithAllColumnTypes(10, withNulls: false);
            DataFrame valueCounts = df.Columns["Bool"].ValueCounts();
            Assert.Equal(2, valueCounts.Rows.Count);
            Assert.Equal((long)5, valueCounts.Columns["Counts"][0]);
            Assert.Equal((long)5, valueCounts.Columns["Counts"][1]);
        }

#pragma warning disable CS0612, CS0618  // Type or member is obsolete
        [Fact]
        public void TestApplyElementwiseNullCount()
        {
            DataFrame df = MakeDataFrameWithTwoColumns(10);
            Int32DataFrameColumn column = df.Columns["Int1"] as Int32DataFrameColumn;
            Assert.Equal(1, column.NullCount);

            // Change all existing values to null

            column.ApplyElementwise((int? value, long rowIndex) =>
            {
                if (!(value is null))
                    return null;
                return value;
            });

            Assert.Equal(column.Length, column.NullCount);

            // Don't change null values
            column.ApplyElementwise((int? value, long rowIndex) =>
            {
                return value;
            });
            Assert.Equal(column.Length, column.NullCount);

            // Change all null values to real values
            column.ApplyElementwise((int? value, long rowIndex) =>
            {
                return 5;
            });
            Assert.Equal(0, column.NullCount);

            // Don't change real values
            column.ApplyElementwise((int? value, long rowIndex) =>
            {
                return value;
            });
            Assert.Equal(0, column.NullCount);

        }
#pragma warning restore CS0612, CS0618 // Type or member is obsolete

        [Theory]
        [InlineData(10, 5)]
        [InlineData(20, 20)]
        public void TestClone(int dfLength, int intDfLength)
        {
            DataFrame df = MakeDataFrameWithAllColumnTypes(dfLength, withNulls: true);
            DataFrame intDf = MakeDataFrameWithTwoColumns(intDfLength, false);
            Int32DataFrameColumn intColumn = intDf.Columns["Int1"] as Int32DataFrameColumn;
            DataFrame clone = df[intColumn];
            Assert.Equal(intDfLength, clone.Rows.Count);
            Assert.Equal(df.Columns.Count, clone.Columns.Count);
            for (int i = 0; i < df.Columns.Count; i++)
            {
                DataFrameColumn dfColumn = df.Columns[i];
                DataFrameColumn cloneColumn = clone.Columns[i];
                for (long r = 0; r < clone.Rows.Count; r++)
                {
                    Assert.Equal(dfColumn[r], cloneColumn[r]);
                }
            }
        }

        [Fact]
        public void TestColumnCreationFromExisitingColumn()
        {
            DataFrame df = MakeDataFrameWithAllColumnTypes(10);
            BooleanDataFrameColumn bigInts = new BooleanDataFrameColumn("BigInts", df.Columns["Int"].ElementwiseGreaterThan(5));
            for (int i = 0; i < 10; i++)
            {
                if (i <= 5)
                    Assert.False(bigInts[i]);
                else
                    Assert.True(bigInts[i]);
            }
        }

        [Fact]
        public void TestColumns()
        {
            DataFrame df = MakeDataFrameWithAllColumnTypes(10);
            IReadOnlyList<DataFrameColumn> columns = df.Columns;
            int i = 0;
            Assert.Equal(columns.Count, df.Columns.Count);
            foreach (DataFrameColumn dataFrameColumn in columns)
            {
                Assert.Equal(dataFrameColumn, df.Columns[i++]);
            }

        }

        [Fact]
        public void TestRows()
        {
            DataFrame df = MakeDataFrameWithAllColumnTypes(10);
            DataFrameRowCollection rows = df.Rows;
            Assert.Equal(10, rows.Count);
            DataFrameRow firstRow = rows[0];
            object firstValue = firstRow[0];
            Assert.Equal(df[0, 0], firstValue);
            long rowCount = 0;
            foreach (DataFrameRow row in rows)
            {
                int columnIndex = 0;
                foreach (var value in row)
                {
                    Assert.Equal(df.Columns[columnIndex][rowCount], value);
                    columnIndex++;
                }
                rowCount++;
            }
            Assert.Equal(df.Rows.Count, rowCount);

            DataFrameRow nullRow = rows[5];
            int intColumnIndex = df.Columns.IndexOf("Int");
            Assert.Equal(1, df.Columns[intColumnIndex].NullCount);
            nullRow[intColumnIndex] = 5;
            Assert.Equal(0, df.Columns[intColumnIndex].NullCount);
            nullRow[intColumnIndex] = null;
            Assert.Equal(1, df.Columns[intColumnIndex].NullCount);
        }

        [Fact]
        public void TestMutationOnRows()
        {
            DataFrame df = MakeDataFrameWithNumericColumns(10);
            DataFrameRowCollection rows = df.Rows;

            foreach (DataFrameRow row in rows)
            {
                for (int i = 0; i < df.Columns.Count; i++)
                {
                    DataFrameColumn column = df.Columns[i];
                    row[i] = Convert.ChangeType(12, column.DataType);
                }
            }

            foreach (var column in df.Columns)
            {
                foreach (var value in column)
                {
                    Assert.Equal("12", value.ToString());
                }
            }
        }

        [Fact]
        public void TestAppendRows()
        {
            DataFrame df = MakeDataFrame<float, bool>(10);
            DataFrame df2 = MakeDataFrame<int, bool>(5);
            Assert.Equal(10, df.Rows.Count);
            Assert.Equal(1, df.Columns[0].NullCount);
            Assert.Equal(1, df.Columns[1].NullCount);

            DataFrame ret = df.Append(df2.Rows, inPlace: false);
            Assert.Equal(10, df.Rows.Count);
            Assert.Equal(1, df.Columns[0].NullCount);
            Assert.Equal(1, df.Columns[1].NullCount);

            Verify(ret, df, df2);

            void Verify(DataFrame ret, DataFrame check1, DataFrame check2)
            {
                Assert.Equal(15, ret.Rows.Count);
                Assert.Equal(2, ret.Columns[0].NullCount);
                Assert.Equal(2, ret.Columns[1].NullCount);
                for (long i = 0; i < ret.Rows.Count; i++)
                {
                    DataFrameRow row = ret.Rows[i];
                    for (int j = 0; j < check1.Columns.Count; j++)
                    {
                        if (i < check1.Rows.Count)
                        {
                            Assert.Equal(row[j], check1.Rows[i][j]);
                        }
                        else
                        {
                            Assert.Equal(row[j]?.ToString(), (check2.Rows[i - check1.Rows.Count][j])?.ToString());
                        }
                    }
                }
            }

            DataFrame dfClone = df.Clone();
            df.Append(df2.Rows, inPlace: true);
            Verify(df, dfClone, df2);
        }

        [Fact]
        public void TestAppendRowsIfColumnAreOutOfOrder()
        {
            var dataFrame = new DataFrame(
                new StringDataFrameColumn("ColumnA", new string[] { "a", "b", "c" }),
                new Int32DataFrameColumn("ColumnB", new int[] { 1, 2, 3 }),
                new Int32DataFrameColumn("ColumnC", new int[] { 10, 20, 30 }));

            //ColumnC and ColumnB are swaped
            var dataFrame2 = new DataFrame(
                new StringDataFrameColumn("ColumnA", new string[] { "d", "e", "f" }),
                new Int32DataFrameColumn("ColumnC", new int[] { 40, 50, 60 }),
                new Int32DataFrameColumn("ColumnB", new int[] { 4, 5, 6 }));

            var resultDataFrame = dataFrame.Append(dataFrame2.Rows);

            Assert.Equal(3, resultDataFrame.Columns.Count);
            Assert.Equal(6, resultDataFrame.Rows.Count);

            Assert.Equal("c", resultDataFrame["ColumnA"][2]);
            Assert.Equal("d", resultDataFrame["ColumnA"][3]);

            Assert.Equal(3, resultDataFrame["ColumnB"][2]);
            Assert.Equal(4, resultDataFrame["ColumnB"][3]);

            Assert.Equal(30, resultDataFrame["ColumnC"][2]);
            Assert.Equal(40, resultDataFrame["ColumnC"][3]);
        }

        [Fact]
        public void TestAppendRow()
        {
            DataFrame df = MakeDataFrame<int, bool>(10);
            df.Append(new List<object> { 5, true }, inPlace: true);
            Assert.Equal(11, df.Rows.Count);
            Assert.Equal(1, df.Columns[0].NullCount);
            Assert.Equal(1, df.Columns[1].NullCount);

            DataFrame ret = df.Append(new List<object> { 5, true });
            Assert.Equal(12, ret.Rows.Count);
            Assert.Equal(1, ret.Columns[0].NullCount);
            Assert.Equal(1, ret.Columns[1].NullCount);

            df.Append(new List<object> { 100 }, inPlace: true);
            Assert.Equal(12, df.Rows.Count);
            Assert.Equal(1, df.Columns[0].NullCount);
            Assert.Equal(2, df.Columns[1].NullCount);

            ret = df.Append(new List<object> { 100 }, inPlace: false);
            Assert.Equal(13, ret.Rows.Count);
            Assert.Equal(1, ret.Columns[0].NullCount);
            Assert.Equal(3, ret.Columns[1].NullCount);

            df.Append(new List<object> { null, null }, inPlace: true);
            Assert.Equal(13, df.Rows.Count);
            Assert.Equal(2, df.Columns[0].NullCount);
            Assert.Equal(3, df.Columns[1].NullCount);
            ret = df.Append(new List<object> { null, null }, inPlace: false);
            Assert.Equal(14, ret.Rows.Count);
            Assert.Equal(3, ret.Columns[0].NullCount);
            Assert.Equal(4, ret.Columns[1].NullCount);

            df.Append(new Dictionary<string, object> { { "Column1", (object)5 }, { "Column2", false } }, inPlace: true);
            Assert.Equal(14, df.Rows.Count);
            Assert.Equal(2, df.Columns[0].NullCount);
            Assert.Equal(3, df.Columns[1].NullCount);
            ret = df.Append(new Dictionary<string, object> { { "Column1", (object)5 }, { "Column2", false } }, inPlace: false);
            Assert.Equal(15, ret.Rows.Count);
            Assert.Equal(2, ret.Columns[0].NullCount);
            Assert.Equal(3, ret.Columns[1].NullCount);

            df.Append(new Dictionary<string, object> { { "Column1", 5 } }, inPlace: true);
            Assert.Equal(15, df.Rows.Count);

            Assert.Equal(15, df.Columns["Column1"].Length);
            Assert.Equal(15, df.Columns["Column2"].Length);
            Assert.Equal(2, df.Columns[0].NullCount);
            Assert.Equal(4, df.Columns[1].NullCount);
            ret = df.Append(new Dictionary<string, object> { { "Column1", 5 } }, inPlace: false);
            Assert.Equal(16, ret.Rows.Count);

            Assert.Equal(16, ret.Columns["Column1"].Length);
            Assert.Equal(16, ret.Columns["Column2"].Length);
            Assert.Equal(2, ret.Columns[0].NullCount);
            Assert.Equal(5, ret.Columns[1].NullCount);

            df.Append(new Dictionary<string, object> { { "Column2", false } }, inPlace: true);
            Assert.Equal(16, df.Rows.Count);
            Assert.Equal(16, df.Columns["Column1"].Length);
            Assert.Equal(16, df.Columns["Column2"].Length);
            Assert.Equal(3, df.Columns[0].NullCount);
            Assert.Equal(4, df.Columns[1].NullCount);
            ret = df.Append(new Dictionary<string, object> { { "Column2", false } }, inPlace: false);
            Assert.Equal(17, ret.Rows.Count);
            Assert.Equal(17, ret.Columns["Column1"].Length);
            Assert.Equal(17, ret.Columns["Column2"].Length);
            Assert.Equal(4, ret.Columns[0].NullCount);
            Assert.Equal(4, ret.Columns[1].NullCount);

            df.Append((IEnumerable<object>)null, inPlace: true);
            Assert.Equal(17, df.Rows.Count);
            Assert.Equal(17, df.Columns["Column1"].Length);
            Assert.Equal(17, df.Columns["Column2"].Length);
            Assert.Equal(4, df.Columns[0].NullCount);
            Assert.Equal(5, df.Columns[1].NullCount);
            ret = df.Append((IEnumerable<object>)null, inPlace: false);
            Assert.Equal(18, ret.Rows.Count);
            Assert.Equal(18, ret.Columns["Column1"].Length);
            Assert.Equal(18, ret.Columns["Column2"].Length);
            Assert.Equal(5, ret.Columns[0].NullCount);
            Assert.Equal(6, ret.Columns[1].NullCount);

            // DataFrame must remain usable even if Append throws
            Assert.Throws<FormatException>(() => df.Append(new List<object> { 5, "str" }, inPlace: true));
            Assert.Throws<FormatException>(() => df.Append(new Dictionary<string, object> { { "Column2", "str" } }, inPlace: true));
            Assert.Throws<ArgumentException>(() => df.Append(new List<object> { 5, true, true }, inPlace: true));

            df.Append(inPlace: true);
            Assert.Equal(18, df.Rows.Count);
            Assert.Equal(18, df.Columns["Column1"].Length);
            Assert.Equal(18, df.Columns["Column2"].Length);
            Assert.Equal(5, df.Columns[0].NullCount);
            Assert.Equal(6, df.Columns[1].NullCount);

            ret = df.Append(inPlace: false);
            Assert.Equal(18, df.Rows.Count);
            Assert.Equal(18, df.Columns["Column1"].Length);
            Assert.Equal(18, df.Columns["Column2"].Length);
            Assert.Equal(5, df.Columns[0].NullCount);
            Assert.Equal(6, df.Columns[1].NullCount);
            Assert.Equal(19, ret.Rows.Count);
            Assert.Equal(19, ret.Columns["Column1"].Length);
            Assert.Equal(19, ret.Columns["Column2"].Length);
            Assert.Equal(6, ret.Columns[0].NullCount);
            Assert.Equal(7, ret.Columns[1].NullCount);
        }

        [Fact]
        public void TestAppendEmptyValue()
        {
            DataFrame df = MakeDataFrame<int, bool>(10);
            df.Append(new List<object> { "", true }, inPlace: true);
            Assert.Equal(11, df.Rows.Count);
            Assert.Equal(2, df.Columns[0].NullCount);
            Assert.Equal(1, df.Columns[1].NullCount);

            StringDataFrameColumn column = new StringDataFrameColumn("Strings", Enumerable.Range(0, 11).Select(x => x.ToString()));
            df.Columns.Add(column);

            df.Append(new List<object> { 1, true, "" }, inPlace: true);
            Assert.Equal(12, df.Rows.Count);
            Assert.Equal(2, df.Columns[0].NullCount);
            Assert.Equal(1, df.Columns[1].NullCount);
            Assert.Equal(0, df.Columns[2].NullCount);

            df.Append(new List<object> { 1, true, null }, inPlace: true);
            Assert.Equal(13, df.Rows.Count);
            Assert.Equal(1, df.Columns[2].NullCount);
        }

        [Fact]
#pragma warning disable CS0612, CS0618 // Type or member is obsolete
        public void TestApply()
        {
            int[] values = { 1, 2, 3, 4, 5 };
            var col = new Int32DataFrameColumn("Ints", values);

            PrimitiveDataFrameColumn<double> newCol = col.Apply(i => i + 0.5d);

            Assert.Equal(values.Length, newCol.Length);

            for (int i = 0; i < newCol.Length; i++)
            {
                Assert.Equal(col[i], values[i]); // Make sure values didn't change
                Assert.Equal(newCol[i], values[i] + 0.5d);
            }
        }
#pragma warning disable CS0612, CS0618 // Type or member is obsolete

        [Fact]
        public void TestDataFrameCreate()
        {
            int length = 10;
            void AssertLengthTypeAndValues(DataFrameColumn column, Type type)
            {
                Assert.Equal(column.DataType, type);
                Assert.Equal(length, column.Length);
                for (long i = 0; i < column.Length; i++)
                {
                    Assert.Equal(i.ToString(), column[i].ToString());
                }
            }
            DataFrameColumn stringColumn = DataFrameColumn.Create("String", Enumerable.Range(0, length).Select(x => x.ToString()));
            AssertLengthTypeAndValues(stringColumn, typeof(string));
            DataFrameColumn byteColumn = DataFrameColumn.Create("Byte", Enumerable.Range(0, length).Select(x => (byte)x));
            AssertLengthTypeAndValues(byteColumn, typeof(byte));
            DataFrameColumn decimalColumn = DataFrameColumn.Create("Decimal", Enumerable.Range(0, length).Select(x => (decimal)x));
            AssertLengthTypeAndValues(decimalColumn, typeof(decimal));
            DataFrameColumn doubleColumn = DataFrameColumn.Create("Double", Enumerable.Range(0, length).Select(x => (double)x));
            AssertLengthTypeAndValues(doubleColumn, typeof(double));
            DataFrameColumn floatColumn = DataFrameColumn.Create("Float", Enumerable.Range(0, length).Select(x => (float)x));
            AssertLengthTypeAndValues(floatColumn, typeof(float));
            DataFrameColumn intColumn = DataFrameColumn.Create("Int", Enumerable.Range(0, length).Select(x => x));
            AssertLengthTypeAndValues(intColumn, typeof(int));
            DataFrameColumn longColumn = DataFrameColumn.Create("Long", Enumerable.Range(0, length).Select(x => (long)x));
            AssertLengthTypeAndValues(longColumn, typeof(long));
            DataFrameColumn sbyteColumn = DataFrameColumn.Create("Sbyte", Enumerable.Range(0, length).Select(x => (sbyte)x));
            AssertLengthTypeAndValues(sbyteColumn, typeof(sbyte));
            DataFrameColumn shortColumn = DataFrameColumn.Create("Short", Enumerable.Range(0, length).Select(x => (short)x));
            AssertLengthTypeAndValues(shortColumn, typeof(short));
            DataFrameColumn uintColumn = DataFrameColumn.Create("Uint", Enumerable.Range(0, length).Select(x => (uint)x));
            AssertLengthTypeAndValues(uintColumn, typeof(uint));
            DataFrameColumn ulongColumn = DataFrameColumn.Create("Ulong", Enumerable.Range(0, length).Select(x => (ulong)x));
            AssertLengthTypeAndValues(ulongColumn, typeof(ulong));
            DataFrameColumn ushortColumn = DataFrameColumn.Create("Ushort", Enumerable.Range(0, length).Select(x => (ushort)x));
            AssertLengthTypeAndValues(ushortColumn, typeof(ushort));
        }

        [Fact]
        public void GetColumnTests()
        {
            DataFrame dataFrame = MakeDataFrameWithAllColumnTypes(10);
            PrimitiveDataFrameColumn<int> primitiveInts = dataFrame.Columns.GetPrimitiveColumn<int>("Int");
            Assert.NotNull(primitiveInts);
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.GetPrimitiveColumn<float>("Int"));

            StringDataFrameColumn strings = dataFrame.Columns.GetStringColumn("String");
            Assert.NotNull(strings);
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.GetStringColumn("ArrowString"));

            ArrowStringDataFrameColumn arrowStrings = dataFrame.Columns.GetArrowStringColumn("ArrowString");
            Assert.NotNull(arrowStrings);
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.GetArrowStringColumn("String"));

            ByteDataFrameColumn bytes = dataFrame.Columns.GetByteColumn("Byte");
            Assert.NotNull(bytes);
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.GetSingleColumn("Byte"));

            Int32DataFrameColumn ints = dataFrame.Columns.GetInt32Column("Int");
            Assert.NotNull(ints);
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.GetSingleColumn("Int"));

            BooleanDataFrameColumn bools = dataFrame.Columns.GetBooleanColumn("Bool");
            Assert.NotNull(bools);
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.GetSingleColumn("Bool"));

            CharDataFrameColumn chars = dataFrame.Columns.GetCharColumn("Char");
            Assert.NotNull(chars);
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.GetSingleColumn("Char"));

            DecimalDataFrameColumn decimals = dataFrame.Columns.GetDecimalColumn("Decimal");
            Assert.NotNull(decimals);
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.GetSingleColumn("Decimal"));

            DoubleDataFrameColumn doubles = dataFrame.Columns.GetDoubleColumn("Double");
            Assert.NotNull(doubles);
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.GetSingleColumn("Double"));

            SingleDataFrameColumn singles = dataFrame.Columns.GetSingleColumn("Float");
            Assert.NotNull(singles);
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.GetDoubleColumn("Float"));

            Int64DataFrameColumn longs = dataFrame.Columns.GetInt64Column("Long");
            Assert.NotNull(longs);
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.GetSingleColumn("Long"));

            SByteDataFrameColumn sbytes = dataFrame.Columns.GetSByteColumn("Sbyte");
            Assert.NotNull(sbytes);
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.GetSingleColumn("Sbyte"));

            Int16DataFrameColumn shorts = dataFrame.Columns.GetInt16Column("Short");
            Assert.NotNull(shorts);
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.GetSingleColumn("Short"));

            UInt32DataFrameColumn uints = dataFrame.Columns.GetUInt32Column("Uint");
            Assert.NotNull(uints);
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.GetSingleColumn("Uint"));

            UInt64DataFrameColumn ulongs = dataFrame.Columns.GetUInt64Column("Ulong");
            Assert.NotNull(ulongs);
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.GetSingleColumn("Ulong"));

            UInt16DataFrameColumn ushorts = dataFrame.Columns.GetUInt16Column("Ushort");
            Assert.NotNull(ushorts);
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.GetSingleColumn("Ushort"));

        }

        [Fact]
        public void TestMean()
        {
            DataFrame df = MakeDataFrameWithNumericColumns(10, true, 0);

            Assert.Equal(40.0 / 9.0, df["Decimal"].Mean());
        }

        [Fact]
        public void TestMedian()
        {
            DataFrame df = MakeDataFrameWithNumericColumns(10, true, 0);

            Assert.Equal(4, df["Decimal"].Median());
        }

        [Fact]
        public void Test_StringColumnNotEqualsNull()
        {
            var col = new StringDataFrameColumn("col", new[] { "One", null, "Two", "Three" });
            var dfTest = new DataFrame(col);

            var filteredNullDf = dfTest.Filter(dfTest["col"].ElementwiseNotEquals(null));

            Assert.True(filteredNullDf.Columns.IndexOf("col") >= 0);
            Assert.Equal(3, filteredNullDf.Columns["col"].Length);

            Assert.Equal("One", filteredNullDf.Columns["col"][0]);
            Assert.Equal("Two", filteredNullDf.Columns["col"][1]);
            Assert.Equal("Three", filteredNullDf.Columns["col"][2]);
        }

        [Fact]
        public void Test_StringColumnEqualsNull()
        {
            var index = new Int32DataFrameColumn("index", new int[] { 1, 2, 3, 4, 5 });
            var col = new StringDataFrameColumn("col", new[] { "One", null, "Three", "Four", null }); ;
            var dfTest = new DataFrame(index, col);

            var filteredNullDf = dfTest.Filter(dfTest["col"].ElementwiseEquals(null));

            Assert.True(filteredNullDf.Columns.IndexOf("col") >= 0);
            Assert.True(filteredNullDf.Columns.IndexOf("index") >= 0);

            Assert.Equal(2, filteredNullDf.Rows.Count);

            Assert.Equal(2, filteredNullDf.Columns["index"][0]);
            Assert.Equal(5, filteredNullDf.Columns["index"][1]);
        }

        public static IEnumerable<object[]> GenerateDataFrameMeltData()
        {
            yield return new object[]
            {
                new DataFrame(
                new Int32DataFrameColumn("id", new int?[] { 1, 2 }),
                new DoubleDataFrameColumn("A", new double?[] { 10, 20 }),
                new DoubleDataFrameColumn("B", new double?[] { 30, 40 })
                ),
                new DataFrame(
                new Int32DataFrameColumn("id", new int?[] { 1, 2, 1, 2 }),
                new StringDataFrameColumn("Variable", new string[] { "A", "A", "B", "B" }),
                new DoubleDataFrameColumn("Value", new double?[] { 10, 20, 30, 40 })
                ),
                new List<string> { "id" },
                new List<string> { "A", "B" },
                "Variable",
                "Value",
                true,
            };
            yield return new object[]
            {
                new DataFrame(
                new Int32DataFrameColumn("id", new int?[] { 1, 2 }),
                new DoubleDataFrameColumn("A", new double?[] { 10, 20 }),
                new DoubleDataFrameColumn("B", new double?[] { 30, 40 })
                ),
                new DataFrame(
                new Int32DataFrameColumn("id", new int?[] { 1, 2, 1, 2 }),
                new StringDataFrameColumn("Variable", new string[] { "A", "A", "B", "B" }),
                new DoubleDataFrameColumn("Value", new double?[] { 10, 20, 30, 40 })
                ),
                new List<string> { "id" },
                null,
                "Variable",
                "Value",
                true,
            };
            yield return new object[]
            {
                new DataFrame(
                new Int32DataFrameColumn("id", new int?[] { 1, 2, 3, 4 }),
                new DoubleDataFrameColumn("A", new double?[] { 10, 20, null, 30 }),
                new DoubleDataFrameColumn("B", new double?[] { 30, 40, 50, null })
                ),
                new DataFrame(
                new Int32DataFrameColumn("id", new int?[] { 1, 2, 3, 4, 1, 2, 3, 4 }),
                new StringDataFrameColumn("Variable", new string[] { "A", "A", "A", "A", "B", "B", "B", "B" }),
                new DoubleDataFrameColumn("Value", new double?[] { 10, 20, null, 30, 30, 40, 50, null })
                ),
                new List<string> { "id" },
                null,
                "Variable",
                "Value",
                false,
            };
            yield return new object[]
            {
                new DataFrame(
                new Int32DataFrameColumn("id", new int?[] { 1, 2, 3, 4 }),
                new DoubleDataFrameColumn("A", new double?[] { 10, 20, null, 30 }),
                new DoubleDataFrameColumn("B", new double?[] { 30, 40, 50, null })
                ),
                new DataFrame(
                new Int32DataFrameColumn("id", new int?[] { 1, 2, 4, 1, 2, 3 }),
                new StringDataFrameColumn("Variable", new string[] { "A", "A", "A", "B", "B", "B" }),
                new DoubleDataFrameColumn("Value", new double?[] { 10, 20, 30, 30, 40, 50 })
                ),
                new List<string> { "id" },
                null,
                "Variable",
                "Value",
                true,
            };
            yield return new object[]
            {
                new DataFrame(
                new Int32DataFrameColumn("id", new int?[] { 1, 2, 3, 4, 5 }),
                new DoubleDataFrameColumn("A", new double?[] { 10, 20, null, 30, 40 }),
                new StringDataFrameColumn("B", new string[] { "30", "40", "50", null, "" })
                ),
                new DataFrame(
                new Int32DataFrameColumn("id", new int?[] { 1, 2, 3, 4, 5, 1, 2, 3, 4, 5 }),
                new StringDataFrameColumn("Variable", new string[] { "A", "A", "A", "A", "A", "B", "B", "B", "B", "B" }),
                new StringDataFrameColumn("Value", new string[] { "10", "20", null, "30", "40", "30", "40", "50", null, "" })
                ),
                new List<string> { "id" },
                null,
                "Variable",
                "Value",
                false,
            };
            yield return new object[]
            {
                new DataFrame(
                new Int32DataFrameColumn("id", new int?[] { 1, 2, 3, 4, 5 }),
                new DoubleDataFrameColumn("A", new double?[] { 10, 20, null, 30, 40 }),
                new StringDataFrameColumn("B", new string[] { "30", "40", "50", null, "" })
                ),
                new DataFrame(
                new Int32DataFrameColumn("id", new int?[] { 1, 2, 4, 5, 1, 2, 3 }),
                new StringDataFrameColumn("Variable", new string[] { "A", "A", "A", "A", "B", "B", "B" }),
                new StringDataFrameColumn("Value", new string[] { "10", "20", "30", "40", "30", "40", "50" })
                ),
                new List<string> { "id" },
                null,
                "Variable",
                "Value",
                true,
            };
            yield return new object[]
            {
                new DataFrame(
                new Int32DataFrameColumn("id", new int?[0]),
                new DoubleDataFrameColumn("A", new double?[0]),
                new StringDataFrameColumn("B", new string[0])
                ),
                new DataFrame(
                new Int32DataFrameColumn("id", new int?[0]),
                new StringDataFrameColumn("Variable", new string[0]),
                new StringDataFrameColumn("Value", new string[0])
                ),
                new List<string> { "id" },
                null,
                "Variable",
                "Value",
                false,
            };
            yield return new object[]
            {
                new DataFrame(
                new Int32DataFrameColumn("id", new int?[0]),
                new DoubleDataFrameColumn("A", new double?[0]),
                new StringDataFrameColumn("B", new string[0])
                ),
                new DataFrame(
                new Int32DataFrameColumn("id", new int?[0]),
                new StringDataFrameColumn("Variable", new string[0]),
                new StringDataFrameColumn("Value", new string[0])
                ),
                new List<string> { "id" },
                null,
                "Variable",
                "Value",
                true,
            };
        }

        [Theory]
        [MemberData(nameof(GenerateDataFrameMeltData))]
        public void TestMelt(DataFrame inputDataFrame, DataFrame outputDataFrame, IEnumerable<string> idColumns, IEnumerable<string> valueColumns, string variableName, string valueName, bool dropNulls)
        {
            DataFrameAssert.Equal(outputDataFrame, inputDataFrame.Melt(idColumns, valueColumns, variableName, valueName, dropNulls));
        }

        [Fact]
        public void TestMelt_InvalidData()
        {
            DataFrame df = new DataFrame(
                new Int32DataFrameColumn("id", new int?[] { 1, 2, 3, 4 }),
                new DoubleDataFrameColumn("A", new double?[] { 10, 20, null, 30 }),
                new DoubleDataFrameColumn("B", new double?[] { 30, 40, 50, null })
                );

            Assert.Throws<ArgumentException>(() => df.Melt(new string[0], new string[] { "id", "A", "B" }));

            Assert.Throws<ArgumentException>(() => df.Melt(new string[] { "id", "A", "B" }, new string[0]));

            Assert.Throws<ArgumentException>(() => df.Melt(new string[] { "id", "A" }, new string[] { "A", "B" }));

            Assert.Throws<InvalidOperationException>(() => df.Melt(new string[] { "id", "A", "B" }));
        }
    }
}
