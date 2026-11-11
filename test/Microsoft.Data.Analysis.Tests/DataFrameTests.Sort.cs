// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace Microsoft.Data.Analysis.Tests
{
    public partial class DataFrameTests
    {
        [Theory]
        [InlineData(true, true)]
        [InlineData(true, false)]
        [InlineData(false, true)]
        [InlineData(false, false)]
        public void TestPrimitiveColumnSortIsStable(bool ascending, bool putNullValuesLast)
        {
            int?[] values = Enumerable.Range(0, 100).Select(i => i % 7 == 0 ? (int?)null : i % 3).ToArray();
            AssertStableSort(new Int32DataFrameColumn("Key", values), values, ascending, putNullValuesLast);
        }

        [Theory]
        [InlineData(true, true, false)]
        [InlineData(true, false, false)]
        [InlineData(false, true, false)]
        [InlineData(false, false, false)]
        [InlineData(true, true, true)]
        [InlineData(true, false, true)]
        [InlineData(false, true, true)]
        [InlineData(false, false, true)]
        public void TestStringColumnSortIsStable(bool ascending, bool putNullValuesLast, bool multipleBuffers)
        {
            int originalCapacity = StringDataFrameColumn.MaxCapacity;
            try
            {
                if (multipleBuffers)
                    StringDataFrameColumn.MaxCapacity = 5;

                string[] values = Enumerable.Range(0, 100).Select(i => i % 7 == 0 ? null : (i % 3).ToString()).ToArray();
                AssertStableSort(new StringDataFrameColumn("Key", values), values, ascending, putNullValuesLast);
            }
            finally
            {
                StringDataFrameColumn.MaxCapacity = originalCapacity;
            }
        }

        private static void AssertStableSort<T>(DataFrameColumn keys, T[] values, bool ascending, bool putNullValuesLast)
        {
            int[] rowIds = Enumerable.Range(0, values.Length).ToArray();
            var frame = new DataFrame(keys, new Int32DataFrameColumn("RowId", rowIds));
            IEnumerable<int> nonNullRows = rowIds.Where(i => values[i] != null);
            IEnumerable<int> orderedRows = ascending
                ? nonNullRows.OrderBy(i => values[i])
                : nonNullRows.OrderByDescending(i => values[i]);
            IEnumerable<int> nullRows = rowIds.Where(i => values[i] == null);
            IEnumerable<int> expected = putNullValuesLast ? orderedRows.Concat(nullRows) : nullRows.Concat(orderedRows);

            DataFrame sorted = frame.OrderBy("Key", ascending, putNullValuesLast);

            Assert.Equal(expected.Select(i => (int?)i), (Int32DataFrameColumn)sorted["RowId"]);
            Assert.Equal(rowIds.Select(i => (int?)i), (Int32DataFrameColumn)frame["RowId"]);
        }

        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void TestPrimitiveColumnSortIsStableForPartitionedInput(bool ascending)
        {
            var values = new List<int?>();
            for (int i = 1; i < 1000; i += 2)
            {
                values.Add(i);
                values.Add(i + 1000);
            }
            for (int i = 2; i <= 2000; i += 2)
                values.Add(i);

            int?[] repeatedValues = values.SelectMany(value => new[] { value, value }).ToArray();
            AssertStableSort(new Int32DataFrameColumn("Key", repeatedValues), repeatedValues, ascending, true);
        }

        [Fact]
        public void TestSplitAndSort()
        {
            DataFrame df = MakeDataFrameWithAllMutableColumnTypes(20);
            df.Columns["Int"][0] = 100000;
            df.Columns["Int"][df.Rows.Count - 1] = -1;
            df.Columns["Int"][5] = 200000;
            DataFrame dfTest;
            DataFrame dfTrain = SplitTrainTest(df, 0.8f, out dfTest);

            // Sort by "Int" in ascending order
            var sortedDf = dfTrain.OrderBy("Int");
            Assert.Null(sortedDf.Columns["Int"][sortedDf.Rows.Count - 1]);
            Assert.Equal(1, sortedDf.Columns["Int"][0]);
            Assert.Equal(100000, sortedDf.Columns["Int"][sortedDf.Rows.Count - 3]);
            Assert.Equal(200000, sortedDf.Columns["Int"][sortedDf.Rows.Count - 2]);
        }

        [Fact]
        public void TestStringColumnSort()
        {
            // StringDataFrameColumn specific sort tests
            StringDataFrameColumn strColumn = new StringDataFrameColumn("String", 0);
            Assert.Equal(0, strColumn.NullCount);
            for (int i = 0; i < 5; i++)
            {
                strColumn.Append(null);
            }
            Assert.Equal(5, strColumn.NullCount);
            // Should handle all nulls
            StringDataFrameColumn sortedStrColumn = strColumn.Sort() as StringDataFrameColumn;
            Assert.Equal(5, sortedStrColumn.NullCount);
            Assert.Null(sortedStrColumn[0]);

            for (int i = 0; i < 5; i++)
            {
                strColumn.Append(i.ToString());
            }
            Assert.Equal(5, strColumn.NullCount);

            // Ascending sort
            sortedStrColumn = strColumn.Sort() as StringDataFrameColumn;
            Assert.Equal("0", sortedStrColumn[0]);
            Assert.Null(sortedStrColumn[9]);

            // Descending sort
            sortedStrColumn = strColumn.Sort(false) as StringDataFrameColumn;
            Assert.Equal("4", sortedStrColumn[0]);
            Assert.Null(sortedStrColumn[9]);
        }

        [Theory]
        [InlineData(5)]
        [InlineData(12)]
        [InlineData(100)]
        [InlineData(1000)]
        public void TestPrimitiveColumnSort(int numberOfNulls)
        {
            // Primitive Column Sort
            Int32DataFrameColumn intColumn = new Int32DataFrameColumn("Int", 0);
            Assert.Equal(0, intColumn.NullCount);
            intColumn.AppendMany(null, numberOfNulls);
            Assert.Equal(numberOfNulls, intColumn.NullCount);

            // Should handle all nulls
            PrimitiveDataFrameColumn<int> sortedIntColumn = intColumn.Sort();
            Assert.Equal(numberOfNulls, sortedIntColumn.NullCount);
            Assert.Null(sortedIntColumn[0]);

            for (int i = 0; i < 5; i++)
            {
                intColumn.Append(i);
            }
            Assert.Equal(numberOfNulls, intColumn.NullCount);

            // Ascending sort
            sortedIntColumn = intColumn.Sort();
            Assert.Equal(0, sortedIntColumn[0]);
            Assert.Null(sortedIntColumn[9]);

            // Descending sort
            sortedIntColumn = intColumn.Sort(ascending: false);
            Assert.Equal(4, sortedIntColumn[0]);
            Assert.Null(sortedIntColumn[9]);
        }

        [Fact]
        public void TestSortWithDifferentNullCountsInColumns()
        {
            DataFrame dataFrame = MakeDataFrameWithAllMutableColumnTypes(10);
            dataFrame["Int"][3] = null;
            dataFrame["String"][3] = null;
            DataFrame sorted = dataFrame.OrderBy("Int");
            void Verify(DataFrame sortedDataFrame)
            {
                Assert.Equal(10, sortedDataFrame.Rows.Count);
                DataFrameRow lastRow = sortedDataFrame.Rows[sortedDataFrame.Rows.Count - 1];
                DataFrameRow penultimateRow = sortedDataFrame.Rows[sortedDataFrame.Rows.Count - 2];
                foreach (object value in lastRow)
                {
                    Assert.Null(value);
                }

                for (int i = 0; i < sortedDataFrame.Columns.Count; i++)
                {
                    string columnName = sortedDataFrame.Columns[i].Name;
                    if (columnName != "String" && columnName != "Int")
                    {
                        Assert.Equal(dataFrame[columnName][3], penultimateRow[i]);
                    }
                    else if (columnName == "String" || columnName == "Int")
                    {
                        Assert.Null(penultimateRow[i]);
                    }
                }
            }

            Verify(sorted);

            sorted = dataFrame.OrderBy("String");
            Verify(sorted);
        }
    }
}
