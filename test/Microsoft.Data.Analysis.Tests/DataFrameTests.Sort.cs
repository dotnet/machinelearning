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

        [Fact]
        public void TestOrderBy_StableSort_PreservesOriginalOrder()
        {
            // Reproduces issue #6443: rows with equal sort keys should preserve original order
            var colA = new Int32DataFrameColumn("A", new int[] { 9, 6, 6, 3, 6, 3, 3, 6 });
            var colB = new Int32DataFrameColumn("B", new int[] { 18, 11, 10, 16, 13, 19, 11, 17 });
            var colC = new Int32DataFrameColumn("C", new int[] { 28, 25, 23, 26, 21, 22, 28, 20 });
            var df = new DataFrame(colA, colB, colC);

            DataFrame sorted = df.OrderBy("A");

            // A=3 rows should preserve original order: B=16, B=19, B=11
            Assert.Equal(3, sorted.Columns["A"][0]);
            Assert.Equal(16, sorted.Columns["B"][0]);
            Assert.Equal(3, sorted.Columns["A"][1]);
            Assert.Equal(19, sorted.Columns["B"][1]);
            Assert.Equal(3, sorted.Columns["A"][2]);
            Assert.Equal(11, sorted.Columns["B"][2]);

            // A=6 rows should preserve original order: B=11, B=10, B=13, B=17
            Assert.Equal(6, sorted.Columns["A"][3]);
            Assert.Equal(11, sorted.Columns["B"][3]);
            Assert.Equal(6, sorted.Columns["A"][4]);
            Assert.Equal(10, sorted.Columns["B"][4]);
            Assert.Equal(6, sorted.Columns["A"][5]);
            Assert.Equal(13, sorted.Columns["B"][5]);
            Assert.Equal(6, sorted.Columns["A"][6]);
            Assert.Equal(17, sorted.Columns["B"][6]);

            // A=9 row
            Assert.Equal(9, sorted.Columns["A"][7]);
            Assert.Equal(18, sorted.Columns["B"][7]);
        }

        [Fact]
        public void TestOrderByDescending_StableSort_PreservesOriginalOrder()
        {
            var colA = new Int32DataFrameColumn("A", new int[] { 1, 2, 1, 2, 1 });
            var colB = new StringDataFrameColumn("B", new[] { "first", "second", "third", "fourth", "fifth" });
            var df = new DataFrame(colA, colB);

            DataFrame sorted = df.OrderByDescending("A");

            // A=2 rows first (descending), preserving original order
            Assert.Equal(2, sorted.Columns["A"][0]);
            Assert.Equal("second", sorted.Columns["B"][0]);
            Assert.Equal(2, sorted.Columns["A"][1]);
            Assert.Equal("fourth", sorted.Columns["B"][1]);

            // A=1 rows next, preserving original order
            Assert.Equal(1, sorted.Columns["A"][2]);
            Assert.Equal("first", sorted.Columns["B"][2]);
            Assert.Equal(1, sorted.Columns["A"][3]);
            Assert.Equal("third", sorted.Columns["B"][3]);
            Assert.Equal(1, sorted.Columns["A"][4]);
            Assert.Equal("fifth", sorted.Columns["B"][4]);
        }

        [Fact]
        public void TestOrderBy_StableSort_WithNullsAndDuplicates()
        {
            var colA = new Int32DataFrameColumn("A", 6);
            colA[0] = 1;
            colA[1] = null;
            colA[2] = 1;
            colA[3] = null;
            colA[4] = 1;
            colA[5] = 2;
            var colB = new StringDataFrameColumn("B", new[] { "a", "b", "c", "d", "e", "f" });
            var df = new DataFrame(colA, colB);

            DataFrame sorted = df.OrderBy("A");

            // A=1 rows should preserve original order: a, c, e
            Assert.Equal(1, sorted.Columns["A"][0]);
            Assert.Equal("a", sorted.Columns["B"][0]);
            Assert.Equal(1, sorted.Columns["A"][1]);
            Assert.Equal("c", sorted.Columns["B"][1]);
            Assert.Equal(1, sorted.Columns["A"][2]);
            Assert.Equal("e", sorted.Columns["B"][2]);

            // A=2
            Assert.Equal(2, sorted.Columns["A"][3]);
            Assert.Equal("f", sorted.Columns["B"][3]);

            // Nulls at the end
            Assert.Null(sorted.Columns["A"][4]);
            Assert.Null(sorted.Columns["A"][5]);
        }

        [Fact]
        public void TestStringColumnSort_StableSort()
        {
            var strCol = new StringDataFrameColumn("Key", new[] { "b", "a", "b", "a", "b" });
            var idCol = new Int32DataFrameColumn("ID", new int[] { 1, 2, 3, 4, 5 });
            var df = new DataFrame(strCol, idCol);

            DataFrame sorted = df.OrderBy("Key");

            // "a" rows preserve order: ID=2, ID=4
            Assert.Equal("a", sorted.Columns["Key"][0]);
            Assert.Equal(2, sorted.Columns["ID"][0]);
            Assert.Equal("a", sorted.Columns["Key"][1]);
            Assert.Equal(4, sorted.Columns["ID"][1]);

            // "b" rows preserve order: ID=1, ID=3, ID=5
            Assert.Equal("b", sorted.Columns["Key"][2]);
            Assert.Equal(1, sorted.Columns["ID"][2]);
            Assert.Equal("b", sorted.Columns["Key"][3]);
            Assert.Equal(3, sorted.Columns["ID"][3]);
            Assert.Equal("b", sorted.Columns["Key"][4]);
            Assert.Equal(5, sorted.Columns["ID"][4]);
        }

        [Fact]
        public void TestOrderBy_StableSort_ManyDuplicates()
        {
            // Regression test for a previously unstable OrderBy implementation that
            // used quicksort on larger partitions. Uses many duplicate keys to hit
            // the scenario where instability used to occur and verify stability now.
            int rowCount = 100;
            var keyValues = new int[rowCount];
            var idValues = new int[rowCount];
            for (int i = 0; i < rowCount; i++)
            {
                keyValues[i] = i % 5; // Only 5 distinct keys → many duplicates
                idValues[i] = i;       // Original position tracker
            }
            var keyCol = new Int32DataFrameColumn("Key", keyValues);
            var idCol = new Int32DataFrameColumn("ID", idValues);
            var df = new DataFrame(keyCol, idCol);

            DataFrame sorted = df.OrderBy("Key");

            // Verify that within each key group, IDs are still in ascending order
            int prevKey = -1;
            int prevId = -1;
            for (int i = 0; i < rowCount; i++)
            {
                int key = (int)sorted.Columns["Key"][i];
                int id = (int)sorted.Columns["ID"][i];

                if (key == prevKey)
                {
                    // Same key group: ID must be greater than previous (stable order)
                    Assert.True(id > prevId,
                        $"Unstable sort at row {i}: Key={key}, ID={id} should be > previous ID={prevId}");
                }
                prevKey = key;
                prevId = id;
            }
            // Note: multi-buffer stability (columns with >536M int elements) cannot be
            // practically tested here. The per-buffer sort is stable (merge sort), and
            // the cross-buffer merge uses a SortedDictionary-based k-way merge. If future
            // changes reduce buffer capacity, consider adding a multi-buffer stability test.
        }
    }
}
