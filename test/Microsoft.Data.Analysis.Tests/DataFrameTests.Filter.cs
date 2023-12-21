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
        public void TestColumnFilter()
        {
            DataFrame df = MakeDataFrameWithNumericColumns(10);
            DataFrameColumn filtered = df.Columns["Int"].Filter(3, 7);
            Assert.Equal(4, filtered.Length);
            Assert.Equal(3, filtered[0]);
            Assert.Equal(4, filtered[1]);
            Assert.Equal(6, filtered[2]);
            Assert.Equal(7, filtered[3]);
        }

        [Fact]
        public void TestDataFrameFilter()
        {
            DataFrame df = MakeDataFrameWithAllMutableColumnTypes(10);
            DataFrame boolColumnFiltered = df[df.Columns["Bool"].ElementwiseEquals(true)];
            List<int> verify = new List<int> { 0, 2, 4, 6, 8 };
            Assert.Equal(5, boolColumnFiltered.Rows.Count);
            for (int i = 0; i < boolColumnFiltered.Columns.Count; i++)
            {
                DataFrameColumn column = boolColumnFiltered.Columns[i];
                if (column.Name == "Char" || column.Name == "Bool" || column.Name == "String" || column.Name == "DateTime")
                    continue;
                for (int j = 0; j < column.Length; j++)
                {
                    Assert.Equal(verify[j].ToString(), column[j].ToString());
                }
            }
            DataFrame intEnumerableFiltered = df[Enumerable.Range(0, 10)];
            DataFrame boolEnumerableFiltered = df[Enumerable.Range(0, 10).Select(x => true)];
            DataFrame longEnumerableFiltered = df[Enumerable.Range(0, 10).Select(x => (long)x)];
            Assert.Equal(intEnumerableFiltered.Columns.Count, df.Columns.Count);
            Assert.Equal(boolEnumerableFiltered.Columns.Count, df.Columns.Count);
            Assert.Equal(longEnumerableFiltered.Columns.Count, df.Columns.Count);
            for (int i = 0; i < intEnumerableFiltered.Columns.Count; i++)
            {
                DataFrameColumn intFilteredColumn = intEnumerableFiltered.Columns[i];
                DataFrameColumn dfColumn = df.Columns[i];
                DataFrameColumn boolFilteredColumn = boolEnumerableFiltered.Columns[i];
                DataFrameColumn longFilteredColumn = longEnumerableFiltered.Columns[i];
                Assert.True(intFilteredColumn.ElementwiseEquals(dfColumn).All());
                Assert.True(boolFilteredColumn.ElementwiseEquals(dfColumn).All());
                Assert.True(longFilteredColumn.ElementwiseEquals(dfColumn).All());
            }
        }
    }
}
