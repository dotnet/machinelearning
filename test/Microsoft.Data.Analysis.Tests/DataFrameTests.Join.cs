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
        public void TestJoin()
        {
            DataFrame left = MakeDataFrameWithAllMutableColumnTypes(10);
            DataFrame right = MakeDataFrameWithAllMutableColumnTypes(5);

            // Tests with right.Rows.Count < left.Rows.Count
            // Left join
            DataFrame join = left.Join(right);
            Assert.Equal(join.Rows.Count, left.Rows.Count);
            Assert.Equal(join.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Null(join.Columns["Int_right"][6]);
            VerifyJoin(join, left, right, JoinAlgorithm.Left);

            // Right join
            join = left.Join(right, joinAlgorithm: JoinAlgorithm.Right);
            Assert.Equal(join.Rows.Count, right.Rows.Count);
            Assert.Equal(join.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(join.Columns["Int_right"][3], right.Columns["Int"][3]);
            Assert.Null(join.Columns["Int_right"][2]);
            VerifyJoin(join, left, right, JoinAlgorithm.Right);

            // Outer join
            join = left.Join(right, joinAlgorithm: JoinAlgorithm.FullOuter);
            Assert.Equal(join.Rows.Count, left.Rows.Count);
            Assert.Equal(join.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Null(join.Columns["Int_right"][6]);
            VerifyJoin(join, left, right, JoinAlgorithm.FullOuter);

            // Inner join
            join = left.Join(right, joinAlgorithm: JoinAlgorithm.Inner);
            Assert.Equal(join.Rows.Count, right.Rows.Count);
            Assert.Equal(join.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(join.Columns["Int_right"][3], right.Columns["Int"][3]);
            Assert.Null(join.Columns["Int_right"][2]);
            VerifyJoin(join, left, right, JoinAlgorithm.Inner);

            // Tests with right.Rows.Count > left.Rows.Count
            // Left join
            right = MakeDataFrameWithAllMutableColumnTypes(15);
            join = left.Join(right);
            Assert.Equal(join.Rows.Count, left.Rows.Count);
            Assert.Equal(join.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(join.Columns["Int_right"][6], right.Columns["Int"][6]);
            VerifyJoin(join, left, right, JoinAlgorithm.Left);

            // Right join
            join = left.Join(right, joinAlgorithm: JoinAlgorithm.Right);
            Assert.Equal(join.Rows.Count, right.Rows.Count);
            Assert.Equal(join.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(join.Columns["Int_right"][2], right.Columns["Int"][2]);
            Assert.Null(join.Columns["Int_left"][12]);
            VerifyJoin(join, left, right, JoinAlgorithm.Right);

            // Outer join
            join = left.Join(right, joinAlgorithm: JoinAlgorithm.FullOuter);
            Assert.Equal(join.Rows.Count, right.Rows.Count);
            Assert.Equal(join.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Null(join.Columns["Int_left"][12]);
            VerifyJoin(join, left, right, JoinAlgorithm.FullOuter);

            // Inner join
            join = left.Join(right, joinAlgorithm: JoinAlgorithm.Inner);
            Assert.Equal(join.Rows.Count, left.Rows.Count);
            Assert.Equal(join.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(join.Columns["Int_right"][2], right.Columns["Int"][2]);
            VerifyJoin(join, left, right, JoinAlgorithm.Inner);
        }

        private void VerifyJoin(DataFrame join, DataFrame left, DataFrame right, JoinAlgorithm joinAlgorithm)
        {
            Int64DataFrameColumn mapIndices = new Int64DataFrameColumn("map", join.Rows.Count);
            for (long i = 0; i < join.Rows.Count; i++)
            {
                mapIndices[i] = i;
            }
            for (int i = 0; i < join.Columns.Count; i++)
            {
                DataFrameColumn joinColumn = join.Columns[i];
                DataFrameColumn isEqual;

                if (joinAlgorithm == JoinAlgorithm.Left)
                {
                    if (i < left.Columns.Count)
                    {
                        DataFrameColumn leftColumn = left.Columns[i];
                        isEqual = joinColumn.ElementwiseEquals(leftColumn);
                    }
                    else
                    {
                        int columnIndex = i - left.Columns.Count;
                        DataFrameColumn rightColumn = right.Columns[columnIndex];
                        DataFrameColumn compareColumn = rightColumn.Length <= join.Rows.Count ? rightColumn.Clone(numberOfNullsToAppend: join.Rows.Count - rightColumn.Length) : rightColumn.Clone(mapIndices);
                        isEqual = joinColumn.ElementwiseEquals(compareColumn);
                    }
                }
                else if (joinAlgorithm == JoinAlgorithm.Right)
                {
                    if (i < left.Columns.Count)
                    {
                        DataFrameColumn leftColumn = left.Columns[i];
                        DataFrameColumn compareColumn = leftColumn.Length <= join.Rows.Count ? leftColumn.Clone(numberOfNullsToAppend: join.Rows.Count - leftColumn.Length) : leftColumn.Clone(mapIndices);
                        isEqual = joinColumn.ElementwiseEquals(compareColumn);
                    }
                    else
                    {
                        int columnIndex = i - left.Columns.Count;
                        DataFrameColumn rightColumn = right.Columns[columnIndex];
                        isEqual = joinColumn.ElementwiseEquals(rightColumn);
                    }
                }
                else if (joinAlgorithm == JoinAlgorithm.Inner)
                {
                    if (i < left.Columns.Count)
                    {
                        DataFrameColumn leftColumn = left.Columns[i];
                        isEqual = joinColumn.ElementwiseEquals(leftColumn.Clone(mapIndices));
                    }
                    else
                    {
                        int columnIndex = i - left.Columns.Count;
                        DataFrameColumn rightColumn = right.Columns[columnIndex];
                        isEqual = joinColumn.ElementwiseEquals(rightColumn.Clone(mapIndices));
                    }
                }
                else
                {
                    if (i < left.Columns.Count)
                    {
                        DataFrameColumn leftColumn = left.Columns[i];
                        isEqual = joinColumn.ElementwiseEquals(leftColumn.Clone(numberOfNullsToAppend: join.Rows.Count - leftColumn.Length));
                    }
                    else
                    {
                        int columnIndex = i - left.Columns.Count;
                        DataFrameColumn rightColumn = right.Columns[columnIndex];
                        isEqual = joinColumn.ElementwiseEquals(rightColumn.Clone(numberOfNullsToAppend: join.Rows.Count - rightColumn.Length));
                    }
                }
                for (int j = 0; j < join.Rows.Count; j++)
                {
                    Assert.Equal(true, isEqual[j]);
                }
            }
        }
    }
}
