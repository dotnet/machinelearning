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
        [InlineData(1, 2)]
        [InlineData(2, 1)]
        public void TestDataCorrectnessForInnerMerge(int leftCount, int rightCount)
        {
            DataFrame left = MakeDataFrameWithNumericColumns(leftCount, false);
            DataFrameColumn leftStringColumn = new StringDataFrameColumn("String", Enumerable.Range(0, leftCount).Select(x => "Left"));
            left.Columns.Insert(left.Columns.Count, leftStringColumn);

            DataFrame right = MakeDataFrameWithNumericColumns(rightCount, false);
            DataFrameColumn rightStringColumn = new StringDataFrameColumn("String", Enumerable.Range(0, rightCount).Select(x => "Right"));
            right.Columns.Insert(right.Columns.Count, rightStringColumn);

            DataFrame merge = left.Merge<int>(right, "Int", "Int", joinAlgorithm: JoinAlgorithm.Inner);

            Assert.Equal("Left", (string)merge.Columns["String_left"][0]);
            Assert.Equal("Right", (string)merge.Columns["String_right"][0]);
        }

        [Fact]
        public void TestMerge()
        {
            DataFrame left = MakeDataFrameWithAllMutableColumnTypes(10);
            DataFrame right = MakeDataFrameWithAllMutableColumnTypes(5);

            // Tests with right.Rows.Count < left.Rows.Count
            // Left merge
            DataFrame merge = left.Merge<int>(right, "Int", "Int");
            Assert.Equal(10, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Null(merge.Columns["Int_right"][6]);
            Assert.Null(merge.Columns["Int_left"][5]);
            VerifyMerge(merge, left, right, JoinAlgorithm.Left);

            // Right merge
            merge = left.Merge<int>(right, "Int", "Int", joinAlgorithm: JoinAlgorithm.Right);
            Assert.Equal(5, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(merge.Columns["Int_right"][3], right.Columns["Int"][3]);
            Assert.Null(merge.Columns["Int_right"][2]);
            VerifyMerge(merge, left, right, JoinAlgorithm.Right);

            // Outer merge
            merge = left.Merge<int>(right, "Int", "Int", joinAlgorithm: JoinAlgorithm.FullOuter);
            Assert.Equal(merge.Rows.Count, left.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Null(merge.Columns["Int_right"][6]);
            VerifyMerge(merge, left, right, JoinAlgorithm.FullOuter);

            // Inner merge
            merge = left.Merge<int>(right, "Int", "Int", joinAlgorithm: JoinAlgorithm.Inner);
            Assert.Equal(merge.Rows.Count, right.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(merge.Columns["Int_right"][2], right.Columns["Int"][3]);
            Assert.Null(merge.Columns["Int_right"][4]);
            VerifyMerge(merge, left, right, JoinAlgorithm.Inner);

            // Tests with right.Rows.Count > left.Rows.Count
            // Left merge
            right = MakeDataFrameWithAllMutableColumnTypes(15);
            merge = left.Merge<int>(right, "Int", "Int");
            Assert.Equal(merge.Rows.Count, left.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(merge.Columns["Int_right"][6], right.Columns["Int"][6]);
            VerifyMerge(merge, left, right, JoinAlgorithm.Left);

            // Right merge
            merge = left.Merge<int>(right, "Int", "Int", joinAlgorithm: JoinAlgorithm.Right);
            Assert.Equal(merge.Rows.Count, right.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(merge.Columns["Int_right"][2], right.Columns["Int"][2]);
            Assert.Null(merge.Columns["Int_left"][12]);
            VerifyMerge(merge, left, right, JoinAlgorithm.Right);

            // Outer merge
            merge = left.Merge<int>(right, "Int", "Int", joinAlgorithm: JoinAlgorithm.FullOuter);
            Assert.Equal(16, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Null(merge.Columns["Int_left"][12]);
            Assert.Null(merge.Columns["Int_left"][15]);
            VerifyMerge(merge, left, right, JoinAlgorithm.FullOuter);

            // Inner merge
            merge = left.Merge<int>(right, "Int", "Int", joinAlgorithm: JoinAlgorithm.Inner);
            Assert.Equal(9, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(merge.Columns["Int_right"][2], right.Columns["Int"][2]);
            VerifyMerge(merge, left, right, JoinAlgorithm.Inner);
        }

        private void MatchRowsOnMergedDataFrame(DataFrame merge, DataFrame left, DataFrame right, long mergeRow, long? leftRow, long? rightRow)
        {
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            DataFrameRow dataFrameMergeRow = merge.Rows[mergeRow];
            int columnIndex = 0;
            foreach (object value in dataFrameMergeRow)
            {
                object compare = null;
                if (columnIndex < left.Columns.Count)
                {
                    if (leftRow != null)
                    {
                        compare = left.Rows[leftRow.Value][columnIndex];
                    }
                }
                else
                {
                    int rightColumnIndex = columnIndex - left.Columns.Count;
                    if (rightRow != null)
                    {
                        compare = right.Rows[rightRow.Value][rightColumnIndex];
                    }
                }
                Assert.Equal(value, compare);
                columnIndex++;
            }
        }

        [Theory]
        [InlineData(10, 5, JoinAlgorithm.Left)]
        [InlineData(5, 10, JoinAlgorithm.Right)]
        public void TestMergeEdgeCases_LeftOrRight(int leftLength, int rightLength, JoinAlgorithm joinAlgorithm)
        {
            DataFrame left = MakeDataFrameWithAllMutableColumnTypes(leftLength);
            if (leftLength > 5)
            {
                left["Int"][8] = null;
            }
            DataFrame right = MakeDataFrameWithAllMutableColumnTypes(rightLength);
            if (rightLength > 5)
            {
                right["Int"][8] = null;
            }

            DataFrame merge = left.Merge<int>(right, "Int", "Int", joinAlgorithm: joinAlgorithm);
            Assert.Equal(10, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            int[] matchedFullRows = new int[] { 0, 1, 3, 4 };
            for (long i = 0; i < matchedFullRows.Length; i++)
            {
                int rowIndex = matchedFullRows[i];
                MatchRowsOnMergedDataFrame(merge, left, right, rowIndex, rowIndex, rowIndex);
            }

            int[] matchedLeftOrRightRowsNullOtherRows = new int[] { 2, 5, 6, 7, 8, 9 };
            for (long i = 0; i < matchedLeftOrRightRowsNullOtherRows.Length; i++)
            {
                int rowIndex = matchedLeftOrRightRowsNullOtherRows[i];
                if (leftLength > 5)
                {
                    MatchRowsOnMergedDataFrame(merge, left, right, rowIndex, rowIndex, null);
                }
                else
                {
                    MatchRowsOnMergedDataFrame(merge, left, right, rowIndex, null, rowIndex);
                }
            }
        }

        [Fact]
        public void TestMergeEdgeCases_Inner()
        {
            DataFrame left = MakeDataFrameWithAllMutableColumnTypes(5);
            DataFrame right = MakeDataFrameWithAllMutableColumnTypes(10);
            left["Int"][3] = null;
            right["Int"][6] = null;
            // Creates this case:
            /*
             * Left:    Right:
             * 0        0
             * 1        1
             * null(2)  2
             * null(3)  3
             * 4        4
             *          null(5)
             *          null(6)
             *          7
             *          8
             *          9
             */
            /*
             * Merge will result in a DataFrame like:
             * Int_Left Int_Right
             * 0        0
             * 1        1
             * 4        4
             * null(2)  null(5)
             * null(3)  null(5)
             * null(2)  null(6)
             * null(3)  null(6)
             */

            DataFrame merge = left.Merge<int>(right, "Int", "Int", joinAlgorithm: JoinAlgorithm.Inner);
            Assert.Equal(7, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);

            int[] mergeRows = new int[] { 0, 1, 2, 3, 4, 5, 6 };
            int[] leftRows = new int[] { 0, 1, 4, 2, 3, 2, 3 };
            int[] rightRows = new int[] { 0, 1, 4, 5, 5, 6, 6 };
            for (long i = 0; i < mergeRows.Length; i++)
            {
                int rowIndex = mergeRows[i];
                int leftRowIndex = leftRows[i];
                int rightRowIndex = rightRows[i];
                MatchRowsOnMergedDataFrame(merge, left, right, rowIndex, leftRowIndex, rightRowIndex);
            }
        }

        [Fact]
        public void TestMergeEdgeCases_Outer()
        {
            DataFrame left = MakeDataFrameWithAllMutableColumnTypes(5);
            left["Int"][3] = null;
            DataFrame right = MakeDataFrameWithAllMutableColumnTypes(5);
            right["Int"][1] = 5;
            right["Int"][3] = null;
            right["Int"][4] = 6;

            // Creates this case:
            /*
             * Left:    Right:    RowIndex:
             * 0        0         0
             * 1        5         1
             * null     null      2
             * null(3)  null(3)   3
             * 4        6         4
             */

            /*
             * Merge will result in a DataFrame like:
             * Int_left:    Int_right:        Merged:    Index:
             * 0            0                 0 - 0      0
             * 1            null              1 - N      1
             * null         null              2 - 2      2
             * null         null(3)           2 - 3      3
             * null(3)      null              3 - 2      4
             * null(3)      null(3)           3 - 3      5
             * 4            null              4 - N      6
             * null         5                 N - 1      7
             * null         6                 N - 4      8
             */

            DataFrame merge = left.Merge<int>(right, "Int", "Int", joinAlgorithm: JoinAlgorithm.FullOuter);
            Assert.Equal(9, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);

            int[] mergeRows = new int[] { 0, 2, 3, 4, 5 };
            int[] leftRows = new int[] { 0, 2, 2, 3, 3 };
            int[] rightRows = new int[] { 0, 2, 3, 2, 3 };
            for (long i = 0; i < mergeRows.Length; i++)
            {
                int rowIndex = mergeRows[i];
                int leftRowIndex = leftRows[i];
                int rightRowIndex = rightRows[i];
                MatchRowsOnMergedDataFrame(merge, left, right, rowIndex, leftRowIndex, rightRowIndex);
            }

            mergeRows = new int[] { 1, 6 };
            leftRows = new int[] { 1, 4 };
            for (long i = 0; i < mergeRows.Length; i++)
            {
                int rowIndex = mergeRows[i];
                int leftRowIndex = leftRows[i];
                MatchRowsOnMergedDataFrame(merge, left, right, rowIndex, leftRowIndex, null);
            }

            mergeRows = new int[] { 7, 8 };
            rightRows = new int[] { 1, 4 };
            for (long i = 0; i < mergeRows.Length; i++)
            {
                int rowIndex = mergeRows[i];
                int rightRowIndex = rightRows[i];
                MatchRowsOnMergedDataFrame(merge, left, right, rowIndex, null, rightRowIndex);
            }
        }

        [Fact]
        public void TestMerge_ByTwoColumns_Complex_LeftJoin()
        {
            //Test left merge by to int type columns

            //Arrange
            var left = new DataFrame();
            left.Columns.Add(new Int32DataFrameColumn("Index", new[] { 0, 1, 2, 3, 4, 5 }));
            left.Columns.Add(new Int32DataFrameColumn("G1", new[] { 0, 1, 1, 2, 2, 3 }));
            left.Columns.Add(new Int32DataFrameColumn("G2", new[] { 3, 1, 2, 1, 2, 1 }));

            var right = new DataFrame();
            right.Columns.Add(new Int32DataFrameColumn("Index", new[] { 0, 1, 2, 3 }));
            right.Columns.Add(new Int32DataFrameColumn("G1", new[] { 1, 1, 1, 2 }));
            right.Columns.Add(new Int32DataFrameColumn("G2", new[] { 1, 2, 1, 1 }));

            // Creates this case:
            /*  -------------------------
             *     Left     |     Right
             *   I  G1 G2   |   I  G1 G2
             *  -------------------------
             *   0  0  3    |   0  1  1
             *   1  1  1    |   1  1  2
             *   2  1  2    |   2  1  1
             *   3  2  1    |   3  2  1
             *   4  2  2
             *   5  3  1
             */

            /*
             * Merge will result in a DataFrame like:
             *   IL G1 G2     IR              Merged:
             *  -------------------------
             *   0  0  3                      0 - N
             *   1  1  1       0  1  1        1 - 0
             *   1  1  1       2  1  1        1 - 2
             *   2  1  2       1  1  2        2 - 1
             *   3  2  1       3  2  1        3 - 3
             *   4  2  2                      4 - N
             *   5  3  1                      5 - N
             */

            //Act
            var merge = left.Merge(right, new[] { "G1", "G2" }, new[] { "G1", "G2" });

            //Assert
            var expectedMerged = new (int? Left, int? Right)[] {
                (0, null),
                (1, 0),
                (1, 2),
                (2, 1),
                (3, 3),
                (4, null),
                (5, null)
            };

            Assert.Equal(expectedMerged.Length, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);

            for (long i = 0; i < expectedMerged.Length; i++)
            {
                MatchRowsOnMergedDataFrame(merge, left, right, i, expectedMerged[i].Left, expectedMerged[i].Right);
            }

        }

        [Fact]
        public void TestMerge_ByTwoColumns_Simple_ManyToMany_LeftJoin()
        {
            //Test left merge by to int type columns

            //Arrange
            var left = new DataFrame();
            left.Columns.Add(new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }));
            left.Columns.Add(new Int32DataFrameColumn("G1", new[] { 1, 1, 3 }));
            left.Columns.Add(new Int32DataFrameColumn("G2", new[] { 1, 1, 3 }));

            var right = new DataFrame();
            right.Columns.Add(new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }));
            right.Columns.Add(new Int32DataFrameColumn("G1", new[] { 1, 1, 0 }));
            right.Columns.Add(new Int32DataFrameColumn("G2", new[] { 1, 1, 0 }));

            // Creates this case:
            /*  ---------------------------
             *     Left    |    Right
             *   I  G1 G2  |   I  G1 G2
             *  ---------------------------
             *   0  1  1   |   0  1  1
             *   1  1  1   |   1  1  1
             *   2  3  3   |   2  0  0
             */

            /*
             * Merge will result in a DataFrame like:
             *   IL G1 G2     IR           Merged:
             *  -------------------------
             *   0  1  1      0  1  1       0 - 0
             *   0  1  1      1  1  1       0 - 1
             *   1  1  1      0  1  1       1 - 0
             *   1  1  1      1  1  1       1 - 1
             *   2  3  3                    2 - N
             */

            //Act
            var merge = left.Merge(right, new[] { "G1", "G2" }, new[] { "G1", "G2" });

            //Assert
            var expectedMerged = new (int? Left, int? Right)[] {
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (2, null)
            };

            Assert.Equal(expectedMerged.Length, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);

            for (long i = 0; i < expectedMerged.Length; i++)
            {
                MatchRowsOnMergedDataFrame(merge, left, right, i, expectedMerged[i].Left, expectedMerged[i].Right);
            }
        }

        [Fact]
        public void TestMerge_ByTwoColumns_Simple_ManyToMany_RightJoin()
        {
            //Test left merge by to int type columns

            //Arrange
            var left = new DataFrame();
            left.Columns.Add(new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }));
            left.Columns.Add(new Int32DataFrameColumn("G1", new[] { 1, 1, 3 }));
            left.Columns.Add(new Int32DataFrameColumn("G2", new[] { 1, 1, 3 }));

            var right = new DataFrame();
            right.Columns.Add(new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }));
            right.Columns.Add(new Int32DataFrameColumn("G1", new[] { 1, 1, 0 }));
            right.Columns.Add(new Int32DataFrameColumn("G2", new[] { 1, 1, 0 }));

            // Creates this case:
            /*  ---------------------------
             *     Left    |    Right
             *   I  G1 G2  |   I  G1 G2
             *  ---------------------------
             *   0  1  1   |   0  1  1
             *   1  1  1   |   1  1  1
             *   2  3  3   |   2  0  0
             */

            /*
             * Merge will result in a DataFrame like:
             *   IL G1 G2     IR           Merged:
             *  -------------------------
             *   0  1  1      0  1  1       0 - 0
             *   1  1  1      0  1  1       1 - 0
             *   0  1  1      1  1  1       0 - 1
             *   1  1  1      1  1  1       1 - 1
             *                2  0  0       N - 2
             */

            //Act
            var merge = left.Merge(right, new[] { "G1", "G2" }, new[] { "G1", "G2" }, joinAlgorithm: JoinAlgorithm.Right);

            //Assert
            var expectedMerged = new (int? Left, int? Right)[] {
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
                (null, 2)
            };

            Assert.Equal(expectedMerged.Length, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);

            for (long i = 0; i < expectedMerged.Length; i++)
            {
                MatchRowsOnMergedDataFrame(merge, left, right, i, expectedMerged[i].Left, expectedMerged[i].Right);
            }
        }

        [Fact]
        public void TestMerge_ByTwoColumns_Simple_ManyToMany_InnerJoin()
        {
            //Test left merge by to int type columns

            //Arrange
            var left = new DataFrame();
            left.Columns.Add(new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }));
            left.Columns.Add(new Int32DataFrameColumn("G1", new[] { 1, 1, 3 }));
            left.Columns.Add(new Int32DataFrameColumn("G2", new[] { 1, 1, 3 }));

            var right = new DataFrame();
            right.Columns.Add(new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }));
            right.Columns.Add(new Int32DataFrameColumn("G1", new[] { 1, 1, 0 }));
            right.Columns.Add(new Int32DataFrameColumn("G2", new[] { 1, 1, 0 }));

            // Creates this case:
            /*  ---------------------------
             *     Left    |    Right
             *   I  G1 G2  |   I  G1 G2
             *  ---------------------------
             *   0  1  1   |   0  1  1
             *   1  1  1   |   1  1  1
             *   2  3  3   |   2  0  0
             */

            /*
             * Merge will result in a DataFrame like:
             *   IL G1 G2     IR           Merged:
             *  -------------------------
             *   0  1  1      0  1  1       0 - 0
             *   1  1  1      0  1  1       1 - 0
             *   0  1  1      1  1  1       0 - 1
             *   1  1  1      1  1  1       1 - 1
             */

            //Act
            var merge = left.Merge(right, new[] { "G1", "G2" }, new[] { "G1", "G2" }, joinAlgorithm: JoinAlgorithm.Inner);

            //Assert
            var expectedMerged = new (int? Left, int? Right)[] {
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1)
            };

            Assert.Equal(expectedMerged.Length, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);

            for (long i = 0; i < expectedMerged.Length; i++)
            {
                MatchRowsOnMergedDataFrame(merge, left, right, i, expectedMerged[i].Left, expectedMerged[i].Right);
            }
        }

        [Fact]
        public void TestMerge_ByTwoColumns_Simple_ManyToMany_OuterJoin()
        {
            //Test left merge by to int type columns

            //Arrange
            var left = new DataFrame();
            left.Columns.Add(new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }));
            left.Columns.Add(new Int32DataFrameColumn("G1", new[] { 1, 1, 3 }));
            left.Columns.Add(new Int32DataFrameColumn("G2", new[] { 1, 1, 3 }));

            var right = new DataFrame();
            right.Columns.Add(new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }));
            right.Columns.Add(new Int32DataFrameColumn("G1", new[] { 1, 1, 0 }));
            right.Columns.Add(new Int32DataFrameColumn("G2", new[] { 1, 1, 0 }));

            // Creates this case:
            /*  ---------------------------
             *     Left    |    Right
             *   I  G1 G2  |   I  G1 G2
             *  ---------------------------
             *   0  1  1   |   0  1  1
             *   1  1  1   |   1  1  1
             *   2  3  3   |   2  0  0
             */

            /*
             * Merge will result in a DataFrame like:
             *   IL G1 G2     IR           Merged:
             *  -------------------------
             *   0  1  1      0  1  1       0 - 0
             *   0  1  1      1  1  1       0 - 1
             *   1  1  1      0  1  1       1 - 0
             *   1  1  1      1  1  1       1 - 1
             *   2  3  3                    2 - N
             *                2  0  0       N - 2
             */

            //Act
            var merge = left.Merge(right, new[] { "G1", "G2" }, new[] { "G1", "G2" }, joinAlgorithm: JoinAlgorithm.FullOuter);

            //Assert
            var expectedMerged = new (int? Left, int? Right)[] {
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (2, null),
                (null, 2)
            };

            Assert.Equal(expectedMerged.Length, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);

            for (long i = 0; i < expectedMerged.Length; i++)
            {
                MatchRowsOnMergedDataFrame(merge, left, right, i, expectedMerged[i].Left, expectedMerged[i].Right);
            }
        }

        [Fact]
        public void TestMerge_ByThreeColumns_OneToOne_LeftJoin()
        {
            //Test merge by LEFT join of int and string columns

            //Arrange
            var left = new DataFrame();
            left.Columns.Add(new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }));
            left.Columns.Add(new Int32DataFrameColumn("G1", new[] { 1, 1, 2 }));
            left.Columns.Add(new Int32DataFrameColumn("G2", new[] { 1, 2, 1 }));
            left.Columns.Add(new StringDataFrameColumn("G3", new[] { "A", "B", "C" }));

            var right = new DataFrame();
            right.Columns.Add(new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }));
            right.Columns.Add(new Int32DataFrameColumn("G1", new[] { 0, 1, 1 }));
            right.Columns.Add(new Int32DataFrameColumn("G2", new[] { 1, 1, 2 }));
            right.Columns.Add(new StringDataFrameColumn("G3", new[] { "Z", "Y", "B" }));

            // Creates this case:
            /*  -----------------------------
             *      Left      |      Right
             *   I  G1 G2 G3  |   I  G1 G2 G3
             *  ------------------------------
             *   0  1  1  A   |   0  0  1  Z
             *   1  1  2  B   |   1  1  1  Y
             *   2  2  1  C   |   2  1  2  B
             */

            /*
             * Merge will result in a DataFrame like:
             *   IL G1 G2 G3    IR              Merged:
             *  -------------------------
             *   0  1  1  A                      0 - N
             *   1  1  2  B     2  1  2  B       1 - 2
             *   2  2  1  C                      2 - N
             */

            //Act
            var merge = left.Merge(right, new[] { "G1", "G2", "G3" }, new[] { "G1", "G2", "G3" });

            //Assert
            var expectedMerged = new (int? Left, int? Right)[] {
                (0, null),
                (1, 2),
                (2, null)
            };

            Assert.Equal(expectedMerged.Length, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);

            for (long i = 0; i < expectedMerged.Length; i++)
            {
                MatchRowsOnMergedDataFrame(merge, left, right, i, expectedMerged[i].Left, expectedMerged[i].Right);
            }
        }

        [Fact]
        public void TestMerge_ByThreeColumns_OneToOne_RightJoin()
        {
            //Test merge by RIGHT join of int and string columns

            //Arrange
            var left = new DataFrame();
            left.Columns.Add(new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }));
            left.Columns.Add(new Int32DataFrameColumn("G1", new[] { 1, 1, 2 }));
            left.Columns.Add(new Int32DataFrameColumn("G2", new[] { 1, 2, 1 }));
            left.Columns.Add(new StringDataFrameColumn("G3", new[] { "A", "B", "C" }));

            var right = new DataFrame();
            right.Columns.Add(new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }));
            right.Columns.Add(new Int32DataFrameColumn("G1", new[] { 0, 1, 1 }));
            right.Columns.Add(new Int32DataFrameColumn("G2", new[] { 1, 1, 2 }));
            right.Columns.Add(new StringDataFrameColumn("G3", new[] { "Z", "Y", "B" }));

            // Creates this case:
            /*  -----------------------------
             *      Left      |      Right
             *   I  G1 G2 G3  |   I  G1 G2 G3
             *  ------------------------------
             *   0  1  1  A   |   0  0  1  Z
             *   1  1  2  B   |   1  1  1  Y
             *   2  2  1  C   |   2  1  2  B
             */

            /*
             * Merge will result in a DataFrame like:
             *   IL G1 G2 G3    IR              Merged:
             *  -------------------------
             *                  0  0  1  Z       N - 0
             *                  1  1  1  Y       N - 1
             *   1  1  2  B     2  1  2  B       1 - 2
             */

            //Act
            var merge = left.Merge(right, new[] { "G1", "G2", "G3" }, new[] { "G1", "G2", "G3" }, joinAlgorithm: JoinAlgorithm.Right);

            //Assert
            var expectedMerged = new (int? Left, int? Right)[] {
                (null, 0),
                (null, 1),
                (1, 2)
            };

            Assert.Equal(expectedMerged.Length, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);

            for (long i = 0; i < expectedMerged.Length; i++)
            {
                MatchRowsOnMergedDataFrame(merge, left, right, i, expectedMerged[i].Left, expectedMerged[i].Right);
            }
        }

        [Fact]
        public void TestMerge_Issue5778()
        {
            DataFrame left = MakeDataFrameWithAllMutableColumnTypes(2, false);
            DataFrame right = MakeDataFrameWithAllMutableColumnTypes(1);

            DataFrame merge = left.Merge<int>(right, "Int", "Int");

            Assert.Equal(2, merge.Rows.Count);
            Assert.Equal(0, (int)merge.Columns["Int_left"][0]);
            Assert.Equal(1, (int)merge.Columns["Int_left"][1]);
            MatchRowsOnMergedDataFrame(merge, left, right, 0, 0, 0);
            MatchRowsOnMergedDataFrame(merge, left, right, 1, 1, 0);
        }

        public static IEnumerable<object[]> GenerateData_TestMerge_EmptyDataFrames()
        {
            yield return new object[]
                {
                    new DataFrame(
                        new Int32DataFrameColumn("Index"),
                        new Int32DataFrameColumn("L1"),
                        new Int32DataFrameColumn("L2"),
                        new StringDataFrameColumn("L3")
                        ),
                    new DataFrame(
                        new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }),
                        new Int32DataFrameColumn("R1", new[] { 0, 1, 1 }),
                        new Int32DataFrameColumn("R2", new[] { 1, 1, 2 }),
                        new StringDataFrameColumn("R3", new[] { "Z", "Y", "B" })
                        ),
                    new string[]{ "L1" },
                    new string[]{ "R1" },
                    JoinAlgorithm.Left,
                    new DataFrame(
                        new Int32DataFrameColumn("Index_left"),
                        new Int32DataFrameColumn("L1"),
                        new Int32DataFrameColumn("L2"),
                        new StringDataFrameColumn("L3"),
                        new Int32DataFrameColumn("Index_right"),
                        new Int32DataFrameColumn("R1"),
                        new Int32DataFrameColumn("R2"),
                        new StringDataFrameColumn("R3")
                        ),
                };
            yield return new object[]
                {
                    new DataFrame(
                        new Int32DataFrameColumn("Index"),
                        new Int32DataFrameColumn("L1"),
                        new Int32DataFrameColumn("L2"),
                        new StringDataFrameColumn("L3")
                        ),
                    new DataFrame(
                        new Int32DataFrameColumn("Index"),
                        new Int32DataFrameColumn("R1"),
                        new Int32DataFrameColumn("R2"),
                        new StringDataFrameColumn("R3")
                        ),
                    new string[]{ "L1" },
                    new string[]{ "R1" },
                    JoinAlgorithm.Inner,
                    new DataFrame(
                        new Int32DataFrameColumn("Index_left"),
                        new Int32DataFrameColumn("L1"),
                        new Int32DataFrameColumn("L2"),
                        new StringDataFrameColumn("L3"),
                        new Int32DataFrameColumn("Index_right"),
                        new Int32DataFrameColumn("R1"),
                        new Int32DataFrameColumn("R2"),
                        new StringDataFrameColumn("R3")
                        ),
                };
            yield return new object[]
                {
                    new DataFrame(
                        new Int32DataFrameColumn("Index"),
                        new Int32DataFrameColumn("L1"),
                        new Int32DataFrameColumn("L2"),
                        new StringDataFrameColumn("L3")
                        ),
                    new DataFrame(
                        new Int32DataFrameColumn("Index"),
                        new Int32DataFrameColumn("R1"),
                        new Int32DataFrameColumn("R2"),
                        new StringDataFrameColumn("R3")
                        ),
                    new string[]{ "L1" },
                    new string[]{ "R1" },
                    JoinAlgorithm.Left,
                    new DataFrame(
                        new Int32DataFrameColumn("Index_left"),
                        new Int32DataFrameColumn("L1"),
                        new Int32DataFrameColumn("L2"),
                        new StringDataFrameColumn("L3"),
                        new Int32DataFrameColumn("Index_right"),
                        new Int32DataFrameColumn("R1"),
                        new Int32DataFrameColumn("R2"),
                        new StringDataFrameColumn("R3")
                        ),
                };
            yield return new object[]
                {
                    new DataFrame(
                        new Int32DataFrameColumn("Index"),
                        new Int32DataFrameColumn("L1"),
                        new Int32DataFrameColumn("L2"),
                        new StringDataFrameColumn("L3")
                        ),
                    new DataFrame(
                        new Int32DataFrameColumn("Index"),
                        new Int32DataFrameColumn("R1"),
                        new Int32DataFrameColumn("R2"),
                        new StringDataFrameColumn("R3")
                        ),
                    new string[]{ "L1" },
                    new string[]{ "R1" },
                    JoinAlgorithm.Right,
                    new DataFrame(
                        new Int32DataFrameColumn("Index_left"),
                        new Int32DataFrameColumn("L1"),
                        new Int32DataFrameColumn("L2"),
                        new StringDataFrameColumn("L3"),
                        new Int32DataFrameColumn("Index_right"),
                        new Int32DataFrameColumn("R1"),
                        new Int32DataFrameColumn("R2"),
                        new StringDataFrameColumn("R3")
                        ),
                };
            yield return new object[]
                {
                    new DataFrame(
                        new Int32DataFrameColumn("Index"),
                        new Int32DataFrameColumn("L1"),
                        new Int32DataFrameColumn("L2"),
                        new StringDataFrameColumn("L3")
                        ),
                    new DataFrame(
                        new Int32DataFrameColumn("Index"),
                        new Int32DataFrameColumn("R1"),
                        new Int32DataFrameColumn("R2"),
                        new StringDataFrameColumn("R3")
                        ),
                    new string[]{ "L1" },
                    new string[]{ "R1" },
                    JoinAlgorithm.FullOuter,
                    new DataFrame(
                        new Int32DataFrameColumn("Index_left"),
                        new Int32DataFrameColumn("L1"),
                        new Int32DataFrameColumn("L2"),
                        new StringDataFrameColumn("L3"),
                        new Int32DataFrameColumn("Index_right"),
                        new Int32DataFrameColumn("R1"),
                        new Int32DataFrameColumn("R2"),
                        new StringDataFrameColumn("R3")
                        ),
                };
        }

        [Theory]
        [MemberData(nameof(GenerateData_TestMerge_EmptyDataFrames))]
        public void TestMerge_EmptyDataFrames(DataFrame left, DataFrame right, string[] leftColumns, string[] rightColumns, JoinAlgorithm joinAlgorithm, DataFrame expectedOutput)
        {
            DataFrame actualOutput = left.Merge(right, leftColumns, rightColumns, joinAlgorithm: joinAlgorithm);

            DataFrameAssert.Equal(expectedOutput, actualOutput);
        }

        public static IEnumerable<object[]> GenerateData_TestMerge_OuterJoinsPreserveUnmatched()
        {
            yield return new object[]
                {
                    new DataFrame(
                        new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }),
                        new Int32DataFrameColumn("L1", new[] { 1, 2, 3 }),
                        new Int32DataFrameColumn("L2", new[] { 1, 2, 1 }),
                        new StringDataFrameColumn("L3", new[] { "A", "B", "C" })
                        ),
                    new DataFrame(
                        new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }),
                        new Int32DataFrameColumn("R1", new[] { 10, 11, 11 }),
                        new Int32DataFrameColumn("R2", new[] { 1, 1, 2 }),
                        new StringDataFrameColumn("R3", new[] { "Z", "Y", "B" })
                        ),
                    new string[]{ "L1" },
                    new string[]{ "R1" },
                    JoinAlgorithm.Left,
                    new DataFrame(
                        new Int32DataFrameColumn("Index_left", new[] { 0, 1, 2 }),
                        new Int32DataFrameColumn("L1", new[] { 1, 2, 3 }),
                        new Int32DataFrameColumn("L2", new[] { 1, 2, 1 }),
                        new StringDataFrameColumn("L3", new[] { "A", "B", "C" }),
                        new Int32DataFrameColumn("Index_right", new int?[] { null, null, null }),
                        new Int32DataFrameColumn("R1", new int?[] { null, null, null }),
                        new Int32DataFrameColumn("R2", new int?[] { null, null, null }),
                        new StringDataFrameColumn("R3", new string[] { null, null, null })
                        ),
                };
            yield return new object[]
                {
                    new DataFrame(
                        new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }),
                        new Int32DataFrameColumn("L1", new[] { 1, 2, 3 }),
                        new Int32DataFrameColumn("L2", new[] { 1, 2, 1 }),
                        new StringDataFrameColumn("L3", new[] { "A", "B", "C" })
                        ),
                    new DataFrame(
                        new Int32DataFrameColumn("Index"),
                        new Int32DataFrameColumn("R1"),
                        new Int32DataFrameColumn("R2"),
                        new StringDataFrameColumn("R3")
                        ),
                    new string[]{ "L1" },
                    new string[]{ "R1" },
                    JoinAlgorithm.Left,
                    new DataFrame(
                        new Int32DataFrameColumn("Index_left", new[] { 0, 1, 2 }),
                        new Int32DataFrameColumn("L1", new[] { 1, 2, 3 }),
                        new Int32DataFrameColumn("L2", new[] { 1, 2, 1 }),
                        new StringDataFrameColumn("L3", new[] { "A", "B", "C" }),
                        new Int32DataFrameColumn("Index_right", new int?[] { null, null, null }),
                        new Int32DataFrameColumn("R1", new int?[] { null, null, null }),
                        new Int32DataFrameColumn("R2", new int?[] { null, null, null }),
                        new StringDataFrameColumn("R3", new string[] { null, null, null })
                        ),
                };
            yield return new object[]
                {
                    new DataFrame(
                        new Int32DataFrameColumn("Index"),
                        new Int32DataFrameColumn("L1"),
                        new Int32DataFrameColumn("L2"),
                        new StringDataFrameColumn("L3")
                        ),
                    new DataFrame(
                        new Int32DataFrameColumn("Index", new[] { 0, 1, 2 }),
                        new Int32DataFrameColumn("R1", new[] { 1, 2, 3 }),
                        new Int32DataFrameColumn("R2", new[] { 1, 2, 1 }),
                        new StringDataFrameColumn("R3", new[] { "A", "B", "C" })
                        ),
                    new string[]{ "L1" },
                    new string[]{ "R1" },
                    JoinAlgorithm.Right,
                    new DataFrame(
                        new Int32DataFrameColumn("Index_left", new int?[] { null, null, null }),
                        new Int32DataFrameColumn("L1", new int?[] { null, null, null }),
                        new Int32DataFrameColumn("L2", new int?[] { null, null, null }),
                        new StringDataFrameColumn("L3", new string[] { null, null, null }),
                        new Int32DataFrameColumn("Index_right", new[] { 0, 1, 2 }),
                        new Int32DataFrameColumn("R1", new[] { 1, 2, 3 }),
                        new Int32DataFrameColumn("R2", new[] { 1, 2, 1 }),
                        new StringDataFrameColumn("R3", new[] { "A", "B", "C" })
                        ),
                };
        }

        [Theory]
        [MemberData(nameof(GenerateData_TestMerge_OuterJoinsPreserveUnmatched))]
        public void TestMerge_OuterJoinsPreserveUnmatched(DataFrame left, DataFrame right, string[] leftColumns, string[] rightColumns, JoinAlgorithm joinAlgorithm, DataFrame expectedOutput)
        {
            DataFrame actualOutput = left.Merge(right, leftColumns, rightColumns, joinAlgorithm: joinAlgorithm);

            DataFrameAssert.Equal(expectedOutput, actualOutput);
        }

        [Fact]
        //Issue 6127
        public void TestMerge_CorrectColumnTypes()
        {
            DataFrame left = MakeDataFrameWithAllMutableColumnTypes(2, false);
            DataFrame right = MakeDataFrameWithAllMutableColumnTypes(1);

            DataFrame merge = left.Merge<int>(right, "Int", "Int");

            Assert.NotNull(merge.Columns.GetBooleanColumn("Bool_left"));
            Assert.NotNull(merge.Columns.GetBooleanColumn("Bool_right"));

            Assert.NotNull(merge.Columns.GetDecimalColumn("Decimal_left"));
            Assert.NotNull(merge.Columns.GetDecimalColumn("Decimal_right"));

            Assert.NotNull(merge.Columns.GetSingleColumn("Float_left"));
            Assert.NotNull(merge.Columns.GetSingleColumn("Float_right"));

            Assert.NotNull(merge.Columns.GetDoubleColumn("Double_left"));
            Assert.NotNull(merge.Columns.GetDoubleColumn("Double_right"));

            Assert.NotNull(merge.Columns.GetByteColumn("Byte_left"));
            Assert.NotNull(merge.Columns.GetByteColumn("Byte_right"));

            Assert.NotNull(merge.Columns.GetCharColumn("Char_left"));
            Assert.NotNull(merge.Columns.GetCharColumn("Char_right"));

            Assert.NotNull(merge.Columns.GetInt16Column("Short_left"));
            Assert.NotNull(merge.Columns.GetInt16Column("Short_right"));

            Assert.NotNull(merge.Columns.GetUInt16Column("Ushort_left"));
            Assert.NotNull(merge.Columns.GetUInt16Column("Ushort_right"));

            Assert.NotNull(merge.Columns.GetInt32Column("Int_left"));
            Assert.NotNull(merge.Columns.GetInt32Column("Int_right"));

            Assert.NotNull(merge.Columns.GetUInt32Column("Uint_left"));
            Assert.NotNull(merge.Columns.GetUInt32Column("Uint_right"));

            Assert.NotNull(merge.Columns.GetInt64Column("Long_left"));
            Assert.NotNull(merge.Columns.GetInt64Column("Long_right"));

            Assert.NotNull(merge.Columns.GetUInt64Column("Ulong_left"));
            Assert.NotNull(merge.Columns.GetUInt64Column("Ulong_right"));

            Assert.NotNull(merge.Columns.GetDateTimeColumn("DateTime_left"));
            Assert.NotNull(merge.Columns.GetDateTimeColumn("DateTime_right"));
        }

        private void VerifyMerge(DataFrame merge, DataFrame left, DataFrame right, JoinAlgorithm joinAlgorithm)
        {
            if (joinAlgorithm == JoinAlgorithm.Left || joinAlgorithm == JoinAlgorithm.Inner)
            {
                HashSet<int> intersection = new HashSet<int>();
                for (int i = 0; i < merge.Columns["Int_left"].Length; i++)
                {
                    if (merge.Columns["Int_left"][i] == null)
                        continue;
                    intersection.Add((int)merge.Columns["Int_left"][i]);
                }
                for (int i = 0; i < left.Columns["Int"].Length; i++)
                {
                    if (left.Columns["Int"][i] != null && intersection.Contains((int)left.Columns["Int"][i]))
                        intersection.Remove((int)left.Columns["Int"][i]);
                }
                Assert.Empty(intersection);
            }
            else if (joinAlgorithm == JoinAlgorithm.Right)
            {
                HashSet<int> intersection = new HashSet<int>();
                for (int i = 0; i < merge.Columns["Int_right"].Length; i++)
                {
                    if (merge.Columns["Int_right"][i] == null)
                        continue;
                    intersection.Add((int)merge.Columns["Int_right"][i]);
                }
                for (int i = 0; i < right.Columns["Int"].Length; i++)
                {
                    if (right.Columns["Int"][i] != null && intersection.Contains((int)right.Columns["Int"][i]))
                        intersection.Remove((int)right.Columns["Int"][i]);
                }
                Assert.Empty(intersection);
            }
            else if (joinAlgorithm == JoinAlgorithm.FullOuter)
            {
                VerifyMerge(merge, left, right, JoinAlgorithm.Left);
                VerifyMerge(merge, left, right, JoinAlgorithm.Right);
            }
        }
    }
}
