// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.Data.Analysis.Tests
{
    public class PrimitiveDataFrameColumnTests : BaseTestClass
    {
        public PrimitiveDataFrameColumnTests(ITestOutputHelper output) : base(output)
        {
        }

        [X64Fact("32-bit doesn't allow to allocate more than 2 Gb")]
        public void TestGetterAndSetterForColumnsGreaterThanMaxCapacity()
        {
            const int MaxCapacityInBytes = 2147483591;

            var length = MaxCapacityInBytes + 5;
            var column = new PrimitiveDataFrameColumn<byte>("LargeColumn", length);
            var index = length - 1;
            column[index] = 33;

            Assert.Equal((byte)33, column[index]);
            Assert.Null(column[index % MaxCapacityInBytes]);
        }

        [Fact]
        public void TestNullCounts()
        {
            PrimitiveDataFrameColumn<int> column2 = new PrimitiveDataFrameColumn<int>("Int2");
            Assert.Equal(0, column2.NullCount);

            PrimitiveDataFrameColumn<int> column3 = new PrimitiveDataFrameColumn<int>("Int3", 10);
            Assert.Equal(10, column3.NullCount);

            // Test null counts with assignments on Primitive Columns
            column2.Append(null);
            column2.Append(1);
            Assert.Equal(1, column2.NullCount);
            column2[1] = 10;
            Assert.Equal(1, column2.NullCount);
            column2[1] = null;
            Assert.Equal(2, column2.NullCount);
            column2[1] = 5;
            Assert.Equal(1, column2.NullCount);
            column2[0] = null;
            Assert.Equal(1, column2.NullCount);

            // Test null counts with assignments on String Columns
            StringDataFrameColumn strCol = new StringDataFrameColumn("String", 0);
            Assert.Equal(0, strCol.NullCount);

            StringDataFrameColumn strCol1 = new StringDataFrameColumn("String1", 5);
            Assert.Equal(5, strCol1.NullCount);

            StringDataFrameColumn strCol2 = new StringDataFrameColumn("String", Enumerable.Range(0, 10).Select(x => x.ToString()));
            Assert.Equal(0, strCol2.NullCount);

            StringDataFrameColumn strCol3 = new StringDataFrameColumn("String", Enumerable.Range(0, 10).Select(x => (string)null));
            Assert.Equal(10, strCol3.NullCount);

            strCol.Append(null);
            Assert.Equal(1, strCol.NullCount);
            strCol.Append("foo");
            Assert.Equal(1, strCol.NullCount);
            strCol[1] = "bar";
            Assert.Equal(1, strCol.NullCount);
            strCol[1] = null;
            Assert.Equal(2, strCol.NullCount);
            strCol[1] = "foo";
            Assert.Equal(1, strCol.NullCount);
            strCol[0] = null;
            Assert.Equal(1, strCol.NullCount);
        }

        [Fact]
        public void TestNullCountWithIndexers()
        {
            PrimitiveDataFrameColumn<int> intColumn = new PrimitiveDataFrameColumn<int>("Int", 5);
            Assert.Equal(5, intColumn.NullCount);
            intColumn[2] = null;
            Assert.Equal(5, intColumn.NullCount);
            intColumn[2] = 5;
            Assert.Equal(4, intColumn.NullCount);
        }

        [Fact]
        public void TestValidity()
        {
            PrimitiveDataFrameColumn<int> dataFrameColumn1 = new PrimitiveDataFrameColumn<int>("Int1", Enumerable.Range(0, 10).Select(x => x));
            dataFrameColumn1.Append(null);
            Assert.False(dataFrameColumn1.IsValid(10));
            for (long i = 0; i < dataFrameColumn1.Length - 1; i++)
            {
                Assert.True(dataFrameColumn1.IsValid(i));
            }
        }

        [Fact]
        public void TestAppendNullToEmptyColumn()
        {
            PrimitiveDataFrameColumn<int> intColumn = new PrimitiveDataFrameColumn<int>("Int1");

            //Act
            intColumn.Append(null);

            Assert.Equal(1, intColumn.NullCount);
            Assert.Equal(1, intColumn.Length);

            for (int i = 0; i < intColumn.Length; i++)
            {
                Assert.False(intColumn.IsValid(i));
            }
        }

        [Fact]
        public void TestAppendNullToColumnWithValues()
        {
            PrimitiveDataFrameColumn<int> dataFrameColumn1 = new PrimitiveDataFrameColumn<int>("Int1", Enumerable.Range(0, 10));
            dataFrameColumn1.Append(null);
            Assert.Equal(1, dataFrameColumn1.NullCount);
            Assert.Equal(11, dataFrameColumn1.Length);
            Assert.Null(dataFrameColumn1[10]);
        }

        [Fact]
        public void TestAppendToColumnWithValues()
        {
            PrimitiveDataFrameColumn<int> intColumn = new PrimitiveDataFrameColumn<int>("Int", Enumerable.Range(0, 10));

            intColumn.Append(0);
            intColumn.Append(1);
            intColumn.Append(null);
            intColumn.Append(2);
            intColumn.Append(null);
            intColumn.Append(3);

            Assert.Equal(16, intColumn.Length);
            Assert.Equal(2, intColumn.NullCount);

            Assert.Equal(0, intColumn[10]);
            Assert.Equal(1, intColumn[11]);
            Assert.Null(intColumn[12]);
            Assert.Equal(2, intColumn[13]);
            Assert.Null(intColumn[14]);
            Assert.Equal(3, intColumn[15]);
        }

        [Fact]
        public void TestAppendManyNullsToEmptyColumn()
        {
            PrimitiveDataFrameColumn<int> intColumn = new PrimitiveDataFrameColumn<int>("Int1");

            //Act
            intColumn.AppendMany(null, 5);
            Assert.Equal(5, intColumn.NullCount);
            Assert.Equal(5, intColumn.Length);
            for (int i = 0; i < intColumn.Length; i++)
            {
                Assert.False(intColumn.IsValid(i));
            }
        }

        [Fact]
        public void TestAppendManyNullsToColumnWithValues()
        {
            //Arrange
            var initialValues = new int?[] { 1, 2, null, 4, 5 };
            PrimitiveDataFrameColumn<int> intColumn = new PrimitiveDataFrameColumn<int>("Int1", initialValues);

            //Act
            intColumn.AppendMany(null, 5);

            //Assert
            Assert.Equal(6, intColumn.NullCount);
            Assert.Equal(10, intColumn.Length);

            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(initialValues[i], intColumn[i]);
            }

            for (int i = 5; i < 10; i++)
            {
                Assert.False(intColumn.IsValid(i));
            }
        }

        [Fact]
        public void TestAppendManyValuesToEmptyColumn()
        {
            //Arrange
            PrimitiveDataFrameColumn<int> intColumn = new PrimitiveDataFrameColumn<int>("Int1");

            //Act
            intColumn.AppendMany(5, 5);

            //Assert
            Assert.Equal(0, intColumn.NullCount);
            Assert.Equal(5, intColumn.Length);

            for (int i = 0; i < intColumn.Length; i++)
            {
                Assert.Equal(5, intColumn[i]);
            }
        }

        [Fact]
        public void TestAppendManyValuesToColumnWithValues()
        {
            //Arrange
            PrimitiveDataFrameColumn<int> intColumn = new PrimitiveDataFrameColumn<int>("Int1", new int?[] { 1, 2, 3, null, null });

            //Act
            intColumn.AppendMany(5, 5);

            //Assert
            Assert.Equal(2, intColumn.NullCount);
            Assert.Equal(10, intColumn.Length);

            Assert.Equal(3, intColumn[2]);
            Assert.Null(intColumn[3]);
            Assert.Null(intColumn[4]);

            for (int i = 5; i < intColumn.Length; i++)
            {
                Assert.Equal(5, intColumn[i]);
            }
        }

        [Fact]
        public void TestNullCountChange()
        {
            //Arrange
            var initialValues = new int?[] { null, null, null, null, null, 5, 5, 5, 5, 5 };
            PrimitiveDataFrameColumn<int> intColumn = new PrimitiveDataFrameColumn<int>("Int1", initialValues);

            //Act
            intColumn[2] = 10;

            //Assert
            Assert.Equal(4, intColumn.NullCount);
            Assert.True(intColumn.IsValid(2));

            //Act
            intColumn[7] = null;

            //Assert
            Assert.Equal(5, intColumn.NullCount);
            Assert.False(intColumn.IsValid(7));
        }

        [Fact]
        public void TestClone_PrimitiveColumn()
        {
            PrimitiveDataFrameColumn<int> intColumn = new PrimitiveDataFrameColumn<int>("Int1", new int?[] { 1, 2, 3, 4, null });
            var copy = intColumn.Clone();

            Assert.Equal(intColumn.Name, copy.Name);
            Assert.Equal(intColumn.Length, copy.Length);
            Assert.Equal(intColumn.DataType, copy.DataType);
            Assert.Equal(intColumn.NullCount, copy.NullCount);

            for (int i = 0; i < intColumn.Length; i++)
                Assert.Equal(intColumn[i], copy[i]);
        }

        [Fact]
        public void TestNotNullableColumnClone()
        {
            //Arrange
            var column = new Int32DataFrameColumn("Int column", values: new[] { -1, 2, 3, 2, 1, -2 });

            //Act
            var clonedColumn = column.Clone();

            //Assert
            Assert.NotSame(column, clonedColumn);
            Assert.Equal(column.Name, clonedColumn.Name);
            Assert.Equal(column.DataType, clonedColumn.DataType);
            Assert.Equal(column.NullCount, clonedColumn.NullCount);
            Assert.Equal(column.Length, clonedColumn.Length);

            for (long i = 0; i < column.Length; i++)
                Assert.Equal(column[i], clonedColumn[i]);
        }

        [Fact]
        public void TestNullableColumnClone()
        {
            //Arrange
            var column = new Int32DataFrameColumn("Int column", values: new int?[] { -1, null, 3, 2, 1, -2 });

            //Act
            var clonedColumn = column.Clone();

            //Assert
            Assert.NotSame(column, clonedColumn);
            Assert.Equal(column.Name, clonedColumn.Name);
            Assert.Equal(column.DataType, clonedColumn.DataType);
            Assert.Equal(column.NullCount, clonedColumn.NullCount);
            Assert.Equal(column.Length, clonedColumn.Length);

            for (long i = 0; i < column.Length; i++)
                Assert.Equal(column[i], clonedColumn[i]);

        }

        [Fact]
        public void TestNotNullableColumnClone_WithIntIndicesMap()
        {
            //Arrange
            var column = new Int32DataFrameColumn("Int column", values: new[] { 0, 5, 2, 4, 1, 3 });
            var indicesMap = new Int32DataFrameColumn("Indices", new[] { 0, 1, 2, 5, 3, 4 });

            //Act
            var clonedColumn = column.Clone(indicesMap);

            //Assert
            Assert.NotSame(column, clonedColumn);
            Assert.Equal(column.Name, clonedColumn.Name);
            Assert.Equal(column.DataType, clonedColumn.DataType);
            Assert.Equal(column.NullCount, clonedColumn.NullCount);
            Assert.Equal(indicesMap.Length, clonedColumn.Length);

            for (int i = 0; i < indicesMap.Length; i++)
                Assert.Equal(column[indicesMap[i].Value], clonedColumn[i]);
        }

        [Fact]
        public void TestNotNullableColumnClone_WithIntIndicesMap_Invert()
        {
            //Arrange
            var column = new Int32DataFrameColumn("Int column", values: new int?[] { 0, 5, null, 4, 1, 3 });
            var indicesMap = new Int32DataFrameColumn("Indices", new[] { 0, 1, 2, 2, 5, 3, 4 });

            //Act
            var clonedColumn = column.Clone(indicesMap, true);

            //Assert
            Assert.NotSame(column, clonedColumn);
            Assert.Equal(column.Name, clonedColumn.Name);
            Assert.Equal(column.DataType, clonedColumn.DataType);
            Assert.Equal(2, clonedColumn.NullCount);
            Assert.Equal(indicesMap.Length, clonedColumn.Length);

            for (int i = 0; i < indicesMap.Length; i++)
                Assert.Equal(column[indicesMap[indicesMap.Length - 1 - i].Value], clonedColumn[i]);
        }

        [Fact]
        public void TestNotNullableColumnClone_WithLongIndicesMap()
        {
            //Arrange
            var column = new Int32DataFrameColumn("Int column", values: new[] { 0, 5, 2, 4, 1, 3 });
            var indicesMap = new Int64DataFrameColumn("Indices", new long[] { 0, 1, 2, 5, 3, 4 });

            //Act
            var clonedColumn = column.Clone(indicesMap);

            //Assert
            Assert.NotSame(column, clonedColumn);
            Assert.Equal(column.Name, clonedColumn.Name);
            Assert.Equal(column.DataType, clonedColumn.DataType);
            Assert.Equal(column.NullCount, clonedColumn.NullCount);
            Assert.Equal(indicesMap.Length, clonedColumn.Length);

            for (int i = 0; i < indicesMap.Length; i++)
                Assert.Equal(column[indicesMap[i].Value], clonedColumn[i]);
        }

        [Fact]
        public void TestNotNullableColumnClone_WithLongIndicesMap_Invert()
        {
            //Arrange
            var column = new Int32DataFrameColumn("Int column", values: new int?[] { 0, 5, null, 4, 1, 3 });
            var indicesMap = new Int64DataFrameColumn("Indices", new long[] { 0, 1, 2, 5, 3, 4, 4, 2 });

            //Act
            var clonedColumn = column.Clone(indicesMap, true);

            //Assert
            Assert.NotSame(column, clonedColumn);
            Assert.Equal(column.Name, clonedColumn.Name);
            Assert.Equal(column.DataType, clonedColumn.DataType);
            Assert.Equal(2, clonedColumn.NullCount);
            Assert.Equal(indicesMap.Length, clonedColumn.Length);

            for (int i = 0; i < indicesMap.Length; i++)
                Assert.Equal(column[indicesMap[indicesMap.Length - 1 - i].Value], clonedColumn[i]);
        }

        [Fact]
        public void TestNotNullableColumnCloneWithIndicesMapAsEnumerableLong()
        {
            //Arrange
            var column = new Int32DataFrameColumn("Int column", values: new[] { 0, 5, 2, 4, 1, 3 });
            var indicesMap = new long[] { 0, 1, 2, 5, 3, 4 };

            //Act
            var clonedColumn = column.Clone(indicesMap);

            //Assert
            Assert.NotSame(column, clonedColumn);
            Assert.Equal(column.Name, clonedColumn.Name);
            Assert.Equal(column.DataType, clonedColumn.DataType);
            Assert.Equal(column.NullCount, clonedColumn.NullCount);
            Assert.Equal(indicesMap.Length, clonedColumn.Length);

            for (int i = 0; i < indicesMap.Length; i++)
                Assert.Equal(column[indicesMap[i]], clonedColumn[i]);
        }

        [Fact]
        public void TestNotNullableColumnCloneWithIndicesMapAsEnumerableInt()
        {
            //Arrange
            var column = new Int32DataFrameColumn("Int column", values: new[] { 0, 5, 2, 4, 1, 3 });
            var indicesMap = new int[] { 0, 1, 2, 5, 3, 4 };

            //Act
            var clonedColumn = column.Clone(indicesMap);

            //Assert
            Assert.NotSame(column, clonedColumn);
            Assert.Equal(column.Name, clonedColumn.Name);
            Assert.Equal(column.DataType, clonedColumn.DataType);
            Assert.Equal(column.NullCount, clonedColumn.NullCount);
            Assert.Equal(indicesMap.Length, clonedColumn.Length);

            for (int i = 0; i < indicesMap.Length; i++)
                Assert.Equal(column[indicesMap[i]], clonedColumn[i]);
        }

        [Fact]
        public void TestNullableColumnCloneWithIndicesMapAndSmallerSize()
        {
            //Arrange
            var column = new Int32DataFrameColumn("Int column", values: new int?[] { null, 5, 2, 4, 1, 3 });
            var indicesMap = new Int32DataFrameColumn("Indices", new[] { 0, 4, 2, 5, 3 });

            //Act
            var clonedColumn = column.Clone(indicesMap);

            //Assert
            Assert.NotSame(column, clonedColumn);
            Assert.Equal(column.Name, clonedColumn.Name);
            Assert.Equal(indicesMap.Length, clonedColumn.Length);
            Assert.Equal(column.DataType, clonedColumn.DataType);

            for (int i = 0; i < indicesMap.Length; i++)
                Assert.Equal(indicesMap.IsValid(i) ? column[indicesMap[i].Value] : null, clonedColumn[i]);
        }

        [Fact]
        public void TestNullableColumnCloneWithIndicesMap_OutOfRange()
        {
            //Arrange
            var column = new Int32DataFrameColumn("Int column", values: new int?[] { null, 1, 1 });
            var indicesMap = new Int32DataFrameColumn("Indices", new[] { 0, 1, 4 });

            //Act and assert
            Assert.Throws<IndexOutOfRangeException>(() => column.Clone(indicesMap));
        }



        [X64Fact("32-bit doesn't allow to allocate more than 2 Gb")]
        public void TestAppend_SizeMoreThanMaxBufferCapacity()
        {
            //Check appending value, than can increase buffer size over MaxCapacity (default strategy is to double buffer capacity)
            PrimitiveDataFrameColumn<byte> intColumn = new PrimitiveDataFrameColumn<byte>("Byte1", int.MaxValue / 2 - 1);
            intColumn.Append(10);
        }

        [X64Fact("32-bit doesn't allow to allocate more than 2 Gb")]
        public void TestAppendMany_SizeMoreThanMaxBufferCapacity()
        {
            const int MaxCapacityInBytes = 0X7FEFFFFF;

            //Check appending values with extending column size over MaxCapacity of ReadOnlyDataFrameBuffer
            PrimitiveDataFrameColumn<byte> intColumn = new PrimitiveDataFrameColumn<byte>("Byte1", MaxCapacityInBytes - 5);
            intColumn.AppendMany(5, 10);

            Assert.Equal(MaxCapacityInBytes + 5, intColumn.Length);

            intColumn = null;
        }

        [Fact]
        public void Test_PrimitiveColumnNotEqualsNull()
        {
            var col = new DoubleDataFrameColumn("col", new double?[] { 1.23, null, 2, 3 });
            var dfTest = new DataFrame(col);

            var filteredNullDf = dfTest.Filter(dfTest["col"].ElementwiseNotEquals(null));

            Assert.True(filteredNullDf.Columns.IndexOf("col") >= 0);
            Assert.Equal(3, filteredNullDf.Columns["col"].Length);

            Assert.Equal(1.23, filteredNullDf.Columns["col"][0]);
            Assert.Equal(2.0, filteredNullDf.Columns["col"][1]);
            Assert.Equal(3.0, filteredNullDf.Columns["col"][2]);
        }

        [Fact]
        public void Test_PrimitiveColumnEqualsNull()
        {
            var index = new Int32DataFrameColumn("index", new int[] { 1, 2, 3, 4, 5 });
            var col = new DoubleDataFrameColumn("col", new double?[] { 1.23, null, 2, 3, null }); ;
            var dfTest = new DataFrame(index, col);

            var filteredNullDf = dfTest.Filter(dfTest["col"].ElementwiseEquals(null));

            Assert.True(filteredNullDf.Columns.IndexOf("col") >= 0);
            Assert.True(filteredNullDf.Columns.IndexOf("index") >= 0);

            Assert.Equal(2, filteredNullDf.Rows.Count);

            Assert.Equal(2, filteredNullDf.Columns["index"][0]);
            Assert.Equal(5, filteredNullDf.Columns["index"][1]);
        }

        [Fact]
        public void Test_ArithmeticsAddWithNull()
        {
            // Arrange
            //Number of elements shoult be higher than 8 to test SIMD
            var left_column = new Int32DataFrameColumn("Left", new int?[] { 1, 1, null, null, 4, 5, 6, 7, 8, 9 });
            var right_column = new Int32DataFrameColumn("Right", new int?[] { 1, null, 1, null, 4, 5, 6, 7, 8, 9 });

            // Act
            var sum = left_column + right_column;

            // Assert
            Assert.Equal(3, sum.NullCount);

            Assert.Equal(2, sum[0]);  // 1 + 1
            Assert.Null(sum[1]); // null + 1
            Assert.Null(sum[2]); // 1 + null
            Assert.Null(sum[3]); // null + null
            Assert.Equal(8, sum[4]);
            Assert.Equal(10, sum[5]);
            Assert.Equal(12, sum[6]);
            Assert.Equal(14, sum[7]);
            Assert.Equal(16, sum[8]);
            Assert.Equal(18, sum[9]);
        }

        [Fact]
        public void Test_ArithmeticsAddScalarWithNull()
        {
            // Arrange
            //Number of elements shoult be higher than 8 to test SIMD
            var left_column = new Int32DataFrameColumn("Left", new int?[] { 0, 1, null, null, 4, 5, 6, 7, 8, null });

            // Act
            var sum = left_column + 5;

            // Assert
            Assert.Equal(3, sum.NullCount);

            Assert.Equal(5, sum[0]);  // 1 + 1
            Assert.Equal(6, sum[1]);  // 1 + 1
            Assert.Null(sum[2]); // 1 + null
            Assert.Null(sum[3]); // null + null
            Assert.Equal(9, sum[4]);
            Assert.Equal(10, sum[5]);
            Assert.Equal(11, sum[6]);
            Assert.Equal(12, sum[7]);
            Assert.Equal(13, sum[8]);
            Assert.Null(sum[9]);
        }

        [Fact]
        public void Test_ArithmeticsDiffWithNull()
        {
            // Arrange
            var left_column = new Int32DataFrameColumn("Left", new int?[] { 1, 1, null, null });
            var right_column = new Int32DataFrameColumn("Right", new int?[] { 1, null, 1, null });

            // Act
            var diff = left_column - (right_column);

            // Assert
            Assert.Equal(3, diff.NullCount);
            Assert.Equal(0, diff[0]);  // 1 - 1
            Assert.Null(diff[1]); // null - 1
            Assert.Null(diff[2]); // 1 - null
            Assert.Null(diff[3]); // null - null
        }

        [Fact]
        public void Test_ArithmeticsMultWithNull()
        {
            // Arrange
            var left_column = new Int32DataFrameColumn("Left", new int?[] { 4, 1, null, null });
            var right_column = new Int32DataFrameColumn("Right", new int?[] { 2, null, 1, null });

            // Act
            var mult = left_column * right_column;

            // Assert
            Assert.Equal(3, mult.NullCount);
            Assert.Equal(8, mult[0]);  // 1 * 1
            Assert.Null(mult[1]); // null * 1
            Assert.Null(mult[2]); // 1 * null
            Assert.Null(mult[3]); // null * null
        }

        [Fact]
        public void Test_ArithmeticsDivWithNull()
        {
            // Arrange
            var left_column = new Int32DataFrameColumn("Left", new int?[] { 4, 1, null, null });
            var right_column = new Int32DataFrameColumn("Right", new int?[] { 2, null, 1, null });

            // Act
            var div = left_column / right_column;

            // Assert
            Assert.Equal(3, div.NullCount);
            Assert.Equal(2, div[0]);  // 1 / 1
            Assert.Null(div[1]); // null / 1
            Assert.Null(div[2]); // 1 / null
            Assert.Null(div[3]); // null / null
        }

        [Fact]
        public void TestApply_InPlace()
        {
            // Arrange
            var column = new Int32DataFrameColumn("int", new int?[] { 0, 1, 2, null, null, 5 });

            column.Apply(x => x * 2, true);

            // Assert
            Assert.Equal(0, column[0]);
            Assert.Equal(2, column[1]);
            Assert.Equal(4, column[2]);
            Assert.Null(column[3]);
            Assert.Null(column[4]);
            Assert.Equal(10, column[5]);
        }

        [Fact]
        public void TestDropNulls()
        {
            // Arrange
            var column = new Int32DataFrameColumn("int", new int?[] { null, 0, 1, 2, null, null, 3, null });

            var res = column.DropNulls();

            // Assert
            Assert.Equal(4, res.Length);
            Assert.Equal(0, res.NullCount);

            Assert.Equal(0, res[0]);
            Assert.Equal(1, res[1]);
            Assert.Equal(2, res[2]);
            Assert.Equal(3, res[3]);
        }

        //#if !NETFRAMEWORK // https://github.com/dotnet/corefxlab/issues/2796
        //        [Fact]
        //        public void TestPrimitiveColumnGetReadOnlyBuffers()
        //        {
        //            RecordBatch recordBatch = new RecordBatch.Builder()
        //                .Append("Column1", false, col => col.Int32(array => array.AppendRange(Enumerable.Range(0, 10)))).Build();
        //            DataFrame df = DataFrame.FromArrowRecordBatch(recordBatch);

        //            PrimitiveDataFrameColumn<int> column = df.Columns["Column1"] as PrimitiveDataFrameColumn<int>;

        //            IEnumerable<ReadOnlyMemory<int>> buffers = column.GetReadOnlyDataBuffers();
        //            IEnumerable<ReadOnlyMemory<byte>> nullBitMaps = column.GetReadOnlyNullBitMapBuffers();

        //            long i = 0;
        //            IEnumerator<ReadOnlyMemory<int>> bufferEnumerator = buffers.GetEnumerator();
        //            IEnumerator<ReadOnlyMemory<byte>> nullBitMapsEnumerator = nullBitMaps.GetEnumerator();
        //            while (bufferEnumerator.MoveNext() && nullBitMapsEnumerator.MoveNext())
        //            {
        //                ReadOnlyMemory<int> dataBuffer = bufferEnumerator.Current;
        //                ReadOnlyMemory<byte> nullBitMap = nullBitMapsEnumerator.Current;

        //                ReadOnlySpan<int> span = dataBuffer.Span;
        //                for (int j = 0; j < span.Length; j++)
        //                {
        //                    // Each buffer has a max length of int.MaxValue
        //                    Assert.Equal(span[j], column[j + i * int.MaxValue]);
        //                }

        //                bool GetBit(byte curBitMap, int index)
        //                {
        //                    return ((curBitMap >> (index & 7)) & 1) != 0;
        //                }
        //                ReadOnlySpan<byte> bitMapSpan = nullBitMap.Span;
        //                // No nulls in this column, so each bit must be set
        //                for (int j = 0; j < bitMapSpan.Length; j++)
        //                {
        //                    for (int k = 0; k < 8; k++)
        //                    {
        //                        if (j * 8 + k == column.Length)
        //                            break;
        //                        Assert.True(GetBit(bitMapSpan[j], k));
        //                    }
        //                }
        //                i++;
        //            }
        //        }

        //        [Fact]
        //        public void TestArrowStringColumnGetReadOnlyBuffers()
        //        {
        //            // Test ArrowStringDataFrameColumn.
        //            StringArray strArray = new StringArray.Builder().Append("foo").Append("bar").Build();
        //            Memory<byte> dataMemory = new byte[] { 102, 111, 111, 98, 97, 114 };
        //            Memory<byte> nullMemory = new byte[] { 1 };
        //            Memory<byte> offsetMemory = new byte[] { 0, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0 };

        //            ArrowStringDataFrameColumn column = new ArrowStringDataFrameColumn("String", dataMemory, offsetMemory, nullMemory, strArray.Length, strArray.NullCount);

        //            IEnumerable<ReadOnlyMemory<byte>> dataBuffers = column.GetReadOnlyDataBuffers();
        //            IEnumerable<ReadOnlyMemory<byte>> nullBitMaps = column.GetReadOnlyNullBitMapBuffers();
        //            IEnumerable<ReadOnlyMemory<int>> offsetsBuffers = column.GetReadOnlyOffsetsBuffers();

        //            using (IEnumerator<ReadOnlyMemory<byte>> bufferEnumerator = dataBuffers.GetEnumerator())
        //            using (IEnumerator<ReadOnlyMemory<int>> offsetsEnumerator = offsetsBuffers.GetEnumerator())
        //            using (IEnumerator<ReadOnlyMemory<byte>> nullBitMapsEnumerator = nullBitMaps.GetEnumerator())
        //            {
        //                while (bufferEnumerator.MoveNext() && nullBitMapsEnumerator.MoveNext() && offsetsEnumerator.MoveNext())
        //                {
        //                    ReadOnlyMemory<byte> dataBuffer = bufferEnumerator.Current;
        //                    ReadOnlyMemory<byte> nullBitMap = nullBitMapsEnumerator.Current;
        //                    ReadOnlyMemory<int> offsets = offsetsEnumerator.Current;

        //                    ReadOnlySpan<byte> dataSpan = dataBuffer.Span;
        //                    ReadOnlySpan<int> offsetsSpan = offsets.Span;
        //                    int dataStart = 0;
        //                    for (int j = 1; j < offsetsSpan.Length; j++)
        //                    {
        //                        int length = offsetsSpan[j] - offsetsSpan[j - 1];
        //                        ReadOnlySpan<byte> str = dataSpan.Slice(dataStart, length);
        //                        ReadOnlySpan<byte> columnStr = dataMemory.Span.Slice(dataStart, length);
        //                        Assert.Equal(str.Length, columnStr.Length);
        //                        for (int s = 0; s < str.Length; s++)
        //                            Assert.Equal(str[s], columnStr[s]);
        //                        dataStart = length;
        //                    }
        //                }
        //            }
        //        }
        //#endif //!NETFRAMEWORK
    }
}
