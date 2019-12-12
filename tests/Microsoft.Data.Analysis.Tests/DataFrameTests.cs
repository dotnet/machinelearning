// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Apache.Arrow;
using Microsoft.ML;
using Xunit;

namespace Microsoft.Data.Analysis.Tests
{
    public partial class DataFrameTests
    {
        public static DataFrame MakeDataFrameWithTwoColumns(int length, bool withNulls = true)
        {
            DataFrameColumn dataFrameColumn1 = new PrimitiveDataFrameColumn<int>("Int1", Enumerable.Range(0, length).Select(x => x));
            DataFrameColumn dataFrameColumn2 = new PrimitiveDataFrameColumn<int>("Int2", Enumerable.Range(10, length).Select(x => x));
            if (withNulls)
            {
                dataFrameColumn1[length / 2] = null;
                dataFrameColumn2[length / 2] = null;
            }
            DataFrame dataFrame = new DataFrame();
            dataFrame.Columns.Insert(0, dataFrameColumn1);
            dataFrame.Columns.Insert(1, dataFrameColumn2);
            return dataFrame;
        }

        public static DataFrameColumn CreateArrowStringColumn(int length, bool withNulls = true)
        {
            byte[] dataMemory = new byte[length * 3];
            byte[] nullMemory = new byte[BitUtility.ByteCount(length)];
            byte[] offsetMemory = new byte[(length + 1) * 4];

            // Initialize offset with 0 as the first value
            offsetMemory[0] = 0;
            offsetMemory[1] = 0;
            offsetMemory[2] = 0;
            offsetMemory[3] = 0;

            // Append "foo" length times, with a possible `null` in the middle
            int validStringsIndex = 0;
            for (int i = 0; i < length; i++)
            {
                if (withNulls && i == length / 2)
                {
                    BitUtility.SetBit(nullMemory, i, false);
                }
                else
                {
                    int dataMemoryIndex = validStringsIndex * 3;
                    dataMemory[dataMemoryIndex++] = 102;
                    dataMemory[dataMemoryIndex++] = 111;
                    dataMemory[dataMemoryIndex++] = 111;
                    BitUtility.SetBit(nullMemory, i, true);

                    validStringsIndex++;
                }

                // write the current length to (index + 1)
                int offsetIndex = (i + 1) * 4;
                offsetMemory[offsetIndex++] = (byte)(3 * validStringsIndex);
                offsetMemory[offsetIndex++] = 0;
                offsetMemory[offsetIndex++] = 0;
                offsetMemory[offsetIndex++] = 0;
            }

            int nullCount = withNulls ? 1 : 0;
            return new ArrowStringDataFrameColumn("ArrowString", dataMemory, offsetMemory, nullMemory, length, nullCount);
        }

        public static DataFrame MakeDataFrameWithAllColumnTypes(int length, bool withNulls = true)
        {
            DataFrame df = MakeDataFrameWithAllMutableColumnTypes(length, withNulls);
            DataFrameColumn arrowStringColumn = CreateArrowStringColumn(length, withNulls);
            df.Columns.Insert(df.Columns.Count, arrowStringColumn);
            return df;
        }

        public static DataFrame MakeDataFrameWithAllMutableColumnTypes(int length, bool withNulls = true)
        {
            DataFrame df = MakeDataFrameWithNumericAndStringColumns(length, withNulls);
            DataFrameColumn boolColumn = new PrimitiveDataFrameColumn<bool>("Bool", Enumerable.Range(0, length).Select(x => x % 2 == 0));
            df.Columns.Insert(df.Columns.Count, boolColumn);
            if (withNulls)
            {
                boolColumn[length / 2] = null;
            }
            return df;
        }

        public static DataFrame MakeDataFrameWithNumericAndBoolColumns(int length)
        {
            DataFrame df = MakeDataFrameWithNumericColumns(length);
            DataFrameColumn boolColumn = new PrimitiveDataFrameColumn<bool>("Bool", Enumerable.Range(0, length).Select(x => x % 2 == 0));
            df.Columns.Insert(df.Columns.Count, boolColumn);
            boolColumn[length / 2] = null;
            return df;
        }

        public static DataFrame MakeDataFrameWithNumericAndStringColumns(int length, bool withNulls = true)
        {
            DataFrame df = MakeDataFrameWithNumericColumns(length, withNulls);
            DataFrameColumn stringColumn = new StringDataFrameColumn("String", Enumerable.Range(0, length).Select(x => x.ToString()));
            df.Columns.Insert(df.Columns.Count, stringColumn);
            if (withNulls)
            {
                stringColumn[length / 2] = null;
            }

            DataFrameColumn charColumn = new PrimitiveDataFrameColumn<char>("Char", Enumerable.Range(0, length).Select(x => (char)(x + 65)));
            df.Columns.Insert(df.Columns.Count, charColumn);
            if (withNulls)
            {
                charColumn[length / 2] = null;
            }
            return df;
        }

        public static DataFrame MakeDataFrameWithNumericColumns(int length, bool withNulls = true)
        {
            DataFrameColumn byteColumn = new PrimitiveDataFrameColumn<byte>("Byte", Enumerable.Range(0, length).Select(x => (byte)x));
            DataFrameColumn decimalColumn = new PrimitiveDataFrameColumn<decimal>("Decimal", Enumerable.Range(0, length).Select(x => (decimal)x));
            DataFrameColumn doubleColumn = new PrimitiveDataFrameColumn<double>("Double", Enumerable.Range(0, length).Select(x => (double)x));
            DataFrameColumn floatColumn = new PrimitiveDataFrameColumn<float>("Float", Enumerable.Range(0, length).Select(x => (float)x));
            DataFrameColumn intColumn = new PrimitiveDataFrameColumn<int>("Int", Enumerable.Range(0, length).Select(x => x));
            DataFrameColumn longColumn = new PrimitiveDataFrameColumn<long>("Long", Enumerable.Range(0, length).Select(x => (long)x));
            DataFrameColumn sbyteColumn = new PrimitiveDataFrameColumn<sbyte>("Sbyte", Enumerable.Range(0, length).Select(x => (sbyte)x));
            DataFrameColumn shortColumn = new PrimitiveDataFrameColumn<short>("Short", Enumerable.Range(0, length).Select(x => (short)x));
            DataFrameColumn uintColumn = new PrimitiveDataFrameColumn<uint>("Uint", Enumerable.Range(0, length).Select(x => (uint)x));
            DataFrameColumn ulongColumn = new PrimitiveDataFrameColumn<ulong>("Ulong", Enumerable.Range(0, length).Select(x => (ulong)x));
            DataFrameColumn ushortColumn = new PrimitiveDataFrameColumn<ushort>("Ushort", Enumerable.Range(0, length).Select(x => (ushort)x));

            DataFrame dataFrame = new DataFrame(new List<DataFrameColumn> { byteColumn, decimalColumn, doubleColumn, floatColumn, intColumn, longColumn, sbyteColumn, shortColumn, uintColumn, ulongColumn, ushortColumn });

            if (withNulls)
            {
                for (int i = 0; i < dataFrame.Columns.Count; i++)
                {
                    dataFrame.Columns[i][length / 2] = null;
                }
            }
            return dataFrame;
        }

        public static DataFrame MakeDataFrame<T1, T2>(int length, bool withNulls = true)
            where T1 : unmanaged
            where T2 : unmanaged
        {
            DataFrameColumn baseColumn1 = new PrimitiveDataFrameColumn<T1>("Column1", Enumerable.Range(0, length).Select(x => (T1)Convert.ChangeType(x % 2 == 0 ? 0 : 1, typeof(T1))));
            DataFrameColumn baseColumn2 = new PrimitiveDataFrameColumn<T2>("Column2", Enumerable.Range(0, length).Select(x => (T2)Convert.ChangeType(x % 2 == 0 ? 0 : 1, typeof(T2))));
            DataFrame dataFrame = new DataFrame(new List<DataFrameColumn> { baseColumn1, baseColumn2 });

            if (withNulls)
            {
                for (int i = 0; i < dataFrame.Columns.Count; i++)
                {
                    dataFrame.Columns[i][length / 2] = null;
                }
            }

            return dataFrame;
        }

        public DataFrame SplitTrainTest(DataFrame input, float testRatio, out DataFrame Test)
        {
            IEnumerable<int> randomIndices = Enumerable.Range(0, (int)input.Rows.Count);
            IEnumerable<int> trainIndices = randomIndices.Take((int)(input.Rows.Count * testRatio));
            IEnumerable<int> testIndices = randomIndices.Skip((int)(input.Rows.Count * testRatio));
            Test = input[testIndices];
            return input[trainIndices];
        }


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

            var column = dataFrame["Int2"] as PrimitiveDataFrameColumn<int>;
            Assert.Equal(1000, (int)column[2]);

            Assert.Throws<ArgumentException>(() => dataFrame["Int5"]);
        }

        [Fact]
        public void ColumnAndTableCreationTest()
        {
            DataFrameColumn intColumn = new PrimitiveDataFrameColumn<int>("IntColumn", Enumerable.Range(0, 10).Select(x => x));
            DataFrameColumn floatColumn = new PrimitiveDataFrameColumn<float>("FloatColumn", Enumerable.Range(0, 10).Select(x => (float)x));
            DataFrame dataFrame = new DataFrame();
            dataFrame.Columns.Insert(0, intColumn);
            dataFrame.Columns.Insert(1, floatColumn);
            Assert.Equal(10, dataFrame.Rows.Count);
            Assert.Equal(2, dataFrame.Columns.Count);
            Assert.Equal(10, dataFrame.Columns[0].Length);
            Assert.Equal("IntColumn", dataFrame.Columns[0].Name);
            Assert.Equal(10, dataFrame.Columns[1].Length);
            Assert.Equal("FloatColumn", dataFrame.Columns[1].Name);

            DataFrameColumn bigColumn = new PrimitiveDataFrameColumn<float>("BigColumn", Enumerable.Range(0, 11).Select(x => (float)x));
            DataFrameColumn repeatedName = new PrimitiveDataFrameColumn<float>("FloatColumn", Enumerable.Range(0, 10).Select(x => (float)x));
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.Insert(2, bigColumn));
            Assert.Throws<ArgumentException>(() => dataFrame.Columns.Insert(2, repeatedName));
            Assert.Throws<ArgumentOutOfRangeException>(() => dataFrame.Columns.Insert(10, repeatedName));

            Assert.Equal(2, dataFrame.Columns.Count);
            DataFrameColumn intColumnCopy = new PrimitiveDataFrameColumn<int>("IntColumn", Enumerable.Range(0, 10).Select(x => x));
            Assert.Throws<ArgumentException>(() => dataFrame.Columns[1] = intColumnCopy);

            DataFrameColumn differentIntColumn = new PrimitiveDataFrameColumn<int>("IntColumn1", Enumerable.Range(0, 10).Select(x => x));
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
        public void InsertAndRemoveColumnTests()
        {
            DataFrame dataFrame = MakeDataFrameWithAllMutableColumnTypes(10);
            DataFrameColumn intColumn = new PrimitiveDataFrameColumn<int>("IntColumn", Enumerable.Range(0, 10).Select(x => x));
            DataFrameColumn charColumn = dataFrame["Char"];
            int insertedIndex = dataFrame.Columns.Count;
            dataFrame.Columns.Insert(dataFrame.Columns.Count, intColumn);
            dataFrame.Columns.RemoveAt(0);
            DataFrameColumn intColumn_1 = dataFrame["IntColumn"];
            DataFrameColumn charColumn_1 = dataFrame["Char"];
            Assert.True(ReferenceEquals(intColumn, intColumn_1));
            Assert.True(ReferenceEquals(charColumn, charColumn_1));
        }

        [Fact]
        public void TestBinaryOperations()
        {
            DataFrame df = MakeDataFrameWithTwoColumns(12);
            IReadOnlyList<int> listOfInts = new List<int>() { 5, 5 };

            // The following binary ops return a copy
            var ret = df.Add(5);
            Assert.Equal(0, df[0, 0]);
            Assert.Equal(5, ret[0, 0]);
            ret = df.Add(listOfInts);
            Assert.Equal(0, df[0, 0]);
            Assert.Equal(5, ret[0, 0]);
            ret = df.Subtract(5);
            Assert.Equal(0, df[0, 0]);
            Assert.Equal(-5, ret[0, 0]);
            ret = df.Subtract(listOfInts);
            Assert.Equal(0, df[0, 0]);
            Assert.Equal(-5, ret[0, 0]);
            ret = df.Multiply(5);
            Assert.Equal(1, df[1, 0]);
            Assert.Equal(5, ret[1, 0]);
            ret = df.Multiply(listOfInts);
            Assert.Equal(1, df[1, 0]);
            Assert.Equal(5, ret[1, 0]);
            ret = df.Divide(5);
            Assert.Equal(5, df[5, 0]);
            Assert.Equal(1, ret[5, 0]);
            ret = df.Divide(listOfInts);
            Assert.Equal(5, df[5, 0]);
            Assert.Equal(1, ret[5, 0]);
            ret = df.Modulo(5);
            Assert.Equal(5, df[5, 0]);
            Assert.Equal(0, ret[5, 0]);
            ret = df.Modulo(listOfInts);
            Assert.Equal(5, df[5, 0]);
            Assert.Equal(0, ret[5, 0]);

            Assert.Equal(true, df.ElementwiseGreaterThanOrEqual(5)[7, 0]);
            Assert.Equal(true, df.ElementwiseGreaterThanOrEqual(listOfInts)[7, 0]);
            Assert.Equal(true, df.ElementwiseLessThanOrEqual(5)[4, 0]);
            Assert.Equal(true, df.ElementwiseLessThanOrEqual(listOfInts)[4, 0]);
            Assert.Equal(false, df.ElementwiseGreaterThan(5)[5, 0]);
            Assert.Equal(false, df.ElementwiseGreaterThan(listOfInts)[5, 0]);
            Assert.Equal(false, df.ElementwiseLessThan(5)[5, 0]);
            Assert.Equal(false, df.ElementwiseLessThan(listOfInts)[5, 0]);
            // The following binary ops are in place
            Assert.Equal(5, df.Add(5, inPlace: true)[0, 0]);
            Assert.Equal(10, df.Add(listOfInts, inPlace: true)[0, 0]);
            Assert.Equal(5, df.Subtract(5, inPlace: true)[0, 0]);
            Assert.Equal(0, df.Subtract(listOfInts, inPlace: true)[0, 0]);
            Assert.Equal(5, df.Multiply(5, inPlace: true)[1, 0]);
            Assert.Equal(25, df.Multiply(listOfInts, inPlace: true)[1, 0]);
            Assert.Equal(5, df.Divide(5, inPlace: true)[1, 0]);
            Assert.Equal(1, df.Divide(listOfInts, inPlace: true)[1, 0]);
            Assert.Equal(1, df.Modulo(5, inPlace: true)[1, 0]);
            Assert.Equal(1, df.Modulo(listOfInts, inPlace: true)[1, 0]);
            Assert.Equal(2, df.LeftShift(1)[1, 0]);
            Assert.Equal(1, df.RightShift(1)[2, 0]);
        }

        [Fact]
        public void TestBinaryOperationsWithColumns()
        {
            int length = 10;
            var df1 = MakeDataFrameWithNumericColumns(length);
            var df2 = MakeDataFrameWithNumericColumns(length);

            DataFrameColumn newColumn;
            DataFrameColumn verify;
            for (int i = 0; i < df1.Columns.Count; i++)
            {
                newColumn = df1[df1.Columns[i].Name] + df2[df2.Columns[i].Name];
                verify = newColumn.ElementwiseEquals(df1.Columns[i] * 2);
                Assert.Equal(true, verify[0]);

                newColumn = df1[df1.Columns[i].Name] - df2[df2.Columns[i].Name];
                verify = newColumn.ElementwiseEquals(0);
                Assert.Equal(true, verify[0]);

                newColumn = df1[df1.Columns[i].Name] * df2[df2.Columns[i].Name];
                verify = newColumn.ElementwiseEquals(df1.Columns[i] * df1.Columns[i]);
                Assert.Equal(true, verify[0]);

                var df1Column = df1.Columns[i] + 1;
                var df2Column = df2.Columns[i] + 1;
                newColumn = df1Column / df2Column;
                verify = newColumn.ElementwiseEquals(1);
                Assert.Equal(true, verify[0]);

                newColumn = df1Column % df2Column;
                verify = newColumn.ElementwiseEquals(0);
                Assert.Equal(true, verify[0]);

                verify = df1[df1.Columns[i].Name].ElementwiseEquals(df2[df2.Columns[i].Name]);
                Assert.True(verify.All());

                verify = df1[df1.Columns[i].Name].ElementwiseNotEquals(df2[df2.Columns[i].Name]);
                Assert.False(verify.Any());

                verify = df1[df1.Columns[i].Name].ElementwiseGreaterThanOrEqual(df2[df2.Columns[i].Name]);
                Assert.True(verify.All());

                verify = df1[df1.Columns[i].Name].ElementwiseLessThanOrEqual(df2[df2.Columns[i].Name]);
                Assert.True(verify.All());

                verify = df1[df1.Columns[i].Name].ElementwiseGreaterThan(df2[df2.Columns[i].Name]);
                Assert.False(verify.Any());

                verify = df1[df1.Columns[i].Name].ElementwiseLessThan(df2[df2.Columns[i].Name]);
                Assert.False(verify.Any());
            }
        }

        [Fact]
        public void TestBinaryOperationsWithConversions()
        {
            DataFrame df = DataFrameTests.MakeDataFrameWithTwoColumns(10);

            // Add a double to an int column
            DataFrame dfd = df.Add(5.0f);
            var dtype = dfd.Columns[0].DataType;
            Assert.True(dtype == typeof(double));

            // Add a decimal to an int column
            DataFrame dfm = df.Add(5.0m);
            dtype = dfm.Columns[0].DataType;
            Assert.True(dtype == typeof(decimal));

            // int + bool should throw
            Assert.Throws<NotSupportedException>(() => df.Add(true));

            var dataFrameColumn1 = new PrimitiveDataFrameColumn<double>("Double1", Enumerable.Range(0, 10).Select(x => (double)x));
            df.Columns[0] = dataFrameColumn1;
            // Double + comparison ops should throw
            Assert.Throws<NotSupportedException>(() => df.And(true));
        }

        [Fact]
        public void TestBinaryOperationsOnBoolColumn()
        {
            var df = new DataFrame();
            var dataFrameColumn1 = new PrimitiveDataFrameColumn<bool>("Bool1", Enumerable.Range(0, 10).Select(x => true));
            var dataFrameColumn2 = new PrimitiveDataFrameColumn<bool>("Bool2", Enumerable.Range(0, 10).Select(x => true));
            df.Columns.Insert(0, dataFrameColumn1);
            df.Columns.Insert(1, dataFrameColumn2);

            // bool + int should throw
            Assert.Throws<NotSupportedException>(() => df.Add(5));
            // Left shift should throw
            Assert.Throws<NotSupportedException>(() => df.LeftShift(5));

            IReadOnlyList<bool> listOfBools = new List<bool>() { true, false };
            // boolean and And should work
            var newdf = df.And(true);
            Assert.Equal(true, newdf[4, 0]);
            var newdf1 = df.And(listOfBools);
            Assert.Equal(false, newdf1[4, 1]);

            newdf = df.Or(true);
            Assert.Equal(true, newdf[4, 0]);
            newdf1 = df.Or(listOfBools);
            Assert.Equal(true, newdf1[4, 1]);

            newdf = df.Xor(true);
            Assert.Equal(false, newdf[4, 0]);
            newdf1 = df.Xor(listOfBools);
            Assert.Equal(true, newdf1[4, 1]);
        }

        [Fact]
        public void TestBinaryOperationsOnArrowStringColumn()
        {
            var df = new DataFrame();
            var strArrayBuilder = new StringArray.Builder();
            for (int i = 0; i < 10; i++)
            {
                strArrayBuilder.Append(i.ToString());
            }
            StringArray strArray = strArrayBuilder.Build();

            ArrowStringDataFrameColumn stringColumn = new ArrowStringDataFrameColumn("String", strArray.ValueBuffer.Memory, strArray.ValueOffsetsBuffer.Memory, strArray.NullBitmapBuffer.Memory, strArray.Length, strArray.NullCount);
            df.Columns.Insert(0, stringColumn);

            DataFrameColumn newCol = stringColumn.ElementwiseEquals(4);
            Assert.Equal(true, newCol[4]);
            Assert.Equal(false, newCol[0]);
            Assert.Equal(false, newCol[5]);

            newCol = stringColumn.ElementwiseEquals("4");
            Assert.Equal(true, newCol[4]);
            Assert.Equal(false, newCol[0]);

            newCol = stringColumn.ElementwiseEquals("foo");
            Assert.False(newCol.All());
            newCol = stringColumn.ElementwiseEquals(null);
            Assert.False(newCol.All());

            ArrowStringDataFrameColumn stringColumnCopy = new ArrowStringDataFrameColumn("String", strArray.ValueBuffer.Memory, strArray.ValueOffsetsBuffer.Memory, strArray.NullBitmapBuffer.Memory, strArray.Length, strArray.NullCount);
            newCol = stringColumn.ElementwiseEquals(stringColumnCopy);
            Assert.True(newCol.All());

            DataFrameColumn stringColumnCopyAsBaseColumn = stringColumnCopy;
            newCol = stringColumn.ElementwiseEquals(stringColumnCopyAsBaseColumn);
            Assert.True(newCol.All());

            newCol = stringColumn.ElementwiseNotEquals(5);
            Assert.Equal(true, newCol[0]);
            Assert.Equal(false, newCol[5]);

            newCol = stringColumn.ElementwiseNotEquals("5");
            Assert.Equal(true, newCol[0]);
            Assert.Equal(false, newCol[5]);

            newCol = stringColumn.ElementwiseNotEquals("foo");
            Assert.True(newCol.All());
            newCol = stringColumn.ElementwiseNotEquals(null);
            Assert.True(newCol.All());

            newCol = stringColumn.ElementwiseNotEquals(stringColumnCopy);
            Assert.False(newCol.All());

            newCol = stringColumn.ElementwiseNotEquals(stringColumnCopyAsBaseColumn);
            Assert.False(newCol.All());
        }

        [Fact]
        public void TestBinaryOperationsOnStringColumn()
        {
            var df = new DataFrame();
            DataFrameColumn stringColumn = new StringDataFrameColumn("String", Enumerable.Range(0, 10).Select(x => x.ToString()));
            df.Columns.Insert(0, stringColumn);

            DataFrameColumn newCol = stringColumn.ElementwiseEquals(5);
            Assert.Equal(true, newCol[5]);
            Assert.Equal(false, newCol[0]);

            newCol = (stringColumn as StringDataFrameColumn).ElementwiseEquals("5");
            Assert.Equal(true, newCol[5]);
            Assert.Equal(false, newCol[0]);

            DataFrameColumn stringColumnCopy = new StringDataFrameColumn("String", Enumerable.Range(0, 10).Select(x => x.ToString()));
            newCol = stringColumn.ElementwiseEquals(stringColumnCopy);
            Assert.Equal(true, newCol[5]);
            Assert.Equal(true, newCol[0]);

            StringDataFrameColumn typedStringColumn = stringColumn as StringDataFrameColumn;
            StringDataFrameColumn typedStringColumnCopy = stringColumnCopy as StringDataFrameColumn;
            newCol = typedStringColumn.ElementwiseEquals(typedStringColumnCopy);
            Assert.True(newCol.All());

            newCol = stringColumn.ElementwiseNotEquals(5);
            Assert.Equal(false, newCol[5]);
            Assert.Equal(true, newCol[0]);

            newCol = typedStringColumn.ElementwiseNotEquals("5");
            Assert.Equal(false, newCol[5]);
            Assert.Equal(true, newCol[0]);

            newCol = stringColumn.ElementwiseNotEquals(stringColumnCopy);
            Assert.Equal(false, newCol[5]);
            Assert.Equal(false, newCol[0]);

            newCol = typedStringColumn.ElementwiseNotEquals(typedStringColumnCopy);
            Assert.False(newCol.All());

            newCol = typedStringColumn.Add("suffix");
            for (int i = 0; i < newCol.Length; i++)
            {
                Assert.Equal(newCol[i], typedStringColumn[i] + "suffix");
            }
            DataFrameColumn addString = typedStringColumn + "suffix";
            for (int i = 0; i < addString.Length; i++)
            {
                Assert.Equal(addString[i], typedStringColumn[i] + "suffix");
            }
            Assert.True(newCol.ElementwiseEquals(addString).All());
            addString = "prefix" + typedStringColumn;
            for (int i = 0; i < addString.Length; i++)
            {
                Assert.Equal(addString[i], "prefix" + typedStringColumn[i]);
            }
        }

        [Fact]
        public void TestBinaryOperatorsWithConversions()
        {
            var df = MakeDataFrameWithNumericColumns(10);

            DataFrame tempDf = df + 1;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] + (double)1);
            tempDf = df + 1.1;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] + 1.1);
            tempDf = df + 1.1m;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] + 1.1m);
            Assert.True(typeof(decimal) == tempDf["Int"].DataType);

            Assert.Throws<NotSupportedException>(() => df + true);

            tempDf = df - 1.1;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] - 1.1);
            tempDf = df - 1.1m;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] - 1.1m);
            Assert.True(typeof(decimal) == tempDf["Int"].DataType);

            tempDf = df * 1.1;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] * 1.1);
            tempDf = df * 1.1m;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] * 1.1m);
            Assert.True(typeof(decimal) == tempDf["Int"].DataType);

            tempDf = df / 1.1;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] / 1.1);
            tempDf = df / 1.1m;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] / 1.1m);
            Assert.True(typeof(decimal) == tempDf["Int"].DataType);

            tempDf = df % 1.1;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] % 1.1);
            tempDf = df % 1.1m;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] % 1.1m);
            Assert.True(typeof(decimal) == tempDf["Int"].DataType);

            tempDf = 1 + df;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] + (double)1);
            tempDf = 1.1 + df;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] + 1.1);
            tempDf = 1.1m + df;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] + 1.1m);
            Assert.True(typeof(decimal) == tempDf["Int"].DataType);

            tempDf = 1.1 - df;
            Assert.Equal(tempDf[0, 0], 1.1 - (byte)df[0, 0]);
            tempDf = 1.1m - df;
            Assert.Equal(tempDf[0, 0], 1.1m - (byte)df[0, 0]);
            Assert.True(typeof(decimal) == tempDf["Int"].DataType);

            tempDf = 1.1 * df;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] * 1.1);
            tempDf = 1.1m * df;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] * 1.1m);
            Assert.True(typeof(decimal) == tempDf["Int"].DataType);

            // To prevent a divide by zero
            var plusOne = df + 1;
            tempDf = 1.1 / plusOne;
            Assert.Equal(tempDf[0, 0], 1.1 / (double)plusOne[0, 0]);
            var plusDecimal = df + 1.1m;
            tempDf = 1.1m / plusDecimal;
            Assert.Equal(tempDf[0, 0], (1.1m) / (decimal)plusDecimal[0, 0]);
            Assert.True(typeof(decimal) == tempDf["Int"].DataType);

            tempDf = 1.1 % plusOne;
            Assert.Equal(tempDf[0, 0], 1.1 % (double)plusOne[0, 0]);
            tempDf = 1.1m % plusDecimal;
            Assert.Equal(tempDf[0, 0], 1.1m % (decimal)plusDecimal[0, 0]);
            Assert.True(typeof(decimal) == tempDf["Int"].DataType);

            Assert.Equal((byte)0, df[0, 0]);
        }

        [Fact]
        public void TestBinaryOperatorsOnBoolColumns()
        {
            var df = new DataFrame();
            var dataFrameColumn1 = new PrimitiveDataFrameColumn<bool>("Bool1", Enumerable.Range(0, 10).Select(x => x % 2 == 0 ? true : false));
            var dataFrameColumn2 = new PrimitiveDataFrameColumn<bool>("Bool2", Enumerable.Range(0, 10).Select(x => x % 2 == 0 ? true : false));
            df.Columns.Insert(0, dataFrameColumn1);
            df.Columns.Insert(1, dataFrameColumn2);

            DataFrame and = df & true;
            for (int i = 0; i < dataFrameColumn1.Length; i++)
            {
                Assert.Equal(dataFrameColumn1[i], and["Bool1"][i]);
            }

            and = true & df;
            for (int i = 0; i < dataFrameColumn1.Length; i++)
            {
                Assert.Equal(dataFrameColumn1[i], and["Bool1"][i]);
            }

            DataFrame or = df | true;
            Assert.True(or.Columns[0].All());
            Assert.True(or.Columns[1].All());
            or = true | df;
            Assert.True(or.Columns[0].All());
            Assert.True(or.Columns[1].All());

            DataFrame xor = df ^ true;
            for (int i = 0; i < xor.Rows.Count; i++)
            {
                if (i % 2 == 0)
                    Assert.False((bool)xor["Bool1"][i]);
                else
                    Assert.True((bool)xor["Bool1"][i]);
            }
            xor = true ^ df;
            for (int i = 0; i < xor.Rows.Count; i++)
            {
                if (i % 2 == 0)
                    Assert.False((bool)xor["Bool1"][i]);
                else
                    Assert.True((bool)xor["Bool1"][i]);
            }
        }

        [Fact]
        public void TestBinaryOperationsOnColumns()
        {
            PrimitiveDataFrameColumn<int> column = new PrimitiveDataFrameColumn<int>("Int", Enumerable.Range(0, 10));
            Assert.ThrowsAny<ArgumentException>(() => column.Add(5.5, inPlace: true));
            Assert.ThrowsAny<ArgumentException>(() => column.ReverseAdd(5.5, inPlace: true));
            string str = "A String";
            Assert.ThrowsAny<ArgumentException>(() => column.Add(str, inPlace: true));
            Assert.ThrowsAny<ArgumentException>(() => column.ReverseAdd(str, inPlace: true));
        }

        [Fact]
        public void TestColumnReverseOrderState()
        {
            var column = new PrimitiveDataFrameColumn<int>("Int", Enumerable.Range(0, 10));
            var newColumn = 1 - column;
            var checkOrderColumn = 1 - newColumn;
            Assert.True(checkOrderColumn.ElementwiseEquals(column).All());
        }

        [Fact]
        public void TestProjectionAndAppend()
        {
            DataFrame df = MakeDataFrameWithTwoColumns(10);

            df["Int3"] = df["Int1"] * 2 + df["Int2"];
            Assert.Equal(16, df["Int3"][2]);
        }

        [Fact]
        public void TestComputations()
        {
            DataFrame df = MakeDataFrameWithAllMutableColumnTypes(10);
            df["Int"][0] = -10;
            Assert.Equal(-10, df["Int"][0]);

            DataFrameColumn absColumn = df["Int"].Abs();
            Assert.Equal(10, absColumn[0]);
            Assert.Equal(-10, df["Int"][0]);
            df["Int"].Abs(true);
            Assert.Equal(10, df["Int"][0]);

            Assert.Throws<NotSupportedException>(() => df["Byte"].All());
            Assert.Throws<NotSupportedException>(() => df["Byte"].Any());
            Assert.Throws<NotSupportedException>(() => df["Char"].All());
            Assert.Throws<NotSupportedException>(() => df["Char"].Any());
            Assert.Throws<NotSupportedException>(() => df["Decimal"].All());
            Assert.Throws<NotSupportedException>(() => df["Decimal"].Any());
            Assert.Throws<NotSupportedException>(() => df["Double"].All());
            Assert.Throws<NotSupportedException>(() => df["Double"].Any());
            Assert.Throws<NotSupportedException>(() => df["Float"].All());
            Assert.Throws<NotSupportedException>(() => df["Float"].Any());
            Assert.Throws<NotSupportedException>(() => df["Int"].All());
            Assert.Throws<NotSupportedException>(() => df["Int"].Any());
            Assert.Throws<NotSupportedException>(() => df["Long"].All());
            Assert.Throws<NotSupportedException>(() => df["Long"].Any());
            Assert.Throws<NotSupportedException>(() => df["Sbyte"].All());
            Assert.Throws<NotSupportedException>(() => df["Sbyte"].Any());
            Assert.Throws<NotSupportedException>(() => df["Short"].All());
            Assert.Throws<NotSupportedException>(() => df["Short"].Any());
            Assert.Throws<NotSupportedException>(() => df["Uint"].All());
            Assert.Throws<NotSupportedException>(() => df["Uint"].Any());
            Assert.Throws<NotSupportedException>(() => df["Ulong"].All());
            Assert.Throws<NotSupportedException>(() => df["Ulong"].Any());
            Assert.Throws<NotSupportedException>(() => df["Ushort"].All());
            Assert.Throws<NotSupportedException>(() => df["Ushort"].Any());

            bool any = df["Bool"].Any();
            bool all = df["Bool"].All();
            Assert.True(any);
            Assert.False(all);

            // Test the computation results
            df["Double"][0] = 100.0;
            DataFrameColumn doubleColumn = df["Double"].CumulativeMax();
            for (int i = 0; i < doubleColumn.Length; i++)
            {
                if (i == 5)
                    Assert.Null(doubleColumn[i]);
                else
                    Assert.Equal(100.0, (double)doubleColumn[i]);
            }
            Assert.Equal(1.0, df["Double"][1]);
            df["Double"].CumulativeMax(true);
            for (int i = 0; i < df["Double"].Length; i++)
            {
                if (i == 5)
                    Assert.Null(df["Double"][i]);
                else
                    Assert.Equal(100.0, (double)df["Double"][i]);
            }

            df["Float"][0] = -10.0f;
            DataFrameColumn floatColumn = df["Float"].CumulativeMin();
            for (int i = 0; i < floatColumn.Length; i++)
            {
                if (i == 5)
                    Assert.Null(floatColumn[i]);
                else
                    Assert.Equal(-10.0f, (float)floatColumn[i]);
            }
            Assert.Equal(9.0f, df["Float"][9]);
            df["Float"].CumulativeMin(true);
            for (int i = 0; i < df["Float"].Length; i++)
            {
                if (i == 5)
                    Assert.Null(df["Float"][i]);
                else
                    Assert.Equal(-10.0f, (float)df["Float"][i]);
            }

            DataFrameColumn uintColumn = df["Uint"].CumulativeProduct();
            Assert.Equal((uint)0, uintColumn[8]);
            Assert.Equal((uint)8, df["Uint"][8]);
            df["Uint"].CumulativeProduct(true);
            Assert.Equal((uint)0, df["Uint"][9]);

            DataFrameColumn ushortColumn = df["Ushort"].CumulativeSum();
            Assert.Equal((ushort)40, ushortColumn[9]);
            Assert.Equal((ushort)9, df["Ushort"][9]);
            df["Ushort"].CumulativeSum(true);
            Assert.Equal((ushort)40, df["Ushort"][9]);

            Assert.Equal(100.0, df["Double"].Max());
            Assert.Equal(-10.0f, df["Float"].Min());
            Assert.Equal((uint)0, df["Uint"].Product());
            Assert.Equal((ushort)140, df["Ushort"].Sum());

            df["Double"][0] = 100.1;
            Assert.Equal(100.1, df["Double"][0]);
            DataFrameColumn roundColumn = df["Double"].Round();
            Assert.Equal(100.0, roundColumn[0]);
            Assert.Equal(100.1, df["Double"][0]);
            df["Double"].Round(true);
            Assert.Equal(100.0, df["Double"][0]);

            // Test that none of the numeric column types throw
            for (int i = 0; i < df.Columns.Count; i++)
            {
                DataFrameColumn column = df.Columns[i];
                if (column.DataType == typeof(bool))
                {
                    Assert.Throws<NotSupportedException>(() => column.CumulativeMax());
                    Assert.Throws<NotSupportedException>(() => column.CumulativeMin());
                    Assert.Throws<NotSupportedException>(() => column.CumulativeProduct());
                    Assert.Throws<NotSupportedException>(() => column.CumulativeSum());
                    Assert.Throws<NotSupportedException>(() => column.Max());
                    Assert.Throws<NotSupportedException>(() => column.Min());
                    Assert.Throws<NotSupportedException>(() => column.Product());
                    Assert.Throws<NotSupportedException>(() => column.Sum());
                    continue;
                }
                else if (column.DataType == typeof(string))
                {
                    Assert.Throws<NotImplementedException>(() => column.CumulativeMax());
                    Assert.Throws<NotImplementedException>(() => column.CumulativeMin());
                    Assert.Throws<NotImplementedException>(() => column.CumulativeProduct());
                    Assert.Throws<NotImplementedException>(() => column.CumulativeSum());
                    Assert.Throws<NotImplementedException>(() => column.Max());
                    Assert.Throws<NotImplementedException>(() => column.Min());
                    Assert.Throws<NotImplementedException>(() => column.Product());
                    Assert.Throws<NotImplementedException>(() => column.Sum());
                    continue;
                }
                column.CumulativeMax();
                column.CumulativeMin();
                column.CumulativeProduct();
                column.CumulativeSum();
                column.Max();
                column.Min();
                column.Product();
                column.Sum();
            }
        }

        [Fact]
        public void TestSort()
        {
            DataFrame df = MakeDataFrameWithAllMutableColumnTypes(20);
            df["Int"][0] = 100;
            df["Int"][19] = -1;
            df["Int"][5] = 2000;

            // Sort by "Int" in ascending order
            var sortedDf = df.Sort("Int");
            Assert.Null(sortedDf["Int"][19]);
            Assert.Equal(-1, sortedDf["Int"][0]);
            Assert.Equal(100, sortedDf["Int"][17]);
            Assert.Equal(2000, sortedDf["Int"][18]);

            // Sort by "Int" in descending order
            sortedDf = df.Sort("Int", false);
            Assert.Null(sortedDf["Int"][19]);
            Assert.Equal(-1, sortedDf["Int"][18]);
            Assert.Equal(100, sortedDf["Int"][1]);
            Assert.Equal(2000, sortedDf["Int"][0]);

            // Sort by "String" in ascending order
            sortedDf = df.Sort("String");
            Assert.Null(sortedDf["Int"][19]);
            Assert.Equal(1, sortedDf["Int"][1]);
            Assert.Equal(8, sortedDf["Int"][17]);
            Assert.Equal(9, sortedDf["Int"][18]);

            sortedDf = df.Sort("String", false);
            Assert.Null(sortedDf["Int"][19]);
            Assert.Equal(8, sortedDf["Int"][1]);
            Assert.Equal(9, sortedDf["Int"][0]);
        }

        [Fact]
        public void TestSplitAndSort()
        {
            DataFrame df = MakeDataFrameWithAllMutableColumnTypes(20);
            df["Int"][0] = 100000;
            df["Int"][df.Rows.Count - 1] = -1;
            df["Int"][5] = 200000;
            DataFrame dfTest;
            DataFrame dfTrain = SplitTrainTest(df, 0.8f, out dfTest);

            // Sort by "Int" in ascending order
            var sortedDf = dfTrain.Sort("Int");
            Assert.Null(sortedDf["Int"][sortedDf.Rows.Count - 1]);
            Assert.Equal(1, sortedDf["Int"][0]);
            Assert.Equal(100000, sortedDf["Int"][sortedDf.Rows.Count - 3]);
            Assert.Equal(200000, sortedDf["Int"][sortedDf.Rows.Count - 2]);
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
            PrimitiveDataFrameColumn<int> intColumn = new PrimitiveDataFrameColumn<int>("Int", 0);
            Assert.Equal(0, intColumn.NullCount);
            intColumn.AppendMany(null, numberOfNulls);
            Assert.Equal(numberOfNulls, intColumn.NullCount);

            // Should handle all nulls
            PrimitiveDataFrameColumn<int> sortedIntColumn = intColumn.Sort() as PrimitiveDataFrameColumn<int>;
            Assert.Equal(numberOfNulls, sortedIntColumn.NullCount);
            Assert.Null(sortedIntColumn[0]);

            for (int i = 0; i < 5; i++)
            {
                intColumn.Append(i);
            }
            Assert.Equal(numberOfNulls, intColumn.NullCount);

            // Ascending sort
            sortedIntColumn = intColumn.Sort() as PrimitiveDataFrameColumn<int>;
            Assert.Equal(0, sortedIntColumn[0]);
            Assert.Null(sortedIntColumn[9]);

            // Descending sort
            sortedIntColumn = intColumn.Sort(false) as PrimitiveDataFrameColumn<int>;
            Assert.Equal(4, sortedIntColumn[0]);
            Assert.Null(sortedIntColumn[9]);
        }

        private void VerifyJoin(DataFrame join, DataFrame left, DataFrame right, JoinAlgorithm joinAlgorithm)
        {
            PrimitiveDataFrameColumn<long> mapIndices = new PrimitiveDataFrameColumn<long>("map", join.Rows.Count);
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

        private void VerifyMerge(DataFrame merge, DataFrame left, DataFrame right, JoinAlgorithm joinAlgorithm)
        {
            if (joinAlgorithm == JoinAlgorithm.Left || joinAlgorithm == JoinAlgorithm.Inner)
            {
                HashSet<int> intersection = new HashSet<int>();
                for (int i = 0; i < merge["Int_left"].Length; i++)
                {
                    if (merge["Int_left"][i] == null)
                        continue;
                    intersection.Add((int)merge["Int_left"][i]);
                }
                for (int i = 0; i < left["Int"].Length; i++)
                {
                    if (left["Int"][i] != null && intersection.Contains((int)left["Int"][i]))
                        intersection.Remove((int)left["Int"][i]);
                }
                Assert.Empty(intersection);
            }
            else if (joinAlgorithm == JoinAlgorithm.Right)
            {
                HashSet<int> intersection = new HashSet<int>();
                for (int i = 0; i < merge["Int_right"].Length; i++)
                {
                    if (merge["Int_right"][i] == null)
                        continue;
                    intersection.Add((int)merge["Int_right"][i]);
                }
                for (int i = 0; i < right["Int"].Length; i++)
                {
                    if (right["Int"][i] != null && intersection.Contains((int)right["Int"][i]))
                        intersection.Remove((int)right["Int"][i]);
                }
                Assert.Empty(intersection);
            }
            else if (joinAlgorithm == JoinAlgorithm.FullOuter)
            {
                VerifyMerge(merge, left, right, JoinAlgorithm.Left);
                VerifyMerge(merge, left, right, JoinAlgorithm.Right);
            }
        }

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
            Assert.Null(join["Int_right"][6]);
            VerifyJoin(join, left, right, JoinAlgorithm.Left);

            // Right join
            join = left.Join(right, joinAlgorithm: JoinAlgorithm.Right);
            Assert.Equal(join.Rows.Count, right.Rows.Count);
            Assert.Equal(join.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(join["Int_right"][3], right["Int"][3]);
            Assert.Null(join["Int_right"][2]);
            VerifyJoin(join, left, right, JoinAlgorithm.Right);

            // Outer join
            join = left.Join(right, joinAlgorithm: JoinAlgorithm.FullOuter);
            Assert.Equal(join.Rows.Count, left.Rows.Count);
            Assert.Equal(join.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Null(join["Int_right"][6]);
            VerifyJoin(join, left, right, JoinAlgorithm.FullOuter);

            // Inner join
            join = left.Join(right, joinAlgorithm: JoinAlgorithm.Inner);
            Assert.Equal(join.Rows.Count, right.Rows.Count);
            Assert.Equal(join.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(join["Int_right"][3], right["Int"][3]);
            Assert.Null(join["Int_right"][2]);
            VerifyJoin(join, left, right, JoinAlgorithm.Inner);

            // Tests with right.Rows.Count > left.Rows.Count
            // Left join
            right = MakeDataFrameWithAllMutableColumnTypes(15);
            join = left.Join(right);
            Assert.Equal(join.Rows.Count, left.Rows.Count);
            Assert.Equal(join.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(join["Int_right"][6], right["Int"][6]);
            VerifyJoin(join, left, right, JoinAlgorithm.Left);

            // Right join
            join = left.Join(right, joinAlgorithm: JoinAlgorithm.Right);
            Assert.Equal(join.Rows.Count, right.Rows.Count);
            Assert.Equal(join.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(join["Int_right"][2], right["Int"][2]);
            Assert.Null(join["Int_left"][12]);
            VerifyJoin(join, left, right, JoinAlgorithm.Right);

            // Outer join
            join = left.Join(right, joinAlgorithm: JoinAlgorithm.FullOuter);
            Assert.Equal(join.Rows.Count, right.Rows.Count);
            Assert.Equal(join.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Null(join["Int_left"][12]);
            VerifyJoin(join, left, right, JoinAlgorithm.FullOuter);

            // Inner join
            join = left.Join(right, joinAlgorithm: JoinAlgorithm.Inner);
            Assert.Equal(join.Rows.Count, left.Rows.Count);
            Assert.Equal(join.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(join["Int_right"][2], right["Int"][2]);
            VerifyJoin(join, left, right, JoinAlgorithm.Inner);
        }

        [Fact]
        public void TestGroupBy()
        {
            DataFrame df = MakeDataFrameWithNumericAndBoolColumns(10);
            DataFrame count = df.GroupBy("Bool").Count();
            Assert.Equal(2, count.Rows.Count);
            Assert.Equal((long)5, count["Int"][0]);
            Assert.Equal((long)4, count["Decimal"][1]);
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
                    DataFrameColumn firstColumn = first[originalColumn.Name];
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
                    DataFrameColumn headColumn = head[originalColumn.Name];
                    Assert.Equal(originalColumn[r].ToString(), headColumn[verify[r]].ToString());
                }
            }
            for (int c = 0; c < count.Columns.Count; c++)
            {
                DataFrameColumn originalColumn = df.Columns[c];
                if (originalColumn.Name == "Bool")
                    continue;
                DataFrameColumn headColumn = head[originalColumn.Name];
                Assert.Equal(originalColumn[5], headColumn[verify[5]]);
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
                    DataFrameColumn tailColumn = tail[originalColumn.Name];
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
                    DataFrameColumn maxColumn = max[originalColumn.Name];
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
                    DataFrameColumn minColumn = min[originalColumn.Name];
                    Assert.Equal("0", minColumn[r].ToString());

                    DataFrameColumn productColumn = product[originalColumn.Name];
                    Assert.Equal("0", productColumn[r].ToString());

                    DataFrameColumn sumColumn = sum[originalColumn.Name];
                    Assert.Equal("20", sumColumn[r].ToString());
                }
            }

            DataFrame columnSum = df.GroupBy("Bool").Sum("Int");
            Assert.Equal(2, columnSum.Columns.Count);
            Assert.Equal(20, columnSum["Int"][0]);
            Assert.Equal(20, columnSum["Int"][1]);
            DataFrame columnMax = df.GroupBy("Bool").Max("Int");
            Assert.Equal(2, columnMax.Columns.Count);
            Assert.Equal(8, columnMax["Int"][0]);
            Assert.Equal(9, columnMax["Int"][1]);
            DataFrame columnProduct = df.GroupBy("Bool").Product("Int");
            Assert.Equal(2, columnProduct.Columns.Count);
            Assert.Equal(0, columnProduct["Int"][0]);
            Assert.Equal(0, columnProduct["Int"][1]);
            DataFrame columnMin = df.GroupBy("Bool").Min("Int");
            Assert.Equal(2, columnMin.Columns.Count);
            Assert.Equal(0, columnMin["Int"][0]);
            Assert.Equal(0, columnMin["Int"][1]);

            DataFrame countIntColumn = df.GroupBy("Bool").Count("Int");
            Assert.Equal(2, countIntColumn.Columns.Count);
            Assert.Equal(2, countIntColumn.Rows.Count);
            Assert.Equal((long)5, countIntColumn["Int"][0]);
            Assert.Equal((long)4, countIntColumn["Int"][1]);

            DataFrame firstDecimalColumn = df.GroupBy("Bool").First("Decimal");
            Assert.Equal(2, firstDecimalColumn.Columns.Count);
            Assert.Equal(2, firstDecimalColumn.Rows.Count);
            Assert.Equal((decimal)0, firstDecimalColumn["Decimal"][0]);
            Assert.Equal((decimal)1, firstDecimalColumn["Decimal"][1]);
        }

        [Fact]
        public void TestGoupByDifferentColumnTypes()
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

            StringDataFrameColumn stringColumn = (StringDataFrameColumn)df["String"];
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

            ArrowStringDataFrameColumn arrowStringColumn = (ArrowStringDataFrameColumn)df["ArrowString"];
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

            PrimitiveDataFrameColumn<float> floatColumn = (PrimitiveDataFrameColumn<float>)df["Float"];
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

            PrimitiveDataFrameColumn<int> intColumn = (PrimitiveDataFrameColumn<int>)df["Int"];
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
        public void TestColumnClip()
        {
            DataFrame df = MakeDataFrameWithNumericColumns(10);
            // Out of place
            DataFrameColumn clipped = df["Int"].Clip(3, 7);
            Assert.Equal(3, clipped[0]);
            Assert.Equal(0, df["Int"][0]);
            Assert.Equal(3, clipped[1]);
            Assert.Equal(1, df["Int"][1]);
            Assert.Equal(3, clipped[2]);
            Assert.Equal(2, df["Int"][2]);
            Assert.Equal(3, clipped[3]);
            Assert.Equal(3, df["Int"][3]);
            Assert.Equal(4, clipped[4]);
            Assert.Equal(4, df["Int"][4]);
            Assert.Null(clipped[5]);
            Assert.Null(df["Int"][5]);
            Assert.Equal(6, clipped[6]);
            Assert.Equal(6, df["Int"][6]);
            Assert.Equal(7, clipped[7]);
            Assert.Equal(7, df["Int"][7]);
            Assert.Equal(7, clipped[8]);
            Assert.Equal(8, df["Int"][8]);
            Assert.Equal(7, clipped[9]);
            Assert.Equal(9, df["Int"][9]);

            // In place
            df["Int"].Clip(3, 7, true);
            Assert.Equal(3, df["Int"][0]);
            Assert.Equal(3, df["Int"][1]);
            Assert.Equal(3, df["Int"][2]);
            Assert.Equal(3, df["Int"][3]);
            Assert.Equal(4, df["Int"][4]);
            Assert.Null(df["Int"][5]);
            Assert.Equal(6, df["Int"][6]);
            Assert.Equal(7, df["Int"][7]);
            Assert.Equal(7, df["Int"][8]);
            Assert.Equal(7, df["Int"][9]);
        }

        [Fact]
        public void TestColumnFilter()
        {
            DataFrame df = MakeDataFrameWithNumericColumns(10);
            DataFrameColumn filtered = df["Int"].Filter(3, 7);
            Assert.Equal(4, filtered.Length);
            Assert.Equal(3, filtered[0]);
            Assert.Equal(4, filtered[1]);
            Assert.Equal(6, filtered[2]);
            Assert.Equal(7, filtered[3]);
        }

        [Fact]
        public void TestDataFrameClip()
        {
            DataFrame df = MakeDataFrameWithAllColumnTypes(10);
            IEnumerable<DataViewSchema.Column> dfColumns = ((IDataView)df).Schema;

            void VerifyDataFrameClip(DataFrame clippedColumn)
            {

                IEnumerable<DataViewSchema.Column> clippedColumns = ((IDataView)clippedColumn).Schema;
                Assert.Equal(df.Columns.Count, clippedColumn.Columns.Count);
                Assert.Equal(dfColumns, clippedColumns);
                for (int c = 0; c < df.Columns.Count; c++)
                {
                    DataFrameColumn column = clippedColumn.Columns[c];
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
            DataFrame clipped = df.Clip(3, 7);
            VerifyDataFrameClip(clipped);
            for (int i = 0; i < 10; i++)
            {
                if (i != 5)
                    Assert.Equal(i, df["Int"][i]);
                else
                    Assert.Null(df["Int"][5]);
            }

            // Inplace
            df.Clip(3, 7, true);
            VerifyDataFrameClip(df);

        }

        [Fact]
        public void TestDataFrameFilter()
        {
            DataFrame df = MakeDataFrameWithAllMutableColumnTypes(10);
            DataFrame boolColumnFiltered = df[df["Bool"].ElementwiseEquals(true)];
            List<int> verify = new List<int> { 0, 2, 4, 6, 8 };
            Assert.Equal(5, boolColumnFiltered.Rows.Count);
            for (int i = 0; i < boolColumnFiltered.Columns.Count; i++)
            {
                DataFrameColumn column = boolColumnFiltered.Columns[i];
                if (column.Name == "Char" || column.Name == "Bool" || column.Name == "String")
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
            DataFrame sampled = df.Sample(3);
            Assert.Equal(3, sampled.Rows.Count);
            Assert.Equal(df.Columns.Count, sampled.Columns.Count);
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
            Assert.Null(merge["Int_right"][6]);
            Assert.Null(merge["Int_left"][5]);
            VerifyMerge(merge, left, right, JoinAlgorithm.Left);

            // Right merge 
            merge = left.Merge<int>(right, "Int", "Int", joinAlgorithm: JoinAlgorithm.Right);
            Assert.Equal(5, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(merge["Int_right"][3], right["Int"][3]);
            Assert.Null(merge["Int_right"][2]);
            VerifyMerge(merge, left, right, JoinAlgorithm.Right);

            // Outer merge 
            merge = left.Merge<int>(right, "Int", "Int", joinAlgorithm: JoinAlgorithm.FullOuter);
            Assert.Equal(merge.Rows.Count, left.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Null(merge["Int_right"][6]);
            VerifyMerge(merge, left, right, JoinAlgorithm.FullOuter);

            // Inner merge 
            merge = left.Merge<int>(right, "Int", "Int", joinAlgorithm: JoinAlgorithm.Inner);
            Assert.Equal(merge.Rows.Count, right.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(merge["Int_right"][2], right["Int"][3]);
            Assert.Null(merge["Int_right"][4]);
            VerifyMerge(merge, left, right, JoinAlgorithm.Inner);

            // Tests with right.Rows.Count > left.Rows.Count 
            // Left merge 
            right = MakeDataFrameWithAllMutableColumnTypes(15);
            merge = left.Merge<int>(right, "Int", "Int");
            Assert.Equal(merge.Rows.Count, left.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(merge["Int_right"][6], right["Int"][6]);
            VerifyMerge(merge, left, right, JoinAlgorithm.Left);

            // Right merge 
            merge = left.Merge<int>(right, "Int", "Int", joinAlgorithm: JoinAlgorithm.Right);
            Assert.Equal(merge.Rows.Count, right.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(merge["Int_right"][2], right["Int"][2]);
            Assert.Null(merge["Int_left"][12]);
            VerifyMerge(merge, left, right, JoinAlgorithm.Right);

            // Outer merge 
            merge = left.Merge<int>(right, "Int", "Int", joinAlgorithm: JoinAlgorithm.FullOuter);
            Assert.Equal(16, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Null(merge["Int_left"][12]);
            Assert.Null(merge["Int_left"][5]);
            VerifyMerge(merge, left, right, JoinAlgorithm.FullOuter);

            // Inner merge 
            merge = left.Merge<int>(right, "Int", "Int", joinAlgorithm: JoinAlgorithm.Inner);
            Assert.Equal(9, merge.Rows.Count);
            Assert.Equal(merge.Columns.Count, left.Columns.Count + right.Columns.Count);
            Assert.Equal(merge["Int_right"][2], right["Int"][2]);
            VerifyMerge(merge, left, right, JoinAlgorithm.Inner);
        }

        [Fact]
        public void TestDescription()
        {
            DataFrame df = MakeDataFrameWithAllMutableColumnTypes(10);

            // Add a column manually here until we fix https://github.com/dotnet/corefxlab/issues/2784
            PrimitiveDataFrameColumn<DateTime> dateTimes = new PrimitiveDataFrameColumn<DateTime>("DateTimes");
            for (int i = 0; i < 10; i++)
            {
                dateTimes.Append(DateTime.Parse("2019/01/01"));
            }
            df.Columns.Add(dateTimes);

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
            Assert.Equal(dateTimeColumn.Name, dateTimes.Name);
            Assert.Equal(4, dateTimeColumn.Length);
            Assert.Equal((float)10, dateTimeColumn[0]);
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
            DataFrame df = MakeDataFrameWithAllMutableColumnTypes(20);
            DataFrame anyNulls = df.DropNulls();
            Assert.Equal(19, anyNulls.Rows.Count);

            DataFrame allNulls = df.DropNulls(DropNullOptions.All);
            Assert.Equal(19, allNulls.Rows.Count);
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
        }

        [Fact]
        public void TestValueCounts()
        {
            DataFrame df = MakeDataFrameWithAllColumnTypes(10, withNulls: false);
            DataFrame valueCounts = df["Bool"].ValueCounts();
            Assert.Equal(2, valueCounts.Rows.Count);
            Assert.Equal((long)5, valueCounts["Counts"][0]);
            Assert.Equal((long)5, valueCounts["Counts"][1]);
        }

        [Fact]
        public void TestApplyElementwiseNullCount()
        {
            DataFrame df = MakeDataFrameWithTwoColumns(10);
            PrimitiveDataFrameColumn<int> column = df["Int1"] as PrimitiveDataFrameColumn<int>;
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

        [Theory]
        [InlineData(10, 5)]
        [InlineData(20, 20)]
        public void TestClone(int dfLength, int intDfLength)
        {
            DataFrame df = MakeDataFrameWithAllColumnTypes(dfLength, withNulls: true);
            DataFrame intDf = MakeDataFrameWithTwoColumns(intDfLength, false);
            PrimitiveDataFrameColumn<int> intColumn = intDf["Int1"] as PrimitiveDataFrameColumn<int>;
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
            PrimitiveDataFrameColumn<bool> bigInts = new PrimitiveDataFrameColumn<bool>("BigInts", df["Int"].ElementwiseGreaterThan(5));
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
        public void TestAppendRow()
        {
            DataFrame df = MakeDataFrame<int, bool>(10);
            df.Append(new List<object> { 5, true });
            Assert.Equal(11, df.Rows.Count);
            Assert.Equal(1, df.Columns[0].NullCount);
            Assert.Equal(1, df.Columns[1].NullCount);

            df.Append(new List<object> { 100 });
            Assert.Equal(12, df.Rows.Count);
            Assert.Equal(1, df.Columns[0].NullCount);
            Assert.Equal(2, df.Columns[1].NullCount);

            df.Append(new List<object> { null, null });
            Assert.Equal(13, df.Rows.Count);
            Assert.Equal(2, df.Columns[0].NullCount);
            Assert.Equal(3, df.Columns[1].NullCount);

            df.Append(new Dictionary<string, object> { { "Column1", (object)5 } ,  { "Column2", false } });
            Assert.Equal(14, df.Rows.Count);
            Assert.Equal(2, df.Columns[0].NullCount);
            Assert.Equal(3, df.Columns[1].NullCount);

            df.Append(new Dictionary<string, object> { { "Column1", 5 } });
            Assert.Equal(15, df.Rows.Count);

            Assert.Equal(15, df["Column1"].Length);
            Assert.Equal(15, df["Column2"].Length);
            Assert.Equal(2, df.Columns[0].NullCount);
            Assert.Equal(4, df.Columns[1].NullCount);

            df.Append(new Dictionary<string, object> { { "Column2", false } });
            Assert.Equal(16, df.Rows.Count);
            Assert.Equal(16, df["Column1"].Length);
            Assert.Equal(16, df["Column2"].Length);
            Assert.Equal(3, df.Columns[0].NullCount);
            Assert.Equal(4, df.Columns[1].NullCount);

            df.Append((IEnumerable<object>)null);
            Assert.Equal(17, df.Rows.Count);
            Assert.Equal(17, df["Column1"].Length);
            Assert.Equal(17, df["Column2"].Length);
            Assert.Equal(4, df.Columns[0].NullCount);
            Assert.Equal(5, df.Columns[1].NullCount);

            // DataFrame must remain usable even if Append throws
            Assert.Throws<FormatException>(() => df.Append(new List<object> { 5, "str" }));
            Assert.Throws<FormatException>(() => df.Append(new Dictionary<string, object> { { "Column2", "str" } }));
            Assert.Throws<ArgumentException>(() => df.Append(new List<object> { 5, true, true }));

            df.Append();
            Assert.Equal(18, df.Rows.Count);
            Assert.Equal(18, df["Column1"].Length);
            Assert.Equal(18, df["Column2"].Length);
            Assert.Equal(5, df.Columns[0].NullCount);
            Assert.Equal(6, df.Columns[1].NullCount);
        }

        [Fact]
        public void TestAppendEmptyValue()
        {
            DataFrame df = MakeDataFrame<int, bool>(10);
            df.Append(new List<object> { "", true });
            Assert.Equal(11, df.Rows.Count);
            Assert.Equal(2, df.Columns[0].NullCount);
            Assert.Equal(1, df.Columns[1].NullCount);

            StringDataFrameColumn column = new StringDataFrameColumn("Strings", Enumerable.Range(0, 11).Select(x => x.ToString()));
            df.Columns.Add(column);

            df.Append(new List<object> { 1, true, "" });
            Assert.Equal(12, df.Rows.Count);
            Assert.Equal(2, df.Columns[0].NullCount);
            Assert.Equal(1, df.Columns[1].NullCount);
            Assert.Equal(0, df.Columns[2].NullCount);

            df.Append(new List<object> { 1, true, null });
            Assert.Equal(13, df.Rows.Count);
            Assert.Equal(1, df.Columns[2].NullCount);
        }
    }
}
