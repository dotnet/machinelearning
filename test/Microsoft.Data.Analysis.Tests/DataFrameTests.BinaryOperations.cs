// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Apache.Arrow;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.Data.Analysis.Tests
{
    public partial class DataFrameTests : BaseTestClass
    {
        public DataFrameTests(ITestOutputHelper output) : base(output, true)
        {
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
                newColumn = df1.Columns[df1.Columns[i].Name] + df2.Columns[df2.Columns[i].Name];
                verify = newColumn.ElementwiseEquals(df1.Columns[i] * 2);
                Assert.Equal(true, verify[0]);

                newColumn = df1.Columns[df1.Columns[i].Name] - df2.Columns[df2.Columns[i].Name];
                verify = newColumn.ElementwiseEquals(0);
                Assert.Equal(true, verify[0]);

                newColumn = df1.Columns[df1.Columns[i].Name] * df2.Columns[df2.Columns[i].Name];
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

                verify = df1.Columns[df1.Columns[i].Name].ElementwiseEquals(df2.Columns[df2.Columns[i].Name]);
                Assert.True(verify.All());

                verify = df1.Columns[df1.Columns[i].Name].ElementwiseNotEquals(df2.Columns[df2.Columns[i].Name]);
                Assert.False(verify.Any());

                verify = df1.Columns[df1.Columns[i].Name].ElementwiseGreaterThanOrEqual(df2.Columns[df2.Columns[i].Name]);
                Assert.True(verify.All());

                verify = df1.Columns[df1.Columns[i].Name].ElementwiseLessThanOrEqual(df2.Columns[df2.Columns[i].Name]);
                Assert.True(verify.All());

                verify = df1.Columns[df1.Columns[i].Name].ElementwiseGreaterThan(df2.Columns[df2.Columns[i].Name]);
                Assert.False(verify.Any());

                verify = df1.Columns[df1.Columns[i].Name].ElementwiseLessThan(df2.Columns[df2.Columns[i].Name]);
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

            var dataFrameColumn1 = new DoubleDataFrameColumn("Double1", Enumerable.Range(0, 10).Select(x => (double)x));
            df.Columns[0] = dataFrameColumn1;
            // Double + comparison ops should throw
            Assert.Throws<NotSupportedException>(() => df.And(true));
        }

        [Fact]
        public void TestBinaryOperationsOnBoolColumn()
        {
            var df = new DataFrame();
            var dataFrameColumn1 = new BooleanDataFrameColumn("Bool1", Enumerable.Range(0, 10).Select(x => true));
            var dataFrameColumn2 = new BooleanDataFrameColumn("Bool2", Enumerable.Range(0, 10).Select(x => true));
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
        public void TestBinaryOperationsOnDateTimeColumn()
        {
            var df = new DataFrame();
            var dataFrameColumn1 = new DateTimeDataFrameColumn("DateTime1", Enumerable.Range(0, 5).Select(x => SampleDateTime.AddDays(x)));
            // Make the second data frame column have one value that is different
            var dataFrameColumn2 = new DateTimeDataFrameColumn("DateTime2", Enumerable.Range(0, 4).Select(x => SampleDateTime.AddDays(x)));
            dataFrameColumn2.Append(SampleDateTime.AddDays(6));
            df.Columns.Insert(0, dataFrameColumn1);
            df.Columns.Insert(1, dataFrameColumn2);

            // DateTime + int should throw
            Assert.Throws<NotSupportedException>(() => df.Add(5));
            // Left shift should throw
            Assert.Throws<NotSupportedException>(() => df.LeftShift(5));
            // Right shift should throw
            Assert.Throws<NotSupportedException>(() => df.RightShift(5));

            // And should throw
            Assert.Throws<NotSupportedException>(() => df.And(true));
            // Or should throw
            Assert.Throws<NotSupportedException>(() => df.Or(true));
            // Xor should throw
            Assert.Throws<NotSupportedException>(() => df.Xor(true));

            var equalsResult = dataFrameColumn1.ElementwiseEquals(dataFrameColumn2);
            Assert.True(equalsResult[0]);
            Assert.False(equalsResult[4]);

            var equalsToScalarResult = df["DateTime1"].ElementwiseEquals(SampleDateTime);
            Assert.True(equalsToScalarResult[0]);
            Assert.False(equalsToScalarResult[1]);

            var notEqualsResult = dataFrameColumn1.ElementwiseNotEquals(dataFrameColumn2);
            Assert.False(notEqualsResult[0]);
            Assert.True(notEqualsResult[4]);

            var notEqualsToScalarResult = df["DateTime1"].ElementwiseNotEquals(SampleDateTime);
            Assert.False(notEqualsToScalarResult[0]);
            Assert.True(notEqualsToScalarResult[1]);
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
            using StringArray strArray = strArrayBuilder.Build();

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
            Assert.True(typeof(decimal) == tempDf.Columns["Int"].DataType);

            tempDf = df - 1.1;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] - 1.1);
            tempDf = df - 1.1m;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] - 1.1m);
            Assert.True(typeof(decimal) == tempDf.Columns["Int"].DataType);

            tempDf = df * 1.1;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] * 1.1);
            tempDf = df * 1.1m;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] * 1.1m);
            Assert.True(typeof(decimal) == tempDf.Columns["Int"].DataType);

            tempDf = df / 1.1;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] / 1.1);
            tempDf = df / 1.1m;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] / 1.1m);
            Assert.True(typeof(decimal) == tempDf.Columns["Int"].DataType);

            tempDf = df % 1.1;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] % 1.1);
            tempDf = df % 1.1m;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] % 1.1m);
            Assert.True(typeof(decimal) == tempDf.Columns["Int"].DataType);

            tempDf = 1 + df;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] + (double)1);
            tempDf = 1.1 + df;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] + 1.1);
            tempDf = 1.1m + df;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] + 1.1m);
            Assert.True(typeof(decimal) == tempDf.Columns["Int"].DataType);

            tempDf = 1.1 - df;
            Assert.Equal(tempDf[0, 0], 1.1 - (byte)df[0, 0]);
            tempDf = 1.1m - df;
            Assert.Equal(tempDf[0, 0], 1.1m - (byte)df[0, 0]);
            Assert.True(typeof(decimal) == tempDf.Columns["Int"].DataType);

            tempDf = 1.1 * df;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] * 1.1);
            tempDf = 1.1m * df;
            Assert.Equal(tempDf[0, 0], (byte)df[0, 0] * 1.1m);
            Assert.True(typeof(decimal) == tempDf.Columns["Int"].DataType);

            // To prevent a divide by zero
            var plusOne = df + 1;
            tempDf = 1.1 / plusOne;
            Assert.Equal(tempDf[0, 0], 1.1 / (double)plusOne[0, 0]);
            var plusDecimal = df + 1.1m;
            tempDf = 1.1m / plusDecimal;
            Assert.Equal(tempDf[0, 0], (1.1m) / (decimal)plusDecimal[0, 0]);
            Assert.True(typeof(decimal) == tempDf.Columns["Int"].DataType);

            tempDf = 1.1 % plusOne;
            Assert.Equal(tempDf[0, 0], 1.1 % (double)plusOne[0, 0]);
            tempDf = 1.1m % plusDecimal;
            Assert.Equal(tempDf[0, 0], 1.1m % (decimal)plusDecimal[0, 0]);
            Assert.True(typeof(decimal) == tempDf.Columns["Int"].DataType);

            Assert.Equal((byte)0, df[0, 0]);
        }

        [Fact]
        public void TestBinaryOperationsOnColumns()
        {
            Int32DataFrameColumn column = new Int32DataFrameColumn("Int", Enumerable.Range(0, 10));
            Assert.ThrowsAny<ArgumentException>(() => column.Add(5.5, inPlace: true));
            Assert.ThrowsAny<ArgumentException>(() => column.ReverseAdd(5.5, inPlace: true));
            string str = "A String";
            Assert.ThrowsAny<ArgumentException>(() => column.Add(str, inPlace: true));
            Assert.ThrowsAny<ArgumentException>(() => column.ReverseAdd(str, inPlace: true));
        }

        [Fact]
        public void TestBinaryOperationsOnExplodedNumericColumns()
        {
            DataFrame df = MakeDataFrameWithNumericAndBoolColumns(10, withNulls: false);
            Int32DataFrameColumn ints = df.Columns["Int"] as Int32DataFrameColumn;
            Int32DataFrameColumn res = ints.Add(1).Subtract(1).Multiply(10).Divide(10).LeftShift(2).RightShift(2);
            Assert.True(res.ElementwiseEquals(ints).All());
            Assert.True(res.ElementwiseGreaterThanOrEqual(ints).All());
            Assert.True(res.ElementwiseLessThanOrEqual(ints).All());
            Assert.False(res.ElementwiseNotEquals(ints).All());
            Assert.False(res.ElementwiseGreaterThan(ints).All());
            Assert.False(res.ElementwiseLessThan(ints).All());

            // Test inPlace
            Int32DataFrameColumn inPlace = ints.Add(1, inPlace: true).Subtract(1, inPlace: true).Multiply(10, inPlace: true).Divide(10, inPlace: true).LeftShift(2, inPlace: true).RightShift(2, inPlace: true).Add(100, inPlace: true);
            Assert.True(inPlace.ElementwiseEquals(ints).All());
            Assert.True(inPlace.ElementwiseGreaterThanOrEqual(ints).All());
            Assert.True(inPlace.ElementwiseLessThanOrEqual(ints).All());
            Assert.False(inPlace.ElementwiseNotEquals(ints).All());
            Assert.False(inPlace.ElementwiseGreaterThan(ints).All());
            Assert.False(inPlace.ElementwiseLessThan(ints).All());

            Assert.False(inPlace.ElementwiseEquals(res).All());
            Assert.True(inPlace.ElementwiseGreaterThanOrEqual(res).All());
            Assert.False(inPlace.ElementwiseLessThanOrEqual(res).All());
            Assert.True(inPlace.ElementwiseNotEquals(res).All());
            Assert.True(inPlace.ElementwiseGreaterThan(res).All());
            Assert.False(inPlace.ElementwiseLessThan(res).All());

            // Test Bool column
            BooleanDataFrameColumn bools = df.Columns["Bool"] as BooleanDataFrameColumn;
            BooleanDataFrameColumn allFalse = bools.Or(true).And(true).Xor(true);
            Assert.True(allFalse.ElementwiseEquals(false).All());

            // Test inPlace
            BooleanDataFrameColumn inPlaceAllFalse = bools.Or(true, inPlace: true).And(true, inPlace: true).Xor(true, inPlace: true);
            Assert.True(inPlaceAllFalse.ElementwiseEquals(bools).All());

            // Test Reverse Operations
            Int32DataFrameColumn reverse = ints.ReverseAdd(1).ReverseSubtract(1).ReverseMultiply(-1);
            Assert.True(reverse.ElementwiseEquals(ints).All());

            // Test inPlace
            Int32DataFrameColumn reverseInPlace = ints.ReverseAdd(1, inPlace: true).ReverseSubtract(1, inPlace: true).ReverseMultiply(-1, inPlace: true).ReverseDivide(100, inPlace: true);
            Assert.True(reverseInPlace.ElementwiseEquals(ints).All());
            Assert.False(reverseInPlace.ElementwiseEquals(reverse).All());
        }
    }
}
