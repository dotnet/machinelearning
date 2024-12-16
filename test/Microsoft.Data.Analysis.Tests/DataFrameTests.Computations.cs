// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.Data.Analysis.Tests
{
    public partial class DataFrameTests : BaseTestClass
    {
        [Fact]
        public void TestComputations()
        {
            DataFrame df = MakeDataFrameWithAllMutableColumnTypes(10);
            df["Int"][0] = -10;
            Assert.Equal(-10, df.Columns["Int"][0]);

            DataFrameColumn absColumn = df.Columns["Int"].Abs();
            Assert.Equal(10, absColumn[0]);
            Assert.Equal(-10, df.Columns["Int"][0]);
            df.Columns["Int"].Abs(true);
            Assert.Equal(10, df.Columns["Int"][0]);

            Assert.Throws<NotSupportedException>(() => df.Columns["Byte"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Byte"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Char"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Char"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["DateTime"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["DateTime"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Decimal"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Decimal"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Double"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Double"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Float"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Float"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Int"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Int"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Long"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Long"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Sbyte"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Sbyte"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Short"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Short"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Uint"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Uint"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Ulong"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Ulong"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Ushort"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Ushort"].Any());

            bool any = df.Columns["Bool"].Any();
            bool all = df.Columns["Bool"].All();
            Assert.True(any);
            Assert.False(all);

            // Test the computation results
            df.Columns["Double"][0] = 100.0;
            DataFrameColumn doubleColumn = df.Columns["Double"].CumulativeMax();
            for (int i = 0; i < doubleColumn.Length; i++)
            {
                if (i == 5)
                    Assert.Null(doubleColumn[i]);
                else
                    Assert.Equal(100.0, (double)doubleColumn[i]);
            }
            Assert.Equal(1.0, df.Columns["Double"][1]);
            df.Columns["Double"].CumulativeMax(true);
            for (int i = 0; i < df.Columns["Double"].Length; i++)
            {
                if (i == 5)
                    Assert.Null(df.Columns["Double"][i]);
                else
                    Assert.Equal(100.0, (double)df.Columns["Double"][i]);
            }

            df.Columns["Float"][0] = -10.0f;
            DataFrameColumn floatColumn = df.Columns["Float"].CumulativeMin();
            for (int i = 0; i < floatColumn.Length; i++)
            {
                if (i == 5)
                    Assert.Null(floatColumn[i]);
                else
                    Assert.Equal(-10.0f, (float)floatColumn[i]);
            }
            Assert.Equal(9.0f, df.Columns["Float"][9]);
            df.Columns["Float"].CumulativeMin(true);
            for (int i = 0; i < df.Columns["Float"].Length; i++)
            {
                if (i == 5)
                    Assert.Null(df.Columns["Float"][i]);
                else
                    Assert.Equal(-10.0f, (float)df.Columns["Float"][i]);
            }

            DataFrameColumn uintColumn = df.Columns["Uint"].CumulativeProduct();
            Assert.Equal((uint)0, uintColumn[8]);
            Assert.Equal((uint)8, df.Columns["Uint"][8]);
            df.Columns["Uint"].CumulativeProduct(true);
            Assert.Equal((uint)0, df.Columns["Uint"][9]);

            DataFrameColumn ushortColumn = df.Columns["Ushort"].CumulativeSum();
            Assert.Equal((ushort)40, ushortColumn[9]);
            Assert.Equal((ushort)9, df.Columns["Ushort"][9]);
            df.Columns["Ushort"].CumulativeSum(true);
            Assert.Equal((ushort)40, df.Columns["Ushort"][9]);

            Assert.Equal(100.0, df.Columns["Double"].Max());
            Assert.Equal(-10.0f, df.Columns["Float"].Min());
            Assert.Equal((uint)0, df.Columns["Uint"].Product());
            Assert.Equal((ushort)130, df.Columns["Ushort"].Sum());

            df.Columns["Double"][0] = 100.1;
            Assert.Equal(100.1, df.Columns["Double"][0]);
            DataFrameColumn roundColumn = df.Columns["Double"].Round();
            Assert.Equal(100.0, roundColumn[0]);
            Assert.Equal(100.1, df.Columns["Double"][0]);
            df.Columns["Double"].Round(true);
            Assert.Equal(100.0, df.Columns["Double"][0]);

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
                else if (column.DataType == typeof(DateTime))
                {
                    column.CumulativeMax();
                    column.CumulativeMin();
                    column.Max();
                    column.Min();

                    Assert.Throws<NotSupportedException>(() => column.CumulativeProduct());
                    Assert.Throws<NotSupportedException>(() => column.CumulativeSum());
                    Assert.Throws<NotSupportedException>(() => column.Product());
                    Assert.Throws<NotSupportedException>(() => column.Sum());
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
        public void TestComputationsIncludingDateTime()
        {
            DataFrame df = MakeDataFrameWithNumericStringAndDateTimeColumns(10);
            df["Int"][0] = -10;
            Assert.Equal(-10, df.Columns["Int"][0]);

            DataFrameColumn absColumn = df.Columns["Int"].Abs();
            Assert.Equal(10, absColumn[0]);
            Assert.Equal(-10, df.Columns["Int"][0]);
            df.Columns["Int"].Abs(true);
            Assert.Equal(10, df.Columns["Int"][0]);

            Assert.Throws<NotSupportedException>(() => df.Columns["Byte"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Byte"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Char"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Char"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Decimal"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Decimal"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Double"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Double"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Float"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Float"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Int"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Int"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Long"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Long"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Sbyte"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Sbyte"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Short"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Short"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Uint"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Uint"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Ulong"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Ulong"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["Ushort"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["Ushort"].Any());
            Assert.Throws<NotSupportedException>(() => df.Columns["DateTime"].All());
            Assert.Throws<NotSupportedException>(() => df.Columns["DateTime"].Any());

            // Test the computation results
            var maxDate = SampleDateTime.AddDays(100);
            df.Columns["DateTime"][0] = maxDate;
            DataFrameColumn dateTimeColumn = df.Columns["DateTime"].CumulativeMax();
            for (int i = 0; i < dateTimeColumn.Length; i++)
            {
                if (i == 5)
                    Assert.Null(dateTimeColumn[i]);
                else
                    Assert.Equal(maxDate, (DateTime)dateTimeColumn[i]);
            }
            Assert.Equal(maxDate, dateTimeColumn.Max());

            df.Columns["Double"][0] = 100.0;
            DataFrameColumn doubleColumn = df.Columns["Double"].CumulativeMax();
            for (int i = 0; i < doubleColumn.Length; i++)
            {
                if (i == 5)
                    Assert.Null(doubleColumn[i]);
                else
                    Assert.Equal(100.0, (double)doubleColumn[i]);
            }
            Assert.Equal(1.0, df.Columns["Double"][1]);
            df.Columns["Double"].CumulativeMax(true);
            for (int i = 0; i < df.Columns["Double"].Length; i++)
            {
                if (i == 5)
                    Assert.Null(df.Columns["Double"][i]);
                else
                    Assert.Equal(100.0, (double)df.Columns["Double"][i]);
            }

            df.Columns["Float"][0] = -10.0f;
            DataFrameColumn floatColumn = df.Columns["Float"].CumulativeMin();
            for (int i = 0; i < floatColumn.Length; i++)
            {
                if (i == 5)
                    Assert.Null(floatColumn[i]);
                else
                    Assert.Equal(-10.0f, (float)floatColumn[i]);
            }
            Assert.Equal(9.0f, df.Columns["Float"][9]);
            df.Columns["Float"].CumulativeMin(true);
            for (int i = 0; i < df.Columns["Float"].Length; i++)
            {
                if (i == 5)
                    Assert.Null(df.Columns["Float"][i]);
                else
                    Assert.Equal(-10.0f, (float)df.Columns["Float"][i]);
            }

            DataFrameColumn uintColumn = df.Columns["Uint"].CumulativeProduct();
            Assert.Equal((uint)0, uintColumn[8]);
            Assert.Equal((uint)8, df.Columns["Uint"][8]);
            df.Columns["Uint"].CumulativeProduct(true);
            Assert.Equal((uint)0, df.Columns["Uint"][9]);

            DataFrameColumn ushortColumn = df.Columns["Ushort"].CumulativeSum();
            Assert.Equal((ushort)40, ushortColumn[9]);
            Assert.Equal((ushort)9, df.Columns["Ushort"][9]);
            df.Columns["Ushort"].CumulativeSum(true);
            Assert.Equal((ushort)40, df.Columns["Ushort"][9]);

            Assert.Equal(100.0, df.Columns["Double"].Max());
            Assert.Equal(-10.0f, df.Columns["Float"].Min());
            Assert.Equal((uint)0, df.Columns["Uint"].Product());
            Assert.Equal((ushort)130, df.Columns["Ushort"].Sum());

            df.Columns["Double"][0] = 100.1;
            Assert.Equal(100.1, df.Columns["Double"][0]);
            DataFrameColumn roundColumn = df.Columns["Double"].Round();
            Assert.Equal(100.0, roundColumn[0]);
            Assert.Equal(100.1, df.Columns["Double"][0]);
            df.Columns["Double"].Round(true);
            Assert.Equal(100.0, df.Columns["Double"][0]);

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
                else if (column.DataType == typeof(DateTime))
                {
                    Assert.Throws<NotSupportedException>(() => column.CumulativeProduct());
                    Assert.Throws<NotSupportedException>(() => column.CumulativeSum());
                    Assert.Throws<NotSupportedException>(() => column.Product());
                    Assert.Throws<NotSupportedException>(() => column.Sum());
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
        public void TestIntComputations_MaxMin_WithNulls()
        {
            var column = new Int32DataFrameColumn("Int", new int?[]
                {
                    null,
                    2,
                    1,
                    4,
                    3,
                    null
                });

            Assert.Equal(1, column.Min());
            Assert.Equal(4, column.Max());
        }

        [Fact]
        public void TestIntSum_OnColumnWithNullsOnly()
        {
            var column = new Int32DataFrameColumn("Int", new int?[] { null, null });
            Assert.Null(column.Sum());
        }

        [Fact]
        public void TestIntSum_OnEmptyColumn()
        {
            var column = new Int32DataFrameColumn("Int");
            Assert.Null(column.Sum());
        }

        [Fact]
        public void TestIntComputations_MaxMin_OnEmptyColumn()
        {
            var column = new Int32DataFrameColumn("Int");

            Assert.Null(column.Min());
            Assert.Null(column.Max());
        }

        [Fact]
        public void TestDateTimeComputations_MaxMin_OnEmptyColumn()
        {
            var column = new DateTimeDataFrameColumn("DateTime");

            Assert.Null(column.Min());
            Assert.Null(column.Max());
        }

        [Fact]
        public void TestDateTimeComputations_MaxMin_WithNulls()
        {
            var dateTimeColumn = new DateTimeDataFrameColumn("DateTime", new DateTime?[]
                {
                    null,
                    new DateTime(2022, 1, 1),
                    new DateTime(2020, 1, 1),
                    new DateTime(2023, 1, 1),
                    new DateTime(2021, 1, 1),
                    null
                });

            Assert.Equal(new DateTime(2020, 1, 1), dateTimeColumn.Min());
            Assert.Equal(new DateTime(2023, 1, 1), dateTimeColumn.Max());
        }

        [Theory]
        [InlineData(5, 10)]
        [InlineData(-15, 10)]
        [InlineData(-5, 10)]
        public void TestComputations_WithNegativeNumbers_MaxMin_Calculated(int startingFrom, int length)
        {
            // Arrange

            IEnumerable<int> range = Enumerable.Range(startingFrom, length);

            int max = range.Max();
            int min = range.Min();

            DataFrame df = MakeDataFrameWithNumericColumns(length, withNulls: false, startingFrom);

            var byteColumn = (PrimitiveDataFrameColumn<byte>)df.Columns["Byte"];
            var decimalColumn = (PrimitiveDataFrameColumn<decimal>)df.Columns["Decimal"];
            var doubleColumn = (PrimitiveDataFrameColumn<double>)df.Columns["Double"];
            var floatColumn = (PrimitiveDataFrameColumn<float>)df.Columns["Float"];
            var intColumn = (PrimitiveDataFrameColumn<int>)df.Columns["Int"];
            var longColumn = (PrimitiveDataFrameColumn<long>)df.Columns["Long"];
            var sbyteColumn = (PrimitiveDataFrameColumn<sbyte>)df.Columns["Sbyte"];
            var shortColumn = (PrimitiveDataFrameColumn<short>)df.Columns["Short"];
            var uintColumn = (PrimitiveDataFrameColumn<uint>)df.Columns["Uint"];
            var ulongColumn = (PrimitiveDataFrameColumn<ulong>)df.Columns["Ulong"];
            var ushortColumn = (PrimitiveDataFrameColumn<ushort>)df.Columns["Ushort"];

            // Act, Assert

            // We need to iterate over all range with conversion to byte due to negative numbers issue
            Assert.Equal((byte)byteColumn.Max(), range.Select(x => (byte)x).Max());

            Assert.Equal((decimal)decimalColumn.Max(), (decimal)max);
            Assert.Equal((double)doubleColumn.Max(), (double)max);
            Assert.Equal((float)floatColumn.Max(), (float)max);
            Assert.Equal((int)intColumn.Max(), (int)max);
            Assert.Equal((long)longColumn.Max(), (long)max);
            Assert.Equal((sbyte)sbyteColumn.Max(), (sbyte)max);
            Assert.Equal((short)shortColumn.Max(), (short)max);

            // We need to iterate over all range with conversion to uint due to negative numbers issue
            Assert.Equal((uint)uintColumn.Max(), range.Select(x => (uint)x).Max());

            // We need to iterate over all range with conversion to ulong due to negative numbers issue
            Assert.Equal((ulong)ulongColumn.Max(), range.Select(x => (ulong)x).Max());

            // We need to iterate over all range with conversion to ushort due to negative numbers issue
            Assert.Equal((ushort)ushortColumn.Max(), range.Select(x => (ushort)x).Max());

            // We need to iterate over all range with conversion to byte due to negative numbers issue
            Assert.Equal((byte)byteColumn.Min(), range.Select(x => (byte)x).Min());

            Assert.Equal((decimal)decimalColumn.Min(), (decimal)min);
            Assert.Equal((double)doubleColumn.Min(), (double)min);
            Assert.Equal((float)floatColumn.Min(), (float)min);
            Assert.Equal((int)intColumn.Min(), (int)min);
            Assert.Equal((long)longColumn.Min(), (long)min);
            Assert.Equal((sbyte)sbyteColumn.Min(), (sbyte)min);
            Assert.Equal((short)shortColumn.Min(), (short)min);

            // We need to iterate over all range with conversion to uint due to negative numbers issue
            Assert.Equal((uint)uintColumn.Min(), range.Select(x => (uint)x).Min());

            // We need to iterate over all range with conversion to ulong due to negative numbers issue
            Assert.Equal((ulong)ulongColumn.Min(), range.Select(x => (ulong)x).Min());

            // We need to iterate over all range with conversion to ushort due to negative numbers issue
            Assert.Equal((ushort)ushortColumn.Min(), range.Select(x => (ushort)x).Min());
        }

        [Fact]
        public void Test_Logical_Computation_And()
        {
            var col1 = new BooleanDataFrameColumn("col1", new Boolean[] { true, false, true });
            var col2 = new BooleanDataFrameColumn("col2", new Boolean[] { false, true, true });
            var dfTest = new DataFrame(col1, col2);
            var col3 = dfTest["col1"].And(dfTest["col2"]);
            var col4 = col1.And(col2);

            for (int i = 0; i < col1.Length; i++)
            {
                var exprectedValue = col1[i] & col2[i];
                Assert.Equal(exprectedValue, col3[i]);
                Assert.Equal(exprectedValue, col4[i]);
            }
        }

        [Fact]
        public void Test_Logical_Computation_Or()
        {
            var col1 = new BooleanDataFrameColumn("col1", new Boolean[] { true, false, true });
            var col2 = new BooleanDataFrameColumn("col2", new Boolean[] { false, true, true });
            var dfTest = new DataFrame(col1, col2);
            var col3 = dfTest["col1"].Or(dfTest["col2"]);
            var col4 = col1.Or(col2);

            for (int i = 0; i < col1.Length; i++)
            {
                var exprectedValue = col1[i] | col2[i];
                Assert.Equal(exprectedValue, col3[i]);
                Assert.Equal(exprectedValue, col4[i]);
            }
        }
    }
}
