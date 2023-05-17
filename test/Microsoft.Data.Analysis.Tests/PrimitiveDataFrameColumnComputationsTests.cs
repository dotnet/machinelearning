

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveDataFrameColumnComputationsTests.tt. Do not modify directly


using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace Microsoft.Data.Analysis.Tests
{
    public partial class DataFrameColumnTests
    {
        IEnumerable<byte?> ByteValues = new byte?[] { 1, null, 2, 3, 4, null, 6, 7 };
        IEnumerable<char?> CharValues = new char?[] { (char)1, null, (char)2, (char)3, (char)4, null, (char)6, (char)7 };
        IEnumerable<decimal?> DecimalValues = new decimal?[] { 1, null, 2, 3, 4, null, 6, 7 };
        IEnumerable<double?> DoubleValues = new double?[] { 1, null, 2, 3, 4, null, 6, 7 };
        IEnumerable<float?> SingleValues = new float?[] { 1, null, 2, 3, 4, null, 6, 7 };
        IEnumerable<int?> Int32Values = new int?[] { 1, null, 2, 3, 4, null, 6, 7 };
        IEnumerable<long?> Int64Values = new long?[] { 1, null, 2, 3, 4, null, 6, 7 };
        IEnumerable<sbyte?> SByteValues = new sbyte?[] { 1, null, 2, 3, 4, null, 6, 7 };
        IEnumerable<short?> Int16Values = new short?[] { 1, null, 2, 3, 4, null, 6, 7 };
        IEnumerable<uint?> UInt32Values = new uint?[] { 1, null, 2, 3, 4, null, 6, 7 };
        IEnumerable<ulong?> UInt64Values = new ulong?[] { 1, null, 2, 3, 4, null, 6, 7 };
        IEnumerable<ushort?> UInt16Values = new ushort?[] { 1, null, 2, 3, 4, null, 6, 7 };


        [Fact]
        public void ByteColumnComputationsTests()
        {

            var column = new ByteDataFrameColumn("byteValues", ByteValues);

            Assert.Equal(Enumerable.Max(Int32Values), Convert.ToInt32(column.Max()));
            Assert.Equal(Enumerable.Min(Int32Values), Convert.ToInt32(column.Min()));
            Assert.Equal(Enumerable.Sum(Int32Values), Convert.ToInt32(column.Sum()));
        }
        [Fact]
        public void CharColumnComputationsTests()
        {

            var column = new CharDataFrameColumn("charValues", CharValues);

            Assert.Equal(Enumerable.Max(Int32Values), Convert.ToInt32(column.Max()));
            Assert.Equal(Enumerable.Min(Int32Values), Convert.ToInt32(column.Min()));
            Assert.Equal(Enumerable.Sum(Int32Values), Convert.ToInt32(column.Sum()));
        }
        [Fact]
        public void DecimalColumnComputationsTests()
        {

            var column = new DecimalDataFrameColumn("decimalValues", DecimalValues);

            Assert.Equal(Enumerable.Max(Int32Values), Convert.ToInt32(column.Max()));
            Assert.Equal(Enumerable.Min(Int32Values), Convert.ToInt32(column.Min()));
            Assert.Equal(Enumerable.Sum(Int32Values), Convert.ToInt32(column.Sum()));
        }
        [Fact]
        public void DoubleColumnComputationsTests()
        {

            var column = new DoubleDataFrameColumn("doubleValues", DoubleValues);

            Assert.Equal(Enumerable.Max(Int32Values), Convert.ToInt32(column.Max()));
            Assert.Equal(Enumerable.Min(Int32Values), Convert.ToInt32(column.Min()));
            Assert.Equal(Enumerable.Sum(Int32Values), Convert.ToInt32(column.Sum()));
        }
        [Fact]
        public void SingleColumnComputationsTests()
        {

            var column = new SingleDataFrameColumn("floatValues", SingleValues);

            Assert.Equal(Enumerable.Max(Int32Values), Convert.ToInt32(column.Max()));
            Assert.Equal(Enumerable.Min(Int32Values), Convert.ToInt32(column.Min()));
            Assert.Equal(Enumerable.Sum(Int32Values), Convert.ToInt32(column.Sum()));
        }
        [Fact]
        public void Int32ColumnComputationsTests()
        {

            var column = new Int32DataFrameColumn("intValues", Int32Values);

            Assert.Equal(Enumerable.Max(Int32Values), Convert.ToInt32(column.Max()));
            Assert.Equal(Enumerable.Min(Int32Values), Convert.ToInt32(column.Min()));
            Assert.Equal(Enumerable.Sum(Int32Values), Convert.ToInt32(column.Sum()));
        }
        [Fact]
        public void Int64ColumnComputationsTests()
        {

            var column = new Int64DataFrameColumn("longValues", Int64Values);

            Assert.Equal(Enumerable.Max(Int32Values), Convert.ToInt32(column.Max()));
            Assert.Equal(Enumerable.Min(Int32Values), Convert.ToInt32(column.Min()));
            Assert.Equal(Enumerable.Sum(Int32Values), Convert.ToInt32(column.Sum()));
        }
        [Fact]
        public void SByteColumnComputationsTests()
        {

            var column = new SByteDataFrameColumn("sbyteValues", SByteValues);

            Assert.Equal(Enumerable.Max(Int32Values), Convert.ToInt32(column.Max()));
            Assert.Equal(Enumerable.Min(Int32Values), Convert.ToInt32(column.Min()));
            Assert.Equal(Enumerable.Sum(Int32Values), Convert.ToInt32(column.Sum()));
        }
        [Fact]
        public void Int16ColumnComputationsTests()
        {

            var column = new Int16DataFrameColumn("shortValues", Int16Values);

            Assert.Equal(Enumerable.Max(Int32Values), Convert.ToInt32(column.Max()));
            Assert.Equal(Enumerable.Min(Int32Values), Convert.ToInt32(column.Min()));
            Assert.Equal(Enumerable.Sum(Int32Values), Convert.ToInt32(column.Sum()));
        }
        [Fact]
        public void UInt32ColumnComputationsTests()
        {

            var column = new UInt32DataFrameColumn("uintValues", UInt32Values);

            Assert.Equal(Enumerable.Max(Int32Values), Convert.ToInt32(column.Max()));
            Assert.Equal(Enumerable.Min(Int32Values), Convert.ToInt32(column.Min()));
            Assert.Equal(Enumerable.Sum(Int32Values), Convert.ToInt32(column.Sum()));
        }
        [Fact]
        public void UInt64ColumnComputationsTests()
        {

            var column = new UInt64DataFrameColumn("ulongValues", UInt64Values);

            Assert.Equal(Enumerable.Max(Int32Values), Convert.ToInt32(column.Max()));
            Assert.Equal(Enumerable.Min(Int32Values), Convert.ToInt32(column.Min()));
            Assert.Equal(Enumerable.Sum(Int32Values), Convert.ToInt32(column.Sum()));
        }
        [Fact]
        public void UInt16ColumnComputationsTests()
        {

            var column = new UInt16DataFrameColumn("ushortValues", UInt16Values);

            Assert.Equal(Enumerable.Max(Int32Values), Convert.ToInt32(column.Max()));
            Assert.Equal(Enumerable.Min(Int32Values), Convert.ToInt32(column.Min()));
            Assert.Equal(Enumerable.Sum(Int32Values), Convert.ToInt32(column.Sum()));
        }
    }
}



