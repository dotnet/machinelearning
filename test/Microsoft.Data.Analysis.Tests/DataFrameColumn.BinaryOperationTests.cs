

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from DataFrameColumn.BinaryOperationTests.tt. Do not modify directly

using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace Microsoft.Data.Analysis.Tests
{
    public partial class DataFrameColumnTests
    {
        [Fact]
        public void AddByteDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn otherColumn = new ByteDataFrameColumn("Byte", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column + otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(2 * x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddDecimalDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (decimal)x);
            DecimalDataFrameColumn otherColumn = new DecimalDataFrameColumn("Decimal", otherColumnEnumerable);
            DecimalDataFrameColumn columnResult = column + otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (decimal)(2 * x));
            var verifyColumn = new DecimalDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddDoubleDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (double)x);
            DoubleDataFrameColumn otherColumn = new DoubleDataFrameColumn("Double", otherColumnEnumerable);
            DoubleDataFrameColumn columnResult = column + otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (double)(2 * x));
            var verifyColumn = new DoubleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddSingleDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (float)x);
            SingleDataFrameColumn otherColumn = new SingleDataFrameColumn("Single", otherColumnEnumerable);
            SingleDataFrameColumn columnResult = column + otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (float)(2 * x));
            var verifyColumn = new SingleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddInt32DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (int)x);
            Int32DataFrameColumn otherColumn = new Int32DataFrameColumn("Int32", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column + otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(2 * x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddInt64DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (long)x);
            Int64DataFrameColumn otherColumn = new Int64DataFrameColumn("Int64", otherColumnEnumerable);
            Int64DataFrameColumn columnResult = column + otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (long)(2 * x));
            var verifyColumn = new Int64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddSByteDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (sbyte)x);
            SByteDataFrameColumn otherColumn = new SByteDataFrameColumn("SByte", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column + otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(2 * x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddInt16DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (short)x);
            Int16DataFrameColumn otherColumn = new Int16DataFrameColumn("Int16", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column + otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(2 * x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddUInt32DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (uint)x);
            UInt32DataFrameColumn otherColumn = new UInt32DataFrameColumn("UInt32", otherColumnEnumerable);
            UInt32DataFrameColumn columnResult = column + otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (uint)(2 * x));
            var verifyColumn = new UInt32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddUInt64DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (ulong)x);
            UInt64DataFrameColumn otherColumn = new UInt64DataFrameColumn("UInt64", otherColumnEnumerable);
            UInt64DataFrameColumn columnResult = column + otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (ulong)(2 * x));
            var verifyColumn = new UInt64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddUInt16DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (ushort)x);
            UInt16DataFrameColumn otherColumn = new UInt16DataFrameColumn("UInt16", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column + otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(2 * x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            byte value = 5;
            Int32DataFrameColumn columnResult = column + value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x + (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddDecimalToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            decimal value = 5;
            DecimalDataFrameColumn columnResult = column + value;
            var verify = Enumerable.Range(1, 10).Select(x => (decimal)((decimal)x + (decimal)value));
            var verifyColumn = new DecimalDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddDoubleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            double value = 5;
            DoubleDataFrameColumn columnResult = column + value;
            var verify = Enumerable.Range(1, 10).Select(x => (double)((double)x + (double)value));
            var verifyColumn = new DoubleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddSingleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            float value = 5;
            SingleDataFrameColumn columnResult = column + value;
            var verify = Enumerable.Range(1, 10).Select(x => (float)((float)x + (float)value));
            var verifyColumn = new SingleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            int value = 5;
            Int32DataFrameColumn columnResult = column + value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x + (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            long value = 5;
            Int64DataFrameColumn columnResult = column + value;
            var verify = Enumerable.Range(1, 10).Select(x => (long)((long)x + (long)value));
            var verifyColumn = new Int64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddSByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            sbyte value = 5;
            Int32DataFrameColumn columnResult = column + value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x + (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            short value = 5;
            Int32DataFrameColumn columnResult = column + value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x + (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddUInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            uint value = 5;
            UInt32DataFrameColumn columnResult = column + value;
            var verify = Enumerable.Range(1, 10).Select(x => (uint)((uint)x + (uint)value));
            var verifyColumn = new UInt32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddUInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ulong value = 5;
            UInt64DataFrameColumn columnResult = column + value;
            var verify = Enumerable.Range(1, 10).Select(x => (ulong)((ulong)x + (ulong)value));
            var verifyColumn = new UInt64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void AddUInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ushort value = 5;
            Int32DataFrameColumn columnResult = column + value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x + (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseAddByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            byte value = 5;
            Int32DataFrameColumn columnResult = value + column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x + (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseAddDecimalToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            decimal value = 5;
            DecimalDataFrameColumn columnResult = value + column;
            var verify = Enumerable.Range(1, 10).Select(x => (decimal)((decimal)x + (decimal)value));
            var verifyColumn = new DecimalDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseAddDoubleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            double value = 5;
            DoubleDataFrameColumn columnResult = value + column;
            var verify = Enumerable.Range(1, 10).Select(x => (double)((double)x + (double)value));
            var verifyColumn = new DoubleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseAddSingleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            float value = 5;
            SingleDataFrameColumn columnResult = value + column;
            var verify = Enumerable.Range(1, 10).Select(x => (float)((float)x + (float)value));
            var verifyColumn = new SingleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseAddInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            int value = 5;
            Int32DataFrameColumn columnResult = value + column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x + (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseAddInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            long value = 5;
            Int64DataFrameColumn columnResult = value + column;
            var verify = Enumerable.Range(1, 10).Select(x => (long)((long)x + (long)value));
            var verifyColumn = new Int64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseAddSByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            sbyte value = 5;
            Int32DataFrameColumn columnResult = value + column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x + (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseAddInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            short value = 5;
            Int32DataFrameColumn columnResult = value + column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x + (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseAddUInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            uint value = 5;
            UInt32DataFrameColumn columnResult = value + column;
            var verify = Enumerable.Range(1, 10).Select(x => (uint)((uint)x + (uint)value));
            var verifyColumn = new UInt32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseAddUInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ulong value = 5;
            UInt64DataFrameColumn columnResult = value + column;
            var verify = Enumerable.Range(1, 10).Select(x => (ulong)((ulong)x + (ulong)value));
            var verifyColumn = new UInt64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseAddUInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ushort value = 5;
            Int32DataFrameColumn columnResult = value + column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x + (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractByteDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn otherColumn = new ByteDataFrameColumn("Byte", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column - otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)0);
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractDecimalDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (decimal)x);
            DecimalDataFrameColumn otherColumn = new DecimalDataFrameColumn("Decimal", otherColumnEnumerable);
            DecimalDataFrameColumn columnResult = column - otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (decimal)0);
            var verifyColumn = new DecimalDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractDoubleDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (double)x);
            DoubleDataFrameColumn otherColumn = new DoubleDataFrameColumn("Double", otherColumnEnumerable);
            DoubleDataFrameColumn columnResult = column - otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (double)0);
            var verifyColumn = new DoubleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractSingleDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (float)x);
            SingleDataFrameColumn otherColumn = new SingleDataFrameColumn("Single", otherColumnEnumerable);
            SingleDataFrameColumn columnResult = column - otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (float)0);
            var verifyColumn = new SingleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractInt32DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (int)x);
            Int32DataFrameColumn otherColumn = new Int32DataFrameColumn("Int32", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column - otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)0);
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractInt64DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (long)x);
            Int64DataFrameColumn otherColumn = new Int64DataFrameColumn("Int64", otherColumnEnumerable);
            Int64DataFrameColumn columnResult = column - otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (long)0);
            var verifyColumn = new Int64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractSByteDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (sbyte)x);
            SByteDataFrameColumn otherColumn = new SByteDataFrameColumn("SByte", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column - otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)0);
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractInt16DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (short)x);
            Int16DataFrameColumn otherColumn = new Int16DataFrameColumn("Int16", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column - otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)0);
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractUInt32DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (uint)x);
            UInt32DataFrameColumn otherColumn = new UInt32DataFrameColumn("UInt32", otherColumnEnumerable);
            UInt32DataFrameColumn columnResult = column - otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (uint)0);
            var verifyColumn = new UInt32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractUInt64DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (ulong)x);
            UInt64DataFrameColumn otherColumn = new UInt64DataFrameColumn("UInt64", otherColumnEnumerable);
            UInt64DataFrameColumn columnResult = column - otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (ulong)0);
            var verifyColumn = new UInt64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractUInt16DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (ushort)x);
            UInt16DataFrameColumn otherColumn = new UInt16DataFrameColumn("UInt16", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column - otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)0);
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            byte value = 5;
            Int32DataFrameColumn columnResult = column - value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x - (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractDecimalToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            decimal value = 5;
            DecimalDataFrameColumn columnResult = column - value;
            var verify = Enumerable.Range(1, 10).Select(x => (decimal)((decimal)x - (decimal)value));
            var verifyColumn = new DecimalDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractDoubleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            double value = 5;
            DoubleDataFrameColumn columnResult = column - value;
            var verify = Enumerable.Range(1, 10).Select(x => (double)((double)x - (double)value));
            var verifyColumn = new DoubleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractSingleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            float value = 5;
            SingleDataFrameColumn columnResult = column - value;
            var verify = Enumerable.Range(1, 10).Select(x => (float)((float)x - (float)value));
            var verifyColumn = new SingleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            int value = 5;
            Int32DataFrameColumn columnResult = column - value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x - (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            long value = 5;
            Int64DataFrameColumn columnResult = column - value;
            var verify = Enumerable.Range(1, 10).Select(x => (long)((long)x - (long)value));
            var verifyColumn = new Int64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractSByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            sbyte value = 5;
            Int32DataFrameColumn columnResult = column - value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x - (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            short value = 5;
            Int32DataFrameColumn columnResult = column - value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x - (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractUInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            uint value = 5;
            UInt32DataFrameColumn columnResult = column - value;
            var verify = Enumerable.Range(1, 10).Select(x => (uint)((uint)x - (uint)value));
            var verifyColumn = new UInt32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractUInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ulong value = 5;
            UInt64DataFrameColumn columnResult = column - value;
            var verify = Enumerable.Range(1, 10).Select(x => (ulong)((ulong)x - (ulong)value));
            var verifyColumn = new UInt64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void SubtractUInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ushort value = 5;
            Int32DataFrameColumn columnResult = column - value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x - (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseSubtractByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            byte value = 5;
            Int32DataFrameColumn columnResult = value - column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)value - (int)x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseSubtractDecimalToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            decimal value = 5;
            DecimalDataFrameColumn columnResult = value - column;
            var verify = Enumerable.Range(1, 10).Select(x => (decimal)((decimal)value - (decimal)x));
            var verifyColumn = new DecimalDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseSubtractDoubleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            double value = 5;
            DoubleDataFrameColumn columnResult = value - column;
            var verify = Enumerable.Range(1, 10).Select(x => (double)((double)value - (double)x));
            var verifyColumn = new DoubleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseSubtractSingleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            float value = 5;
            SingleDataFrameColumn columnResult = value - column;
            var verify = Enumerable.Range(1, 10).Select(x => (float)((float)value - (float)x));
            var verifyColumn = new SingleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseSubtractInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            int value = 5;
            Int32DataFrameColumn columnResult = value - column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)value - (int)x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseSubtractInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            long value = 5;
            Int64DataFrameColumn columnResult = value - column;
            var verify = Enumerable.Range(1, 10).Select(x => (long)((long)value - (long)x));
            var verifyColumn = new Int64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseSubtractSByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            sbyte value = 5;
            Int32DataFrameColumn columnResult = value - column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)value - (int)x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseSubtractInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            short value = 5;
            Int32DataFrameColumn columnResult = value - column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)value - (int)x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseSubtractUInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            uint value = 5;
            UInt32DataFrameColumn columnResult = value - column;
            var verify = Enumerable.Range(1, 10).Select(x => (uint)((uint)value - (uint)x));
            var verifyColumn = new UInt32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseSubtractUInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ulong value = 5;
            UInt64DataFrameColumn columnResult = value - column;
            var verify = Enumerable.Range(1, 10).Select(x => (ulong)((ulong)value - (ulong)x));
            var verifyColumn = new UInt64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseSubtractUInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ushort value = 5;
            Int32DataFrameColumn columnResult = value - column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)value - (int)x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyByteDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn otherColumn = new ByteDataFrameColumn("Byte", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column * otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(x * x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyDecimalDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (decimal)x);
            DecimalDataFrameColumn otherColumn = new DecimalDataFrameColumn("Decimal", otherColumnEnumerable);
            DecimalDataFrameColumn columnResult = column * otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (decimal)(x * x));
            var verifyColumn = new DecimalDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyDoubleDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (double)x);
            DoubleDataFrameColumn otherColumn = new DoubleDataFrameColumn("Double", otherColumnEnumerable);
            DoubleDataFrameColumn columnResult = column * otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (double)(x * x));
            var verifyColumn = new DoubleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplySingleDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (float)x);
            SingleDataFrameColumn otherColumn = new SingleDataFrameColumn("Single", otherColumnEnumerable);
            SingleDataFrameColumn columnResult = column * otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (float)(x * x));
            var verifyColumn = new SingleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyInt32DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (int)x);
            Int32DataFrameColumn otherColumn = new Int32DataFrameColumn("Int32", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column * otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(x * x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyInt64DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (long)x);
            Int64DataFrameColumn otherColumn = new Int64DataFrameColumn("Int64", otherColumnEnumerable);
            Int64DataFrameColumn columnResult = column * otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (long)(x * x));
            var verifyColumn = new Int64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplySByteDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (sbyte)x);
            SByteDataFrameColumn otherColumn = new SByteDataFrameColumn("SByte", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column * otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(x * x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyInt16DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (short)x);
            Int16DataFrameColumn otherColumn = new Int16DataFrameColumn("Int16", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column * otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(x * x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyUInt32DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (uint)x);
            UInt32DataFrameColumn otherColumn = new UInt32DataFrameColumn("UInt32", otherColumnEnumerable);
            UInt32DataFrameColumn columnResult = column * otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (uint)(x * x));
            var verifyColumn = new UInt32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyUInt64DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (ulong)x);
            UInt64DataFrameColumn otherColumn = new UInt64DataFrameColumn("UInt64", otherColumnEnumerable);
            UInt64DataFrameColumn columnResult = column * otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (ulong)(x * x));
            var verifyColumn = new UInt64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyUInt16DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (ushort)x);
            UInt16DataFrameColumn otherColumn = new UInt16DataFrameColumn("UInt16", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column * otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(x * x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            byte value = 5;
            Int32DataFrameColumn columnResult = column * value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x * (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyDecimalToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            decimal value = 5;
            DecimalDataFrameColumn columnResult = column * value;
            var verify = Enumerable.Range(1, 10).Select(x => (decimal)((decimal)x * (decimal)value));
            var verifyColumn = new DecimalDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyDoubleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            double value = 5;
            DoubleDataFrameColumn columnResult = column * value;
            var verify = Enumerable.Range(1, 10).Select(x => (double)((double)x * (double)value));
            var verifyColumn = new DoubleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplySingleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            float value = 5;
            SingleDataFrameColumn columnResult = column * value;
            var verify = Enumerable.Range(1, 10).Select(x => (float)((float)x * (float)value));
            var verifyColumn = new SingleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            int value = 5;
            Int32DataFrameColumn columnResult = column * value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x * (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            long value = 5;
            Int64DataFrameColumn columnResult = column * value;
            var verify = Enumerable.Range(1, 10).Select(x => (long)((long)x * (long)value));
            var verifyColumn = new Int64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplySByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            sbyte value = 5;
            Int32DataFrameColumn columnResult = column * value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x * (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            short value = 5;
            Int32DataFrameColumn columnResult = column * value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x * (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyUInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            uint value = 5;
            UInt32DataFrameColumn columnResult = column * value;
            var verify = Enumerable.Range(1, 10).Select(x => (uint)((uint)x * (uint)value));
            var verifyColumn = new UInt32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyUInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ulong value = 5;
            UInt64DataFrameColumn columnResult = column * value;
            var verify = Enumerable.Range(1, 10).Select(x => (ulong)((ulong)x * (ulong)value));
            var verifyColumn = new UInt64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void MultiplyUInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ushort value = 5;
            Int32DataFrameColumn columnResult = column * value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x * (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseMultiplyByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            byte value = 5;
            Int32DataFrameColumn columnResult = value * column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x * (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseMultiplyDecimalToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            decimal value = 5;
            DecimalDataFrameColumn columnResult = value * column;
            var verify = Enumerable.Range(1, 10).Select(x => (decimal)((decimal)x * (decimal)value));
            var verifyColumn = new DecimalDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseMultiplyDoubleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            double value = 5;
            DoubleDataFrameColumn columnResult = value * column;
            var verify = Enumerable.Range(1, 10).Select(x => (double)((double)x * (double)value));
            var verifyColumn = new DoubleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseMultiplySingleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            float value = 5;
            SingleDataFrameColumn columnResult = value * column;
            var verify = Enumerable.Range(1, 10).Select(x => (float)((float)x * (float)value));
            var verifyColumn = new SingleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseMultiplyInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            int value = 5;
            Int32DataFrameColumn columnResult = value * column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x * (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseMultiplyInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            long value = 5;
            Int64DataFrameColumn columnResult = value * column;
            var verify = Enumerable.Range(1, 10).Select(x => (long)((long)x * (long)value));
            var verifyColumn = new Int64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseMultiplySByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            sbyte value = 5;
            Int32DataFrameColumn columnResult = value * column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x * (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseMultiplyInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            short value = 5;
            Int32DataFrameColumn columnResult = value * column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x * (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseMultiplyUInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            uint value = 5;
            UInt32DataFrameColumn columnResult = value * column;
            var verify = Enumerable.Range(1, 10).Select(x => (uint)((uint)x * (uint)value));
            var verifyColumn = new UInt32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseMultiplyUInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ulong value = 5;
            UInt64DataFrameColumn columnResult = value * column;
            var verify = Enumerable.Range(1, 10).Select(x => (ulong)((ulong)x * (ulong)value));
            var verifyColumn = new UInt64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseMultiplyUInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ushort value = 5;
            Int32DataFrameColumn columnResult = value * column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x * (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideByteDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn otherColumn = new ByteDataFrameColumn("Byte", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column / otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(1));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideDecimalDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (decimal)x);
            DecimalDataFrameColumn otherColumn = new DecimalDataFrameColumn("Decimal", otherColumnEnumerable);
            DecimalDataFrameColumn columnResult = column / otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (decimal)(1));
            var verifyColumn = new DecimalDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideDoubleDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (double)x);
            DoubleDataFrameColumn otherColumn = new DoubleDataFrameColumn("Double", otherColumnEnumerable);
            DoubleDataFrameColumn columnResult = column / otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (double)(1));
            var verifyColumn = new DoubleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideSingleDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (float)x);
            SingleDataFrameColumn otherColumn = new SingleDataFrameColumn("Single", otherColumnEnumerable);
            SingleDataFrameColumn columnResult = column / otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (float)(1));
            var verifyColumn = new SingleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideInt32DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (int)x);
            Int32DataFrameColumn otherColumn = new Int32DataFrameColumn("Int32", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column / otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(1));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideInt64DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (long)x);
            Int64DataFrameColumn otherColumn = new Int64DataFrameColumn("Int64", otherColumnEnumerable);
            Int64DataFrameColumn columnResult = column / otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (long)(1));
            var verifyColumn = new Int64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideSByteDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (sbyte)x);
            SByteDataFrameColumn otherColumn = new SByteDataFrameColumn("SByte", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column / otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(1));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideInt16DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (short)x);
            Int16DataFrameColumn otherColumn = new Int16DataFrameColumn("Int16", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column / otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(1));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideUInt32DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (uint)x);
            UInt32DataFrameColumn otherColumn = new UInt32DataFrameColumn("UInt32", otherColumnEnumerable);
            UInt32DataFrameColumn columnResult = column / otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (uint)(1));
            var verifyColumn = new UInt32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideUInt64DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (ulong)x);
            UInt64DataFrameColumn otherColumn = new UInt64DataFrameColumn("UInt64", otherColumnEnumerable);
            UInt64DataFrameColumn columnResult = column / otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (ulong)(1));
            var verifyColumn = new UInt64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideUInt16DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (ushort)x);
            UInt16DataFrameColumn otherColumn = new UInt16DataFrameColumn("UInt16", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column / otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(1));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            byte value = 5;
            Int32DataFrameColumn columnResult = column / value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x / (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideDecimalToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            decimal value = 5;
            DecimalDataFrameColumn columnResult = column / value;
            var verify = Enumerable.Range(1, 10).Select(x => (decimal)((decimal)x / (decimal)value));
            var verifyColumn = new DecimalDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideDoubleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            double value = 5;
            DoubleDataFrameColumn columnResult = column / value;
            var verify = Enumerable.Range(1, 10).Select(x => (double)((double)x / (double)value));
            var verifyColumn = new DoubleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideSingleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            float value = 5;
            SingleDataFrameColumn columnResult = column / value;
            var verify = Enumerable.Range(1, 10).Select(x => (float)((float)x / (float)value));
            var verifyColumn = new SingleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            int value = 5;
            Int32DataFrameColumn columnResult = column / value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x / (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            long value = 5;
            Int64DataFrameColumn columnResult = column / value;
            var verify = Enumerable.Range(1, 10).Select(x => (long)((long)x / (long)value));
            var verifyColumn = new Int64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideSByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            sbyte value = 5;
            Int32DataFrameColumn columnResult = column / value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x / (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            short value = 5;
            Int32DataFrameColumn columnResult = column / value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x / (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideUInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            uint value = 5;
            UInt32DataFrameColumn columnResult = column / value;
            var verify = Enumerable.Range(1, 10).Select(x => (uint)((uint)x / (uint)value));
            var verifyColumn = new UInt32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideUInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ulong value = 5;
            UInt64DataFrameColumn columnResult = column / value;
            var verify = Enumerable.Range(1, 10).Select(x => (ulong)((ulong)x / (ulong)value));
            var verifyColumn = new UInt64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void DivideUInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ushort value = 5;
            Int32DataFrameColumn columnResult = column / value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x / (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseDivideByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            byte value = 5;
            Int32DataFrameColumn columnResult = value / column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)value / (int)x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseDivideDecimalToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            decimal value = 5;
            DecimalDataFrameColumn columnResult = value / column;
            var verify = Enumerable.Range(1, 10).Select(x => (decimal)((decimal)value / (decimal)x));
            var verifyColumn = new DecimalDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseDivideDoubleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            double value = 5;
            DoubleDataFrameColumn columnResult = value / column;
            var verify = Enumerable.Range(1, 10).Select(x => (double)((double)value / (double)x));
            var verifyColumn = new DoubleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseDivideSingleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            float value = 5;
            SingleDataFrameColumn columnResult = value / column;
            var verify = Enumerable.Range(1, 10).Select(x => (float)((float)value / (float)x));
            var verifyColumn = new SingleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseDivideInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            int value = 5;
            Int32DataFrameColumn columnResult = value / column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)value / (int)x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseDivideInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            long value = 5;
            Int64DataFrameColumn columnResult = value / column;
            var verify = Enumerable.Range(1, 10).Select(x => (long)((long)value / (long)x));
            var verifyColumn = new Int64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseDivideSByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            sbyte value = 5;
            Int32DataFrameColumn columnResult = value / column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)value / (int)x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseDivideInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            short value = 5;
            Int32DataFrameColumn columnResult = value / column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)value / (int)x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseDivideUInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            uint value = 5;
            UInt32DataFrameColumn columnResult = value / column;
            var verify = Enumerable.Range(1, 10).Select(x => (uint)((uint)value / (uint)x));
            var verifyColumn = new UInt32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseDivideUInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ulong value = 5;
            UInt64DataFrameColumn columnResult = value / column;
            var verify = Enumerable.Range(1, 10).Select(x => (ulong)((ulong)value / (ulong)x));
            var verifyColumn = new UInt64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseDivideUInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ushort value = 5;
            Int32DataFrameColumn columnResult = value / column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)value / (int)x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloByteDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn otherColumn = new ByteDataFrameColumn("Byte", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column % otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(0));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloDecimalDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (decimal)x);
            DecimalDataFrameColumn otherColumn = new DecimalDataFrameColumn("Decimal", otherColumnEnumerable);
            DecimalDataFrameColumn columnResult = column % otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (decimal)(0));
            var verifyColumn = new DecimalDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloDoubleDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (double)x);
            DoubleDataFrameColumn otherColumn = new DoubleDataFrameColumn("Double", otherColumnEnumerable);
            DoubleDataFrameColumn columnResult = column % otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (double)(0));
            var verifyColumn = new DoubleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloSingleDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (float)x);
            SingleDataFrameColumn otherColumn = new SingleDataFrameColumn("Single", otherColumnEnumerable);
            SingleDataFrameColumn columnResult = column % otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (float)(0));
            var verifyColumn = new SingleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloInt32DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (int)x);
            Int32DataFrameColumn otherColumn = new Int32DataFrameColumn("Int32", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column % otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(0));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloInt64DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (long)x);
            Int64DataFrameColumn otherColumn = new Int64DataFrameColumn("Int64", otherColumnEnumerable);
            Int64DataFrameColumn columnResult = column % otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (long)(0));
            var verifyColumn = new Int64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloSByteDataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (sbyte)x);
            SByteDataFrameColumn otherColumn = new SByteDataFrameColumn("SByte", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column % otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(0));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloInt16DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (short)x);
            Int16DataFrameColumn otherColumn = new Int16DataFrameColumn("Int16", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column % otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(0));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloUInt32DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (uint)x);
            UInt32DataFrameColumn otherColumn = new UInt32DataFrameColumn("UInt32", otherColumnEnumerable);
            UInt32DataFrameColumn columnResult = column % otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (uint)(0));
            var verifyColumn = new UInt32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloUInt64DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (ulong)x);
            UInt64DataFrameColumn otherColumn = new UInt64DataFrameColumn("UInt64", otherColumnEnumerable);
            UInt64DataFrameColumn columnResult = column % otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (ulong)(0));
            var verifyColumn = new UInt64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloUInt16DataFrameColumnToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (ushort)x);
            UInt16DataFrameColumn otherColumn = new UInt16DataFrameColumn("UInt16", otherColumnEnumerable);
            Int32DataFrameColumn columnResult = column % otherColumn;
            var verify = Enumerable.Range(1, 10).Select(x => (int)(0));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            byte value = 5;
            Int32DataFrameColumn columnResult = column % value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x % (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloDecimalToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            decimal value = 5;
            DecimalDataFrameColumn columnResult = column % value;
            var verify = Enumerable.Range(1, 10).Select(x => (decimal)((decimal)x % (decimal)value));
            var verifyColumn = new DecimalDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloDoubleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            double value = 5;
            DoubleDataFrameColumn columnResult = column % value;
            var verify = Enumerable.Range(1, 10).Select(x => (double)((double)x % (double)value));
            var verifyColumn = new DoubleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloSingleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            float value = 5;
            SingleDataFrameColumn columnResult = column % value;
            var verify = Enumerable.Range(1, 10).Select(x => (float)((float)x % (float)value));
            var verifyColumn = new SingleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            int value = 5;
            Int32DataFrameColumn columnResult = column % value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x % (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            long value = 5;
            Int64DataFrameColumn columnResult = column % value;
            var verify = Enumerable.Range(1, 10).Select(x => (long)((long)x % (long)value));
            var verifyColumn = new Int64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloSByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            sbyte value = 5;
            Int32DataFrameColumn columnResult = column % value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x % (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            short value = 5;
            Int32DataFrameColumn columnResult = column % value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x % (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloUInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            uint value = 5;
            UInt32DataFrameColumn columnResult = column % value;
            var verify = Enumerable.Range(1, 10).Select(x => (uint)((uint)x % (uint)value));
            var verifyColumn = new UInt32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloUInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ulong value = 5;
            UInt64DataFrameColumn columnResult = column % value;
            var verify = Enumerable.Range(1, 10).Select(x => (ulong)((ulong)x % (ulong)value));
            var verifyColumn = new UInt64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ModuloUInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ushort value = 5;
            Int32DataFrameColumn columnResult = column % value;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)x % (int)value));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseModuloByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            byte value = 5;
            Int32DataFrameColumn columnResult = value % column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)value % (int)x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseModuloDecimalToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            decimal value = 5;
            DecimalDataFrameColumn columnResult = value % column;
            var verify = Enumerable.Range(1, 10).Select(x => (decimal)((decimal)value % (decimal)x));
            var verifyColumn = new DecimalDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseModuloDoubleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            double value = 5;
            DoubleDataFrameColumn columnResult = value % column;
            var verify = Enumerable.Range(1, 10).Select(x => (double)((double)value % (double)x));
            var verifyColumn = new DoubleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseModuloSingleToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            float value = 5;
            SingleDataFrameColumn columnResult = value % column;
            var verify = Enumerable.Range(1, 10).Select(x => (float)((float)value % (float)x));
            var verifyColumn = new SingleDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseModuloInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            int value = 5;
            Int32DataFrameColumn columnResult = value % column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)value % (int)x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseModuloInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            long value = 5;
            Int64DataFrameColumn columnResult = value % column;
            var verify = Enumerable.Range(1, 10).Select(x => (long)((long)value % (long)x));
            var verifyColumn = new Int64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseModuloSByteToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            sbyte value = 5;
            Int32DataFrameColumn columnResult = value % column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)value % (int)x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseModuloInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            short value = 5;
            Int32DataFrameColumn columnResult = value % column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)value % (int)x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseModuloUInt32ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            uint value = 5;
            UInt32DataFrameColumn columnResult = value % column;
            var verify = Enumerable.Range(1, 10).Select(x => (uint)((uint)value % (uint)x));
            var verifyColumn = new UInt32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseModuloUInt64ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ulong value = 5;
            UInt64DataFrameColumn columnResult = value % column;
            var verify = Enumerable.Range(1, 10).Select(x => (ulong)((ulong)value % (ulong)x));
            var verifyColumn = new UInt64DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ReverseModuloUInt16ToByteDataFrameColumn()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ushort value = 5;
            Int32DataFrameColumn columnResult = value % column;
            var verify = Enumerable.Range(1, 10).Select(x => (int)((int)value % (int)x));
            var verifyColumn = new Int32DataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToByte()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn otherColumn = new ByteDataFrameColumn("Byte", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToDecimal()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (decimal)x);
            DecimalDataFrameColumn otherColumn = new DecimalDataFrameColumn("Decimal", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToDouble()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (double)x);
            DoubleDataFrameColumn otherColumn = new DoubleDataFrameColumn("Double", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToSingle()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (float)x);
            SingleDataFrameColumn otherColumn = new SingleDataFrameColumn("Single", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToInt32()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (int)x);
            Int32DataFrameColumn otherColumn = new Int32DataFrameColumn("Int32", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToInt64()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (long)x);
            Int64DataFrameColumn otherColumn = new Int64DataFrameColumn("Int64", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToSByte()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (sbyte)x);
            SByteDataFrameColumn otherColumn = new SByteDataFrameColumn("SByte", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToInt16()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (short)x);
            Int16DataFrameColumn otherColumn = new Int16DataFrameColumn("Int16", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToUInt32()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (uint)x);
            UInt32DataFrameColumn otherColumn = new UInt32DataFrameColumn("UInt32", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToUInt64()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (ulong)x);
            UInt64DataFrameColumn otherColumn = new UInt64DataFrameColumn("UInt64", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToUInt16()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (ushort)x);
            UInt16DataFrameColumn otherColumn = new UInt16DataFrameColumn("UInt16", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToScalarByte()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            byte value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToScalarDecimal()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            decimal value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToScalarDouble()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            double value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToScalarSingle()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            float value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToScalarInt32()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            int value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToScalarInt64()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            long value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToScalarSByte()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            sbyte value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToScalarInt16()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            short value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToScalarUInt32()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            uint value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToScalarUInt64()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ulong value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseEqualsBooleanToScalarUInt16()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ushort value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToByte()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn otherColumn = new ByteDataFrameColumn("Byte", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToDecimal()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (decimal)x);
            DecimalDataFrameColumn otherColumn = new DecimalDataFrameColumn("Decimal", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToDouble()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (double)x);
            DoubleDataFrameColumn otherColumn = new DoubleDataFrameColumn("Double", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToSingle()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (float)x);
            SingleDataFrameColumn otherColumn = new SingleDataFrameColumn("Single", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToInt32()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (int)x);
            Int32DataFrameColumn otherColumn = new Int32DataFrameColumn("Int32", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToInt64()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (long)x);
            Int64DataFrameColumn otherColumn = new Int64DataFrameColumn("Int64", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToSByte()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (sbyte)x);
            SByteDataFrameColumn otherColumn = new SByteDataFrameColumn("SByte", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToInt16()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (short)x);
            Int16DataFrameColumn otherColumn = new Int16DataFrameColumn("Int16", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToUInt32()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (uint)x);
            UInt32DataFrameColumn otherColumn = new UInt32DataFrameColumn("UInt32", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToUInt64()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (ulong)x);
            UInt64DataFrameColumn otherColumn = new UInt64DataFrameColumn("UInt64", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToUInt16()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            var otherColumnEnumerable = Enumerable.Range(1, 10).Select(x => (ushort)x);
            UInt16DataFrameColumn otherColumn = new UInt16DataFrameColumn("UInt16", otherColumnEnumerable);
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(otherColumn);
            var verify = Enumerable.Range(1, 10).Select(x => true);
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());

            // If this is equals, change thisx to false
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToScalarByte()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            byte value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToScalarDecimal()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            decimal value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToScalarDouble()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            double value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToScalarSingle()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            float value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToScalarInt32()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            int value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToScalarInt64()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            long value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToScalarSByte()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            sbyte value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToScalarInt16()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            short value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToScalarUInt32()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            uint value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToScalarUInt64()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ulong value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
        [Fact]
        public void ElementwiseNotEqualsBooleanToScalarUInt16()
        {
            var columnEnumerable = Enumerable.Range(1, 10).Select(x => (byte)x);
            ByteDataFrameColumn column = new ByteDataFrameColumn("Byte", columnEnumerable);
            ushort value = 100;
            BooleanDataFrameColumn columnResult = column.ElementwiseNotEquals(value);
            var verify = Enumerable.Range(1, 10).Select(x => (bool)(false));
            var verifyColumn = new BooleanDataFrameColumn("Verify", verify);
            Assert.Equal(columnResult.Length, verify.Count());
            Assert.True(columnResult.ElementwiseNotEquals(verifyColumn).All());
        }
    }
}
