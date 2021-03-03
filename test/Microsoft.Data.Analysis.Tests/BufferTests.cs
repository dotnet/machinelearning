// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Apache.Arrow;
using Xunit;

namespace Microsoft.Data.Analysis.Tests
{
    public class BufferTests
    {
        [Fact]
        public void TestNullCounts()
        {
            PrimitiveDataFrameColumn<int> dataFrameColumn1 = new PrimitiveDataFrameColumn<int>("Int1", Enumerable.Range(0, 10).Select(x => x));
            dataFrameColumn1.Append(null);
            Assert.Equal(1, dataFrameColumn1.NullCount);

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
            Assert.Equal(0, strCol1.NullCount);

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

            PrimitiveDataFrameColumn<int> intColumn = new PrimitiveDataFrameColumn<int>("Int");
            intColumn.Append(0);
            intColumn.Append(1);
            intColumn.Append(null);
            intColumn.Append(2);
            intColumn.Append(null);
            intColumn.Append(3);
            Assert.Equal(0, intColumn[0]);
            Assert.Equal(1, intColumn[1]);
            Assert.Null(intColumn[2]);
            Assert.Equal(2, intColumn[3]);
            Assert.Null(intColumn[4]);
            Assert.Equal(3, intColumn[5]);

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
        public void TestAppendMany()
        {
            PrimitiveDataFrameColumn<int> intColumn = new PrimitiveDataFrameColumn<int>("Int1");
            intColumn.AppendMany(null, 5);
            Assert.Equal(5, intColumn.NullCount);
            Assert.Equal(5, intColumn.Length);
            for (int i = 0; i < intColumn.Length; i++)
            {
                Assert.False(intColumn.IsValid(i));
            }

            intColumn.AppendMany(5, 5);
            Assert.Equal(5, intColumn.NullCount);
            Assert.Equal(10, intColumn.Length);
            for (int i = 5; i < intColumn.Length; i++)
            {
                Assert.True(intColumn.IsValid(i));
            }

            intColumn[2] = 10;
            Assert.Equal(4, intColumn.NullCount);
            Assert.True(intColumn.IsValid(2));

            intColumn[7] = null;
            Assert.Equal(5, intColumn.NullCount);
            Assert.False(intColumn.IsValid(7));
        }

        [Fact]
        public void TestBasicArrowStringColumn()
        {
            StringArray strArray = new StringArray.Builder().Append("foo").Append("bar").Build();
            Memory<byte> dataMemory = new byte[] { 102, 111, 111, 98, 97, 114 };
            Memory<byte> nullMemory = new byte[] { 0, 0, 0, 0 };
            Memory<byte> offsetMemory = new byte[] { 0, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0 };

            ArrowStringDataFrameColumn stringColumn = new ArrowStringDataFrameColumn("String", dataMemory, offsetMemory, nullMemory, strArray.Length, strArray.NullCount);
            Assert.Equal(2, stringColumn.Length);
            Assert.Equal("foo", stringColumn[0]);
            Assert.Equal("bar", stringColumn[1]);
        }

        [Fact]
        public void TestArrowStringColumnWithNulls()
        {
            string data = "joemark";
            byte[] bytes = Encoding.UTF8.GetBytes(data);
            Memory<byte> dataMemory = new Memory<byte>(bytes);
            Memory<byte> nullMemory = new byte[] { 0b1101 };
            Memory<byte> offsetMemory = new byte[] { 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0 };
            ArrowStringDataFrameColumn stringColumn = new ArrowStringDataFrameColumn("String", dataMemory, offsetMemory, nullMemory, 4, 1);

            Assert.Equal(4, stringColumn.Length);
            Assert.Equal("joe", stringColumn[0]);
            Assert.Null(stringColumn[1]);
            Assert.Equal("mark", stringColumn[2]);
            Assert.Equal("", stringColumn[3]);

            List<string> ret = stringColumn[0, 4];
            Assert.Equal("joe", ret[0]);
            Assert.Null(ret[1]);
            Assert.Equal("mark", ret[2]);
            Assert.Equal("", ret[3]);
        }

        [Fact]
        public void TestArrowStringColumnClone()
        {
            StringArray strArray = new StringArray.Builder().Append("foo").Append("bar").Build();
            Memory<byte> dataMemory = new byte[] { 102, 111, 111, 98, 97, 114 };
            Memory<byte> nullMemory = new byte[] { 0, 0, 0, 0 };
            Memory<byte> offsetMemory = new byte[] { 0, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0 };

            ArrowStringDataFrameColumn stringColumn = new ArrowStringDataFrameColumn("String", dataMemory, offsetMemory, nullMemory, strArray.Length, strArray.NullCount);

            DataFrameColumn clone = stringColumn.Clone(numberOfNullsToAppend: 5);
            Assert.Equal(7, clone.Length);
            Assert.Equal(stringColumn[0], clone[0]);
            Assert.Equal(stringColumn[1], clone[1]);
            for (int i = 2; i < 7; i++)
                Assert.Null(clone[i]);
        }

#if !NETFRAMEWORK // https://github.com/dotnet/corefxlab/issues/2796
        [Fact]
        public void TestPrimitiveColumnGetReadOnlyBuffers()
        {
            RecordBatch recordBatch = new RecordBatch.Builder()
                .Append("Column1", false, col => col.Int32(array => array.AppendRange(Enumerable.Range(0, 10)))).Build();
            DataFrame df = DataFrame.FromArrowRecordBatch(recordBatch);

            PrimitiveDataFrameColumn<int> column = df.Columns["Column1"] as PrimitiveDataFrameColumn<int>;

            IEnumerable<ReadOnlyMemory<int>> buffers = column.GetReadOnlyDataBuffers();
            IEnumerable<ReadOnlyMemory<byte>> nullBitMaps = column.GetReadOnlyNullBitMapBuffers();

            long i = 0;
            IEnumerator<ReadOnlyMemory<int>> bufferEnumerator = buffers.GetEnumerator();
            IEnumerator<ReadOnlyMemory<byte>> nullBitMapsEnumerator = nullBitMaps.GetEnumerator();
            while (bufferEnumerator.MoveNext() && nullBitMapsEnumerator.MoveNext())
            {
                ReadOnlyMemory<int> dataBuffer = bufferEnumerator.Current;
                ReadOnlyMemory<byte> nullBitMap = nullBitMapsEnumerator.Current;

                ReadOnlySpan<int> span = dataBuffer.Span;
                for (int j = 0; j < span.Length; j++)
                {
                    // Each buffer has a max length of int.MaxValue
                    Assert.Equal(span[j], column[j + i * int.MaxValue]);
                }

                bool GetBit(byte curBitMap, int index)
                {
                    return ((curBitMap >> (index & 7)) & 1) != 0;
                }
                ReadOnlySpan<byte> bitMapSpan = nullBitMap.Span;
                // No nulls in this column, so each bit must be set
                for (int j = 0; j < bitMapSpan.Length; j++)
                {
                    for (int k = 0; k < 8; k++)
                    {
                        if (j * 8 + k == column.Length)
                            break;
                        Assert.True(GetBit(bitMapSpan[j], k));
                    }
                }
                i++;
            }
        }

        [Fact]
        public void TestArrowStringColumnGetReadOnlyBuffers()
        {
            // Test ArrowStringDataFrameColumn.
            StringArray strArray = new StringArray.Builder().Append("foo").Append("bar").Build();
            Memory<byte> dataMemory = new byte[] { 102, 111, 111, 98, 97, 114 };
            Memory<byte> nullMemory = new byte[] { 1 };
            Memory<byte> offsetMemory = new byte[] { 0, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0 };

            ArrowStringDataFrameColumn column = new ArrowStringDataFrameColumn("String", dataMemory, offsetMemory, nullMemory, strArray.Length, strArray.NullCount);

            IEnumerable<ReadOnlyMemory<byte>> dataBuffers = column.GetReadOnlyDataBuffers();
            IEnumerable<ReadOnlyMemory<byte>> nullBitMaps = column.GetReadOnlyNullBitMapBuffers();
            IEnumerable<ReadOnlyMemory<int>> offsetsBuffers = column.GetReadOnlyOffsetsBuffers();

            using (IEnumerator<ReadOnlyMemory<byte>> bufferEnumerator = dataBuffers.GetEnumerator())
            using (IEnumerator<ReadOnlyMemory<int>> offsetsEnumerator = offsetsBuffers.GetEnumerator())
            using (IEnumerator<ReadOnlyMemory<byte>> nullBitMapsEnumerator = nullBitMaps.GetEnumerator())
            {
                while (bufferEnumerator.MoveNext() && nullBitMapsEnumerator.MoveNext() && offsetsEnumerator.MoveNext())
                {
                    ReadOnlyMemory<byte> dataBuffer = bufferEnumerator.Current;
                    ReadOnlyMemory<byte> nullBitMap = nullBitMapsEnumerator.Current;
                    ReadOnlyMemory<int> offsets = offsetsEnumerator.Current;

                    ReadOnlySpan<byte> dataSpan = dataBuffer.Span;
                    ReadOnlySpan<int> offsetsSpan = offsets.Span;
                    int dataStart = 0;
                    for (int j = 1; j < offsetsSpan.Length; j++)
                    {
                        int length = offsetsSpan[j] - offsetsSpan[j - 1];
                        ReadOnlySpan<byte> str = dataSpan.Slice(dataStart, length);
                        ReadOnlySpan<byte> columnStr = dataMemory.Span.Slice(dataStart, length);
                        Assert.Equal(str.Length, columnStr.Length);
                        for (int s = 0; s < str.Length; s++)
                            Assert.Equal(str[s], columnStr[s]);
                        dataStart = length;
                    }
                }
            }
        }
#endif //!NETFRAMEWORK
    }
}
