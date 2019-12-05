// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Apache.Arrow;
using Apache.Arrow.Ipc;
using Xunit;

namespace Microsoft.Data.Analysis.Tests
{
    public class ArrowIntegrationTests
    {
        [Fact]
        public void TestArrowIntegration()
        {
            RecordBatch originalBatch = new RecordBatch.Builder()
                .Append("Column1", false, col => col.Int32(array => array.AppendRange(Enumerable.Range(0, 10))))
                .Append("Column2", true, new Int32Array(
                    valueBuffer: new ArrowBuffer.Builder<int>().AppendRange(Enumerable.Range(0, 10)).Build(),
                    nullBitmapBuffer: new ArrowBuffer.Builder<byte>().Append(0xfd).Append(0xff).Build(),
                    length: 10,
                    nullCount: 1,
                    offset: 0))
                .Append("Column3", true, new Int32Array(
                    valueBuffer: new ArrowBuffer.Builder<int>().AppendRange(Enumerable.Range(0, 10)).Build(),
                    nullBitmapBuffer: new ArrowBuffer.Builder<byte>().Append(0x00).Append(0x00).Build(),
                    length: 10,
                    nullCount: 10,
                    offset: 0))
                .Append("NullableBooleanColumn", true, new BooleanArray(
                    valueBuffer: new ArrowBuffer.Builder<byte>().Append(0xfd).Append(0xff).Build(),
                    nullBitmapBuffer: new ArrowBuffer.Builder<byte>().Append(0xed).Append(0xff).Build(),
                    length: 10,
                    nullCount: 2,
                    offset: 0))
                .Append("StringDataFrameColumn", false, new StringArray.Builder().AppendRange(Enumerable.Range(0, 10).Select(x => x.ToString())).Build())
                .Append("DoubleColumn", false, new DoubleArray.Builder().AppendRange(Enumerable.Repeat(1.0, 10)).Build())
                .Append("FloatColumn", false, new FloatArray.Builder().AppendRange(Enumerable.Repeat(1.0f, 10)).Build())
                .Append("ShortColumn", false, new Int16Array.Builder().AppendRange(Enumerable.Repeat((short)1, 10)).Build())
                .Append("LongColumn", false, new Int64Array.Builder().AppendRange(Enumerable.Repeat((long)1, 10)).Build())
                .Append("UIntColumn", false, new UInt32Array.Builder().AppendRange(Enumerable.Repeat((uint)1, 10)).Build())
                .Append("UShortColumn", false, new UInt16Array.Builder().AppendRange(Enumerable.Repeat((ushort)1, 10)).Build())
                .Append("ULongColumn", false, new UInt64Array.Builder().AppendRange(Enumerable.Repeat((ulong)1, 10)).Build())
                .Append("ByteColumn", false, new Int8Array.Builder().AppendRange(Enumerable.Repeat((sbyte)1, 10)).Build())
                .Append("UByteColumn", false, new UInt8Array.Builder().AppendRange(Enumerable.Repeat((byte)1, 10)).Build())
                .Build();

            DataFrame df = DataFrame.FromArrowRecordBatch(originalBatch);

            IEnumerable<RecordBatch> recordBatches = df.ToArrowRecordBatches();

            foreach (RecordBatch batch in recordBatches)
            {
                RecordBatchComparer.CompareBatches(originalBatch, batch);
            }
        }

        [Fact]
        public void TestEmptyDataFrameRecordBatch()
        {
            PrimitiveDataFrameColumn<int> ageColumn = new PrimitiveDataFrameColumn<int>("Age");
            PrimitiveDataFrameColumn<int> lengthColumn = new PrimitiveDataFrameColumn<int>("CharCount");
            ArrowStringDataFrameColumn stringColumn = new ArrowStringDataFrameColumn("Empty");
            DataFrame df = new DataFrame(new List<DataFrameColumn>() { ageColumn, lengthColumn, stringColumn });

            IEnumerable<RecordBatch> recordBatches = df.ToArrowRecordBatches();
            bool foundARecordBatch = false;
            foreach (RecordBatch recordBatch in recordBatches)
            {
                foundARecordBatch = true;
                MemoryStream stream = new MemoryStream();
                ArrowStreamWriter writer = new ArrowStreamWriter(stream, recordBatch.Schema);
                writer.WriteRecordBatchAsync(recordBatch).GetAwaiter().GetResult();

                stream.Position = 0;
                ArrowStreamReader reader = new ArrowStreamReader(stream);
                RecordBatch readRecordBatch = reader.ReadNextRecordBatch();
                while (readRecordBatch != null)
                {
                    RecordBatchComparer.CompareBatches(recordBatch, readRecordBatch);
                    readRecordBatch = reader.ReadNextRecordBatch();
                }
            }
            Assert.True(foundARecordBatch);
        }

        [Fact]
        public void TestMutationOnArrowColumn()
        {
            RecordBatch originalBatch = new RecordBatch.Builder()
                .Append("Column1", false, col => col.Int32(array => array.AppendRange(Enumerable.Range(0, 10)))).Build();
            DataFrame df = DataFrame.FromArrowRecordBatch(originalBatch);
            Assert.Equal(1, df["Column1"][1]);
            df["Column1"][1] = 100;
            Assert.Equal(100, df["Column1"][1]);
            Assert.Equal(0, df["Column1"].NullCount);
        }

        [Fact]
        public void TestEmptyArrowColumns()
        {
            // Tests to ensure that we don't crash and the internal NullCounts stay consistent on encountering:
            // 1. Data + Empty null bitmaps
            // 2. Empty Data + Null bitmaps
            // 3. Empty Data + Empty null bitmaps
            RecordBatch originalBatch = new RecordBatch.Builder()
                .Append("EmptyNullBitMapColumn", false, col => col.Int32(array => array.AppendRange(Enumerable.Range(0, 10))))
                .Append("EmptyDataColumn", true, new Int32Array(
                    valueBuffer: ArrowBuffer.Empty,
                    nullBitmapBuffer: new ArrowBuffer.Builder<byte>().Append(0x00).Append(0x00).Build(),
                    length: 10,
                    nullCount: 10,
                    offset: 0)).Build();
            DataFrame df = DataFrame.FromArrowRecordBatch(originalBatch);
            Assert.Equal(0, df["EmptyNullBitMapColumn"].NullCount);
            Assert.Equal(10, df["EmptyNullBitMapColumn"].Length);
            df["EmptyNullBitMapColumn"][9] = null;
            Assert.Equal(1, df["EmptyNullBitMapColumn"].NullCount);
            Assert.Equal(10, df["EmptyDataColumn"].NullCount);
            Assert.Equal(10, df["EmptyDataColumn"].Length);
            df["EmptyDataColumn"][9] = 9;
            Assert.Equal(9, df["EmptyDataColumn"].NullCount);
            Assert.Equal(10, df["EmptyDataColumn"].Length);
            for (int i = 0; i < 9; i++)
            {
                Assert.Equal(i, (int)df["EmptyNullBitMapColumn"][i]);
                Assert.Null(df["EmptyDataColumn"][i]);
            }

            RecordBatch batch1 = new RecordBatch.Builder()
                .Append("EmptyDataAndNullColumns", false, col => col.Int32(array => array.Clear())).Build();
            DataFrame emptyDataFrame = DataFrame.FromArrowRecordBatch(batch1);
            Assert.Equal(0, emptyDataFrame.Rows.Count);
            Assert.Equal(0, emptyDataFrame["EmptyDataAndNullColumns"].Length);
            Assert.Equal(0, emptyDataFrame["EmptyDataAndNullColumns"].NullCount);
        }

        [Fact]
        public void TestInconsistentNullBitMapLength()
        {
            // Arrow allocates buffers of length 64 by default. 64 * 8 = 512 bits in the NullBitMapBuffer. Anything lesser than 512 will not trigger a throw
            Int32Array int32 = new Int32Array.Builder().AppendRange(Enumerable.Range(0, 520)).Build();
            RecordBatch originalBatch = new RecordBatch.Builder()
                .Append("EmptyDataColumn", true, new Int32Array(
                    valueBuffer: int32.ValueBuffer,
                    nullBitmapBuffer: new ArrowBuffer.Builder<byte>().Append(0x00).Build(),
                    length: 520,
                    nullCount: 520,
                    offset: 0)).Build();

            Assert.ThrowsAny<ArgumentException>(() => DataFrame.FromArrowRecordBatch(originalBatch));
        }
    }
}
