// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Apache.Arrow;
using Apache.Arrow.Ipc;
using Apache.Arrow.Types;
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
            DataFrameIOTests.VerifyColumnTypes(df, testArrowStringColumn: true);

            IEnumerable<RecordBatch> recordBatches = df.ToArrowRecordBatches();

            foreach (RecordBatch batch in recordBatches)
            {
                RecordBatchComparer.CompareBatches(originalBatch, batch);
            }
        }

        [Fact]
        public void TestRecordBatchWithStructArrays()
        {
            RecordBatch CreateRecordBatch(string prependColumnNamesWith = "")
            {
                RecordBatch ret = new RecordBatch.Builder()
                    .Append(prependColumnNamesWith + "Column1", false, col => col.Int32(array => array.AppendRange(Enumerable.Range(0, 10))))
                    .Append(prependColumnNamesWith + "Column2", true, new Int32Array(
                        valueBuffer: new ArrowBuffer.Builder<int>().AppendRange(Enumerable.Range(0, 10)).Build(),
                        nullBitmapBuffer: new ArrowBuffer.Builder<byte>().Append(0xfd).Append(0xff).Build(),
                        length: 10,
                        nullCount: 1,
                        offset: 0))
                    .Append(prependColumnNamesWith + "Column3", true, new Int32Array(
                        valueBuffer: new ArrowBuffer.Builder<int>().AppendRange(Enumerable.Range(0, 10)).Build(),
                        nullBitmapBuffer: new ArrowBuffer.Builder<byte>().Append(0x00).Append(0x00).Build(),
                        length: 10,
                        nullCount: 10,
                        offset: 0))
                    .Append(prependColumnNamesWith + "NullableBooleanColumn", true, new BooleanArray(
                        valueBuffer: new ArrowBuffer.Builder<byte>().Append(0xfd).Append(0xff).Build(),
                        nullBitmapBuffer: new ArrowBuffer.Builder<byte>().Append(0xed).Append(0xff).Build(),
                        length: 10,
                        nullCount: 2,
                        offset: 0))
                    .Append(prependColumnNamesWith + "StringDataFrameColumn", false, new StringArray.Builder().AppendRange(Enumerable.Range(0, 10).Select(x => x.ToString())).Build())
                    .Append(prependColumnNamesWith + "DoubleColumn", false, new DoubleArray.Builder().AppendRange(Enumerable.Repeat(1.0, 10)).Build())
                    .Append(prependColumnNamesWith + "FloatColumn", false, new FloatArray.Builder().AppendRange(Enumerable.Repeat(1.0f, 10)).Build())
                    .Append(prependColumnNamesWith + "ShortColumn", false, new Int16Array.Builder().AppendRange(Enumerable.Repeat((short)1, 10)).Build())
                    .Append(prependColumnNamesWith + "LongColumn", false, new Int64Array.Builder().AppendRange(Enumerable.Repeat((long)1, 10)).Build())
                    .Append(prependColumnNamesWith + "UIntColumn", false, new UInt32Array.Builder().AppendRange(Enumerable.Repeat((uint)1, 10)).Build())
                    .Append(prependColumnNamesWith + "UShortColumn", false, new UInt16Array.Builder().AppendRange(Enumerable.Repeat((ushort)1, 10)).Build())
                    .Append(prependColumnNamesWith + "ULongColumn", false, new UInt64Array.Builder().AppendRange(Enumerable.Repeat((ulong)1, 10)).Build())
                    .Append(prependColumnNamesWith + "ByteColumn", false, new Int8Array.Builder().AppendRange(Enumerable.Repeat((sbyte)1, 10)).Build())
                    .Append(prependColumnNamesWith + "UByteColumn", false, new UInt8Array.Builder().AppendRange(Enumerable.Repeat((byte)1, 10)).Build())
                    .Build();

                return ret;
            }

            RecordBatch originalBatch = CreateRecordBatch();
            ArrowBuffer.BitmapBuilder validityBitmapBuilder = new ArrowBuffer.BitmapBuilder();
            for (int i = 0; i < originalBatch.Length; i++)
            {
                validityBitmapBuilder.Append(true);
            }
            ArrowBuffer validityBitmap = validityBitmapBuilder.Build();

            StructType structType = new StructType(originalBatch.Schema.Fields.Select((KeyValuePair<string, Field> pair) => pair.Value).ToList());
            StructArray structArray = new StructArray(structType, originalBatch.Length, originalBatch.Arrays.Cast<Apache.Arrow.Array>(), validityBitmap);
            Schema schema = new Schema.Builder().Field(new Field("Struct", structType, false)).Build();
            RecordBatch recordBatch = new RecordBatch(schema, new[] { structArray }, originalBatch.Length);

            DataFrame df = DataFrame.FromArrowRecordBatch(recordBatch);
            DataFrameIOTests.VerifyColumnTypes(df, testArrowStringColumn: true);

            IEnumerable<RecordBatch> recordBatches = df.ToArrowRecordBatches();

            RecordBatch expected = CreateRecordBatch("Struct_");
            foreach (RecordBatch batch in recordBatches)
            {
                RecordBatchComparer.CompareBatches(expected, batch);
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
            Assert.Equal(1, df.Columns["Column1"][1]);
            df.Columns["Column1"][1] = 100;
            Assert.Equal(100, df.Columns["Column1"][1]);
            Assert.Equal(0, df.Columns["Column1"].NullCount);
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
            Assert.Equal(0, df.Columns["EmptyNullBitMapColumn"].NullCount);
            Assert.Equal(10, df.Columns["EmptyNullBitMapColumn"].Length);
            df.Columns["EmptyNullBitMapColumn"][9] = null;
            Assert.Equal(1, df.Columns["EmptyNullBitMapColumn"].NullCount);
            Assert.Equal(10, df.Columns["EmptyDataColumn"].NullCount);
            Assert.Equal(10, df.Columns["EmptyDataColumn"].Length);
            df.Columns["EmptyDataColumn"][9] = 9;
            Assert.Equal(9, df.Columns["EmptyDataColumn"].NullCount);
            Assert.Equal(10, df.Columns["EmptyDataColumn"].Length);
            for (int i = 0; i < 9; i++)
            {
                Assert.Equal(i, (int)df.Columns["EmptyNullBitMapColumn"][i]);
                Assert.Null(df.Columns["EmptyDataColumn"][i]);
            }

            RecordBatch batch1 = new RecordBatch.Builder()
                .Append("EmptyDataAndNullColumns", false, col => col.Int32(array => array.Clear())).Build();
            DataFrame emptyDataFrame = DataFrame.FromArrowRecordBatch(batch1);
            Assert.Equal(0, emptyDataFrame.Rows.Count);
            Assert.Equal(0, emptyDataFrame.Columns["EmptyDataAndNullColumns"].Length);
            Assert.Equal(0, emptyDataFrame.Columns["EmptyDataAndNullColumns"].NullCount);
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
