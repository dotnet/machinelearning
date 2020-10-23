// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Apache.Arrow;
using Apache.Arrow.Types;

namespace Microsoft.Data.Analysis
{
    public partial class DataFrame
    {
        private static void AppendDataFrameColumnFromArrowArray(Field field, IArrowArray arrowArray, DataFrame ret, string fieldNamePrefix = "")
        {
            IArrowType fieldType = field.DataType;
            DataFrameColumn dataFrameColumn = null;
            string fieldName = fieldNamePrefix + field.Name;
            switch (fieldType.TypeId)
            {
                case ArrowTypeId.Boolean:
                    BooleanArray arrowBooleanArray = (BooleanArray)arrowArray;
                    ReadOnlyMemory<byte> valueBuffer = arrowBooleanArray.ValueBuffer.Memory;
                    ReadOnlyMemory<byte> nullBitMapBuffer = arrowBooleanArray.NullBitmapBuffer.Memory;
                    dataFrameColumn = new BooleanDataFrameColumn(fieldName, valueBuffer, nullBitMapBuffer, arrowArray.Length, arrowArray.NullCount);
                    break;
                case ArrowTypeId.Double:
                    PrimitiveArray<double> arrowDoubleArray = (PrimitiveArray<double>)arrowArray;
                    ReadOnlyMemory<byte> doubleValueBuffer = arrowDoubleArray.ValueBuffer.Memory;
                    ReadOnlyMemory<byte> doubleNullBitMapBuffer = arrowDoubleArray.NullBitmapBuffer.Memory;
                    dataFrameColumn = new DoubleDataFrameColumn(fieldName, doubleValueBuffer, doubleNullBitMapBuffer, arrowArray.Length, arrowArray.NullCount);
                    break;
                case ArrowTypeId.Float:
                    PrimitiveArray<float> arrowFloatArray = (PrimitiveArray<float>)arrowArray;
                    ReadOnlyMemory<byte> floatValueBuffer = arrowFloatArray.ValueBuffer.Memory;
                    ReadOnlyMemory<byte> floatNullBitMapBuffer = arrowFloatArray.NullBitmapBuffer.Memory;
                    dataFrameColumn = new SingleDataFrameColumn(fieldName, floatValueBuffer, floatNullBitMapBuffer, arrowArray.Length, arrowArray.NullCount);
                    break;
                case ArrowTypeId.Int8:
                    PrimitiveArray<sbyte> arrowsbyteArray = (PrimitiveArray<sbyte>)arrowArray;
                    ReadOnlyMemory<byte> sbyteValueBuffer = arrowsbyteArray.ValueBuffer.Memory;
                    ReadOnlyMemory<byte> sbyteNullBitMapBuffer = arrowsbyteArray.NullBitmapBuffer.Memory;
                    dataFrameColumn = new SByteDataFrameColumn(fieldName, sbyteValueBuffer, sbyteNullBitMapBuffer, arrowArray.Length, arrowArray.NullCount);
                    break;
                case ArrowTypeId.Int16:
                    PrimitiveArray<short> arrowshortArray = (PrimitiveArray<short>)arrowArray;
                    ReadOnlyMemory<byte> shortValueBuffer = arrowshortArray.ValueBuffer.Memory;
                    ReadOnlyMemory<byte> shortNullBitMapBuffer = arrowshortArray.NullBitmapBuffer.Memory;
                    dataFrameColumn = new Int16DataFrameColumn(fieldName, shortValueBuffer, shortNullBitMapBuffer, arrowArray.Length, arrowArray.NullCount);
                    break;
                case ArrowTypeId.Int32:
                    PrimitiveArray<int> arrowIntArray = (PrimitiveArray<int>)arrowArray;
                    ReadOnlyMemory<byte> intValueBuffer = arrowIntArray.ValueBuffer.Memory;
                    ReadOnlyMemory<byte> intNullBitMapBuffer = arrowIntArray.NullBitmapBuffer.Memory;
                    dataFrameColumn = new Int32DataFrameColumn(fieldName, intValueBuffer, intNullBitMapBuffer, arrowArray.Length, arrowArray.NullCount);
                    break;
                case ArrowTypeId.Int64:
                    PrimitiveArray<long> arrowLongArray = (PrimitiveArray<long>)arrowArray;
                    ReadOnlyMemory<byte> longValueBuffer = arrowLongArray.ValueBuffer.Memory;
                    ReadOnlyMemory<byte> longNullBitMapBuffer = arrowLongArray.NullBitmapBuffer.Memory;
                    dataFrameColumn = new Int64DataFrameColumn(fieldName, longValueBuffer, longNullBitMapBuffer, arrowArray.Length, arrowArray.NullCount);
                    break;
                case ArrowTypeId.String:
                    StringArray stringArray = (StringArray)arrowArray;
                    ReadOnlyMemory<byte> dataMemory = stringArray.ValueBuffer.Memory;
                    ReadOnlyMemory<byte> offsetsMemory = stringArray.ValueOffsetsBuffer.Memory;
                    ReadOnlyMemory<byte> nullMemory = stringArray.NullBitmapBuffer.Memory;
                    dataFrameColumn = new ArrowStringDataFrameColumn(fieldName, dataMemory, offsetsMemory, nullMemory, stringArray.Length, stringArray.NullCount);
                    break;
                case ArrowTypeId.UInt8:
                    PrimitiveArray<byte> arrowbyteArray = (PrimitiveArray<byte>)arrowArray;
                    ReadOnlyMemory<byte> byteValueBuffer = arrowbyteArray.ValueBuffer.Memory;
                    ReadOnlyMemory<byte> byteNullBitMapBuffer = arrowbyteArray.NullBitmapBuffer.Memory;
                    dataFrameColumn = new ByteDataFrameColumn(fieldName, byteValueBuffer, byteNullBitMapBuffer, arrowArray.Length, arrowArray.NullCount);
                    break;
                case ArrowTypeId.UInt16:
                    PrimitiveArray<ushort> arrowUshortArray = (PrimitiveArray<ushort>)arrowArray;
                    ReadOnlyMemory<byte> ushortValueBuffer = arrowUshortArray.ValueBuffer.Memory;
                    ReadOnlyMemory<byte> ushortNullBitMapBuffer = arrowUshortArray.NullBitmapBuffer.Memory;
                    dataFrameColumn = new UInt16DataFrameColumn(fieldName, ushortValueBuffer, ushortNullBitMapBuffer, arrowArray.Length, arrowArray.NullCount);
                    break;
                case ArrowTypeId.UInt32:
                    PrimitiveArray<uint> arrowUintArray = (PrimitiveArray<uint>)arrowArray;
                    ReadOnlyMemory<byte> uintValueBuffer = arrowUintArray.ValueBuffer.Memory;
                    ReadOnlyMemory<byte> uintNullBitMapBuffer = arrowUintArray.NullBitmapBuffer.Memory;
                    dataFrameColumn = new UInt32DataFrameColumn(fieldName, uintValueBuffer, uintNullBitMapBuffer, arrowArray.Length, arrowArray.NullCount);
                    break;
                case ArrowTypeId.UInt64:
                    PrimitiveArray<ulong> arrowUlongArray = (PrimitiveArray<ulong>)arrowArray;
                    ReadOnlyMemory<byte> ulongValueBuffer = arrowUlongArray.ValueBuffer.Memory;
                    ReadOnlyMemory<byte> ulongNullBitMapBuffer = arrowUlongArray.NullBitmapBuffer.Memory;
                    dataFrameColumn = new UInt64DataFrameColumn(fieldName, ulongValueBuffer, ulongNullBitMapBuffer, arrowArray.Length, arrowArray.NullCount);
                    break;
                case ArrowTypeId.Struct:
                    StructArray structArray = (StructArray)arrowArray;
                    StructType structType = (StructType)field.DataType;
                    IEnumerator<Field> fieldsEnumerator = structType.Fields.GetEnumerator();
                    IEnumerator<IArrowArray> structArrayEnumerator = structArray.Fields.GetEnumerator();
                    while (fieldsEnumerator.MoveNext() && structArrayEnumerator.MoveNext())
                    {
                        AppendDataFrameColumnFromArrowArray(fieldsEnumerator.Current, structArrayEnumerator.Current, ret, field.Name + "_");
                    }
                    break;
                case ArrowTypeId.Decimal:
                case ArrowTypeId.Binary:
                case ArrowTypeId.Date32:
                case ArrowTypeId.Date64:
                case ArrowTypeId.Dictionary:
                case ArrowTypeId.FixedSizedBinary:
                case ArrowTypeId.HalfFloat:
                case ArrowTypeId.Interval:
                case ArrowTypeId.List:
                case ArrowTypeId.Map:
                case ArrowTypeId.Null:
                case ArrowTypeId.Time32:
                case ArrowTypeId.Time64:
                default:
                    throw new NotImplementedException(nameof(fieldType.Name));
            }

            if (dataFrameColumn != null)
            {
                ret.Columns.Insert(ret.Columns.Count, dataFrameColumn);
            }
        }

        /// <summary>
        /// Wraps a <see cref="DataFrame"/> around an Arrow <see cref="RecordBatch"/> without copying data
        /// </summary>
        /// <param name="recordBatch"></param>
        /// <returns><see cref="DataFrame"/></returns>
        public static DataFrame FromArrowRecordBatch(RecordBatch recordBatch)
        {
            DataFrame ret = new DataFrame();
            Apache.Arrow.Schema arrowSchema = recordBatch.Schema;
            int fieldIndex = 0;
            IEnumerable<IArrowArray> arrowArrays = recordBatch.Arrays;
            foreach (IArrowArray arrowArray in arrowArrays)
            {
                Field field = arrowSchema.GetFieldByIndex(fieldIndex);
                AppendDataFrameColumnFromArrowArray(field, arrowArray, ret);
                fieldIndex++;
            }
            return ret;
        }

        /// <summary>
        /// Returns an <see cref="IEnumerable{RecordBatch}"/> without copying data
        /// </summary>
        public IEnumerable<RecordBatch> ToArrowRecordBatches()
        {
            Apache.Arrow.Schema.Builder schemaBuilder = new Apache.Arrow.Schema.Builder();

            int columnCount = Columns.Count;
            for (int i = 0; i < columnCount; i++)
            {
                DataFrameColumn column = Columns[i];
                Field field = column.GetArrowField();
                schemaBuilder.Field(field);
            }

            Schema schema = schemaBuilder.Build();
            List<Apache.Arrow.Array> arrays = new List<Apache.Arrow.Array>();

            int recordBatchLength = Int32.MaxValue;
            int numberOfRowsInThisRecordBatch = (int)Math.Min(recordBatchLength, Rows.Count);
            long numberOfRowsProcessed = 0;

            // Sometimes .NET for Spark passes in DataFrames with no rows. In those cases, we just return a RecordBatch with the right Schema and no rows
            do
            {
                for (int i = 0; i < columnCount; i++)
                {
                    DataFrameColumn column = Columns[i];
                    numberOfRowsInThisRecordBatch = (int)Math.Min(numberOfRowsInThisRecordBatch, column.GetMaxRecordBatchLength(numberOfRowsProcessed));
                }
                for (int i = 0; i < columnCount; i++)
                {
                    DataFrameColumn column = Columns[i];
                    arrays.Add(column.ToArrowArray(numberOfRowsProcessed, numberOfRowsInThisRecordBatch));
                }
                numberOfRowsProcessed += numberOfRowsInThisRecordBatch;
                yield return new RecordBatch(schema, arrays, numberOfRowsInThisRecordBatch);
            } while (numberOfRowsProcessed < Rows.Count);
        }

    }
}
