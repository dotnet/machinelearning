// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Featurizers;
using Microsoft.ML.Runtime;
using Microsoft.Win32.SafeHandles;
using static Microsoft.ML.Featurizers.CommonExtensions;
using static Microsoft.ML.Featurizers.TimeSeriesImputerEstimator;

namespace Microsoft.ML.Transforms
{

    internal sealed class TimeSeriesImputerDataView : IDataTransform
    {
        #region Typed Columns
        private readonly TimeSeriesImputerTransformer _parent;
        public class SharedColumnState
        {
            public SharedColumnState()
            {
                SourceCanMoveNext = true;
                MemStream = new MemoryStream(4096);
                BinWriter = new BinaryWriter(MemStream, Encoding.UTF8);
            }

            public bool SourceCanMoveNext { get; set; }
            public int TransformedDataPosition { get; set; }

            // This array is used to hold the returned row data from the native transformer. Because we create rows in this transformer, the number
            // of rows returned from the native code is not always consistent and so this has to be an array.
            public NativeBinaryArchiveData[] TransformedData { get; set; }

            // Hold the serialized data that we are going to send to the native code for processing.
            public MemoryStream MemStream { get; set; }
            public BinaryWriter BinWriter { get; set; }
            public TransformedDataSafeHandle TransformedDataHandler { get; set; }
        }

        private abstract class TypedColumn
        {
            private protected SharedColumnState SharedState;

            internal readonly DataViewSchema.Column Column;
            internal readonly bool IsImputed;
            internal TypedColumn(DataViewSchema.Column column, bool isImputed, SharedColumnState state)
            {
                Column = column;
                SharedState = state;
                IsImputed = isImputed;
            }

            internal abstract Delegate GetGetter();
            internal abstract void InitializeGetter(DataViewRowCursor cursor, TransformerEstimatorSafeHandle transformerParent, string timeSeriesColumn,
                string[] grainColumns, string[] dataColumns, string[] allColumnNames, Dictionary<string, TypedColumn> allColumns);

            internal abstract TypeId GetTypeId();
            internal abstract void SerializeValue(BinaryWriter binaryWriter);
            internal abstract unsafe int GetDataSizeInBytes(byte* data, int currentOffset);
            internal abstract void QueueNonImputedColumnValue();

            public bool MoveNext(DataViewRowCursor cursor)
            {
                SharedState.TransformedDataPosition++;

                if (SharedState.TransformedData == null || SharedState.TransformedDataPosition >= SharedState.TransformedData.Length)
                    SharedState.SourceCanMoveNext = cursor.MoveNext();

                if (!SharedState.SourceCanMoveNext)
                    if (SharedState.TransformedDataPosition >= SharedState.TransformedData.Length)
                    {
                        if (!SharedState.TransformedDataHandler.IsClosed)
                            SharedState.TransformedDataHandler.Dispose();
                        return false;
                    }

                return true;
            }

            internal static TypedColumn CreateTypedColumn(DataViewSchema.Column column, string[] optionalColumns, string[] allImputedColumns, SharedColumnState state)
            {
                var type = column.Type.RawType.ToString();
                if (type == typeof(sbyte).ToString())
                    return new SByteTypedColumn(column, optionalColumns.Contains(column.Name), allImputedColumns.Contains(column.Name), state);
                else if (type == typeof(short).ToString())
                    return new ShortTypedColumn(column, optionalColumns.Contains(column.Name), allImputedColumns.Contains(column.Name), state);
                else if (type == typeof(int).ToString())
                    return new IntTypedColumn(column, optionalColumns.Contains(column.Name), allImputedColumns.Contains(column.Name), state);
                else if (type == typeof(long).ToString())
                    return new LongTypedColumn(column, optionalColumns.Contains(column.Name), allImputedColumns.Contains(column.Name), state);
                else if (type == typeof(byte).ToString())
                    return new ByteTypedColumn(column, optionalColumns.Contains(column.Name), allImputedColumns.Contains(column.Name), state);
                else if (type == typeof(ushort).ToString())
                    return new UShortTypedColumn(column, optionalColumns.Contains(column.Name), allImputedColumns.Contains(column.Name), state);
                else if (type == typeof(uint).ToString())
                    return new UIntTypedColumn(column, optionalColumns.Contains(column.Name), allImputedColumns.Contains(column.Name), state);
                else if (type == typeof(ulong).ToString())
                    return new ULongTypedColumn(column, optionalColumns.Contains(column.Name), allImputedColumns.Contains(column.Name), state);
                else if (type == typeof(float).ToString())
                    return new FloatTypedColumn(column, optionalColumns.Contains(column.Name), allImputedColumns.Contains(column.Name), state);
                else if (type == typeof(double).ToString())
                    return new DoubleTypedColumn(column, optionalColumns.Contains(column.Name), allImputedColumns.Contains(column.Name), state);
                else if (type == typeof(ReadOnlyMemory<char>).ToString())
                    return new StringTypedColumn(column, optionalColumns.Contains(column.Name), allImputedColumns.Contains(column.Name), state);
                else if (type == typeof(bool).ToString())
                    return new BoolTypedColumn(column, optionalColumns.Contains(column.Name), allImputedColumns.Contains(column.Name), state);
                else if (type == typeof(DateTime).ToString())
                    return new DateTimeTypedColumn(column, optionalColumns.Contains(column.Name), allImputedColumns.Contains(column.Name), state);

                throw new InvalidOperationException($"Unsupported type {type}");
            }
        }

        private abstract class TypedColumn<T> : TypedColumn
        {
            private ValueGetter<T> _getter;
            private ValueGetter<T> _sourceGetter;
            private long _position;
            private T _cache;

            // When columns are not being imputed, we need to store the column values in memory until they are used.
            private protected Queue<T> SourceQueue;

            internal TypedColumn(DataViewSchema.Column column, bool isImputed, SharedColumnState state) :
                base(column, isImputed, state)
            {
                SourceQueue = new Queue<T>();
                _position = -1;
            }

            internal override Delegate GetGetter()
            {
                return _getter;
            }

            internal override unsafe void InitializeGetter(DataViewRowCursor cursor, TransformerEstimatorSafeHandle transformer, string timeSeriesColumn,
                string[] grainColumns, string[] dataColumns, string[] allImputedColumnNames, Dictionary<string, TypedColumn> allColumns)
            {
                if (Column.Name != IsRowImputedColumnName)
                    _sourceGetter = cursor.GetGetter<T>(Column);

                _getter = (ref T dst) =>
                {
                    IntPtr errorHandle = IntPtr.Zero;
                    bool success = false;
                    if (SharedState.TransformedData == null || SharedState.TransformedDataPosition >= SharedState.TransformedData.Length)
                    {
                        // Free native memory if we are about to get more
                        if (SharedState.TransformedData != null && SharedState.TransformedDataPosition >= SharedState.TransformedData.Length)
                            SharedState.TransformedDataHandler.Dispose();

                        var outputDataSize = IntPtr.Zero;
                        NativeBinaryArchiveData* outputData = default;
                        while (outputDataSize == IntPtr.Zero && SharedState.SourceCanMoveNext)
                        {
                            BuildColumnByteArray(allColumns, allImputedColumnNames);
                            QueueDataForNonImputedColumns(allColumns, allImputedColumnNames);
                            fixed (byte* bufferPointer = SharedState.MemStream.GetBuffer())
                            {
                                var binaryArchiveData = new NativeBinaryArchiveData() { Data = bufferPointer, DataSize = new IntPtr(SharedState.MemStream.Position) };
                                success = TransformDataNative(transformer, binaryArchiveData, out outputData, out outputDataSize, out errorHandle);
                                if (!success)
                                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                            }

                            if (outputDataSize == IntPtr.Zero)
                                SharedState.SourceCanMoveNext = cursor.MoveNext();

                            SharedState.MemStream.Position = 0;
                        }

                        if (!SharedState.SourceCanMoveNext)
                            success = FlushDataNative(transformer, out outputData, out outputDataSize, out errorHandle);

                        if (!success)
                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                        if (outputDataSize.ToInt32() > 0)
                        {
                            SharedState.TransformedDataHandler = new TransformedDataSafeHandle((IntPtr)outputData, outputDataSize);
                            SharedState.TransformedData = new NativeBinaryArchiveData[outputDataSize.ToInt32()];
                            for (int i = 0; i < outputDataSize.ToInt32(); i++)
                            {
                                SharedState.TransformedData[i] = *(outputData + i);
                            }
                            SharedState.TransformedDataPosition = 0;
                        }
                    }

                    // Base case where we didn't impute the column
                    if (!allImputedColumnNames.Contains(Column.Name))
                    {
                        var imputedData = SharedState.TransformedData[SharedState.TransformedDataPosition];
                        // If the row was imputed we want to just return the default value for the type.
                        if (BoolTypedColumn.GetBoolFromNativeBinaryArchiveData(imputedData.Data, 0))
                        {
                            dst = default;
                        }
                        else
                        {
                            // If the row wasn't imputed, get the original value for that row we stored in the queue and return that.
                            if (_position != cursor.Position)
                            {
                                _position = cursor.Position;
                                _cache = SourceQueue.Dequeue();
                            }
                            dst = _cache;
                        }
                    }
                    // If we did impute the column then parse the data from the returned byte array.
                    else
                    {
                        var imputedData = SharedState.TransformedData[SharedState.TransformedDataPosition];
                        int offset = 0;
                        foreach (var columnName in allImputedColumnNames)
                        {
                            var col = allColumns[columnName];
                            if (col.Column.Name == Column.Name)
                            {
                                dst = GetDataFromNativeBinaryArchiveData(imputedData.Data, offset);
                                return;
                            }

                            offset += col.GetDataSizeInBytes(imputedData.Data, offset);
                        }

                        // This should never be hit.
                        dst = default;
                    }
                };
            }

            private void QueueDataForNonImputedColumns(Dictionary<string, TypedColumn> allColumns, string[] allImputedColumnNames)
            {
                foreach (var column in allColumns.Where(x => !allImputedColumnNames.Contains(x.Value.Column.Name)).Select(x => x.Value))
                    column.QueueNonImputedColumnValue();
            }

            internal override void QueueNonImputedColumnValue()
            {
                SourceQueue.Enqueue(GetSourceValue());
            }

            private void BuildColumnByteArray(Dictionary<string, TypedColumn> allColumns, string[] columns)
            {
                foreach (var column in columns.Where(x => x != IsRowImputedColumnName))
                {
                    allColumns[column].SerializeValue(SharedState.BinWriter);
                }
            }

            private protected T GetSourceValue()
            {
                T value = default;
                _sourceGetter(ref value);
                return value;
            }

            internal override TypeId GetTypeId()
            {
                return typeof(T).GetNativeTypeIdFromType();
            }

            internal abstract unsafe T GetDataFromNativeBinaryArchiveData(byte* data, int offset);
        }

        private abstract class NumericTypedColumn<T> : TypedColumn<T>
        {
            private protected readonly bool IsNullable;

            internal NumericTypedColumn(DataViewSchema.Column column, bool isNullable, bool isImputed, SharedColumnState state) :
                base(column, isImputed, state)
            {
                IsNullable = isNullable;
            }

            internal override void SerializeValue(BinaryWriter binaryWriter)
            {
                dynamic value = GetSourceValue();

                if (IsNullable && value.GetType() != typeof(float) && value.GetType() != typeof(double))
                    binaryWriter.Write(true);

                binaryWriter.Write(value);
            }

            internal override unsafe int GetDataSizeInBytes(byte* data, int currentOffset)
            {
                if (IsNullable && typeof(T) != typeof(float) && typeof(T) != typeof(double))
                    return Marshal.SizeOf(default(T)) + sizeof(bool);
                else
                    return Marshal.SizeOf(default(T));
            }
        }

        private class ByteTypedColumn : NumericTypedColumn<byte>
        {
            internal ByteTypedColumn(DataViewSchema.Column column, bool isNullable, bool isImputed, SharedColumnState state) :
                base(column, isNullable, isImputed, state)
            {
            }

            internal override unsafe byte GetDataFromNativeBinaryArchiveData(byte* data, int offset)
            {
                if (IsNullable)
                {
                    if (BoolTypedColumn.GetBoolFromNativeBinaryArchiveData(data, offset))
                        return *(byte*)(data + offset + sizeof(bool));
                    else
                        return default;
                }
                else
                    return *(byte*)(data + offset);
            }
        }

        private class SByteTypedColumn : NumericTypedColumn<sbyte>
        {
            internal SByteTypedColumn(DataViewSchema.Column column, bool isNullable, bool isImputed, SharedColumnState state) :
                base(column, isNullable, isImputed, state)
            {
            }

            internal override unsafe sbyte GetDataFromNativeBinaryArchiveData(byte* data, int offset)
            {
                if (IsNullable)
                {
                    if (BoolTypedColumn.GetBoolFromNativeBinaryArchiveData(data, offset))
                        return *(sbyte*)(data + offset + sizeof(bool));
                    else
                        return default;
                }
                else
                    return *(sbyte*)(data + offset);
            }
        }

        private class ShortTypedColumn : NumericTypedColumn<short>
        {
            internal ShortTypedColumn(DataViewSchema.Column column, bool isNullable, bool isImputed, SharedColumnState state) :
                base(column, isNullable, isImputed, state)
            {
            }

            internal override unsafe short GetDataFromNativeBinaryArchiveData(byte* data, int offset)
            {
                if (IsNullable)
                {
                    if (BoolTypedColumn.GetBoolFromNativeBinaryArchiveData(data, offset))
                        return *(short*)(data + offset + sizeof(bool));
                    else
                        return default;
                }
                else
                    return *(short*)(data + offset);
            }
        }

        private class UShortTypedColumn : NumericTypedColumn<ushort>
        {
            internal UShortTypedColumn(DataViewSchema.Column column, bool isNullable, bool isImputed, SharedColumnState state) :
                base(column, isNullable, isImputed, state)
            {
            }

            internal override unsafe ushort GetDataFromNativeBinaryArchiveData(byte* data, int offset)
            {
                if (IsNullable)
                {
                    if (BoolTypedColumn.GetBoolFromNativeBinaryArchiveData(data, offset))
                        return *(ushort*)(data + offset + sizeof(bool));
                    else
                        return default;
                }
                else
                    return *(ushort*)(data + offset);
            }
        }

        private class IntTypedColumn : NumericTypedColumn<int>
        {
            internal IntTypedColumn(DataViewSchema.Column column, bool isNullable, bool isImputed, SharedColumnState state) :
                base(column, isNullable, isImputed, state)
            {
            }

            internal override unsafe int GetDataFromNativeBinaryArchiveData(byte* data, int offset)
            {
                if (IsNullable)
                {
                    if (BoolTypedColumn.GetBoolFromNativeBinaryArchiveData(data, offset))
                        return *(int*)(data + offset + sizeof(bool));
                    else
                        return default;
                }
                else
                    return *(int*)(data + offset);
            }
        }

        private class UIntTypedColumn : NumericTypedColumn<uint>
        {
            internal UIntTypedColumn(DataViewSchema.Column column, bool isNullable, bool isImputed, SharedColumnState state) :
                base(column, isNullable, isImputed, state)
            {
            }

            internal override unsafe uint GetDataFromNativeBinaryArchiveData(byte* data, int offset)
            {
                if (IsNullable)
                {
                    if (BoolTypedColumn.GetBoolFromNativeBinaryArchiveData(data, offset))
                        return *(uint*)(data + offset + sizeof(bool));
                    else
                        return default;
                }
                else
                    return *(uint*)(data + offset);
            }
        }

        private class LongTypedColumn : NumericTypedColumn<long>
        {
            internal LongTypedColumn(DataViewSchema.Column column, bool isNullable, bool isImputed, SharedColumnState state) :
                base(column, isNullable, isImputed, state)
            {
            }

            internal override unsafe long GetDataFromNativeBinaryArchiveData(byte* data, int offset)
            {
                if (IsNullable)
                {
                    if (BoolTypedColumn.GetBoolFromNativeBinaryArchiveData(data, offset))
                        return *(long*)(data + offset + sizeof(bool));
                    else
                        return default;
                }
                else
                    return *(long*)(data + offset);
            }
        }

        private class ULongTypedColumn : NumericTypedColumn<ulong>
        {
            internal ULongTypedColumn(DataViewSchema.Column column, bool isNullable, bool isImputed, SharedColumnState state) :
                base(column, isNullable, isImputed, state)
            {
            }

            internal override unsafe ulong GetDataFromNativeBinaryArchiveData(byte* data, int offset)
            {
                if (IsNullable)
                {
                    if (BoolTypedColumn.GetBoolFromNativeBinaryArchiveData(data, offset))
                        return *(ulong*)(data + offset + sizeof(bool));
                    else
                        return default;
                }
                else
                    return *(ulong*)(data + offset);
            }
        }

        private class FloatTypedColumn : NumericTypedColumn<float>
        {
            internal FloatTypedColumn(DataViewSchema.Column column, bool isNullable, bool isImputed, SharedColumnState state) :
                base(column, isNullable, isImputed, state)
            {
            }

            internal override unsafe float GetDataFromNativeBinaryArchiveData(byte* data, int offset)
            {
                var bytes = new byte[sizeof(float)];
                Marshal.Copy((IntPtr)(data + offset), bytes, 0, sizeof(float));
                return BitConverter.ToSingle(bytes, 0);
            }
        }

        private class DoubleTypedColumn : NumericTypedColumn<double>
        {
            internal DoubleTypedColumn(DataViewSchema.Column column, bool isNullable, bool isImputed, SharedColumnState state) :
                base(column, isNullable, isImputed, state)
            {
            }

            internal override unsafe double GetDataFromNativeBinaryArchiveData(byte* data, int offset)
            {
                var bytes = new byte[sizeof(double)];
                Marshal.Copy((IntPtr)(data + offset), bytes, 0, sizeof(double));
                return BitConverter.ToDouble(bytes, 0);
            }
        }

        private class BoolTypedColumn : NumericTypedColumn<bool>
        {
            internal BoolTypedColumn(DataViewSchema.Column column, bool isNullable, bool isImputed, SharedColumnState state) :
                base(column, isNullable, isImputed, state)
            {
            }

            internal override unsafe bool GetDataFromNativeBinaryArchiveData(byte* data, int offset)
            {
                if (IsNullable)
                {
                    if (GetBoolFromNativeBinaryArchiveData(data, offset))
                        return *(bool*)(data + offset + sizeof(bool));
                    else
                        return default;
                }
                else
                    return *(bool*)(data + offset);
            }

            internal static unsafe bool GetBoolFromNativeBinaryArchiveData(byte* data, int offset)
            {
                return *(bool*)(data + offset);
            }

            internal override unsafe int GetDataSizeInBytes(byte* data, int currentOffset)
            {
                return sizeof(bool);
            }
        }

        private class StringTypedColumn : TypedColumn<ReadOnlyMemory<char>>
        {
            private readonly bool _isNullable;
            internal StringTypedColumn(DataViewSchema.Column column, bool isNullable, bool isImputed, SharedColumnState state) :
                base(column, isImputed, state)
            {
                _isNullable = isNullable;
            }

            internal override void SerializeValue(BinaryWriter binaryWriter)
            {
                var value = GetSourceValue().ToString();
                var stringBytes = Encoding.UTF8.GetBytes(value);

                if (_isNullable)
                    binaryWriter.Write(true);

                binaryWriter.Write(stringBytes.Length);

                binaryWriter.Write(stringBytes);
            }

            internal override unsafe ReadOnlyMemory<char> GetDataFromNativeBinaryArchiveData(byte* data, int offset)
            {
                if (_isNullable)
                {
                    if (!BoolTypedColumn.GetBoolFromNativeBinaryArchiveData(data, offset)) // If value not present return empty string
                        return new ReadOnlyMemory<char>("".ToCharArray());

                    var size = *(uint*)(data + offset + 1); // Add 1 for the byte bool flag

                    var bytes = new byte[size];
                    Marshal.Copy((IntPtr)(data + offset + sizeof(uint) + 1), bytes, 0, (int)size);
                    return Encoding.UTF8.GetString(bytes).AsMemory();
                }
                else
                {
                    var size = *(uint*)(data + offset);

                    var bytes = new byte[size];
                    Marshal.Copy((IntPtr)(data + offset + sizeof(uint)), bytes, 0, (int)size);
                    return Encoding.UTF8.GetString(bytes).AsMemory();
                }
            }

            internal override unsafe int GetDataSizeInBytes(byte* data, int currentOffset)
            {
                var size = *(uint*)(data + currentOffset);
                if (_isNullable)
                    return 1 + (int)size + sizeof(uint); // + 1 for the byte bool flag

                return (int)size + sizeof(uint);
            }
        }

        private class DateTimeTypedColumn : TypedColumn<DateTime>
        {
            private static readonly DateTime _unixEpoch = new DateTime(1970, 1, 1);
            private readonly bool _isNullable;

            internal DateTimeTypedColumn(DataViewSchema.Column column, bool isNullable, bool isImputed, SharedColumnState state) :
                base(column, isImputed, state)
            {
                _isNullable = isNullable;
            }

            internal override void SerializeValue(BinaryWriter binaryWriter)
            {
                var dateTime = GetSourceValue();

                var value = dateTime.Subtract(_unixEpoch).Ticks / TimeSpan.TicksPerSecond;

                if (_isNullable)
                    binaryWriter.Write(true);

                binaryWriter.Write(value);
            }

            internal override unsafe DateTime GetDataFromNativeBinaryArchiveData(byte* data, int offset)
            {
                long value;
                if (_isNullable)
                {
                    if (!BoolTypedColumn.GetBoolFromNativeBinaryArchiveData(data, offset)) // If value not present return empty string
                        return new DateTime();

                    value = *(long*)(data + offset + 1); // Add 1 for the byte bool flag

                }
                else
                {
                    value = *(long*)(data + offset);
                }

                return new DateTime(_unixEpoch.Ticks + (value * TimeSpan.TicksPerSecond));

            }

            internal override unsafe int GetDataSizeInBytes(byte* data, int currentOffset)
            {
                if (_isNullable)
                    return 1 + sizeof(long); // + 1 for the byte bool flag

                return sizeof(long);
            }
        }

        #endregion

        #region Native Exports

        [DllImport("Featurizers", EntryPoint = "TimeSeriesImputerFeaturizer_BinaryArchive_Transform"), SuppressUnmanagedCodeSecurity]
        private static extern unsafe bool TransformDataNative(TransformerEstimatorSafeHandle transformer, /*in*/ NativeBinaryArchiveData data, out NativeBinaryArchiveData* outputData, out IntPtr outputDataSize, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "TimeSeriesImputerFeaturizer_BinaryArchive_Transform"), SuppressUnmanagedCodeSecurity]
        private static extern unsafe bool FlushDataNative(TransformerEstimatorSafeHandle transformer, out NativeBinaryArchiveData* outputData, out IntPtr outputDataSize, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "TimeSeriesImputerFeaturizer_BinaryArchive_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
        private static extern unsafe bool DestroyTransformedDataNative(IntPtr data, IntPtr dataSize, out IntPtr errorHandle);

        #endregion

        #region Native SafeHandles

        internal class TransformedDataSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
        {
            private readonly IntPtr _size;
            public TransformedDataSafeHandle(IntPtr handle, IntPtr size) : base(true)
            {
                SetHandle(handle);
                _size = size;
            }

            protected override bool ReleaseHandle()
            {
                // Not sure what to do with error stuff here.  There shouldn't ever be one though.
                return DestroyTransformedDataNative(handle, _size, out IntPtr errorHandle);
            }
        }

        #endregion

        private readonly IHostEnvironment _host;
        private readonly IDataView _source;
        private readonly string _timeSeriesColumn;
        private readonly string[] _dataColumns;
        private readonly string[] _grainColumns;
        private readonly string[] _allImputedColumnNames;
        private readonly DataViewSchema _schema;

        internal TimeSeriesImputerDataView(IHostEnvironment env, IDataView input, string timeSeriesColumn, string[] grainColumns, string[] dataColumns, string[] allColumnNames, TimeSeriesImputerTransformer parent)
        {
            _host = env;
            _source = input;

            _timeSeriesColumn = timeSeriesColumn;
            _grainColumns = grainColumns;
            _dataColumns = dataColumns;
            _allImputedColumnNames = new string[] { IsRowImputedColumnName }.Concat(allColumnNames).ToArray();
            _parent = parent;
            // Build new schema.
            var schemaColumns = _source.Schema.ToDictionary(x => x.Name);
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumns(_source.Schema.AsEnumerable());
            schemaBuilder.AddColumn(IsRowImputedColumnName, BooleanDataViewType.Instance);

            _schema = schemaBuilder.ToSchema();
        }

        public bool CanShuffle => false;

        public DataViewSchema Schema => _schema;

        public IDataView Source => _source;

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            _host.AssertValueOrNull(rand);

            var input = _source.GetRowCursorForAllColumns();
            return new Cursor(_host, input, _parent.CloneTransformer(), _timeSeriesColumn, _grainColumns, _dataColumns, _allImputedColumnNames, _schema);
        }

        // Can't use parallel cursors so this defaults to calling non-parallel version
        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null) =>
             new DataViewRowCursor[] { GetRowCursor(columnsNeeded, rand) };

        // Since we may add rows we don't know the row count
        public long? GetRowCount() => null;

        public void Save(ModelSaveContext ctx)
        {
            _parent.Save(ctx);
        }

        private sealed class Cursor : DataViewRowCursor
        {
            private readonly IChannelProvider _ch;
            private readonly DataViewRowCursor _input;
            private long _position;
            private bool _isGood;
            private readonly Dictionary<string, TypedColumn> _allColumns;
            private readonly DataViewSchema _schema;
            private readonly TransformerEstimatorSafeHandle _transformer;

            public Cursor(IChannelProvider provider, DataViewRowCursor input, TransformerEstimatorSafeHandle transformer, string timeSeriesColumn,
                string[] grainColumns, string[] dataColumns, string[] allImputedColumnNames, DataViewSchema schema)
            {
                _ch = provider;
                _ch.CheckValue(input, nameof(input));

                _input = input;
                var length = input.Schema.Count;
                _position = -1;
                _schema = schema;
                _transformer = transformer;

                var sharedState = new SharedColumnState();

                _allColumns = _schema.Select(x => TypedColumn.CreateTypedColumn(x, dataColumns, allImputedColumnNames, sharedState)).ToDictionary(x => x.Column.Name);
                _allColumns[IsRowImputedColumnName] = new BoolTypedColumn(_schema[IsRowImputedColumnName], false, true, sharedState);

                foreach (var column in _allColumns.Values)
                {
                    column.InitializeGetter(_input, transformer, timeSeriesColumn, grainColumns, dataColumns, allImputedColumnNames, _allColumns);
                }
            }

            public sealed override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return
                       (ref DataViewRowId val) =>
                       {
                           _ch.Check(_isGood, RowCursorUtils.FetchValueStateError);
                           val = new DataViewRowId((ulong)Position, 0);
                       };
            }

            public sealed override DataViewSchema Schema => _schema;

            /// <summary>
            /// Since rows will be generated all columns are active
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column) => true;

            protected override void Dispose(bool disposing)
            {
                if (!_transformer.IsClosed)
                    _transformer.Close();
            }

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                _ch.Check(IsColumnActive(column));

                var originFn = _allColumns[column.Name].GetGetter();
                var fn = originFn as ValueGetter<TValue>;
                if (fn == null)
                    throw _ch.Except($"Invalid TValue in GetGetter: '{typeof(TValue)}', " +
                            $"expected type: '{originFn.GetType().GetGenericArguments().First()}'.");
                return fn;
            }

            public override bool MoveNext()
            {
                _position++;
                _isGood = _allColumns[IsRowImputedColumnName].MoveNext(_input);
                return _isGood;
            }

            public sealed override long Position => _position;

            public sealed override long Batch => _input.Batch;
        }
    }
}
