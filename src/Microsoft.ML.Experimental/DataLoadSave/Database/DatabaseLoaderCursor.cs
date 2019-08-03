// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Data.Common;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    public sealed partial class DatabaseLoader
    {
        private sealed class Cursor : RootCursorBase
        {
            private readonly Bindings _bindings;
            private readonly bool[] _active; // Which columns are active.
            private readonly DatabaseSource _source;
            private readonly Delegate[] _getters;

            private DbConnection _connection;
            private DbCommand _command;
            private DbDataReader _dataReader;

            // This holds the overall count of rows currently served up in the cursor.
            private long _total;
            private bool _disposed;

            public override long Batch => 0;

            private Cursor(DatabaseLoader parent, DatabaseSource source, bool[] active)
                : base(parent._host)
            {
                Ch.Assert(active == null || active.Length == parent._bindings.OutputSchema.Count);
                Ch.CheckValue(source, nameof(source));

                _total = -1;
                _bindings = parent._bindings;
                _active = active;
                _source = source;

                _getters = new Delegate[_bindings.Infos.Length];
                for (int i = 0; i < _getters.Length; i++)
                {
                    if (_active != null && !_active[i])
                        continue;
                    _getters[i] = CreateGetterDelegate(i);
                    Ch.Assert(_getters[i] != null);
                }
            }

            public DbConnection Connection
            {
                get
                {
                    if (_connection is null)
                    {
                        _connection = _source.ProviderFactory.CreateConnection();
                        _connection.ConnectionString = _source.ConnectionString;
                        _connection.Open();
                    }
                    return _connection;
                }
            }

            public DbCommand Command
            {
                get
                {
                    if (_command is null)
                    {
                        _command = Connection.CreateCommand();
                        _command.CommandText = _source.CommandText;
                    }
                    return _command;
                }
            }

            public DbDataReader DataReader
            {
                get
                {
                    if (_dataReader is null)
                    {
                        _dataReader = Command.ExecuteReader();
                    }
                    return _dataReader;
                }
            }

            public static DataViewRowCursor Create(DatabaseLoader parent, DatabaseSource source, bool[] active)
            {
                Contracts.AssertValue(parent);
                Contracts.AssertValue(source);
                Contracts.Assert(active == null || active.Length == parent._bindings.OutputSchema.Count);

                return new Cursor(parent, source, active);
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return
                    (ref DataViewRowId val) =>
                    {
                        Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                        val = new DataViewRowId((ulong)_total, 0);
                    };
            }

            public override DataViewSchema Schema => _bindings.OutputSchema;

            protected override void Dispose(bool disposing)
            {
                if (_disposed)
                    return;
                if (disposing)
                {
                    _dataReader?.Dispose();
                    _command?.Dispose();
                    _connection?.Dispose();
                }
                _disposed = true;
                base.Dispose(disposing);
            }

            protected override bool MoveNextCore()
            {
                if (DataReader.Read())
                {
                    _total++;
                    return true;
                }

                return false;
            }

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                Ch.Check(column.Index < _bindings.Infos.Length);
                return _active == null || _active[column.Index];
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
                Ch.CheckParam(column.Index < _getters.Length, nameof(column), "requested column not valid.");
                Ch.Check(IsColumnActive(column));

                var fn = _getters[column.Index] as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }

            private Delegate CreateGetterDelegate(int col)
            {
                return Utils.MarshalInvoke(CreateGetterDelegate<int>, _bindings.Infos[col].ColType.RawType, col);
            }

            private Delegate CreateGetterDelegate<TValue>(int col)
            {
                Delegate getterDelegate;

                var colInfo = _bindings.Infos[col];
                Ch.Assert(colInfo.ColType.RawType == typeof(TValue));

                if (typeof(TValue) == typeof(bool))
                {
                    getterDelegate = CreateBooleanGetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(byte))
                {
                    getterDelegate = CreateByteGetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(DateTime))
                {
                    getterDelegate = CreateDateTimeGetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(double))
                {
                    getterDelegate = CreateDoubleGetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(short))
                {
                    getterDelegate = CreateInt16GetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(int))
                {
                    getterDelegate = CreateInt32GetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(long))
                {
                    getterDelegate = CreateInt64GetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(sbyte))
                {
                    getterDelegate = CreateSByteGetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(float))
                {
                    getterDelegate = CreateSingleGetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(ReadOnlyMemory<char>))
                {
                    getterDelegate = CreateStringGetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(ushort))
                {
                    getterDelegate = CreateUInt16GetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(uint))
                {
                    getterDelegate = CreateUInt32GetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(ulong))
                {
                    getterDelegate = CreateUInt64GetterDelegate(colInfo);
                }
                else
                {
                    throw new NotSupportedException();
                }

                return getterDelegate as ValueGetter<TValue>;
            }

            private ValueGetter<bool> CreateBooleanGetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref bool value) => value = DataReader.GetBoolean(columnIndex);
            }

            private ValueGetter<byte> CreateByteGetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref byte value) => value = DataReader.GetByte(columnIndex);
            }

            private ValueGetter<DateTime> CreateDateTimeGetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref DateTime value) => value = DataReader.GetDateTime(columnIndex);
            }

            private ValueGetter<double> CreateDoubleGetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref double value) => value = DataReader.GetDouble(columnIndex);
            }

            private ValueGetter<short> CreateInt16GetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref short value) => value = DataReader.GetInt16(columnIndex);
            }

            private ValueGetter<int> CreateInt32GetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref int value) => value = DataReader.GetInt32(columnIndex);
            }

            private ValueGetter<long> CreateInt64GetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref long value) => value = DataReader.GetInt64(columnIndex);
            }

            private ValueGetter<sbyte> CreateSByteGetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref sbyte value) => value = (sbyte)DataReader.GetByte(columnIndex);
            }

            private ValueGetter<float> CreateSingleGetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref float value) => value = DataReader.GetFloat(columnIndex);
            }

            private ValueGetter<ReadOnlyMemory<char>> CreateStringGetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref ReadOnlyMemory<char> value) => value = DataReader.GetString(columnIndex).AsMemory();
            }

            private ValueGetter<ushort> CreateUInt16GetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref ushort value) => value = (ushort)DataReader.GetInt16(columnIndex);
            }

            private ValueGetter<uint> CreateUInt32GetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref uint value) => value = (uint)DataReader.GetInt32(columnIndex);
            }

            private ValueGetter<ulong> CreateUInt64GetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref ulong value) => value = (ulong)DataReader.GetInt64(columnIndex);
            }

            private int GetColumnIndex(ColInfo colInfo)
            {
                return colInfo.SourceIndex ?? DataReader.GetOrdinal(colInfo.Name);
            }
        }
    }
}
