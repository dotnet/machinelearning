// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Data.Common;
using System.Linq;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    public sealed partial class DatabaseLoader
    {
        private sealed class Cursor : RootCursorBase
        {
            private static readonly FuncInstanceMethodInfo1<Cursor, int, Delegate> _createGetterDelegateMethodInfo
                = FuncInstanceMethodInfo1<Cursor, int, Delegate>.Create(target => target.CreateGetterDelegate<int>);

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
                        _command.CommandTimeout = _source.CommandTimeoutInSeconds;
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

                var originFn = _getters[column.Index];
                var fn = originFn as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except($"Invalid TValue in GetGetter: '{typeof(TValue)}', " +
                        $"expected type: '{originFn.GetType().GetGenericArguments().First()}'.");
                return fn;
            }

            private Delegate CreateGetterDelegate(int col)
            {
                return Utils.MarshalInvoke(_createGetterDelegateMethodInfo, this, _bindings.Infos[col].ColType.RawType, col);
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
                else if (typeof(TValue) == typeof(VBuffer<bool>))
                {
                    getterDelegate = CreateVBufferBooleanGetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(VBuffer<byte>))
                {
                    getterDelegate = CreateVBufferByteGetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(VBuffer<DateTime>))
                {
                    getterDelegate = CreateVBufferDateTimeGetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(VBuffer<double>))
                {
                    getterDelegate = CreateVBufferDoubleGetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(VBuffer<short>))
                {
                    getterDelegate = CreateVBufferInt16GetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(VBuffer<int>))
                {
                    getterDelegate = CreateVBufferInt32GetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(VBuffer<long>))
                {
                    getterDelegate = CreateVBufferInt64GetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(VBuffer<sbyte>))
                {
                    getterDelegate = CreateVBufferSByteGetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(VBuffer<float>))
                {
                    getterDelegate = CreateVBufferSingleGetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(VBuffer<ReadOnlyMemory<char>>))
                {
                    getterDelegate = CreateVBufferStringGetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(VBuffer<ushort>))
                {
                    getterDelegate = CreateVBufferUInt16GetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(VBuffer<uint>))
                {
                    getterDelegate = CreateVBufferUInt32GetterDelegate(colInfo);
                }
                else if (typeof(TValue) == typeof(VBuffer<ulong>))
                {
                    getterDelegate = CreateVBufferUInt64GetterDelegate(colInfo);
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
                return (ref bool value) => value = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetBoolean(columnIndex);
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
                return (ref double value) => value = DataReader.IsDBNull(columnIndex) ? double.NaN : DataReader.GetDouble(columnIndex);
            }

            private ValueGetter<short> CreateInt16GetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref short value) => value = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetInt16(columnIndex);
            }

            private ValueGetter<int> CreateInt32GetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref int value) => value = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetInt32(columnIndex);
            }

            private ValueGetter<long> CreateInt64GetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref long value) => value = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetInt64(columnIndex);
            }

            private ValueGetter<sbyte> CreateSByteGetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref sbyte value) => value = DataReader.IsDBNull(columnIndex) ? default : (sbyte)DataReader.GetByte(columnIndex);
            }

            private ValueGetter<float> CreateSingleGetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref float value) => value = DataReader.IsDBNull(columnIndex) ? float.NaN : DataReader.GetFloat(columnIndex);
            }

            private ValueGetter<ReadOnlyMemory<char>> CreateStringGetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref ReadOnlyMemory<char> value) => value = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetString(columnIndex).AsMemory();
            }

            private ValueGetter<ushort> CreateUInt16GetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref ushort value) => value = DataReader.IsDBNull(columnIndex) ? default : (ushort)DataReader.GetInt16(columnIndex);
            }

            private ValueGetter<uint> CreateUInt32GetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref uint value) => value = DataReader.IsDBNull(columnIndex) ? default : (uint)DataReader.GetInt32(columnIndex);
            }

            private ValueGetter<ulong> CreateUInt64GetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref ulong value) => value = DataReader.IsDBNull(columnIndex) ? default : (ulong)DataReader.GetInt64(columnIndex);
            }

            private ValueGetter<VBuffer<bool>> CreateVBufferBooleanGetterDelegate(ColInfo colInfo)
            {
                return (ref VBuffer<bool> value) =>
                {
                    int length = colInfo.SizeBase;
                    var editor = VBufferEditor.Create(ref value, length);

                    int i = 0;
                    var segs = colInfo.Segments;

                    foreach (var seg in segs)
                    {
                        if (seg.Name is null)
                        {
                            for (int columnIndex = seg.Min; columnIndex < seg.Lim; columnIndex++, i++)
                            {
                                editor.Values[i] = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetBoolean(columnIndex);
                            }
                        }
                        else
                        {
                            var columnIndex = DataReader.GetOrdinal(seg.Name);
                            editor.Values[i++] = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetBoolean(columnIndex);
                        }
                    }

                    value = editor.Commit();
                };
            }

            private ValueGetter<VBuffer<byte>> CreateVBufferByteGetterDelegate(ColInfo colInfo)
            {
                return (ref VBuffer<byte> value) =>
                {
                    int length = colInfo.SizeBase;
                    var editor = VBufferEditor.Create(ref value, length);

                    int i = 0;
                    var segs = colInfo.Segments;

                    foreach (var seg in segs)
                    {
                        if (seg.Name is null)
                        {
                            for (int columnIndex = seg.Min; columnIndex < seg.Lim; columnIndex++, i++)
                            {
                                editor.Values[i] = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetByte(columnIndex);
                            }
                        }
                        else
                        {
                            var columnIndex = DataReader.GetOrdinal(seg.Name);
                            editor.Values[i++] = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetByte(columnIndex);
                        }
                    }

                    value = editor.Commit();
                };
            }

            private ValueGetter<VBuffer<DateTime>> CreateVBufferDateTimeGetterDelegate(ColInfo colInfo)
            {
                return (ref VBuffer<DateTime> value) =>
                {
                    int length = colInfo.SizeBase;
                    var editor = VBufferEditor.Create(ref value, length);

                    int i = 0;
                    var segs = colInfo.Segments;

                    foreach (var seg in segs)
                    {
                        if (seg.Name is null)
                        {
                            for (int columnIndex = seg.Min; columnIndex < seg.Lim; columnIndex++, i++)
                            {
                                editor.Values[i] = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetDateTime(columnIndex);
                            }
                        }
                        else
                        {
                            var columnIndex = DataReader.GetOrdinal(seg.Name);
                            editor.Values[i++] = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetDateTime(columnIndex);
                        }
                    }

                    value = editor.Commit();
                };
            }

            private ValueGetter<VBuffer<double>> CreateVBufferDoubleGetterDelegate(ColInfo colInfo)
            {
                return (ref VBuffer<double> value) =>
                {
                    int length = colInfo.SizeBase;
                    var editor = VBufferEditor.Create(ref value, length);

                    int i = 0;
                    var segs = colInfo.Segments;

                    foreach (var seg in segs)
                    {
                        if (seg.Name is null)
                        {
                            for (int columnIndex = seg.Min; columnIndex < seg.Lim; columnIndex++, i++)
                            {
                                editor.Values[i] = DataReader.IsDBNull(columnIndex) ? double.NaN : DataReader.GetDouble(columnIndex);
                            }
                        }
                        else
                        {
                            var columnIndex = DataReader.GetOrdinal(seg.Name);
                            editor.Values[i++] = DataReader.IsDBNull(columnIndex) ? double.NaN : DataReader.GetDouble(columnIndex);
                        }
                    }

                    value = editor.Commit();
                };
            }

            private ValueGetter<VBuffer<short>> CreateVBufferInt16GetterDelegate(ColInfo colInfo)
            {
                return (ref VBuffer<short> value) =>
                {
                    int length = colInfo.SizeBase;
                    var editor = VBufferEditor.Create(ref value, length);

                    int i = 0;
                    var segs = colInfo.Segments;

                    foreach (var seg in segs)
                    {
                        if (seg.Name is null)
                        {
                            for (int columnIndex = seg.Min; columnIndex < seg.Lim; columnIndex++, i++)
                            {
                                editor.Values[i] = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetInt16(columnIndex);
                            }
                        }
                        else
                        {
                            var columnIndex = DataReader.GetOrdinal(seg.Name);
                            editor.Values[i++] = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetInt16(columnIndex);
                        }
                    }

                    value = editor.Commit();
                };
            }

            private ValueGetter<VBuffer<int>> CreateVBufferInt32GetterDelegate(ColInfo colInfo)
            {
                return (ref VBuffer<int> value) =>
                {
                    int length = colInfo.SizeBase;
                    var editor = VBufferEditor.Create(ref value, length);

                    int i = 0;
                    var segs = colInfo.Segments;

                    foreach (var seg in segs)
                    {
                        if (seg.Name is null)
                        {
                            for (int columnIndex = seg.Min; columnIndex < seg.Lim; columnIndex++, i++)
                            {
                                editor.Values[i] = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetInt32(columnIndex);
                            }
                        }
                        else
                        {
                            var columnIndex = DataReader.GetOrdinal(seg.Name);
                            editor.Values[i++] = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetInt32(columnIndex);
                        }
                    }

                    value = editor.Commit();
                };
            }

            private ValueGetter<VBuffer<long>> CreateVBufferInt64GetterDelegate(ColInfo colInfo)
            {
                return (ref VBuffer<long> value) =>
                {
                    int length = colInfo.SizeBase;
                    var editor = VBufferEditor.Create(ref value, length);

                    int i = 0;
                    var segs = colInfo.Segments;

                    foreach (var seg in segs)
                    {
                        if (seg.Name is null)
                        {
                            for (int columnIndex = seg.Min; columnIndex < seg.Lim; columnIndex++, i++)
                            {
                                editor.Values[i] = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetInt64(columnIndex);
                            }
                        }
                        else
                        {
                            var columnIndex = DataReader.GetOrdinal(seg.Name);
                            editor.Values[i++] = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetInt64(columnIndex);
                        }
                    }

                    value = editor.Commit();
                };
            }

            private ValueGetter<VBuffer<sbyte>> CreateVBufferSByteGetterDelegate(ColInfo colInfo)
            {
                return (ref VBuffer<sbyte> value) =>
                {
                    int length = colInfo.SizeBase;
                    var editor = VBufferEditor.Create(ref value, length);

                    int i = 0;
                    var segs = colInfo.Segments;

                    foreach (var seg in segs)
                    {
                        if (seg.Name is null)
                        {
                            for (int columnIndex = seg.Min; columnIndex < seg.Lim; columnIndex++, i++)
                            {
                                editor.Values[i] = DataReader.IsDBNull(columnIndex) ? default : (sbyte)DataReader.GetByte(columnIndex);
                            }
                        }
                        else
                        {
                            var columnIndex = DataReader.GetOrdinal(seg.Name);
                            editor.Values[i++] = DataReader.IsDBNull(columnIndex) ? default : (sbyte)DataReader.GetByte(columnIndex);
                        }
                    }

                    value = editor.Commit();
                };
            }

            private ValueGetter<VBuffer<float>> CreateVBufferSingleGetterDelegate(ColInfo colInfo)
            {
                return (ref VBuffer<float> value) =>
                {
                    int length = colInfo.SizeBase;
                    var editor = VBufferEditor.Create(ref value, length);

                    int i = 0;
                    var segs = colInfo.Segments;

                    foreach (var seg in segs)
                    {
                        if (seg.Name is null)
                        {
                            for (int columnIndex = seg.Min; columnIndex < seg.Lim; columnIndex++, i++)
                            {
                                editor.Values[i] = DataReader.IsDBNull(columnIndex) ? float.NaN : DataReader.GetFloat(columnIndex);
                            }
                        }
                        else
                        {
                            var columnIndex = DataReader.GetOrdinal(seg.Name);
                            editor.Values[i++] = DataReader.IsDBNull(columnIndex) ? float.NaN : DataReader.GetFloat(columnIndex);
                        }
                    }

                    value = editor.Commit();
                };
            }

            private ValueGetter<VBuffer<ReadOnlyMemory<char>>> CreateVBufferStringGetterDelegate(ColInfo colInfo)
            {
                return (ref VBuffer<ReadOnlyMemory<char>> value) =>
                {
                    int length = colInfo.SizeBase;
                    var editor = VBufferEditor.Create(ref value, length);

                    int i = 0;
                    var segs = colInfo.Segments;

                    foreach (var seg in segs)
                    {
                        if (seg.Name is null)
                        {
                            for (int columnIndex = seg.Min; columnIndex < seg.Lim; columnIndex++, i++)
                            {
                                editor.Values[i] = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetString(columnIndex).AsMemory();
                            }
                        }
                        else
                        {
                            var columnIndex = DataReader.GetOrdinal(seg.Name);
                            editor.Values[i++] = DataReader.IsDBNull(columnIndex) ? default : DataReader.GetString(columnIndex).AsMemory();
                        }
                    }

                    value = editor.Commit();
                };
            }

            private ValueGetter<VBuffer<ushort>> CreateVBufferUInt16GetterDelegate(ColInfo colInfo)
            {
                return (ref VBuffer<ushort> value) =>
                {
                    int length = colInfo.SizeBase;
                    var editor = VBufferEditor.Create(ref value, length);

                    int i = 0;
                    var segs = colInfo.Segments;

                    foreach (var seg in segs)
                    {
                        if (seg.Name is null)
                        {
                            for (int columnIndex = seg.Min; columnIndex < seg.Lim; columnIndex++, i++)
                            {
                                editor.Values[i] = DataReader.IsDBNull(columnIndex) ? default : (ushort)DataReader.GetInt16(columnIndex);
                            }
                        }
                        else
                        {
                            var columnIndex = DataReader.GetOrdinal(seg.Name);
                            editor.Values[i++] = DataReader.IsDBNull(columnIndex) ? default : (ushort)DataReader.GetInt16(columnIndex);
                        }
                    }

                    value = editor.Commit();
                };
            }

            private ValueGetter<VBuffer<uint>> CreateVBufferUInt32GetterDelegate(ColInfo colInfo)
            {
                return (ref VBuffer<uint> value) =>
                {
                    int length = colInfo.SizeBase;
                    var editor = VBufferEditor.Create(ref value, length);

                    int i = 0;
                    var segs = colInfo.Segments;

                    foreach (var seg in segs)
                    {
                        if (seg.Name is null)
                        {
                            for (int columnIndex = seg.Min; columnIndex < seg.Lim; columnIndex++, i++)
                            {
                                editor.Values[i] = DataReader.IsDBNull(columnIndex) ? default : (uint)DataReader.GetInt32(columnIndex);
                            }
                        }
                        else
                        {
                            var columnIndex = DataReader.GetOrdinal(seg.Name);
                            editor.Values[i++] = DataReader.IsDBNull(columnIndex) ? default : (uint)DataReader.GetInt32(columnIndex);
                        }
                    }

                    value = editor.Commit();
                };
            }

            private ValueGetter<VBuffer<ulong>> CreateVBufferUInt64GetterDelegate(ColInfo colInfo)
            {
                return (ref VBuffer<ulong> value) =>
                {
                    int length = colInfo.SizeBase;
                    var editor = VBufferEditor.Create(ref value, length);

                    int i = 0;
                    var segs = colInfo.Segments;

                    foreach (var seg in segs)
                    {
                        if (seg.Name is null)
                        {
                            for (int columnIndex = seg.Min; columnIndex < seg.Lim; columnIndex++, i++)
                            {
                                editor.Values[i] = DataReader.IsDBNull(columnIndex) ? default : (ulong)DataReader.GetInt64(columnIndex);
                            }
                        }
                        else
                        {
                            var columnIndex = DataReader.GetOrdinal(seg.Name);
                            editor.Values[i++] = DataReader.IsDBNull(columnIndex) ? default : (ulong)DataReader.GetInt64(columnIndex);
                        }
                    }

                    value = editor.Commit();
                };
            }

            private int GetColumnIndex(ColInfo colInfo)
            {
                var segs = colInfo.Segments;

                if ((segs is null) || (segs.Length == 0))
                {
                    return DataReader.GetOrdinal(colInfo.Name);
                }

                Contracts.Check(segs.Length == 1);

                var seg = segs[0];

                if (seg.Name is null)
                {
                    Contracts.Check(seg.Lim == (seg.Min + 1));
                    return seg.Min;
                }
                else
                {
                    return DataReader.GetOrdinal(seg.Name);
                }
            }
        }
    }
}
