// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    public sealed partial class DatabaseLoader
    {
        private sealed class Cursor : RootCursorBase
        {
            private readonly Bindings _bindings;
            private readonly IDataReader _input;
            private readonly bool[] _active; // Which columns are active.
            private readonly Delegate[] _getters;

            // This holds the overall count of rows currently served up in the cursor.
            private long _total;
            private bool _disposed;

            public override long Batch => 0;

            private Cursor(DatabaseLoader parent, IDataReader input, bool[] active)
                : base(parent._host)
            {
                Ch.Assert(active == null || active.Length == parent._bindings.OutputSchema.Count);
                Ch.CheckValue(input, nameof(input));

                _total = -1;
                _bindings = parent._bindings;
                _active = active;
                _input = input;

                _getters = new Delegate[_bindings.Infos.Length];
                for (int i = 0; i < _getters.Length; i++)
                {
                    if (_active != null && !_active[i])
                        continue;
                    _getters[i] = CreateGetterDelegate(i);
                    Ch.Assert(_getters[i] != null);
                }
            }

            public static DataViewRowCursor Create(DatabaseLoader parent, Func<IDataReader> input, bool[] active)
            {
                Contracts.AssertValue(parent);
                Contracts.AssertValue(input);
                Contracts.Assert(active == null || active.Length == parent._bindings.OutputSchema.Count);

                return new Cursor(parent, input(), active);
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
                    _input.Dispose();
                }
                _disposed = true;
                base.Dispose(disposing);
            }

            protected override bool MoveNextCore()
            {
                if (_input.Read())
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
                var colInfo = _bindings.Infos[col];
                Ch.Assert(colInfo.ColType.RawType == typeof(TValue));

                if (typeof(TValue) == typeof(int))
                {
                    return CreateInt32GetterDelegate(colInfo) as ValueGetter<TValue>;
                }
                if (typeof(TValue) == typeof(float))
                {
                    return CreateFloatGetterDelegate(colInfo) as ValueGetter<TValue>;
                }

                throw new NotSupportedException();
            }

            private ValueGetter<int> CreateInt32GetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref int value) => value = _input.GetInt32(columnIndex);
            }

            private ValueGetter<float> CreateFloatGetterDelegate(ColInfo colInfo)
            {
                int columnIndex = GetColumnIndex(colInfo);
                return (ref float value) => value = _input.GetFloat(columnIndex);
            }

            private int GetColumnIndex(ColInfo colInfo)
            {
                return colInfo.SourceIndex ?? _input.GetOrdinal(colInfo.Name);
            }
        }
    }
}
