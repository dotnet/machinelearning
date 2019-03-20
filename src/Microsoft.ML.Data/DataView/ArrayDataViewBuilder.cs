// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    using BitArray = System.Collections.BitArray;

    /// <summary>
    /// This is a class for composing an in memory IDataView.
    /// </summary>
    [BestFriend]
    internal sealed class ArrayDataViewBuilder
    {
        private readonly IHost _host;
        private readonly List<Column> _columns;
        private readonly List<string> _names;
        private readonly Dictionary<string, ValueGetter<VBuffer<ReadOnlyMemory<char>>>> _getSlotNames;
        private readonly Dictionary<string, ValueGetter<VBuffer<ReadOnlyMemory<char>>>> _getKeyValues;

        private int? RowCount
        {
            get
            {
                if (_columns.Count == 0)
                    return null;
                return _columns[0].Length;
            }
        }

        public ArrayDataViewBuilder(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register("ArrayDataViewBuilder");

            _columns = new List<Column>();
            _names = new List<string>();
            _getSlotNames = new Dictionary<string, ValueGetter<VBuffer<ReadOnlyMemory<char>>>>();
            _getKeyValues = new Dictionary<string, ValueGetter<VBuffer<ReadOnlyMemory<char>>>>();
        }

        /// <summary>
        /// Verifies that the input array to one of the add routines is of the same length
        /// as previously added arrays, assuming there were any.
        /// </summary>
        private void CheckLength<T>(string name, T[] values)
        {
            _host.CheckValue(name, nameof(name));
            _host.CheckValue(values, nameof(values));
            if (_columns.Count > 0 && values.Length != _columns[0].Length)
                throw _host.Except("Previous inputs were of length {0}, but new input is of length {1}", _columns[0].Length, values.Length);
        }

        /// <summary>
        /// Constructs a new column from an array where values are copied to output simply
        /// by being assigned. Output values are returned simply by being assigned, so the
        /// type <typeparamref name="T"/> should be a type where assigning to a different
        /// value does not compromise the immutability of the source object (so, for example,
        /// a scalar, string, or <c>ReadOnlyMemory</c> would be perfectly acceptable, but a
        /// <c>HashSet</c> or <c>VBuffer</c> would not be).
        /// </summary>
        public void AddColumn<T>(string name, PrimitiveDataViewType type, params T[] values)
        {
            _host.CheckParam(type != null && type.RawType == typeof(T), nameof(type));
            CheckLength(name, values);
            _columns.Add(new AssignmentColumn<T>(type, values));
            _names.Add(name);
        }

        /// <summary>
        /// Constructs a new key column from an array where values are copied to output simply
        /// by being assigned.
        /// </summary>
        /// <param name="name">The name of the column.</param>
        /// <param name="getKeyValues">The delegate that does a reverse lookup based upon the given key. This is for annotation creation</param>
        /// <param name="keyCount">The count of unique keys specified in values</param>
        /// <param name="values">The values to add to the column. Note that since this is creating a <see cref="KeyType"/> column, the values will be offset by 1.</param>
        public void AddColumn<T1>(string name, ValueGetter<VBuffer<ReadOnlyMemory<char>>> getKeyValues, ulong keyCount, params T1[] values)
        {
            _host.CheckValue(getKeyValues, nameof(getKeyValues));
            _host.CheckParam(keyCount > 0, nameof(keyCount));
            CheckLength(name, values);
            values.GetType().GetElementType().TryGetDataKind(out InternalDataKind kind);
            _columns.Add(new AssignmentColumn<T1>(new KeyType(kind.ToType(), keyCount), values));
            _getKeyValues.Add(name, getKeyValues);
            _names.Add(name);
        }

        /// <summary>
        /// Creates a column with slot names from arrays. The added column will be re-interpreted as a buffer.
        /// </summary>
        public void AddColumn<T>(string name, ValueGetter<VBuffer<ReadOnlyMemory<char>>> getNames, PrimitiveDataViewType itemType, params T[][] values)
        {
            _host.CheckValue(getNames, nameof(getNames));
            _host.CheckParam(itemType != null && itemType.RawType == typeof(T), nameof(itemType));
            CheckLength(name, values);
            var col = new ArrayToVBufferColumn<T>(itemType, values);
            _columns.Add(col);
            _getSlotNames.Add(name, getNames);
            _names.Add(name);
        }

        /// <summary>
        /// Creates a column from arrays. The added column will be re-interpreted as a buffer.
        /// </summary>
        public void AddColumn<T>(string name, PrimitiveDataViewType itemType, params T[][] values)
        {
            _host.CheckParam(itemType != null && itemType.RawType == typeof(T), nameof(itemType));
            CheckLength(name, values);
            _columns.Add(new ArrayToVBufferColumn<T>(itemType, values));
            _names.Add(name);
        }

        /// <summary>
        /// Creates a column with slot names from arrays. The added column will be re-interpreted as a buffer and possibly sparsified.
        /// </summary>
        public void AddColumn<T>(string name, ValueGetter<VBuffer<ReadOnlyMemory<char>>> getNames, PrimitiveDataViewType itemType, Combiner<T> combiner, params T[][] values)
        {
            _host.CheckValue(getNames, nameof(getNames));
            _host.CheckParam(itemType != null && itemType.RawType == typeof(T), nameof(itemType));
            CheckLength(name, values);
            var col = new ArrayToSparseVBufferColumn<T>(itemType, combiner, values);
            _columns.Add(col);
            _getSlotNames.Add(name, getNames);
            _names.Add(name);
        }

        /// <summary>
        /// Creates a column from arrays. The added column will be re-interpreted as a buffer and possibly sparsified.
        /// </summary>
        public void AddColumn<T>(string name, PrimitiveDataViewType itemType, Combiner<T> combiner, params T[][] values)
        {
            _host.CheckParam(itemType != null && itemType.RawType == typeof(T), nameof(itemType));
            CheckLength(name, values);
            _columns.Add(new ArrayToSparseVBufferColumn<T>(itemType, combiner, values));
            _names.Add(name);
        }

        /// <summary>
        /// Adds a VBuffer{T} valued column.
        /// </summary>
        public void AddColumn<T>(string name, PrimitiveDataViewType itemType, params VBuffer<T>[] values)
        {
            _host.CheckParam(itemType != null && itemType.RawType == typeof(T), nameof(itemType));
            CheckLength(name, values);
            _columns.Add(new VBufferColumn<T>(itemType, values));
            _names.Add(name);
        }

        /// <summary>
        /// Adds a VBuffer{T} valued column.
        /// </summary>
        public void AddColumn<T>(string name, ValueGetter<VBuffer<ReadOnlyMemory<char>>> getNames, PrimitiveDataViewType itemType, params VBuffer<T>[] values)
        {
            _host.CheckValue(getNames, nameof(getNames));
            _host.CheckParam(itemType != null && itemType.RawType == typeof(T), nameof(itemType));
            CheckLength(name, values);
            _columns.Add(new VBufferColumn<T>(itemType, values));
            _getSlotNames.Add(name, getNames);
            _names.Add(name);
        }

        /// <summary>
        /// Adds a <c>ReadOnlyMemory</c> valued column from an array of strings.
        /// </summary>
        public void AddColumn(string name, params string[] values)
        {
            CheckLength(name, values);
            _columns.Add(new StringToTextColumn(values));
            _names.Add(name);
        }

        /// <summary>
        /// Constructs a data view from the columns added so far. Note that it is perfectly acceptable
        /// to continue adding columns to the builder, but these additions will not be reflected in the
        /// returned dataview.
        /// </summary>
        /// <param name="rowCount"></param>
        public IDataView GetDataView(int? rowCount = null)
        {
            if (rowCount.HasValue)
            {
                _host.Check(!RowCount.HasValue || RowCount.Value == rowCount.Value, "Specified row count incompatible with existing columns");
                return new DataView(_host, this, rowCount.Value);
            }
            _host.Check(_columns.Count > 0, "Cannot construct data-view with neither any columns nor a specified row count");
            return new DataView(_host, this, RowCount.Value);
        }

        private sealed class DataView : IDataView
        {
            private readonly int _rowCount;
            private readonly Column[] _columns;
            private readonly DataViewSchema _schema;
            private readonly IHost _host;

            public DataViewSchema Schema { get { return _schema; } }

            public long? GetRowCount() { return _rowCount; }

            public bool CanShuffle { get { return true; } }

            public DataView(IHostEnvironment env, ArrayDataViewBuilder builder, int rowCount)
            {
                Contracts.AssertValue(env, "env");
                _host = env.Register("ArrayDataView");

                _host.AssertValue(builder);
                _host.Assert(rowCount >= 0);
                _host.Assert(builder._names.Count == builder._columns.Count);
                _columns = builder._columns.ToArray();

                var schemaBuilder = new DataViewSchema.Builder();
                for(int i=0; i< _columns.Length; i++)
                {
                    var meta = new DataViewSchema.Annotations.Builder();

                    if (builder._getSlotNames.TryGetValue(builder._names[i], out var slotNamesGetter))
                        meta.AddSlotNames(_columns[i].Type.GetVectorSize(), slotNamesGetter);

                    if (builder._getKeyValues.TryGetValue(builder._names[i], out var keyValueGetter))
                        meta.AddKeyValues(_columns[i].Type.GetKeyCountAsInt32(_host), TextDataViewType.Instance, keyValueGetter);
                    schemaBuilder.AddColumn(builder._names[i], _columns[i].Type, meta.ToAnnotations());
                }

                _schema = schemaBuilder.ToSchema();
                _rowCount = rowCount;
            }

            public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, Schema);

                _host.CheckValueOrNull(rand);
                return new Cursor(_host, this, predicate, rand);
            }

            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            {
                var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, Schema);

                _host.CheckValueOrNull(rand);
                return new DataViewRowCursor[] { new Cursor(_host, this, predicate, rand) };
            }

            private sealed class Cursor : RootCursorBase
            {
                private readonly DataView _view;
                private readonly BitArray _active;
                private readonly int[] _indices;

                public override DataViewSchema Schema => _view.Schema;

                public override long Batch
                {
                    // REVIEW: Implement cursor set support.
                    get { return 0; }
                }

                public Cursor(IChannelProvider provider, DataView view, Func<int, bool> predicate, Random rand)
                    : base(provider)
                {
                    Ch.AssertValue(view);
                    Ch.AssertValueOrNull(rand);
                    Ch.Assert(view.Schema.Count >= 0);

                    _view = view;
                    _active = new BitArray(view.Schema.Count);
                    if (predicate == null)
                        _active.SetAll(true);
                    else
                    {
                        for (int i = 0; i < view.Schema.Count; ++i)
                            _active[i] = predicate(i);
                    }
                    if (rand != null)
                        _indices = Utils.GetRandomPermutation(rand, view._rowCount);
                }

                public override ValueGetter<DataViewRowId> GetIdGetter()
                {
                    if (_indices == null)
                    {
                        return
                            (ref DataViewRowId val) =>
                            {
                                Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                                val = new DataViewRowId((ulong)Position, 0);
                            };
                    }
                    else
                    {
                        return
                            (ref DataViewRowId val) =>
                            {
                                Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                                val = new DataViewRowId((ulong)MappedIndex(), 0);
                            };
                    }
                }

                /// <summary>
                /// Returns whether the given column is active in this row.
                /// </summary>
                public override bool IsColumnActive(DataViewSchema.Column column)
                {
                    Ch.Check(column.Index < Schema.Count);
                    return _active[column.Index];
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
                    Ch.Check(column.Index < Schema.Count);
                    Ch.Check(column.Index < _active.Length && _active[column.Index], "the requested column is not active");

                    var columnValue = _view._columns[column.Index] as Column<TValue>;
                    if (columnValue == null)
                        throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));

                    return
                        (ref TValue value) =>
                        {
                            Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                            columnValue.CopyOut(MappedIndex(), ref value);
                        };
                }

                protected override bool MoveNextCore()
                {
                    Ch.Assert(Position < _view._rowCount);
                    return 1 < _view._rowCount - Position;
                }

                private int MappedIndex()
                {
                    Ch.Assert(IsGood);
                    Ch.Assert(0 <= Position && Position < _view._rowCount);
                    if (_indices == null)
                        return (int)Position;
                    return _indices[(int)Position];
                }
            }
        }

        #region Column implementations

        private abstract class Column
        {
            public readonly DataViewType Type;

            public abstract int Length { get; }

            public Column(DataViewType type)
            {
                Type = type;
            }
        }

        private abstract class Column<TOut> : Column
        {
            /// <summary>
            /// Produce the output value given the index.
            /// </summary>
            public abstract void CopyOut(int index, ref TOut value);

            public Column(DataViewType type)
                : base(type)
            {
                Contracts.Assert(typeof(TOut) == type.RawType);
            }
        }

        private abstract class Column<TIn, TOut> : Column<TOut>
        {
            private readonly TIn[] _values;

            public override int Length { get { return _values.Length; } }

            public Column(DataViewType type, TIn[] values)
                : base(type)
            {
                Contracts.AssertValue(values);
                _values = values;
            }

            /// <summary>
            /// Assigns dst in such a way that the caller has ownership of <c>dst</c> without
            /// compromising this object's ownership of <c>src</c>. What that operation will be
            /// will depend on the types.
            /// </summary>
            protected abstract void CopyOut(in TIn src, ref TOut dst);

            /// <summary>
            /// Produce the output value given the index. This overload utilizes the <c>CopyOut</c>
            /// helper function.
            /// </summary>
            public override void CopyOut(int index, ref TOut value)
            {
                Contracts.Assert(0 <= index & index < _values.Length);
                CopyOut(in _values[index], ref value);
            }
        }

        /// <summary>
        /// A column where the input and output types are the same, and simple assignment does
        /// not compromise ownership of the internal vlaues.
        /// </summary>
        private sealed class AssignmentColumn<T> : Column<T, T>
        {
            public AssignmentColumn(PrimitiveDataViewType type, T[] values)
                : base(type, values)
            {
            }

            protected override void CopyOut(in T src, ref T dst)
            {
                dst = src;
            }
        }

        /// <summary>
        /// A convenience column for converting strings into textspans.
        /// </summary>
        private sealed class StringToTextColumn : Column<string, ReadOnlyMemory<char>>
        {
            public StringToTextColumn(string[] values)
                : base(TextDataViewType.Instance, values)
            {
            }

            protected override void CopyOut(in string src, ref ReadOnlyMemory<char> dst)
            {
                dst = src.AsMemory();
            }
        }

        private abstract class VectorColumn<TIn, TOut> : Column<TIn, VBuffer<TOut>>
        {
            public VectorColumn(PrimitiveDataViewType itemType, TIn[] values, Func<TIn, int> lengthFunc)
                : base(InferType(itemType, values, lengthFunc), values)
            {
            }

            /// <summary>
            /// A utility function for subclasses that want to get the type with a dimension based
            /// on the input value array and some length function over the input type.
            /// </summary>
            private static DataViewType InferType(PrimitiveDataViewType itemType, TIn[] values, Func<TIn, int> lengthFunc)
            {
                Contracts.AssertValue(itemType);
                Contracts.Assert(itemType.RawType == typeof(TOut));

                int degree = 0;
                if (Utils.Size(values) > 0)
                {
                    degree = lengthFunc(values[0]);
                    for (int i = 1; i < values.Length; ++i)
                    {
                        if (degree != lengthFunc(values[i]))
                        {
                            degree = 0;
                            break;
                        }
                    }
                }
                return new VectorType(itemType, degree);
            }
        }

        /// <summary>
        /// A column of buffers.
        /// </summary>
        private sealed class VBufferColumn<T> : VectorColumn<VBuffer<T>, T>
        {
            public VBufferColumn(PrimitiveDataViewType itemType, VBuffer<T>[] values)
                : base(itemType, values, v => v.Length)
            {
            }

            protected override void CopyOut(in VBuffer<T> src, ref VBuffer<T> dst)
            {
                src.CopyTo(ref dst);
            }
        }

        private sealed class ArrayToVBufferColumn<T> : VectorColumn<T[], T>
        {
            public ArrayToVBufferColumn(PrimitiveDataViewType itemType, T[][] values)
                : base(itemType, values, Utils.Size)
            {
            }

            protected override void CopyOut(in T[] src, ref VBuffer<T> dst)
            {
                VBuffer<T>.Copy(src, 0, ref dst, Utils.Size(src));
            }
        }

        private sealed class ArrayToSparseVBufferColumn<T> : VectorColumn<T[], T>
        {
            private readonly BufferBuilder<T> _bldr;

            public ArrayToSparseVBufferColumn(PrimitiveDataViewType itemType, Combiner<T> combiner, T[][] values)
                : base(itemType, values, Utils.Size)
            {
                _bldr = new BufferBuilder<T>(combiner);
            }

            protected override void CopyOut(in T[] src, ref VBuffer<T> dst)
            {
                var length = Utils.Size(src);
                _bldr.Reset(length, false);
                for (int i = 0; i < length; i++)
                    _bldr.AddFeature(i, src[i]);
                _bldr.GetResult(ref dst);
            }
        }

        #endregion
    }
}
