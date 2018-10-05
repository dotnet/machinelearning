// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    using BitArray = System.Collections.BitArray;

    /// <summary>
    /// This is a class for composing an in memory IDataView.
    /// </summary>
    public sealed class ArrayDataViewBuilder
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
        public void AddColumn<T>(string name, PrimitiveType type, params T[] values)
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
        public void AddColumn(string name, ValueGetter<VBuffer<ReadOnlyMemory<char>>> getKeyValues, ulong keyMin, int keyCount, params uint[] values)
        {
            _host.CheckValue(getKeyValues, nameof(getKeyValues));
            _host.CheckParam(keyCount > 0, nameof(keyCount));
            CheckLength(name, values);
            _columns.Add(new AssignmentColumn<uint>(new KeyType(DataKind.U4, keyMin, keyCount), values));
            _getKeyValues.Add(name, getKeyValues);
            _names.Add(name);
        }

        /// <summary>
        /// Creates a column with slot names from arrays. The added column will be re-interpreted as a buffer.
        /// </summary>
        public void AddColumn<T>(string name, ValueGetter<VBuffer<ReadOnlyMemory<char>>> getNames, PrimitiveType itemType, params T[][] values)
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
        public void AddColumn<T>(string name, PrimitiveType itemType, params T[][] values)
        {
            _host.CheckParam(itemType != null && itemType.RawType == typeof(T), nameof(itemType));
            CheckLength(name, values);
            _columns.Add(new ArrayToVBufferColumn<T>(itemType, values));
            _names.Add(name);
        }

        /// <summary>
        /// Creates a column with slot names from arrays. The added column will be re-interpreted as a buffer and possibly sparsified.
        /// </summary>
        public void AddColumn<T>(string name, ValueGetter<VBuffer<ReadOnlyMemory<char>>> getNames, PrimitiveType itemType, Combiner<T> combiner, params T[][] values)
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
        public void AddColumn<T>(string name, PrimitiveType itemType, Combiner<T> combiner, params T[][] values)
        {
            _host.CheckParam(itemType != null && itemType.RawType == typeof(T), nameof(itemType));
            CheckLength(name, values);
            _columns.Add(new ArrayToSparseVBufferColumn<T>(itemType, combiner, values));
            _names.Add(name);
        }

        /// <summary>
        /// Adds a VBuffer{T} valued column.
        /// </summary>
        public void AddColumn<T>(string name, PrimitiveType itemType, params VBuffer<T>[] values)
        {
            _host.CheckParam(itemType != null && itemType.RawType == typeof(T), nameof(itemType));
            CheckLength(name, values);
            _columns.Add(new VBufferColumn<T>(itemType, values));
            _names.Add(name);
        }

        /// <summary>
        /// Adds a VBuffer{T} valued column.
        /// </summary>
        public void AddColumn<T>(string name, ValueGetter<VBuffer<ReadOnlyMemory<char>>> getNames, PrimitiveType itemType, params VBuffer<T>[] values)
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
            private readonly Schema _schema;
            private readonly IHost _host;

            public ISchema Schema { get { return _schema; } }

            public long? GetRowCount(bool lazy = true) { return _rowCount; }

            public bool CanShuffle { get { return true; } }

            public DataView(IHostEnvironment env, ArrayDataViewBuilder builder, int rowCount)
            {
                Contracts.AssertValue(env, "env");
                _host = env.Register("ArrayDataView");

                _host.AssertValue(builder);
                _host.Assert(rowCount >= 0);
                _host.Assert(builder._names.Count == builder._columns.Count);
                _columns = builder._columns.ToArray();

                var schemaCols = new Schema.Column[_columns.Length];
                for(int i=0; i<schemaCols.Length; i++)
                {
                    var meta = new Schema.MetadataRow.Builder();

                    if (builder._getSlotNames.TryGetValue(builder._names[i], out var slotNamesGetter))
                        meta.AddSlotNames(_columns[i].Type.VectorSize, slotNamesGetter);

                    if (builder._getKeyValues.TryGetValue(builder._names[i], out var keyValueGetter))
                        meta.AddKeyValues(_columns[i].Type.KeyCount, TextType.Instance, keyValueGetter);
                    schemaCols[i] = new Schema.Column(builder._names[i], _columns[i].Type, meta.GetMetadataRow());
                }

                _schema = new Schema(schemaCols);
                _rowCount = rowCount;
            }

            public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
            {
                _host.CheckValue(predicate, nameof(predicate));
                _host.CheckValueOrNull(rand);
                return new RowCursor(_host, this, predicate, rand);
            }

            public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
                Func<int, bool> predicate, int n, IRandom rand = null)
            {
                _host.CheckValue(predicate, nameof(predicate));
                _host.CheckValueOrNull(rand);
                consolidator = null;
                return new IRowCursor[] { new RowCursor(_host, this, predicate, rand) };
            }

            private sealed class RowCursor : RootCursorBase, IRowCursor
            {
                private readonly DataView _view;
                private readonly BitArray _active;
                private readonly int[] _indices;

                public ISchema Schema { get { return _view.Schema; } }

                public override long Batch
                {
                    // REVIEW: Implement cursor set support.
                    get { return 0; }
                }

                public RowCursor(IChannelProvider provider, DataView view, Func<int, bool> predicate, IRandom rand)
                    : base(provider)
                {
                    Ch.AssertValue(view);
                    Ch.AssertValueOrNull(rand);
                    Ch.Assert(view.Schema.ColumnCount >= 0);

                    _view = view;
                    _active = new BitArray(view.Schema.ColumnCount);
                    if (predicate == null)
                        _active.SetAll(true);
                    else
                    {
                        for (int i = 0; i < view.Schema.ColumnCount; ++i)
                            _active[i] = predicate(i);
                    }
                    if (rand != null)
                        _indices = Utils.GetRandomPermutation(rand, view._rowCount);
                }

                public override ValueGetter<UInt128> GetIdGetter()
                {
                    if (_indices == null)
                    {
                        return
                            (ref UInt128 val) =>
                            {
                                Ch.Check(IsGood, "Cannot call ID getter in current state");
                                val = new UInt128((ulong)Position, 0);
                            };
                    }
                    else
                    {
                        return
                            (ref UInt128 val) =>
                            {
                                Ch.Check(IsGood, "Cannot call ID getter in current state");
                                val = new UInt128((ulong)MappedIndex(), 0);
                            };
                    }
                }

                public bool IsColumnActive(int col)
                {
                    Ch.Check(0 <= col & col < Schema.ColumnCount);
                    return _active[col];
                }

                public ValueGetter<TValue> GetGetter<TValue>(int col)
                {
                    Ch.Check(0 <= col & col < Schema.ColumnCount);
                    Ch.Check(_active[col], "column is not active");
                    var column = _view._columns[col] as Column<TValue>;
                    if (column == null)
                        throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));

                    return
                        (ref TValue value) =>
                        {
                            Ch.Check(IsGood);
                            column.CopyOut(MappedIndex(), ref value);
                        };
                }

                protected override bool MoveNextCore()
                {
                    Ch.Assert(State != CursorState.Done);
                    Ch.Assert(Position < _view._rowCount);
                    return 1 < _view._rowCount - Position;
                }

                protected override bool MoveManyCore(long count)
                {
                    Ch.Assert(State != CursorState.Done);
                    Ch.Assert(Position < _view._rowCount);
                    return count < _view._rowCount - Position;
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
            public readonly ColumnType Type;

            public abstract int Length { get; }

            public Column(ColumnType type)
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

            public Column(ColumnType type)
                : base(type)
            {
                Contracts.Assert(typeof(TOut) == type.RawType);
            }
        }

        private abstract class Column<TIn, TOut> : Column<TOut>
        {
            private readonly TIn[] _values;

            public override int Length { get { return _values.Length; } }

            public Column(ColumnType type, TIn[] values)
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
            protected abstract void CopyOut(ref TIn src, ref TOut dst);

            /// <summary>
            /// Produce the output value given the index. This overload utilizes the <c>CopyOut</c>
            /// helper function.
            /// </summary>
            public override void CopyOut(int index, ref TOut value)
            {
                Contracts.Assert(0 <= index & index < _values.Length);
                CopyOut(ref _values[index], ref value);
            }
        }

        /// <summary>
        /// A column where the input and output types are the same, and simple assignment does
        /// not compromise ownership of the internal vlaues.
        /// </summary>
        private sealed class AssignmentColumn<T> : Column<T, T>
        {
            public AssignmentColumn(PrimitiveType type, T[] values)
                : base(type, values)
            {
            }

            protected override void CopyOut(ref T src, ref T dst)
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
                : base(TextType.Instance, values)
            {
            }

            protected override void CopyOut(ref string src, ref ReadOnlyMemory<char> dst)
            {
                dst = src.AsMemory();
            }
        }

        private abstract class VectorColumn<TIn, TOut> : Column<TIn, VBuffer<TOut>>
        {
            public VectorColumn(PrimitiveType itemType, TIn[] values, Func<TIn, int> lengthFunc)
                : base(InferType(itemType, values, lengthFunc), values)
            {
            }

            /// <summary>
            /// A utility function for subclasses that want to get the type with a dimension based
            /// on the input value array and some length function over the input type.
            /// </summary>
            private static ColumnType InferType(PrimitiveType itemType, TIn[] values, Func<TIn, int> lengthFunc)
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
            public VBufferColumn(PrimitiveType itemType, VBuffer<T>[] values)
                : base(itemType, values, v => v.Length)
            {
            }

            protected override void CopyOut(ref VBuffer<T> src, ref VBuffer<T> dst)
            {
                src.CopyTo(ref dst);
            }
        }

        private sealed class ArrayToVBufferColumn<T> : VectorColumn<T[], T>
        {
            public ArrayToVBufferColumn(PrimitiveType itemType, T[][] values)
                : base(itemType, values, Utils.Size)
            {
            }

            protected override void CopyOut(ref T[] src, ref VBuffer<T> dst)
            {
                VBuffer<T>.Copy(src, 0, ref dst, Utils.Size(src));
            }
        }

        private sealed class ArrayToSparseVBufferColumn<T> : VectorColumn<T[], T>
        {
            private readonly BufferBuilder<T> _bldr;

            public ArrayToSparseVBufferColumn(PrimitiveType itemType, Combiner<T> combiner, T[][] values)
                : base(itemType, values, Utils.Size)
            {
                _bldr = new BufferBuilder<T>(combiner);
            }

            protected override void CopyOut(ref T[] src, ref VBuffer<T> dst)
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
