// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This interface is an analogy to <see cref="IRow"/> that encapsulates the contents of a single
    /// column.
    ///
    /// Note that in the same sense that <see cref="IRow"/> is not thread safe, implementors of this interface
    /// by similar token must not be considered thread safe by users of the interface, and by the same token
    /// implementors should feel free to write their implementations with the expectation that only one thread
    /// will be calling it at a time.
    ///
    /// Similarly, in the same sense that an <see cref="IRow"/> can have its values "change under it" by having
    /// the underlying cursor move, so too might this item have its values change under it, and they will if
    /// they were directly instantiated from a row.
    ///
    /// Generally actual implementors of this interface should not implement this directly, but instead implement
    /// <see cref="IValueColumn{T}"/>.
    /// </summary>
    // REVIEW: It is possible we may want to make this ICounted, but let's not start with
    // that assumption. The use cases I have in mind are that we'll still, on the side, have an
    // IRow lying around.
    public interface IColumn
    {
        /// <summary>
        /// The name of a column. This string should always be non-empty.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// The type of the column.
        /// </summary>
        ColumnType Type { get; }

        // REVIEW: This property anticipates a time when we get away with metadata accessors
        // altogether, and just have the metadata for a column be represented as a row.
        /// <summary>
        /// The metadata for a column, or <c>null</c> if this column has no metadata.
        /// </summary>
        IRow Metadata { get; }

        /// <summary>
        /// Whether the column should be considered active or not.
        /// </summary>
        bool IsActive { get; }

        /// <summary>
        /// The value getter, as a <see cref="Delegate"/>. Implementators should just pass through
        /// <see cref="IValueColumn{T}.GetGetter"/>.
        /// </summary>
        /// <returns>The generic getter delegate</returns>
        Delegate GetGetter();
    }

    /// <summary>
    /// The type specific interface for a <see cref="IColumn"/>.
    /// </summary>
    /// <typeparam name="T">The type of values in this column. This should agree with the <see cref="ColumnType.RawType"/>
    /// field of <see name="IRowColumn.Type"/>.</typeparam>
    public interface IValueColumn<T> : IColumn
    {
        new ValueGetter<T> GetGetter();
    }

    public static class RowColumnUtils
    {
        /// <summary>
        /// Exposes a single column in a row.
        /// </summary>
        /// <param name="row">The row to wrap</param>
        /// <param name="col">The column to expose</param>
        /// <returns>A row column instance</returns>
        public static IColumn GetColumn(IRow row, int col)
        {
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckParam(0 <= col && col < row.Schema.ColumnCount, nameof(col));

            Func<IRow, int, IColumn> func = GetColumnCore<int>;
            return Utils.MarshalInvoke(func, row.Schema.GetColumnType(col).RawType, row, col);
        }

        private static IColumn GetColumnCore<T>(IRow row, int col)
        {
            Contracts.AssertValue(row);
            Contracts.Assert(0 <= col && col < row.Schema.ColumnCount);
            Contracts.Assert(row.Schema.GetColumnType(col).RawType == typeof(T));

            return new RowWrap<T>(row, col);
        }

        /// <summary>
        /// Exposes a single column in a schema. The column is considered inactive.
        /// </summary>
        /// <param name="schema">The schema to get the data for</param>
        /// <param name="col">The column to get</param>
        /// <returns>A column with <see cref="IColumn.IsActive"/> false</returns>
        public static IColumn GetColumn(ISchema schema, int col)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckParam(0 <= col && col < schema.ColumnCount, nameof(col));

            Func<ISchema, int, IColumn> func = GetColumnCore<int>;
            return Utils.MarshalInvoke(func, schema.GetColumnType(col).RawType, schema, col);
        }

        private static IColumn GetColumnCore<T>(ISchema schema, int col)
        {
            Contracts.AssertValue(schema);
            Contracts.Assert(0 <= col && col < schema.ColumnCount);
            Contracts.Assert(schema.GetColumnType(col).RawType == typeof(T));

            return new SchemaWrap<T>(schema, col);
        }

        /// <summary>
        /// Wraps the metadata of a column as a row.
        /// </summary>
        public static IRow GetMetadataAsRow(ISchema schema, int col, Func<string, bool> takeMetadata)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckParam(0 <= col && col < schema.ColumnCount, nameof(col));
            Contracts.CheckValue(takeMetadata, nameof(takeMetadata));

            return new MetadataRow(schema, col, takeMetadata);
        }

        /// <summary>
        /// Constructs a column out of a value. This will store the input value, not make a copy.
        /// </summary>
        /// <typeparam name="T">The type of the value</typeparam>
        /// <param name="name">The column name, which must be non-empty</param>
        /// <param name="type">The type of the column, whose raw type must be <typeparamref name="T"/></param>
        /// <param name="value">The value to store in the column</param>
        /// <param name="meta">Optionally, metadata for the column</param>
        /// <returns>A column with this value</returns>
        public static IColumn GetColumn<T>(string name, ColumnType type, ref T value, IRow meta = null)
        {
            Contracts.CheckNonEmpty(name, nameof(name));
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(type.RawType == typeof(T), nameof(type), "Mismatch on object type and column type");
            if (type.IsVector)
                return Utils.MarshalInvoke(GetColumnVecCore<int>, type.ItemType.RawType, name, type.AsVector, (object)value, meta);
            Contracts.CheckParam(type.IsPrimitive, nameof(type), "Type must be either vector or primitive");
            Contracts.CheckValueOrNull(meta);
            return Utils.MarshalInvoke(GetColumnOneCore<int>, type.RawType, name, type, (object)value, meta);
        }

        private static IColumn GetColumnVecCore<T>(string name, VectorType type, object value, IRow meta)
        {
            // REVIEW: Ugh. Nasty. Any alternative to boxing?
            Contracts.AssertNonEmpty(name);
            Contracts.AssertValue(type);
            Contracts.Assert(type.IsVector);
            Contracts.Assert(type.ItemType.RawType == typeof(T));
            Contracts.Assert(value is VBuffer<T>);
            Contracts.AssertValueOrNull(meta);
            VBuffer<T> typedVal = (VBuffer<T>)value;
            return new ConstVecImpl<T>(name, meta, type, typedVal);
        }

        private static IColumn GetColumnOneCore<T>(string name, ColumnType type, object value, IRow meta)
        {
            Contracts.AssertNonEmpty(name);
            Contracts.AssertValue(type);
            Contracts.Assert(type.IsPrimitive);
            Contracts.Assert(type.RawType == typeof(T));
            Contracts.Assert(value is T);
            Contracts.AssertValueOrNull(meta);
            T typedVal = (T)value;
            return new ConstOneImpl<T>(name, meta, type, typedVal);
        }

        /// <summary>
        /// Constructs a column out of a getter.
        /// </summary>
        /// <typeparam name="T">The type of the value</typeparam>
        /// <param name="name">The column name, which must be non-empty</param>
        /// <param name="type">The type of the column, whose raw type must be <typeparamref name="T"/></param>
        /// <param name="getter">The getter for the column</param>
        /// <param name="meta">Optionally, metadata for the column</param>
        /// <returns>A column with this getter</returns>
        public static IColumn GetColumn<T>(string name, ColumnType type, ValueGetter<T> getter, IRow meta = null)
        {
            Contracts.CheckNonEmpty(name, nameof(name));
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckParam(type.RawType == typeof(T), nameof(type), "Mismatch on object type and column type");
            Contracts.CheckValue(getter, nameof(getter));
            Contracts.CheckValueOrNull(meta);

            return new GetterImpl<T>(name, meta, type, getter);
        }

        /// <summary>
        /// Wraps a set of row columns as a row.
        /// </summary>
        /// <param name="counted">The counted object that the output row will wrap for its own implementation of
        /// <see cref="ICounted"/>, or if null, the output row will yield default values for those implementations,
        /// that is, a totally static row</param>
        /// <param name="columns">A set of row columns</param>
        /// <returns>A row with items derived from <paramref name="columns"/></returns>
        public static IRow GetRow(ICounted counted, params IColumn[] columns)
        {
            Contracts.CheckValueOrNull(counted);
            Contracts.CheckValue(columns, nameof(columns));
            return new RowColumnRow(counted, columns);
        }

        /// <summary>
        /// Given a column, returns a deep-copied memory-materialized version of it. Note that
        /// it is acceptable for the column to be inactive: the returned column will likewise
        /// be inactive.
        /// </summary>
        /// <param name="column"></param>
        /// <returns>A memory materialized version of <paramref name="column"/> which may be,
        /// under appropriate circumstances, the input object itself</returns>
        public static IColumn CloneColumn(IColumn column)
        {
            Contracts.CheckValue(column, nameof(column));
            return Utils.MarshalInvoke(CloneColumnCore<int>, column.Type.RawType, column);
        }

        private static IColumn CloneColumnCore<T>(IColumn column)
        {
            Contracts.Assert(column is IValueColumn<T>);
            IRow meta = column.Metadata;
            if (meta != null)
                meta = RowCursorUtils.CloneRow(meta);

            var tcolumn = (IValueColumn<T>)column;
            if (!tcolumn.IsActive)
                return new InactiveImpl<T>(tcolumn.Name, meta, tcolumn.Type);
            T val = default(T);
            tcolumn.GetGetter()(ref val);
            return GetColumn(tcolumn.Name, tcolumn.Type, ref val, meta);
        }

        /// <summary>
        /// The implementation for a simple wrapping of an <see cref="IRow"/>.
        /// </summary>
        private sealed class RowWrap<T> : IValueColumn<T>
        {
            private readonly IRow _row;
            private readonly int _col;
            private MetadataRow _meta;

            public string Name => _row.Schema.GetColumnName(_col);
            public ColumnType Type => _row.Schema.GetColumnType(_col);
            public bool IsActive => _row.IsColumnActive(_col);

            public IRow Metadata
            {
                get
                {
                    if (_meta == null)
                        Interlocked.CompareExchange(ref _meta, new MetadataRow(_row.Schema, _col, x => true), null);
                    return _meta;
                }
            }

            public RowWrap(IRow row, int col)
            {
                Contracts.AssertValue(row);
                Contracts.Assert(0 <= col && col < row.Schema.ColumnCount);
                Contracts.Assert(row.Schema.GetColumnType(col).RawType == typeof(T));

                _row = row;
                _col = col;
            }

            Delegate IColumn.GetGetter()
                => GetGetter();

            public ValueGetter<T> GetGetter()
                => _row.GetGetter<T>(_col);
        }

        /// <summary>
        /// The base class for a few <see cref="ICounted"/> implementations that do not "go" anywhere.
        /// </summary>
        public abstract class DefaultCounted : ICounted
        {
            public long Position => 0;
            public long Batch => 0;
            public ValueGetter<UInt128> GetIdGetter()
                => IdGetter;

            private static void IdGetter(ref UInt128 id)
                => id = default;
        }

        /// <summary>
        /// Simple wrapper for a schema column, considered inctive with no getter.
        /// </summary>
        /// <typeparam name="T">The type of the getter</typeparam>
        private sealed class SchemaWrap<T> : IValueColumn<T>
        {
            private readonly ISchema _schema;
            private readonly int _col;
            private MetadataRow _meta;

            public string Name => _schema.GetColumnName(_col);
            public ColumnType Type => _schema.GetColumnType(_col);
            public bool IsActive => false;

            public IRow Metadata
            {
                get
                {
                    if (_meta == null)
                        Interlocked.CompareExchange(ref _meta, new MetadataRow(_schema, _col, x => true), null);
                    return _meta;
                }
            }

            public SchemaWrap(ISchema schema, int col)
            {
                Contracts.AssertValue(schema);
                Contracts.Assert(0 <= col && col < schema.ColumnCount);
                Contracts.Assert(schema.GetColumnType(col).RawType == typeof(T));

                _schema = schema;
                _col = col;
            }

            Delegate IColumn.GetGetter()
                => GetGetter();

            public ValueGetter<T> GetGetter()
                => throw Contracts.Except("Column not active");
        }

        /// <summary>
        /// This class exists to present metadata as stored in an <see cref="ISchema"/> for one particular
        /// column as an <see cref="IRow"/>. This class will cease to be necessary at the point when all
        /// metadata implementations are just simple <see cref="IRow"/>s.
        /// </summary>
        public sealed class MetadataRow : DefaultCounted, IRow
        {
            public ISchema Schema => _schema;

            private readonly ISchema _metaSchema;
            private readonly int _col;
            private readonly SchemaImpl _schema;

            private readonly KeyValuePair<string, ColumnType>[] _map;

            private sealed class SchemaImpl : ISchema
            {
                private readonly MetadataRow _parent;
                private readonly Dictionary<string, int> _nameToCol;

                public int ColumnCount { get { return _parent._map.Length; } }

                public SchemaImpl(MetadataRow parent)
                {
                    Contracts.AssertValue(parent);
                    _parent = parent;
                    _nameToCol = new Dictionary<string, int>(ColumnCount);
                    for (int i = 0; i < _parent._map.Length; ++i)
                        _nameToCol[_parent._map[i].Key] = i;
                }

                public string GetColumnName(int col)
                {
                    Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    return _parent._map[col].Key;
                }

                public ColumnType GetColumnType(int col)
                {
                    Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    return _parent._map[col].Value;
                }

                public bool TryGetColumnIndex(string name, out int col)
                {
                    return _nameToCol.TryGetValue(name, out col);
                }

                public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
                {
                    return Enumerable.Empty<KeyValuePair<string, ColumnType>>();
                }

                public void GetMetadata<TValue>(string kind, int col, ref TValue value)
                {
                    throw MetadataUtils.ExceptGetMetadata();
                }

                public ColumnType GetMetadataTypeOrNull(string kind, int col)
                {
                    Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    return null;
                }
            }

            public MetadataRow(ISchema schema, int col, Func<string, bool> takeMetadata)
            {
                Contracts.CheckValue(schema, nameof(schema));
                Contracts.CheckParam(0 <= col && col < schema.ColumnCount, nameof(col));
                Contracts.CheckValue(takeMetadata, nameof(takeMetadata));

                _metaSchema = schema;
                _col = col;
                _map = _metaSchema.GetMetadataTypes(_col).Where(x => takeMetadata(x.Key)).ToArray();
                _schema = new SchemaImpl(this);
            }

            public bool IsColumnActive(int col)
            {
                Contracts.CheckParam(0 <= col && col < _map.Length, nameof(col));
                return true;
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Contracts.CheckParam(0 <= col && col < _map.Length, nameof(col));
                // REVIEW: On type mismatch, this will throw a metadata exception, which is not really
                // appropriate. However, since this meant to be a shim anyway, we will tolerate imperfection.
                return (ref TValue dst) => _metaSchema.GetMetadata(_map[col].Key, _col, ref dst);
            }
        }

        /// <summary>
        /// This is used for a few <see cref="IColumn"/> implementations that need to store their own name,
        /// metadata, and type themselves.
        /// </summary>
        private abstract class SimpleColumnBase<T> : IValueColumn<T>
        {
            public string Name { get; }
            public IRow Metadata { get; }
            public ColumnType Type { get; }
            public abstract bool IsActive { get; }

            public SimpleColumnBase(string name, IRow meta, ColumnType type)
            {
                Contracts.CheckNonEmpty(name, nameof(name));
                Contracts.CheckValueOrNull(meta);
                Contracts.CheckValue(type, nameof(type));
                Contracts.CheckParam(type.RawType == typeof(T), nameof(type), "Mismatch between CLR type and column type");

                Name = name;
                Metadata = meta;
                Type = type;
            }

            Delegate IColumn.GetGetter()
            {
                return GetGetter();
            }

            public abstract ValueGetter<T> GetGetter();
        }

        private sealed class InactiveImpl<T> : SimpleColumnBase<T>
        {
            public override bool IsActive { get { return false; } }

            public InactiveImpl(string name, IRow meta, ColumnType type)
                : base(name, meta, type)
            {
            }

            public override ValueGetter<T> GetGetter()
            {
                throw Contracts.Except("Can't get getter for inactive column");
            }
        }

        private sealed class ConstOneImpl<T> : SimpleColumnBase<T>
        {
            private readonly T _value;

            public override bool IsActive => true;

            public ConstOneImpl(string name, IRow meta, ColumnType type, T value)
                : base(name, meta, type)
            {
                Contracts.Assert(type.IsPrimitive);
                _value = value;
            }

            public override ValueGetter<T> GetGetter()
            {
                return Getter;
            }

            private void Getter(ref T val)
            {
                val = _value;
            }
        }

        private sealed class ConstVecImpl<T> : SimpleColumnBase<VBuffer<T>>
        {
            private readonly VBuffer<T> _value;

            public override bool IsActive { get { return true; } }

            public ConstVecImpl(string name, IRow meta, ColumnType type, VBuffer<T> value)
                : base(name, meta, type)
            {
                _value = value;
            }

            public override ValueGetter<VBuffer<T>> GetGetter()
            {
                return Getter;
            }

            private void Getter(ref VBuffer<T> val)
            {
                _value.CopyTo(ref val);
            }
        }

        private sealed class GetterImpl<T> : SimpleColumnBase<T>
        {
            private readonly ValueGetter<T> _getter;

            public override bool IsActive => _getter != null;

            public GetterImpl(string name, IRow meta, ColumnType type, ValueGetter<T> getter)
                : base(name, meta, type)
            {
                Contracts.CheckValueOrNull(getter);
                _getter = getter;
            }

            public override ValueGetter<T> GetGetter()
            {
                Contracts.Check(IsActive, "column is not active");
                return _getter;
            }
        }

        /// <summary>
        /// An <see cref="IRow"/> that is an amalgation of multiple <see cref="IColumn"/> implementers.
        /// </summary>
        private sealed class RowColumnRow : IRow
        {
            private static readonly DefaultCountedImpl _defCount = new DefaultCountedImpl();
            private readonly ICounted _counted;
            private readonly IColumn[] _columns;
            private readonly SchemaImpl _schema;

            public ISchema Schema => _schema;
            public long Position => _counted.Position;
            public long Batch => _counted.Batch;

            public RowColumnRow(ICounted counted, IColumn[] columns)
            {
                Contracts.AssertValueOrNull(counted);
                Contracts.AssertValue(columns);
                _counted = counted ?? _defCount;
                _columns = columns;
                _schema = new SchemaImpl(this);
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Contracts.CheckParam(IsColumnActive(col), nameof(col), "requested column not active");
                var rowCol = _columns[col] as IValueColumn<TValue>;
                if (rowCol == null)
                    throw Contracts.Except("Invalid TValue: '{0}'", typeof(TValue));
                return rowCol.GetGetter();
            }

            public bool IsColumnActive(int col)
            {
                Contracts.CheckParam(0 <= col && col < _columns.Length, nameof(col));
                return _columns[col].IsActive;
            }

            public ValueGetter<UInt128> GetIdGetter()
            {
                return _counted.GetIdGetter();
            }

            private sealed class SchemaImpl : ISchema
            {
                private readonly RowColumnRow _parent;
                private readonly Dictionary<string, int> _nameToIndex;

                public int ColumnCount => _parent._columns.Length;

                public SchemaImpl(RowColumnRow parent)
                {
                    Contracts.AssertValue(parent);
                    _parent = parent;
                    _nameToIndex = new Dictionary<string, int>();
                    for (int i = 0; i < _parent._columns.Length; ++i)
                        _nameToIndex[_parent._columns[i].Name] = i;
                }

                public void GetMetadata<TValue>(string kind, int col, ref TValue value)
                {
                    Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    var meta = _parent._columns[col].Metadata;
                    int mcol;
                    if (meta == null || !meta.Schema.TryGetColumnIndex(kind, out mcol))
                        throw MetadataUtils.ExceptGetMetadata();
                    // REVIEW: Again, since this is a shim, not going to sweat the potential for inappropriate exception message.
                    meta.GetGetter<TValue>(mcol)(ref value);
                }

                public ColumnType GetMetadataTypeOrNull(string kind, int col)
                {
                    Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    var meta = _parent._columns[col].Metadata;
                    int mcol;
                    if (meta == null || !meta.Schema.TryGetColumnIndex(kind, out mcol))
                        return null;
                    return meta.Schema.GetColumnType(mcol);
                }

                public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
                {
                    Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    // REVIEW: An IRow can have collisions in names, whereas there is no notion of this in metadata types.
                    // Since I intend to remove this soon anyway and the number of usages of this will be very low, I am just going
                    // to tolerate the potential for strangeness here, since it will practically never arise until we reorganize
                    // the whole thing.
                    var meta = _parent._columns[col].Metadata;
                    if (meta == null)
                        yield break;
                    var schema = meta.Schema;
                    for (int i = 0; i < schema.ColumnCount; ++i)
                        yield return new KeyValuePair<string, ColumnType>(schema.GetColumnName(i), schema.GetColumnType(i));
                }

                public ColumnType GetColumnType(int col)
                {
                    Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    return _parent._columns[col].Type;
                }

                public string GetColumnName(int col)
                {
                    Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                    return _parent._columns[col].Name;
                }

                public bool TryGetColumnIndex(string name, out int col)
                {
                    return _nameToIndex.TryGetValue(name, out col);
                }
            }

            private sealed class DefaultCountedImpl : DefaultCounted
            {
            }
        }
    }
}
