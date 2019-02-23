// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Microsoft.Data.DataView
{
    /// <summary>
    /// This class represents the <see cref="DataViewSchema"/> of an object like, for interstance, an <see cref="IDataView"/> or an <see cref="DataViewRow"/>.
    /// On the high level, the schema is a collection of <see cref="DataViewSchema.Column"/>.
    /// </summary>
    [DebuggerTypeProxy(typeof(SchemaDebuggerProxy))]
    public sealed class DataViewSchema : IReadOnlyList<DataViewSchema.Column>
    {
        private readonly Column[] _columns;
        private readonly Dictionary<string, int> _nameMap;

        /// <summary>
        /// Number of columns in the schema.
        /// </summary>
        public int Count => _columns.Length;

        /// <summary>
        /// Get the column by name. Throws an exception if such column does not exist.
        /// Note that if multiple columns exist with the same name, the one with the biggest index is returned.
        /// The other columns are considered 'hidden', and only accessible by their index.
        /// </summary>
        public Column this[string name]
        {
            get
            {
                if (string.IsNullOrEmpty(name)) throw new ArgumentNullException(nameof(name));
                if (!_nameMap.TryGetValue(name, out int col))
                    throw new ArgumentOutOfRangeException(nameof(name), $"Column '{name}' not found");
                return _columns[col];
            }
        }

        /// <summary>
        /// Get the column by index.
        /// </summary>
        public Column this[int columnIndex]
        {
            get
            {
                if (!(0 <= columnIndex && columnIndex < _columns.Length))
                    throw new ArgumentOutOfRangeException(nameof(columnIndex));
                return _columns[columnIndex];
            }
        }

        /// <summary>
        /// Get the column by name, or <c>null</c> if the column is not present.
        /// </summary>
        public Column? GetColumnOrNull(string name)
        {
            if (string.IsNullOrEmpty(name)) throw new ArgumentNullException(nameof(name));
            if (_nameMap.TryGetValue(name, out int col))
                return _columns[col];
            return null;
        }

        public IEnumerator<Column> GetEnumerator() => ((IEnumerable<Column>)_columns).GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public override string ToString() => $"{Count} columns";

        /// <summary>
        /// This class describes one column in the particular schema.
        /// </summary>
        public struct Column
        {
            /// <summary>
            /// The name of the column.
            /// </summary>
            public string Name { get; }

            /// <summary>
            /// The column's index in the schema.
            /// </summary>
            public int Index { get; }

            /// <summary>
            /// Whether this column is hidden (accessible only by index).
            /// </summary>
            public bool IsHidden { get; }

            /// <summary>
            /// The type of the column.
            /// </summary>
            public DataViewType Type { get; }

            /// <summary>
            /// The metadata of the column.
            /// </summary>
            public Metadata Metadata { get; }

            internal Column(string name, int index, bool isHidden, DataViewType type, Metadata metadata)
            {
                if (string.IsNullOrEmpty(name))
                    throw new ArgumentNullException(nameof(name));
                if (index < 0)
                    throw new ArgumentOutOfRangeException(nameof(index));

                Name = name;
                Index = index;
                IsHidden = isHidden;
                Type = type ?? throw new ArgumentNullException(nameof(type));
                Metadata = metadata ?? Metadata.Empty;
            }

            public override string ToString()
            {
                var metadataString = (Metadata == null || Metadata.Schema.Count == 0) ?
                    null : $" {{{string.Join(", ", Metadata.Schema.Select(x => x.Name))}}}";
                return $"{Name}: {Type}{metadataString}";
            }
        }

        /// <summary>
        /// This class represents the schema of one column of a data view, without an attachment to a particular <see cref="DataViewSchema"/>.
        /// </summary>
        public struct DetachedColumn
        {
            /// <summary>
            /// The name of the column.
            /// </summary>
            public string Name { get; }
            /// <summary>
            /// The type of the column.
            /// </summary>
            public DataViewType Type { get; }
            /// <summary>
            /// The metadata associated with the column.
            /// </summary>
            public Metadata Metadata { get; }

            /// <summary>
            /// Creates an instance of a <see cref="DetachedColumn"/>.
            /// </summary>
            public DetachedColumn(string name, DataViewType type, Metadata metadata = null)
            {
                if (string.IsNullOrEmpty(name))
                    throw new ArgumentNullException(nameof(name));

                Name = name;
                Type = type ?? throw new ArgumentNullException(nameof(type));
                Metadata = metadata ?? DataViewSchema.Metadata.Empty;
            }

            /// <summary>
            /// Create an instance of <see cref="DetachedColumn"/> from an existing schema's column.
            /// </summary>
            public DetachedColumn(Column column)
            {
                Name = column.Name;
                Type = column.Type;
                Metadata = column.Metadata;
            }
        }

        /// <summary>
        /// The metadata of one <see cref="Column"/>.
        /// </summary>
        [DebuggerTypeProxy(typeof(MetadataDebuggerProxy))]
        public sealed class Metadata
        {
            /// <summary>
            /// Metadata getter delegates. Useful to construct metadata out of other metadata.
            /// </summary>
            private readonly Delegate[] _getters;

            /// <summary>
            /// The schema of the metadata row. It is different from the schema that the column belongs to.
            /// </summary>
            public DataViewSchema Schema { get; }

            public static Metadata Empty { get; } = new Metadata(new DataViewSchema(new Column[0]), new Delegate[0]);

            /// <summary>
            /// Create a metadata row by supplying the schema columns and the getter delegates for all the values.
            /// </summary>
            /// <remarks>
            /// Note: The <paramref name="getters"/> array will be owned by this Metadata instance.
            /// </remarks>
            internal Metadata(DataViewSchema schema, Delegate[] getters)
            {
                Debug.Assert(schema != null);
                Debug.Assert(getters != null);

                Debug.Assert(schema.Count == getters.Length);
                // Check all getters.
                for (int i = 0; i < schema.Count; i++)
                {
                    var getter = getters[i];
                    if (getter == null)
                        throw new ArgumentNullException(nameof(getter), $"Delegate at index '{i}' of {nameof(getters)} was null.");
                    Utils.MarshalActionInvoke(CheckGetter<int>, schema[i].Type.RawType, getter);
                }
                Schema = schema;
                _getters = getters;
            }

            private void CheckGetter<TValue>(Delegate getter)
            {
                var typedGetter = getter as ValueGetter<TValue>;
                if (typedGetter == null)
                    throw new ArgumentNullException(nameof(getter), $"Getter of type '{typeof(TValue)}' expected, but {getter.GetType()} found");
            }

            /// <summary>
            /// Get a getter delegate for one value of the metadata row.
            /// </summary>
            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                if (!(0 <= col && col < Schema.Count))
                    throw new ArgumentOutOfRangeException(nameof(col));
                var typedGetter = _getters[col] as ValueGetter<TValue>;
                if (typedGetter == null)
                {
                    Debug.Assert(_getters[col] != null);
                    throw new InvalidOperationException("Invalid call to GetMetadata");
                }
                return typedGetter;
            }

            /// <summary>
            /// Get the value of the metadata, by metadata kind (aka column name).
            /// </summary>
            public void GetValue<TValue>(string kind, ref TValue value)
            {
                var column = Schema.GetColumnOrNull(kind);
                if (column == null)
                    throw new InvalidOperationException("Invalid call to GetMetadata");
                GetGetter<TValue>(column.Value.Index)(ref value);
            }

            public override string ToString() => string.Join(", ", Schema.Select(x => x.Name));

            internal Delegate GetGetterInternal(int index)
            {
                Debug.Assert(0 <= index && index < Schema.Count);
                return _getters[index];
            }

            /// <summary>
            /// The class that incrementally builds a <see cref="Metadata"/>.
            /// </summary>
            public sealed class Builder
            {
                private readonly List<(string Name, DataViewType Type, Delegate Getter, Metadata Metadata)> _items;

                public Builder()
                {
                    _items = new List<(string Name, DataViewType Type, Delegate Getter, Metadata Metadata)>();
                }

                /// <summary>
                /// Add some columns from <paramref name="metadata"/> into our new metadata, by applying <paramref name="selector"/>
                /// to all the names.
                /// </summary>
                /// <param name="metadata">The metadata row to take values from.</param>
                /// <param name="selector">The predicate describing which metadata columns to keep.</param>
                public void Add(Metadata metadata, Func<string, bool> selector)
                {
                    if (metadata == null)
                        return;

                    if (selector == null)
                        throw new ArgumentNullException(nameof(selector));

                    foreach (var column in metadata.Schema)
                    {
                        if (selector(column.Name))
                            _items.Add((column.Name, column.Type, metadata.GetGetterInternal(column.Index), column.Metadata));
                    }
                }

                /// <summary>
                /// Add one metadata column, strongly-typed version.
                /// </summary>
                /// <typeparam name="TValue">The type of the value.</typeparam>
                /// <param name="name">The metadata name.</param>
                /// <param name="type">The metadata type.</param>
                /// <param name="getter">The getter delegate.</param>
                /// <param name="metadata">Metadata of the input column. Note that metadata on a metadata column is somewhat rare
                /// except for certain types (for example, slot names for a vector, key values for something of key type).</param>
                public void Add<TValue>(string name, DataViewType type, ValueGetter<TValue> getter, Metadata metadata = null)
                {
                    if (string.IsNullOrEmpty(name))
                        throw new ArgumentNullException(nameof(name));
                    if (type == null)
                        throw new ArgumentNullException(nameof(type));
                    if (getter == null)
                        throw new ArgumentNullException(nameof(getter));
                    if (type.RawType != typeof(TValue))
                        throw new ArgumentException($"{nameof(type)}.{nameof(type.RawType)} must be of type '{typeof(TValue).FullName}'.", nameof(type));

                    _items.Add((name, type, getter, metadata));
                }

                /// <summary>
                /// Add one metadata column, weakly-typed version.
                /// </summary>
                /// <param name="name">The metadata name.</param>
                /// <param name="type">The metadata type.</param>
                /// <param name="getter">The getter delegate that provides the value. Note that the type of the getter is still checked
                /// inside this method.</param>
                /// <param name="metadata">Metadata of the input column. Note that metadata on a metadata column is somewhat rare
                /// except for certain types (for example, slot names for a vector, key values for something of key type).</param>
                public void Add(string name, DataViewType type, Delegate getter, Metadata metadata = null)
                {
                    if (string.IsNullOrEmpty(name))
                        throw new ArgumentNullException(nameof(name));
                    if (type == null)
                        throw new ArgumentNullException(nameof(type));
                    if (getter == null)
                        throw new ArgumentNullException(nameof(getter));

                    Utils.MarshalActionInvoke(AddDelegate<int>, type.RawType, name, type, getter, metadata);
                }

                /// <summary>
                /// Add one metadata column for a primitive value type.
                /// </summary>
                /// <param name="name">The metadata name.</param>
                /// <param name="type">The metadata type.</param>
                /// <param name="value">The value of the metadata.</param>
                /// <param name="metadata">Metadata of the input column. Note that metadata on a metadata column is somewhat rare
                /// except for certain types (for example, slot names for a vector, key values for something of key type).</param>
                public void AddPrimitiveValue<TValue>(string name, PrimitiveDataViewType type, TValue value, Metadata metadata = null)
                {
                    if (string.IsNullOrEmpty(name))
                        throw new ArgumentNullException(nameof(name));
                    if (type == null)
                        throw new ArgumentNullException(nameof(type));
                    if (type.RawType != typeof(TValue))
                        throw new ArgumentException($"{nameof(type)}.{nameof(type.RawType)} must be of type '{typeof(TValue).FullName}'.", nameof(type));

                    Add(name, type, (ref TValue dst) => dst = value, metadata);
                }

                /// <summary>
                /// Returns a <see cref="Metadata"/> row that contains the current contents of this <see cref="Builder"/>.
                /// </summary>
                public Metadata ToMetadata()
                {
                    var builder = new DataViewSchema.Builder();
                    foreach (var item in _items)
                        builder.AddColumn(item.Name, item.Type, item.Metadata);
                    return new Metadata(builder.ToSchema(), _items.Select(x => x.Getter).ToArray());
                }

                private void AddDelegate<TValue>(string name, DataViewType type, Delegate getter, Metadata metadata)
                {
                    Debug.Assert(!string.IsNullOrEmpty(name));
                    Debug.Assert(type != null);
                    Debug.Assert(getter != null);

                    var typedGetter = getter as ValueGetter<TValue>;
                    if (typedGetter == null)
                        throw new ArgumentException($"{nameof(getter)} must be of type '{typeof(ValueGetter<TValue>).FullName}'", nameof(getter));
                    _items.Add((name, type, typedGetter, metadata));
                }
            }
        }

        /// <summary>
        /// The class that incrementally builds a <see cref="DataViewSchema"/>.
        /// </summary>
        public sealed class Builder
        {
            private readonly List<(string Name, DataViewType Type, Metadata Metadata)> _items;

            /// <summary>
            /// Create a new instance of <see cref="Builder"/>.
            /// </summary>
            public Builder()
            {
                _items = new List<(string Name, DataViewType Type, Metadata Metadata)>();
            }

            /// <summary>
            /// Add one column to the schema being built.
            /// </summary>
            /// <param name="name">The column name.</param>
            /// <param name="type">The column type.</param>
            /// <param name="metadata">The column metadata.</param>
            public void AddColumn(string name, DataViewType type, Metadata metadata = null)
            {
                if (string.IsNullOrEmpty(name))
                    throw new ArgumentNullException(nameof(name));
                if (type == null)
                    throw new ArgumentNullException(nameof(type));

                _items.Add((name, type, metadata));
            }

            /// <summary>
            /// Add multiple existing columns to the schema being built.
            /// </summary>
            /// <param name="source">Columns to add.</param>
            public void AddColumns(IEnumerable<Column> source)
            {
                foreach (var column in source)
                    AddColumn(column.Name, column.Type, column.Metadata);
            }

            /// <summary>
            /// Add multiple existing columns to the schema being built.
            /// </summary>
            /// <param name="source">Columns to add.</param>
            public void AddColumns(IEnumerable<DetachedColumn> source)
            {
                foreach (var column in source)
                    AddColumn(column.Name, column.Type, column.Metadata);
            }

            /// <summary>
            /// Returns a <see cref="DataViewSchema"/> that contains the current contents of this <see cref="Builder"/>.
            /// </summary>
            public DataViewSchema ToSchema()
            {
                var nameMap = new Dictionary<string, int>();
                for (int i = 0; i < _items.Count; i++)
                    nameMap[_items[i].Name] = i;

                var columns = new Column[_items.Count];
                for (int i = 0; i < columns.Length; i++)
                    columns[i] = new Column(_items[i].Name, i, nameMap[_items[i].Name] != i, _items[i].Type, _items[i].Metadata);

                return new DataViewSchema(columns);
            }
        }

        /// <summary>
        /// This constructor should only be called by <see cref="Builder"/>.
        /// </summary>
        /// <param name="columns">The input columns. The constructed instance takes ownership of the array.</param>
        private DataViewSchema(Column[] columns)
        {
            if (columns == null)
                throw new ArgumentNullException(nameof(columns));

            _columns = columns;
            _nameMap = new Dictionary<string, int>();
            for (int i = 0; i < _columns.Length; i++)
            {
                Debug.Assert(_columns[i].Index == i);
                _nameMap[_columns[i].Name] = i;
            }
        }
    }
}
