// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This class represents the schema of an <see cref="ISchematized"/> object (like an <see cref="IDataView"/> or an <see cref="IRow"/>).
    /// On the high level, the schema is a collection of 'columns'. Each column has the following properties:
    /// - Column name.
    /// - Column type.
    /// - Metadata. The metadata itself is a 'single-row dataset' (namely, an instance of <see cref="Metadata"/>), that contains its own schema
    /// and values.
    /// </summary>
    [System.Diagnostics.DebuggerTypeProxy(typeof(SchemaDebuggerProxy))]
    public sealed class Schema : ISchema
    {
        private readonly Column[] _columns;
        private readonly Dictionary<string, int> _nameMap;

        /// <summary>
        /// Number of columns in the schema.
        /// </summary>
        public int ColumnCount => _columns.Length;

        /// <summary>
        /// Get the column by name. Throws an exception if such column does not exist.
        /// Note that if multiple columns exist with the same name, the one with the biggest index is returned.
        /// The other columns are considered 'hidden', and only accessible by their index.
        /// </summary>
        public Column this[string name]
        {
            get
            {
                Contracts.CheckValue(name, nameof(name));
                if (!_nameMap.TryGetValue(name, out int col))
                    throw Contracts.ExceptParam(nameof(name), $"Column '{name}' not found");
                return _columns[col];
            }
        }

        /// <summary>
        /// Get the column by index.
        /// </summary>
        public Column this[int col]
        {
            get
            {
                Contracts.CheckParam(0 <= col && col < _columns.Length, nameof(col));
                return _columns[col];
            }
        }

        /// <summary>
        /// Retrieves an index of a column by name.
        /// </summary>
        /// <returns>True iff the column is present in the schema.</returns>
        public bool TryGetColumnIndex(string name, out int col) => _nameMap.TryGetValue(name, out col);

        /// <summary>
        /// Get the column by name, or <c>null</c> if the column is not present.
        /// </summary>
        public Column GetColumnOrNull(string name)
        {
            Contracts.CheckNonEmpty(name, nameof(name));
            if (_nameMap.TryGetValue(name, out int col))
                return _columns[col];
            return null;
        }

        public override string ToString()
        {
            return $"{ColumnCount} columns";
        }

        /// <summary>
        /// This class describes one column in the schema.
        /// </summary>
        public sealed class Column
        {
            /// <summary>
            /// The name of the column.
            /// </summary>
            public string Name { get; }

            /// <summary>
            /// The type of the column.
            /// </summary>
            public ColumnType Type { get; }

            /// <summary>
            /// The metadata of the column.
            /// </summary>
            public Metadata Metadata { get; }

            public Column(string name, ColumnType type, Metadata metadata)
            {
                Contracts.CheckNonEmpty(name, nameof(name));
                Contracts.CheckValue(type, nameof(type));
                Contracts.CheckValueOrNull(metadata);

                Name = name;
                Type = type;
                Metadata = metadata;
            }

            public override string ToString()
            {
                var metadataString = (Metadata == null || Metadata.Schema.ColumnCount == 0) ?
                    null : $" {{{string.Join(", ", Metadata.Schema.GetColumns().Select(x => x.column.Name))}}}";
                return $"{Name}: {Type}{metadataString}";
            }
        }

        /// <summary>
        /// The metadata of one <see cref="Column"/>.
        /// </summary>
        [System.Diagnostics.DebuggerTypeProxy(typeof(MetadataDebuggerProxy))]
        public sealed class Metadata
        {
            private readonly (Column column, Delegate getter)[] _values;

            /// <summary>
            /// The schema of the metadata row. It is different from the schema that the column belongs to.
            /// </summary>
            public Schema Schema { get; }

            /// <summary>
            /// Create a metadata row by supplying the schema columns and the getter delegates for all the values.
            /// </summary>
            public Metadata(IEnumerable<(Column column, Delegate getter)> values)
            {
                Contracts.CheckValue(values, nameof(values));
                // Check all getters.
                foreach (var (column, getter) in values)
                {
                    Contracts.CheckValue(column, nameof(column));
                    Contracts.CheckValue(getter, nameof(getter));
                    Utils.MarshalActionInvoke(CheckGetter<int>, column.Type.RawType, getter);
                }
                _values = values.ToArray();
                Schema = new Schema(_values.Select(x => x.column));
            }

            private void CheckGetter<TValue>(Delegate getter)
            {
                var typedGetter = getter as ValueGetter<TValue>;
                if (typedGetter == null)
                    throw Contracts.ExceptParam(nameof(getter), $"Getter of type '{typeof(TValue)}' expected, but {getter.GetType()} found");
            }

            /// <summary>
            /// Get a getter delegate for one value of the metadata row.
            /// </summary>
            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Contracts.CheckParam(0 <= col && col < _values.Length, nameof(col));
                var typedGetter = _values[col].getter as ValueGetter<TValue>;
                if (typedGetter == null)
                {
                    Contracts.Assert(_values[col].getter != null);
                    throw MetadataUtils.ExceptGetMetadata();
                }
                return typedGetter;
            }

            /// <summary>
            /// Get the value of the metadata, by metadata kind (aka column name).
            /// </summary>
            public void GetValue<TValue>(string kind, ref TValue value)
            {
                if (!Schema.TryGetColumnIndex(kind, out int col))
                    throw MetadataUtils.ExceptGetMetadata();
                GetGetter<TValue>(col)(ref value);
            }

            public override string ToString() => string.Join(", ", Schema.GetColumns().Select(x => x.column.Name));

            /// <summary>
            /// The class that incrementally builds a <see cref="Metadata"/>.
            /// </summary>
            public sealed class Builder
            {
                private readonly List<(Column column, Delegate getter)> _items;

                public Builder()
                {
                    _items = new List<(Column column, Delegate getter)>();
                }

                /// <summary>
                /// Add some columns from <paramref name="metadata"/> into our new metadata, by applying <paramref name="selector"/>
                /// to all the names.
                /// </summary>
                /// <param name="metadata">The metadata row to take values from.</param>
                /// <param name="selector">The predicate describing which metadata columns to keep.</param>
                public void Add(Metadata metadata, Func<string, bool> selector)
                {
                    Contracts.CheckValueOrNull(metadata);
                    Contracts.CheckValue(selector, nameof(selector));

                    if (metadata == null)
                        return;

                    foreach (var pair in metadata._values)
                    {
                        if (selector(pair.column.Name))
                            _items.Add(pair);
                    }
                }

                /// <summary>
                /// Add one metadata column, strongly-typed version.
                /// </summary>
                /// <typeparam name="TValue">The type of the value.</typeparam>
                /// <param name="column">The column information for the schema.</param>
                /// <param name="getter">The getter delegate that provides the value.</param>
                public void Add<TValue>(Column column, ValueGetter<TValue> getter)
                {
                    Contracts.CheckValue(column, nameof(column));
                    Contracts.CheckValue(getter, nameof(getter));
                    Contracts.CheckParam(column.Type.RawType == typeof(TValue), nameof(getter));
                    _items.Add((column, getter));
                }

                /// <summary>
                /// Add one metadata column, weakly-typed version.
                /// </summary>
                /// <param name="column">The column information for the schema.</param>
                /// <param name="getter">The getter delegate that provides the value. Note that the type of the getter is still checked
                /// inside this method.</param>
                public void Add(Column column, Delegate getter)
                {
                    Contracts.CheckValue(column, nameof(column));
                    Utils.MarshalActionInvoke(AddDelegate<int>, column.Type.RawType, column, getter);
                }

                /// <summary>
                /// Add slot names metadata.
                /// </summary>
                /// <param name="size">The size of the slot names vector.</param>
                /// <param name="getter">The getter delegate for the slot names.</param>
                public void AddSlotNames(int size, ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter)
                    => Add(new Column(MetadataUtils.Kinds.SlotNames, new VectorType(TextType.Instance, size), null), getter);

                /// <summary>
                /// Add key values metadata.
                /// </summary>
                /// <typeparam name="TValue">The value type of key values.</typeparam>
                /// <param name="size">The size of key values vector.</param>
                /// <param name="valueType">The value type of key values. Its raw type must match <typeparamref name="TValue"/>.</param>
                /// <param name="getter">The getter delegate for the key values.</param>
                public void AddKeyValues<TValue>(int size, PrimitiveType valueType, ValueGetter<VBuffer<TValue>> getter)
                    => Add(new Column(MetadataUtils.Kinds.KeyValues, new VectorType(valueType, size), null), getter);

                /// <summary>
                /// Produce the metadata row that the builder has so far.
                /// Can be called multiple times.
                /// </summary>
                public Metadata GetMetadata() => new Metadata(_items);

                private void AddDelegate<TValue>(Schema.Column column, Delegate getter)
                {
                    Contracts.CheckValue(column, nameof(column));
                    Contracts.CheckValue(getter, nameof(getter));
                    var typedGetter = getter as ValueGetter<TValue>;
                    Contracts.CheckParam(typedGetter != null, nameof(getter));
                    _items.Add((column, typedGetter));
                }
            }
        }

        public Schema(IEnumerable<Column> columns)
        {
            Contracts.CheckValue(columns, nameof(columns));

            _columns = columns.ToArray();
            _nameMap = new Dictionary<string, int>();
            for (int i = 0; i < _columns.Length; i++)
                _nameMap[_columns[i].Name] = i;
        }

        /// <summary>
        /// Get all non-hidden columns as pairs of (index, <see cref="Column"/>).
        /// </summary>
        public IEnumerable<(int index, Column column)> GetColumns() => _nameMap.Values.Select(idx => (idx, _columns[idx]));

        /// <summary>
        /// Manufacture an instance of <see cref="Schema"/> out of any <see cref="ISchema"/>.
        /// </summary>
        public static Schema Create(ISchema inputSchema)
        {
            Contracts.CheckValue(inputSchema, nameof(inputSchema));

            if (inputSchema is Schema s)
                return s;

            var columns = new Column[inputSchema.ColumnCount];
            for (int i = 0; i < columns.Length; i++)
            {
                var meta = new Metadata.Builder();
                foreach (var kvp in inputSchema.GetMetadataTypes(i))
                {
                    var getter = Utils.MarshalInvoke(GetMetadataGetterDelegate<int>, kvp.Value.RawType, inputSchema, i, kvp.Key);
                    meta.Add(new Column(kvp.Key, kvp.Value, null), getter);
                }
                columns[i] = new Column(inputSchema.GetColumnName(i), inputSchema.GetColumnType(i), meta.GetMetadata());
            }

            return new Schema(columns);
        }

        private static Delegate GetMetadataGetterDelegate<TValue>(ISchema schema, int col, string kind)
        {
            // REVIEW: We are facing a choice here: cache 'value' and get rid of 'schema' reference altogether,
            // or retain the reference but be more memory efficient. This code should not stick around for too long
            // anyway, so let's not sweat too much, and opt for the latter.
            ValueGetter<TValue> getter = (ref TValue value) => schema.GetMetadata(kind, col, ref value);
            return getter;
        }

        #region Legacy schema API to be removed
        public string GetColumnName(int col) => this[col].Name;

        public ColumnType GetColumnType(int col) => this[col].Type;

        public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
        {
            var meta = this[col].Metadata;
            if (meta == null)
                return Enumerable.Empty<KeyValuePair<string, ColumnType>>();
            return meta.Schema.GetColumns().Select(c => new KeyValuePair<string, ColumnType>(c.column.Name, c.column.Type));
        }

        public ColumnType GetMetadataTypeOrNull(string kind, int col)
        {
            var meta = this[col].Metadata;
            if (meta == null)
                return null;
            if (meta.Schema.TryGetColumnIndex(kind, out int metaCol))
                return meta.Schema[metaCol].Type;
            return null;
        }

        public void GetMetadata<TValue>(string kind, int col, ref TValue value)
        {
            var meta = this[col].Metadata;
            if (meta == null)
                throw MetadataUtils.ExceptGetMetadata();
            meta.GetValue(kind, ref value);
        }
        #endregion
    }

}
