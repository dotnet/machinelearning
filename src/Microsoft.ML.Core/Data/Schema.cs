// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This class represents the <see cref="Schema"/> of an object like, for interstance, an <see cref="IDataView"/> or an <see cref="Row"/>.
    /// On the high level, the schema is a collection of 'columns'. Each column has the following properties:
    /// - Column name.
    /// - Column type.
    /// - Metadata. The metadata itself is a 'single-row dataset' (namely, an instance of <see cref="Metadata"/>), that contains its own schema
    /// and values.
    /// </summary>
    [System.Diagnostics.DebuggerTypeProxy(typeof(SchemaDebuggerProxy))]
    public sealed class Schema : ISchema, IReadOnlyList<Schema.Column>
    {
        private readonly Column[] _columns;
        private readonly Dictionary<string, int> _nameMap;

        /// <summary>
        /// Number of columns in the schema.
        /// </summary>
        public int ColumnCount => _columns.Length;

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
                Contracts.CheckValue(name, nameof(name));
                if (!_nameMap.TryGetValue(name, out int col))
                    throw Contracts.ExceptParam(nameof(name), $"Column '{name}' not found");
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
                Contracts.CheckParam(0 <= columnIndex && columnIndex < _columns.Length, nameof(columnIndex));
                return _columns[columnIndex];
            }
        }

        /// <summary>
        /// Get the column by name, or <c>null</c> if the column is not present.
        /// </summary>
        public Column? GetColumnOrNull(string name)
        {
            Contracts.CheckNonEmpty(name, nameof(name));
            if (_nameMap.TryGetValue(name, out int col))
                return _columns[col];
            return null;
        }

        public IEnumerator<Column> GetEnumerator() => ((IEnumerable<Column>)_columns).GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public override string ToString()
        {
            return $"{Count} columns";
        }

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
            public ColumnType Type { get; }

            /// <summary>
            /// The metadata of the column.
            /// </summary>
            public Metadata Metadata { get; }

            internal Column(string name, int index, bool isHidden, ColumnType type, Metadata metadata)
            {
                Contracts.AssertNonEmpty(name);
                Contracts.Assert(index >= 0);
                Contracts.AssertValue(type);
                Contracts.AssertValueOrNull(metadata);

                Name = name;
                Index = index;
                IsHidden = isHidden;
                Type = type;
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
        /// This class represents the schema of one column of a data view, without an attachment to a particular <see cref="Schema"/>.
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
            public ColumnType Type { get; }
            /// <summary>
            /// The metadata associated with the column.
            /// </summary>
            public Metadata Metadata { get; }

            /// <summary>
            /// Creates an instance of a <see cref="DetachedColumn"/>.
            /// </summary>
            public DetachedColumn(string name, ColumnType type, Metadata metadata = null)
            {
                Contracts.CheckNonEmpty(name, nameof(name));
                Contracts.CheckValue(type, nameof(type));
                Contracts.CheckValueOrNull(metadata);
                Name = name;
                Type = type;
                Metadata = metadata ?? Schema.Metadata.Empty;
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
        [System.Diagnostics.DebuggerTypeProxy(typeof(MetadataDebuggerProxy))]
        public sealed class Metadata
        {
            /// <summary>
            /// Metadata getter delegates. Useful to construct metadata out of other metadata.
            /// </summary>
            internal ImmutableArray<Delegate> Getters { get; }
            /// <summary>
            /// The schema of the metadata row. It is different from the schema that the column belongs to.
            /// </summary>
            public Schema Schema { get; }

            public static Metadata Empty { get; } = new Metadata(new Schema(new Column[0]), new Delegate[0]);

            /// <summary>
            /// Create a metadata row by supplying the schema columns and the getter delegates for all the values.
            /// </summary>
            internal Metadata(Schema schema, Delegate[] getters)
            {
                Contracts.AssertValue(schema);
                Contracts.AssertValue(getters);

                Contracts.Assert(schema.Count == getters.Length);
                // Check all getters.
                for (int i = 0; i < schema.Count; i++)
                {
                    var getter = getters[i];
                    Contracts.CheckValue(getter, nameof(getter));
                    Utils.MarshalActionInvoke(CheckGetter<int>, schema[i].Type.RawType, getter);
                }
                Schema = schema;
                Getters = getters.ToImmutableArray();
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
                Contracts.CheckParam(0 <= col && col < Schema.Count, nameof(col));
                var typedGetter = Getters[col] as ValueGetter<TValue>;
                if (typedGetter == null)
                {
                    Contracts.Assert(Getters[col] != null);
                    throw MetadataUtils.ExceptGetMetadata();
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
                    throw MetadataUtils.ExceptGetMetadata();
                GetGetter<TValue>(column.Value.Index)(ref value);
            }

            public override string ToString() => string.Join(", ", Schema.GetColumns().Select(x => x.column.Name));

        }

        /// <summary>
        /// This constructor should only be called by <see cref="SchemaBuilder"/>.
        /// </summary>
        /// <param name="columns">The input columns. The constructed instance takes ownership of the array.</param>
        internal Schema(Column[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));

            _columns = columns;
            _nameMap = new Dictionary<string, int>();
            for (int i = 0; i < _columns.Length; i++)
            {
                Contracts.Assert(_columns[i].Index == i);
                _nameMap[_columns[i].Name] = i;
            }
        }

        /// <summary>
        /// Get all non-hidden columns as pairs of (index, <see cref="Column"/>).
        /// </summary>
        public IEnumerable<(int index, Column column)> GetColumns() => _nameMap.Values.Select(idx => (idx, _columns[idx]));

        /// <summary>
        /// Manufacture an instance of <see cref="Schema"/> out of any <see cref="ISchema"/>.
        /// </summary>
        [BestFriend]
        internal static Schema Create(ISchema inputSchema)
        {
            Contracts.CheckValue(inputSchema, nameof(inputSchema));

            if (inputSchema is Schema s)
                return s;

            var builder = new SchemaBuilder();
            for (int i = 0; i < inputSchema.ColumnCount; i++)
            {
                var meta = new MetadataBuilder();
                foreach (var kvp in inputSchema.GetMetadataTypes(i))
                {
                    var getter = Utils.MarshalInvoke(GetMetadataGetterDelegate<int>, kvp.Value.RawType, inputSchema, i, kvp.Key);
                    meta.Add(kvp.Key, kvp.Value, getter);
                }
                builder.AddColumn(inputSchema.GetColumnName(i), inputSchema.GetColumnType(i), meta.GetMetadata());
            }

            return builder.GetSchema();
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

        public bool TryGetColumnIndex(string name, out int col)
        {
            col = GetColumnOrNull(name)?.Index ?? -1;
            return col >= 0;
        }
        #endregion
    }
}
