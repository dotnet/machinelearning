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
    [System.Diagnostics.DebuggerTypeProxy(typeof(SchemaDebuggerProxy))]
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
        [System.Diagnostics.DebuggerTypeProxy(typeof(MetadataDebuggerProxy))]
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
        }

        /// <summary>
        /// This constructor should only be called by <see cref="SchemaBuilder"/>.
        /// </summary>
        /// <param name="columns">The input columns. The constructed instance takes ownership of the array.</param>
        internal DataViewSchema(Column[] columns)
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
