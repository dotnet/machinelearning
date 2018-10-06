// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.Runtime.Data
{
#pragma warning disable CS0618 // Type or member is obsolete
    public sealed class Schema : ISchema
#pragma warning restore CS0618 // Type or member is obsolete
    {
        private readonly Column[] _columns;
        private readonly Dictionary<string, int> _nameMap;

        public int ColumnCount => _columns.Length;

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

        public Column this[int col]
        {
            get
            {
                Contracts.CheckParam(0 <= col && col < _columns.Length, nameof(col));
                return _columns[col];
            }
        }

        public bool TryGetColumnIndex(string name, out int col) => _nameMap.TryGetValue(name, out col);

        public Column GetColumnOrNull(string name)
        {
            Contracts.CheckNonEmpty(name, nameof(name));
            if (_nameMap.TryGetValue(name, out int col))
                return _columns[col];
            return null;
        }

        public sealed class Column
        {
            public string Name { get; }

            public ColumnType Type { get; }

            public MetadataRow Metadata { get; }

            public Column(string name, ColumnType type, MetadataRow metadata)
            {
                Contracts.CheckNonEmpty(name, nameof(name));
                Contracts.CheckValue(type, nameof(type));
                Contracts.CheckValueOrNull(metadata);

                Name = name;
                Type = type;
                Metadata = metadata;
            }
        }

        public sealed class MetadataRow
        {
            private readonly (Column column, Delegate getter)[] _values;
            public Schema Schema { get; }

            public MetadataRow(IEnumerable<(Column column, Delegate getter)> values)
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

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Contracts.CheckParam(0 <= col && col < _values.Length, nameof(col));
                var typedGetter = _values[col].getter as ValueGetter<TValue>;
                return typedGetter;
            }

            /// <summary>
            /// The class that incrementally builds a <see cref="MetadataRow"/>.
            /// </summary>
            public sealed class Builder
            {
                private readonly List<(Column column, Delegate getter)> _items;

                public Builder()
                {
                    _items = new List<(Column column, Delegate getter)>();
                }

                public void Add(MetadataRow metadata, Func<string, bool> selector)
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

                public void Add<TValue>(Column column, ValueGetter<TValue> getter)
                {
                    Contracts.CheckValue(column, nameof(column));
                    Contracts.CheckValue(getter, nameof(getter));
                    Contracts.CheckParam(column.Type.RawType == typeof(TValue), nameof(getter));
                    _items.Add((column, getter));
                }

                public void Add(Column column, Delegate getter)
                {
                    Contracts.CheckValue(column, nameof(column));
                    Utils.MarshalActionInvoke(AddDelegate<int>, column.Type.RawType, column, getter);
                }

                public void AddSlotNames(int size, ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter)
                    => Add(new Column(MetadataUtils.Kinds.SlotNames, new VectorType(TextType.Instance, size), null), getter);

                public void AddKeyValues<TValue>(int size, PrimitiveType valueType, ValueGetter<VBuffer<TValue>> getter)
                    => Add(new Column(MetadataUtils.Kinds.KeyValues, new VectorType(valueType, size), null), getter);

                public MetadataRow GetMetadataRow() => new MetadataRow(_items);

                private void AddDelegate<TValue>(Schema.Column column, Delegate getter)
                {
                    Contracts.CheckValue(column, nameof(column));
                    Contracts.CheckValue(getter, nameof(getter));
                    var typedGetter = getter as ValueGetter<TValue>;
                    Contracts.CheckParam(typedGetter != null, nameof(getter));
                    _items.Add((column, typedGetter));
                }
            }

            public void GetValue<TValue>(int col, ref TValue value) => GetGetter<TValue>(col)(ref value);

            public void GetValue<TValue>(string kind, ref TValue value)
            {
                if (!Schema.TryGetColumnIndex(kind, out int col))
                    throw MetadataUtils.ExceptGetMetadata();
                GetValue(col, ref value);
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

        public IEnumerable<(int index, Column column)> GetColumns() => _nameMap.Values.Select(idx => (idx, _columns[idx]));

#pragma warning disable CS0618 // Type or member is obsolete
                              /// <summary>
                              /// Manufacture an instance of <see cref="Schema"/> out of any <see cref="ISchema"/>.
                              /// </summary>
        public static Schema Create(ISchema inputSchema)
#pragma warning restore CS0618 // Type or member is obsolete
        {
            Contracts.CheckValue(inputSchema, nameof(inputSchema));

            if (inputSchema is Schema s)
                return s;

            var columns = new Column[inputSchema.ColumnCount];
            for (int i = 0; i < columns.Length; i++)
            {
                var meta = new MetadataRow.Builder();
                foreach (var kvp in inputSchema.GetMetadataTypes(i))
                {
                    var getter = Utils.MarshalInvoke(GetMetadataGetterDelegate<int>, kvp.Value.RawType, inputSchema, i, kvp.Key);
                    meta.Add(new Column(kvp.Key, kvp.Value, null), getter);
                }
                columns[i] = new Column(inputSchema.GetColumnName(i), inputSchema.GetColumnType(i), meta.GetMetadataRow());
            }

            return new Schema(columns);
        }

#pragma warning disable CS0618 // Type or member is obsolete
        private static Delegate GetMetadataGetterDelegate<TValue>(ISchema schema, int col, string kind)
#pragma warning restore CS0618 // Type or member is obsolete
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
            if (!meta.Schema.TryGetColumnIndex(kind, out int metaCol))
                throw MetadataUtils.ExceptGetMetadata();
            meta.GetValue(metaCol, ref value);
        }
        #endregion
    }

}
