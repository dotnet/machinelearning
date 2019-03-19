// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML
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
            /// The annotations of the column.
            /// </summary>
            public Annotations Annotations { get; }

            internal Column(string name, int index, bool isHidden, DataViewType type, Annotations annotations)
            {
                if (string.IsNullOrEmpty(name))
                    throw new ArgumentNullException(nameof(name));
                if (index < 0)
                    throw new ArgumentOutOfRangeException(nameof(index));

                Name = name;
                Index = index;
                IsHidden = isHidden;
                Type = type ?? throw new ArgumentNullException(nameof(type));
                Annotations = annotations ?? Annotations.Empty;
            }

            public override string ToString()
            {
                var annotationsString = (Annotations == null || Annotations.Schema.Count == 0) ?
                    null : $" {{{string.Join(", ", Annotations.Schema.Select(x => x.Name))}}}";
                return $"{Name}: {Type}{annotationsString}";
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
            /// The annotations associated with the column.
            /// </summary>
            public Annotations Annotations { get; }

            /// <summary>
            /// Creates an instance of a <see cref="DetachedColumn"/>.
            /// </summary>
            public DetachedColumn(string name, DataViewType type, Annotations annotations = null)
            {
                if (string.IsNullOrEmpty(name))
                    throw new ArgumentNullException(nameof(name));

                Name = name;
                Type = type ?? throw new ArgumentNullException(nameof(type));
                Annotations = annotations ?? Annotations.Empty;
            }

            /// <summary>
            /// Create an instance of <see cref="DetachedColumn"/> from an existing schema's column.
            /// </summary>
            public DetachedColumn(Column column)
            {
                Name = column.Name;
                Type = column.Type;
                Annotations = column.Annotations;
            }
        }

        /// <summary>
        /// The annotations of one <see cref="Column"/>.
        /// </summary>
        [DebuggerTypeProxy(typeof(AnnotationsDebuggerProxy))]
        public sealed class Annotations
        {
            /// <summary>
            /// Annotation getter delegates. Useful to construct annotations out of other annotations.
            /// </summary>
            private readonly Delegate[] _getters;

            /// <summary>
            /// The schema of the annotations row. It is different from the schema that the column belongs to.
            /// </summary>
            public DataViewSchema Schema { get; }

            public static Annotations Empty { get; } = new Annotations(new DataViewSchema(new Column[0]), new Delegate[0]);

            /// <summary>
            /// Create an annotations row by supplying the schema columns and the getter delegates for all the values.
            /// </summary>
            /// <remarks>
            /// Note: The <paramref name="getters"/> array will be owned by this <see cref="Annotations"/> instance.
            /// </remarks>
            internal Annotations(DataViewSchema schema, Delegate[] getters)
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
            /// Get a getter delegate for one value of the annotations row.
            /// </summary>
            public ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                if (column.Index >= _getters.Length)
                    throw new ArgumentException(nameof(column));
                var typedGetter = _getters[column.Index] as ValueGetter<TValue>;
                if (typedGetter == null)
                {
                    Debug.Assert(_getters[column.Index] != null);
                    throw new InvalidOperationException($"Invalid call to '{nameof(GetGetter)}'");
                }
                return typedGetter;
            }

            /// <summary>
            /// Get the value of an annotation, by annotation kind (aka column name).
            /// </summary>
            public void GetValue<TValue>(string kind, ref TValue value)
            {
                var column = Schema.GetColumnOrNull(kind);
                if (column == null)
                    throw new InvalidOperationException($"Invalid call to '{nameof(GetValue)}'");
                GetGetter<TValue>(column.Value)(ref value);
            }

            public override string ToString() => string.Join(", ", Schema.Select(x => x.Name));

            internal Delegate GetGetterInternal(int index)
            {
                Debug.Assert(0 <= index && index < Schema.Count);
                return _getters[index];
            }

            /// <summary>
            /// The class that incrementally builds an <see cref="Annotations"/>.
            /// </summary>
            public sealed class Builder
            {
                private readonly List<(string Name, DataViewType Type, Delegate Getter, Annotations Annotations)> _items;

                public Builder()
                {
                    _items = new List<(string Name, DataViewType Type, Delegate Getter, Annotations Annotations)>();
                }

                /// <summary>
                /// Add some columns from <paramref name="annotations"/> into our new annotations, by applying <paramref name="selector"/>
                /// to all the names.
                /// </summary>
                /// <param name="annotations">The annotations row to take values from.</param>
                /// <param name="selector">The predicate describing which annotation columns to keep.</param>
                public void Add(Annotations annotations, Func<string, bool> selector)
                {
                    if (annotations == null)
                        return;

                    if (selector == null)
                        throw new ArgumentNullException(nameof(selector));

                    foreach (var column in annotations.Schema)
                    {
                        if (selector(column.Name))
                            _items.Add((column.Name, column.Type, annotations.GetGetterInternal(column.Index), column.Annotations));
                    }
                }

                /// <summary>
                /// Add one annotation column, strongly-typed version.
                /// </summary>
                /// <typeparam name="TValue">The type of the value.</typeparam>
                /// <param name="name">The annotation name.</param>
                /// <param name="type">The annotation type.</param>
                /// <param name="getter">The getter delegate.</param>
                /// <param name="annotations">Annotations of the input column. Note that annotations on an annotation column is somewhat rare
                /// except for certain types (for example, slot names for a vector, key values for something of key type).</param>
                public void Add<TValue>(string name, DataViewType type, ValueGetter<TValue> getter, Annotations annotations = null)
                {
                    if (string.IsNullOrEmpty(name))
                        throw new ArgumentNullException(nameof(name));
                    if (type == null)
                        throw new ArgumentNullException(nameof(type));
                    if (getter == null)
                        throw new ArgumentNullException(nameof(getter));
                    if (type.RawType != typeof(TValue))
                        throw new ArgumentException($"{nameof(type)}.{nameof(type.RawType)} must be of type '{typeof(TValue).FullName}'.", nameof(type));

                    _items.Add((name, type, getter, annotations));
                }

                /// <summary>
                /// Add one annotation column, weakly-typed version.
                /// </summary>
                /// <param name="name">The annotation name.</param>
                /// <param name="type">The annotation type.</param>
                /// <param name="getter">The getter delegate that provides the value. Note that the type of the getter is still checked
                /// inside this method.</param>
                /// <param name="annotations">Annotations of the input column. Note that annotations on an annotation column is somewhat rare
                /// except for certain types (for example, slot names for a vector, key values for something of key type).</param>
                public void Add(string name, DataViewType type, Delegate getter, Annotations annotations = null)
                {
                    if (string.IsNullOrEmpty(name))
                        throw new ArgumentNullException(nameof(name));
                    if (type == null)
                        throw new ArgumentNullException(nameof(type));
                    if (getter == null)
                        throw new ArgumentNullException(nameof(getter));

                    Utils.MarshalActionInvoke(AddDelegate<int>, type.RawType, name, type, getter, annotations);
                }

                /// <summary>
                /// Add one annotation column for a primitive value type.
                /// </summary>
                /// <param name="name">The annotation name.</param>
                /// <param name="type">The annotation type.</param>
                /// <param name="value">The value of the annotation.</param>
                /// <param name="annotations">Annotations of the input column. Note that annotations on an annotation column is somewhat rare
                /// except for certain types (for example, slot names for a vector, key values for something of key type).</param>
                public void AddPrimitiveValue<TValue>(string name, PrimitiveDataViewType type, TValue value, Annotations annotations = null)
                {
                    if (string.IsNullOrEmpty(name))
                        throw new ArgumentNullException(nameof(name));
                    if (type == null)
                        throw new ArgumentNullException(nameof(type));
                    if (type.RawType != typeof(TValue))
                        throw new ArgumentException($"{nameof(type)}.{nameof(type.RawType)} must be of type '{typeof(TValue).FullName}'.", nameof(type));

                    Add(name, type, (ref TValue dst) => dst = value, annotations);
                }

                /// <summary>
                /// Returns a <see cref="Annotations"/> row that contains the current contents of this <see cref="Builder"/>.
                /// </summary>
                public Annotations ToAnnotations()
                {
                    var builder = new DataViewSchema.Builder();
                    foreach (var item in _items)
                        builder.AddColumn(item.Name, item.Type, item.Annotations);
                    return new Annotations(builder.ToSchema(), _items.Select(x => x.Getter).ToArray());
                }

                private void AddDelegate<TValue>(string name, DataViewType type, Delegate getter, Annotations annotations)
                {
                    Debug.Assert(!string.IsNullOrEmpty(name));
                    Debug.Assert(type != null);
                    Debug.Assert(getter != null);

                    var typedGetter = getter as ValueGetter<TValue>;
                    if (typedGetter == null)
                        throw new ArgumentException($"{nameof(getter)} must be of type '{typeof(ValueGetter<TValue>).FullName}'", nameof(getter));
                    _items.Add((name, type, typedGetter, annotations));
                }
            }
        }

        /// <summary>
        /// The class that incrementally builds a <see cref="DataViewSchema"/>.
        /// </summary>
        public sealed class Builder
        {
            private readonly List<(string Name, DataViewType Type, Annotations Annotations)> _items;

            /// <summary>
            /// Create a new instance of <see cref="Builder"/>.
            /// </summary>
            public Builder()
            {
                _items = new List<(string Name, DataViewType Type, Annotations Annotations)>();
            }

            /// <summary>
            /// Add one column to the schema being built.
            /// </summary>
            /// <param name="name">The column name.</param>
            /// <param name="type">The column type.</param>
            /// <param name="annotations">The column annotations.</param>
            public void AddColumn(string name, DataViewType type, Annotations annotations = null)
            {
                if (string.IsNullOrEmpty(name))
                    throw new ArgumentNullException(nameof(name));
                if (type == null)
                    throw new ArgumentNullException(nameof(type));

                _items.Add((name, type, annotations));
            }

            /// <summary>
            /// Add multiple existing columns to the schema being built.
            /// </summary>
            /// <param name="source">Columns to add.</param>
            public void AddColumns(IEnumerable<Column> source)
            {
                foreach (var column in source)
                    AddColumn(column.Name, column.Type, column.Annotations);
            }

            /// <summary>
            /// Add multiple existing columns to the schema being built.
            /// </summary>
            /// <param name="source">Columns to add.</param>
            public void AddColumns(IEnumerable<DetachedColumn> source)
            {
                foreach (var column in source)
                    AddColumn(column.Name, column.Type, column.Annotations);
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
                    columns[i] = new Column(_items[i].Name, i, nameMap[_items[i].Name] != i, _items[i].Type, _items[i].Annotations);

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
