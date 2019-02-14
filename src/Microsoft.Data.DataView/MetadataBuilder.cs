// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Microsoft.Data.DataView
{
    /// <summary>
    /// The class that incrementally builds a <see cref="DataViewSchema.Metadata"/>.
    /// </summary>
    public sealed class MetadataBuilder
    {
        private readonly List<(string Name, DataViewType Type, Delegate Getter, DataViewSchema.Metadata Metadata)> _items;

        public MetadataBuilder()
        {
            _items = new List<(string Name, DataViewType Type, Delegate Getter, DataViewSchema.Metadata Metadata)>();
        }

        /// <summary>
        /// Add some columns from <paramref name="metadata"/> into our new metadata, by applying <paramref name="selector"/>
        /// to all the names.
        /// </summary>
        /// <param name="metadata">The metadata row to take values from.</param>
        /// <param name="selector">The predicate describing which metadata columns to keep.</param>
        public void Add(DataViewSchema.Metadata metadata, Func<string, bool> selector)
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
        public void Add<TValue>(string name, DataViewType type, ValueGetter<TValue> getter, DataViewSchema.Metadata metadata = null)
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
        public void Add(string name, DataViewType type, Delegate getter, DataViewSchema.Metadata metadata = null)
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
        public void AddPrimitiveValue<TValue>(string name, PrimitiveDataViewType type, TValue value, DataViewSchema.Metadata metadata = null)
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
        /// Produce the metadata row that the builder has so far.
        /// Can be called multiple times.
        /// </summary>
        public DataViewSchema.Metadata GetMetadata()
        {
            var builder = new SchemaBuilder();
            foreach (var item in _items)
                builder.AddColumn(item.Name, item.Type, item.Metadata);
            return new DataViewSchema.Metadata(builder.GetSchema(), _items.Select(x => x.Getter).ToArray());
        }

        private void AddDelegate<TValue>(string name, DataViewType type, Delegate getter, DataViewSchema.Metadata metadata)
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
