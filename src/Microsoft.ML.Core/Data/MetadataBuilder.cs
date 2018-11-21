// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// The class that incrementally builds a <see cref="Schema.Metadata"/>.
    /// </summary>
    public sealed class MetadataBuilder
    {
        private readonly List<(string Name, ColumnType Type, Delegate Getter)> _items;

        public MetadataBuilder()
        {
            _items = new List<(string Name, ColumnType Type, Delegate Getter)>();
        }

        /// <summary>
        /// Add some columns from <paramref name="metadata"/> into our new metadata, by applying <paramref name="selector"/>
        /// to all the names.
        /// </summary>
        /// <param name="metadata">The metadata row to take values from.</param>
        /// <param name="selector">The predicate describing which metadata columns to keep.</param>
        public void Add(Schema.Metadata metadata, Func<string, bool> selector)
        {
            Contracts.CheckValueOrNull(metadata);
            Contracts.CheckValue(selector, nameof(selector));

            if (metadata == null)
                return;

            foreach (var column in metadata.Schema)
            {
                if (selector(column.Name))
                    _items.Add((column.Name, column.Type, metadata.Getters[column.Index]));
            }
        }

        /// <summary>
        /// Add one metadata column, strongly-typed version.
        /// </summary>
        /// <typeparam name="TValue">The type of the value.</typeparam>
        /// <param name="name">The metadata name.</param>
        /// <param name="type">The metadata type.</param>
        /// <param name="getter">The getter delegate.</param>
        public void Add<TValue>(string name, ColumnType type, ValueGetter<TValue> getter)
        {
            Contracts.CheckNonEmpty(name, nameof(name));
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckValue(getter, nameof(getter));
            Contracts.CheckParam(type.RawType == typeof(TValue), nameof(getter));
            _items.Add((name, type, getter));
        }

        /// <summary>
        /// Add one metadata column, weakly-typed version.
        /// </summary>
        /// <param name="name">The metadata name.</param>
        /// <param name="type">The metadata type.</param>
        /// <param name="getter">The getter delegate that provides the value. Note that the type of the getter is still checked
        /// inside this method.</param>
        public void Add(string name, ColumnType type, Delegate getter)
        {
            Contracts.CheckNonEmpty(name, nameof(name));
            Contracts.CheckValue(type, nameof(type));
            Utils.MarshalActionInvoke(AddDelegate<int>, type.RawType, name, type, getter);
        }

        /// <summary>
        /// Add slot names metadata.
        /// </summary>
        /// <param name="size">The size of the slot names vector.</param>
        /// <param name="getter">The getter delegate for the slot names.</param>
        public void AddSlotNames(int size, ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter)
            => Add(MetadataUtils.Kinds.SlotNames, new VectorType(TextType.Instance, size), getter);

        /// <summary>
        /// Add key values metadata.
        /// </summary>
        /// <typeparam name="TValue">The value type of key values.</typeparam>
        /// <param name="size">The size of key values vector.</param>
        /// <param name="valueType">The value type of key values. Its raw type must match <typeparamref name="TValue"/>.</param>
        /// <param name="getter">The getter delegate for the key values.</param>
        public void AddKeyValues<TValue>(int size, PrimitiveType valueType, ValueGetter<VBuffer<TValue>> getter)
            => Add(MetadataUtils.Kinds.KeyValues, new VectorType(valueType, size), getter);

        /// <summary>
        /// Produce the metadata row that the builder has so far.
        /// Can be called multiple times.
        /// </summary>
        public Schema.Metadata GetMetadata()
        {
            var builder = new SchemaBuilder();
            foreach (var item in _items)
                builder.AddColumn(item.Name, item.Type, null);
            return new Schema.Metadata(builder.GetSchema(), _items.Select(x => x.Getter).ToArray());
        }

        private void AddDelegate<TValue>(string name, ColumnType type, Delegate getter)
        {
            Contracts.AssertNonEmpty(name);
            Contracts.AssertValue(type);
            Contracts.AssertValue(getter);

            var typedGetter = getter as ValueGetter<TValue>;
            Contracts.CheckParam(typedGetter != null, nameof(getter));
            _items.Add((name, type, typedGetter));
        }
    }
}
