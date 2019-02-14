﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.Data.DataView
{
    /// <summary>
    /// A builder for <see cref="DataViewSchema"/>.
    /// </summary>
    public sealed class SchemaBuilder
    {
        private readonly List<(string Name, DataViewType Type, DataViewSchema.Metadata Metadata)> _items;

        /// <summary>
        /// Create a new instance of <see cref="SchemaBuilder"/>.
        /// </summary>
        public SchemaBuilder()
        {
            _items = new List<(string Name, DataViewType Type, DataViewSchema.Metadata Metadata)>();
        }

        /// <summary>
        /// Add one column to the schema being built.
        /// </summary>
        /// <param name="name">The column name.</param>
        /// <param name="type">The column type.</param>
        /// <param name="metadata">The column metadata.</param>
        public void AddColumn(string name, DataViewType type, DataViewSchema.Metadata metadata = null)
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
        public void AddColumns(IEnumerable<DataViewSchema.Column> source)
        {
            foreach (var column in source)
                AddColumn(column.Name, column.Type, column.Metadata);
        }

        /// <summary>
        /// Add multiple existing columns to the schema being built.
        /// </summary>
        /// <param name="source">Columns to add.</param>
        public void AddColumns(IEnumerable<DataViewSchema.DetachedColumn> source)
        {
            foreach (var column in source)
                AddColumn(column.Name, column.Type, column.Metadata);
        }

        /// <summary>
        /// Generate the final <see cref="DataViewSchema"/>.
        /// </summary>
        public DataViewSchema GetSchema()
        {
            var nameMap = new Dictionary<string, int>();
            for (int i = 0; i < _items.Count; i++)
                nameMap[_items[i].Name] = i;

            var columns = new DataViewSchema.Column[_items.Count];
            for (int i = 0; i < columns.Length; i++)
                columns[i] = new DataViewSchema.Column(_items[i].Name, i, nameMap[_items[i].Name] != i, _items[i].Type, _items[i].Metadata);

            return new DataViewSchema(columns);
        }
    }
}
