// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// A builder for <see cref="Schema"/>.
    /// </summary>
    public sealed class SchemaBuilder
    {
        private readonly List<(string Name, ColumnType Type, Schema.Metadata Metadata)> _items;

        /// <summary>
        /// Create a new instance of <see cref="SchemaBuilder"/>.
        /// </summary>
        public SchemaBuilder()
        {
            _items = new List<(string Name, ColumnType Type, Schema.Metadata Metadata)>();
        }

        /// <summary>
        /// Add one column to the schema being built.
        /// </summary>
        /// <param name="name">The column name.</param>
        /// <param name="type">The column type.</param>
        /// <param name="metadata">The column metadata.</param>
        public void AddColumn(string name, ColumnType type, Schema.Metadata metadata = null)
        {
            Contracts.CheckNonEmpty(name, nameof(name));
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckValueOrNull(metadata);
            _items.Add((name, type, metadata));
        }

        /// <summary>
        /// Add multiple existing columns to the schema being built.
        /// </summary>
        /// <param name="source">Columns to add.</param>
        public void AddColumns(IEnumerable<Schema.Column> source)
        {
            foreach (var column in source)
                AddColumn(column.Name, column.Type, column.Metadata);
        }

        /// <summary>
        /// Add multiple existing columns to the schema being built.
        /// </summary>
        /// <param name="source">Columns to add.</param>
        public void AddColumns(IEnumerable<Schema.DetachedColumn> source)
        {
            foreach (var column in source)
                AddColumn(column.Name, column.Type, column.Metadata);
        }

        /// <summary>
        /// Generate the final <see cref="Schema"/>.
        /// </summary>
        public Schema GetSchema()
        {
            var nameMap = new Dictionary<string, int>();
            for (int i = 0; i < _items.Count; i++)
                nameMap[_items[i].Name] = i;

            var columns = new Schema.Column[_items.Count];
            for (int i = 0; i < columns.Length; i++)
                columns[i] = new Schema.Column(_items[i].Name, i, nameMap[_items[i].Name] != i, _items[i].Type, _items[i].Metadata);

            return new Schema(columns);
        }

        [BestFriend]
        internal static Schema MakeSchema(IEnumerable<Schema.DetachedColumn> columns)
        {
            var builder = new SchemaBuilder();
            builder.AddColumns(columns);
            return builder.GetSchema();
        }
    }
}
