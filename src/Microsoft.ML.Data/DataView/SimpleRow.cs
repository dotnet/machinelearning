// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// An implementation of <see cref="Row"/> that gets its <see cref="Row.Position"/>, <see cref="Row.Batch"/>,
    /// and <see cref="Row.GetIdGetter"/> from an input row. The constructor requires a schema and array of getter
    /// delegates. A <see langword="null"/> delegate indicates an inactive column. The delegates are assumed to be of the appropriate type
    /// (this does not validate the type).
    /// REVIEW: Should this validate that the delegates are of the appropriate type? It wouldn't be difficult
    /// to do so.
    /// </summary>
    [BestFriend]
    internal sealed class SimpleRow : WrappingRow
    {
        private readonly Delegate[] _getters;

        public override Schema Schema { get; }

        public SimpleRow(Schema schema, Row input, Delegate[] getters)
            : base(input)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckValue(input, nameof(input));
            Contracts.Check(Utils.Size(getters) == schema.ColumnCount);
            Schema = schema;
            _getters = getters ?? new Delegate[0];
        }

        public override ValueGetter<T> GetGetter<T>(int col)
        {
            Contracts.CheckParam(0 <= col && col < _getters.Length, nameof(col), "Invalid col value in GetGetter");
            Contracts.Check(IsColumnActive(col));
            if (_getters[col] is ValueGetter<T> fn)
                return fn;
            throw Contracts.Except("Unexpected TValue in GetGetter");
        }

        public override bool IsColumnActive(int col)
        {
            Contracts.Check(0 <= col && col < _getters.Length);
            return _getters[col] != null;
        }
    }

    /// <summary>
    /// An <see cref="ISchema"/> that takes all column names and types as constructor parameters.
    /// The columns do not have metadata.
    /// </summary>
    public abstract class SimpleSchemaBase : ISchema
    {
        protected readonly IExceptionContext Ectx;
        private readonly string[] _names;
        protected readonly ColumnType[] Types;
        protected readonly Dictionary<string, int> ColumnNameMap;

        public int ColumnCount => Types.Length;

        protected SimpleSchemaBase(IExceptionContext ectx, params KeyValuePair<string, ColumnType>[] columns)
        {
            Contracts.CheckValueOrNull(ectx);
            Ectx = ectx;
            Ectx.CheckValue(columns, nameof(columns));

            _names = new string[columns.Length];
            Types = new ColumnType[columns.Length];
            ColumnNameMap = new Dictionary<string, int>();
            for (int i = 0; i < columns.Length; i++)
            {
                _names[i] = columns[i].Key;
                Types[i] = columns[i].Value;
                if (ColumnNameMap.ContainsKey(columns[i].Key))
                    throw ectx.ExceptParam(nameof(columns), $"Duplicate column name: '{columns[i].Key}'");
                ColumnNameMap[columns[i].Key] = i;
            }
        }

        public bool TryGetColumnIndex(string name, out int col)
        {
            return ColumnNameMap.TryGetValue(name, out col);
        }

        public string GetColumnName(int col)
        {
            Ectx.CheckParam(0 <= col && col < ColumnCount, nameof(col));
            return _names[col];
        }

        public ColumnType GetColumnType(int col)
        {
            Ectx.CheckParam(0 <= col && col < ColumnCount, nameof(col));
            return Types[col];
        }

        public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
        {
            Ectx.Assert(0 <= col && col < ColumnCount);
            return GetMetadataTypesCore(col);
        }

        protected abstract IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypesCore(int col);

        public ColumnType GetMetadataTypeOrNull(string kind, int col)
        {
            Ectx.CheckParam(0 <= col && col < ColumnCount, nameof(col));
            return GetMetadataTypeOrNullCore(kind, col);
        }

        protected abstract ColumnType GetMetadataTypeOrNullCore(string kind, int col);

        public void GetMetadata<TValue>(string kind, int col, ref TValue value)
        {
            Ectx.CheckParam(0 <= col && col < ColumnCount, nameof(col));
            GetMetadataCore(kind, col, ref value);
        }

        protected abstract void GetMetadataCore<TValue>(string kind, int col, ref TValue value);
    }

    public static class SimpleSchemaUtils
    {
        public static Schema Create(IExceptionContext ectx, params KeyValuePair<string, ColumnType>[] columns)
        {
            Contracts.CheckValueOrNull(ectx);
            ectx.CheckValue(columns, nameof(columns));

            var builder = new SchemaBuilder();
            builder.AddColumns(columns.Select(kvp => new Schema.DetachedColumn(kvp.Key, kvp.Value)));
            return builder.GetSchema();
        }
    }
}