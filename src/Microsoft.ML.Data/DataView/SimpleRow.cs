// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// An implementation of <see cref="IRow"/> that gets its <see cref="ICounted.Position"/>, <see cref="ICounted.Batch"/>,
    /// and <see cref="ICounted.GetIdGetter"/> from an input row. The constructor requires a schema and array of getter
    /// delegates. A null delegate indicates an inactive column. The delegates are assumed to be of the appropriate type
    /// (this does not validate the type).
    /// REVIEW: Should this validate that the delegates are of the appropriate type? It wouldn't be difficult
    /// to do so.
    /// </summary>
    public sealed class SimpleRow : IRow
    {
        private readonly ISchema _schema;
        private readonly IRow _input;
        private readonly Delegate[] _getters;

        public ISchema Schema { get { return _schema; } }

        public long Position { get { return _input.Position; } }

        public long Batch { get { return _input.Batch; } }

        public SimpleRow(ISchema schema, IRow input, Delegate[] getters)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckValue(input, nameof(input));
            Contracts.Check(Utils.Size(getters) == schema.ColumnCount);
            _schema = schema;
            _input = input;
            _getters = getters ?? new Delegate[0];
        }

        public ValueGetter<UInt128> GetIdGetter()
        {
            return _input.GetIdGetter();
        }

        public ValueGetter<T> GetGetter<T>(int col)
        {
            Contracts.CheckParam(0 <= col && col < _getters.Length, nameof(col), "Invalid col value in GetGetter");
            Contracts.Check(IsColumnActive(col));
            var fn = _getters[col] as ValueGetter<T>;
            if (fn == null)
                throw Contracts.Except("Unexpected TValue in GetGetter");
            return fn;
        }

        public bool IsColumnActive(int col)
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

    /// <summary>
    /// An <see cref="ISchema"/> that takes all column names and types as constructor parameters.
    /// The columns can optionally have text <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.
    /// </summary>
    public sealed class SimpleSchema : SimpleSchemaBase
    {
        private readonly MetadataUtils.MetadataGetter<VBuffer<DvText>>[] _keyValueGetters;

        public SimpleSchema(IExceptionContext ectx, params KeyValuePair<string, ColumnType>[] columns)
            : base(ectx, columns)
        {
            _keyValueGetters = new MetadataUtils.MetadataGetter<VBuffer<DvText>>[ColumnCount];
        }

        public SimpleSchema(IExceptionContext ectx, KeyValuePair<string, ColumnType>[] columns, Dictionary<string, MetadataUtils.MetadataGetter<VBuffer<DvText>>> keyValues)
            : this(ectx, columns)
        {
            foreach (var kvp in keyValues)
            {
                var name = kvp.Key;
                var getter = kvp.Value;
                if (!ColumnNameMap.TryGetValue(name, out int col))
                    throw Ectx.ExceptParam(nameof(keyValues), $"Output schema does not contain column '{name}'");
                if (!Types[col].ItemType.IsKey)
                    throw Ectx.ExceptParam(nameof(keyValues), $"Column '{name}' is not a key column, so it cannot have key value metadata");
                _keyValueGetters[col] = getter;
            }
        }

        protected override IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypesCore(int col)
        {
            Ectx.Assert(0 <= col && col < ColumnCount);
            if (_keyValueGetters[col] != null)
            {
                Ectx.Assert(Types[col].ItemType.IsKey);
                yield return new KeyValuePair<string, ColumnType>(MetadataUtils.Kinds.KeyValues,
                    new VectorType(TextType.Instance, Types[col].ItemType.KeyCount));
            }
        }

        protected override ColumnType GetMetadataTypeOrNullCore(string kind, int col)
        {
            Ectx.Assert(0 <= col && col < ColumnCount);
            if (kind == MetadataUtils.Kinds.KeyValues && _keyValueGetters[col] != null)
            {
                Ectx.Assert(Types[col].ItemType.IsKey);
                return new VectorType(TextType.Instance, Types[col].ItemType.KeyCount);
            }
            return null;
        }

        protected override void GetMetadataCore<TValue>(string kind, int col, ref TValue value)
        {
            Ectx.Assert(0 <= col && col < ColumnCount);
            if (kind == MetadataUtils.Kinds.KeyValues && _keyValueGetters[col] != null)
                _keyValueGetters[col].Marshal(col, ref value);
            else
                throw Ectx.ExceptGetMetadata();
        }
    }
}