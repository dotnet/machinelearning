// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace Microsoft.ML.Transforms.Conversions
{
    /// <include file='doc.xml' path='doc/members/member[@name="ValueToKeyMappingEstimator"]/*' />
    public sealed class ValueToKeyMappingEstimator: IEstimator<ValueToKeyMappingTransformer>
    {
        public static class Defaults
        {
            public const int MaxNumTerms = 1000000;
            public const ValueToKeyMappingTransformer.SortOrder Sort = ValueToKeyMappingTransformer.SortOrder.Occurrence;
        }

        private readonly IHost _host;
        private readonly ValueToKeyMappingTransformer.ColumnInfo[] _columns;
        private readonly string _file;
        private readonly string _termsColumn;
        private readonly IComponentFactory<IMultiStreamSource, IDataLoader> _loaderFactory;

        /// <summary>
        /// Initializes a new instance of <see cref="ValueToKeyMappingEstimator"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="inputColumn">Name of the column to be transformed.</param>
        /// <param name="outputColumn">Name of the output column. If this is null '<paramref name="inputColumn"/>' will be used.</param>
        /// <param name="maxNumTerms">Maximum number of keys to keep per column when auto-training.</param>
        /// <param name="sort">How items should be ordered when vectorized. If <see cref="ValueToKeyMappingTransformer.SortOrder.Occurrence"/> choosen they will be in the order encountered.
        /// If <see cref="ValueToKeyMappingTransformer.SortOrder.Value"/>, items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').</param>
        public ValueToKeyMappingEstimator(IHostEnvironment env, string inputColumn, string outputColumn = null, int maxNumTerms = Defaults.MaxNumTerms, ValueToKeyMappingTransformer.SortOrder sort = Defaults.Sort) :
           this(env, new [] { new ValueToKeyMappingTransformer.ColumnInfo(inputColumn, outputColumn ?? inputColumn, maxNumTerms, sort) })
        {
        }

        public ValueToKeyMappingEstimator(IHostEnvironment env, ValueToKeyMappingTransformer.ColumnInfo[] columns,
            string file = null, string termsColumn = null,
            IComponentFactory<IMultiStreamSource, IDataLoader> loaderFactory = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(ValueToKeyMappingEstimator));
            _columns = columns;
            _file = file;
            _termsColumn = termsColumn;
            _loaderFactory = loaderFactory;
        }

        public ValueToKeyMappingTransformer Fit(IDataView input) => new ValueToKeyMappingTransformer(_host, input, _columns, _file, _termsColumn, _loaderFactory);

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in _columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.Input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);

                if ((col.ItemType.ItemType.RawKind == default) || !(col.ItemType.IsVector || col.ItemType is PrimitiveType))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                SchemaShape metadata;

                // In the event that we are transforming something that is of type key, we will get their type of key value
                // metadata, unless it has none or is not vector in which case we back off to having key values over the item type.
                if (!col.IsKey || !col.Metadata.TryFindColumn(MetadataUtils.Kinds.KeyValues, out var kv) || kv.Kind != SchemaShape.Column.VectorKind.Vector)
                {
                    kv = new SchemaShape.Column(MetadataUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector,
                        colInfo.TextKeyValues ? TextType.Instance : col.ItemType, col.IsKey);
                }
                Contracts.Assert(kv.IsValid);

                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.SlotNames, out var slotMeta))
                    metadata = new SchemaShape(new[] { slotMeta, kv });
                else
                    metadata = new SchemaShape(new[] { kv });
                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, col.Kind, NumberType.U4, true, metadata);
            }

            return new SchemaShape(result.Values);
        }
    }

    public enum KeyValueOrder : byte
    {
        /// <summary>
        /// Terms will be assigned ID in the order in which they appear.
        /// </summary>
        Occurence = ValueToKeyMappingTransformer.SortOrder.Occurrence,

        /// <summary>
        /// Terms will be assigned ID according to their sort via an ordinal comparison for the type.
        /// </summary>
        Value = ValueToKeyMappingTransformer.SortOrder.Value
    }

    /// <summary>
    /// Information on the result of fitting a to-key transform.
    /// </summary>
    /// <typeparam name="T">The type of the values.</typeparam>
    public sealed class ToKeyFitResult<T>
    {
        /// <summary>
        /// For user defined delegates that accept instances of the containing type.
        /// </summary>
        /// <param name="result"></param>
        public delegate void OnFit(ToKeyFitResult<T> result);

        // At the moment this is empty. Once PR #863 clears, we can change this class to hold the output
        // key-values metadata.

        [BestFriend]
        internal ToKeyFitResult(ValueToKeyMappingTransformer.TermMap map)
        {
        }
    }
}
