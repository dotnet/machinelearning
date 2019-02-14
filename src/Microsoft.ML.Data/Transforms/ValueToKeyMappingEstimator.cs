// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;

namespace Microsoft.ML.Transforms.Conversions
{
    /// <include file='doc.xml' path='doc/members/member[@name="ValueToKeyMappingEstimator"]/*' />
    public sealed class ValueToKeyMappingEstimator : IEstimator<ValueToKeyMappingTransformer>
    {
        [BestFriend]
        internal static class Defaults
        {
            public const int MaxNumKeys = 1000000;
            public const SortOrder Sort = SortOrder.Occurrence;
        }

        /// <summary>
        /// Controls how the order of the output keys.
        /// </summary>
        public enum SortOrder : byte
        {
            Occurrence = 0,
            Value = 1,
            // REVIEW: We can think about having a frequency order option. What about
            // other things, like case insensitive (where appropriate), culturally aware, etc.?
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        public class ColumnInfo
        {
            public readonly string OutputColumnName;
            public readonly string InputColumnName;
            public readonly SortOrder Sort;
            public readonly int MaxNumKeys;
            public readonly string[] Term;
            public readonly bool TextKeyValues;

            protected internal string Terms { get; set; }

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
            /// <param name="maxNumKeys">Maximum number of keys to keep per column when auto-training.</param>
            /// <param name="sort">How items should be ordered when vectorized. If <see cref="SortOrder.Occurrence"/> choosen they will be in the order encountered.
            /// If <see cref="SortOrder.Value"/>, items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').</param>
            /// <param name="term">List of terms.</param>
            /// <param name="textKeyValues">Whether key value metadata should be text, regardless of the actual input type.</param>
            public ColumnInfo(string outputColumnName, string inputColumnName = null,
                int maxNumKeys = Defaults.MaxNumKeys,
                SortOrder sort = Defaults.Sort,
                string[] term = null,
                bool textKeyValues = false
                )
            {
                Contracts.CheckNonWhiteSpace(outputColumnName, nameof(outputColumnName));
                OutputColumnName = outputColumnName;
                InputColumnName = inputColumnName ?? outputColumnName;
                Sort = sort;
                MaxNumKeys = maxNumKeys;
                Term = term;
                TextKeyValues = textKeyValues;
            }
        }

        private readonly IHost _host;
        private readonly ColumnInfo[] _columns;
        private readonly IDataView _keyData;

        /// <summary>
        /// Initializes a new instance of <see cref="ValueToKeyMappingEstimator"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="maxNumKeys">Maximum number of keys to keep per column when auto-training.</param>
        /// <param name="sort">How items should be ordered when vectorized. If <see cref="SortOrder.Occurrence"/> choosen they will be in the order encountered.
        /// If <see cref="SortOrder.Value"/>, items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').</param>
        internal ValueToKeyMappingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null, int maxNumKeys = Defaults.MaxNumKeys, SortOrder sort = Defaults.Sort) :
           this(env, new [] { new ColumnInfo(outputColumnName, inputColumnName ?? outputColumnName, maxNumKeys, sort) })
        {
        }

        internal ValueToKeyMappingEstimator(IHostEnvironment env, ColumnInfo[] columns, IDataView keyData = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(ValueToKeyMappingEstimator));
            _host.CheckNonEmpty(columns, nameof(columns));
            _host.CheckValueOrNull(keyData);
            if (keyData != null && keyData.Schema.Count != 1)
            {
                throw _host.ExceptParam(nameof(keyData), "If specified, this data view should contain only a single column " +
                    $"containing the terms to map, but this had {keyData.Schema.Count} columns.");

            }

            _columns = columns;
            _keyData = keyData;
        }

        /// <summary>
        /// Trains and returns a <see cref="ValueToKeyMappingTransformer"/>.
        /// </summary>
        public ValueToKeyMappingTransformer Fit(IDataView input) => new ValueToKeyMappingTransformer(_host, input, _columns, _keyData, false);

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in _columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.InputColumnName, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName);

                if (!col.ItemType.IsStandardScalar())
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName);
                SchemaShape metadata;

                // In the event that we are transforming something that is of type key, we will get their type of key value
                // metadata, unless it has none or is not vector in which case we back off to having key values over the item type.
                if (!col.IsKey || !col.Metadata.TryFindColumn(MetadataUtils.Kinds.KeyValues, out var kv) || kv.Kind != SchemaShape.Column.VectorKind.Vector)
                {
                    kv = new SchemaShape.Column(MetadataUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector,
                        colInfo.TextKeyValues ? TextDataViewType.Instance : col.ItemType, col.IsKey);
                }
                Contracts.Assert(kv.IsValid);

                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.SlotNames, out var slotMeta))
                    metadata = new SchemaShape(new[] { slotMeta, kv });
                else
                    metadata = new SchemaShape(new[] { kv });
                result[colInfo.OutputColumnName] = new SchemaShape.Column(colInfo.OutputColumnName, col.Kind, NumberDataViewType.UInt32, true, metadata);
            }

            return new SchemaShape(result.Values);
        }
    }

    public enum KeyValueOrder : byte
    {
        /// <summary>
        /// Terms will be assigned ID in the order in which they appear.
        /// </summary>
        Occurence = ValueToKeyMappingEstimator.SortOrder.Occurrence,

        /// <summary>
        /// Terms will be assigned ID according to their sort via an ordinal comparison for the type.
        /// </summary>
        Value = ValueToKeyMappingEstimator.SortOrder.Value
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
