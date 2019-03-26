// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    /// <include file='doc.xml' path='doc/members/member[@name="ValueToKeyMappingEstimator"]/*' />
    public sealed class ValueToKeyMappingEstimator : IEstimator<ValueToKeyMappingTransformer>
    {
        [BestFriend]
        internal static class Defaults
        {
            public const int MaximumNumberOfKeys = 1000000;
            public const KeyOrdinality Ordinality = KeyOrdinality.ByOccurrence;
            public const bool AddKeyValueAnnotationsAsText = false;
        }

        /// <summary>
        /// Controls how the order of the output keys.
        /// </summary>
        public enum KeyOrdinality : byte
        {
            /// <summary>
            /// Values will be assigned keys in the order in which they appear.
            /// </summary>
            ByOccurrence = 0,

            /// <summary>
            /// Values will be assigned keys according to their sort via an ordinal comparison for the type.
            /// </summary>
            ByValue = 1,
            // REVIEW: We can think about having a frequency order option. What about
            // other things, like case insensitive (where appropriate), culturally aware, etc.?
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        [BestFriend]
        internal abstract class ColumnOptionsBase
        {
            public readonly string OutputColumnName;
            public readonly string InputColumnName;
            public readonly KeyOrdinality KeyOrdinality;
            public readonly int MaximumNumberOfKeys;
            public readonly bool AddKeyValueAnnotationsAsText;

            [BestFriend]
            internal string[] Keys { get; set; }

            [BestFriend]
            internal string Key { get; set; }

            [BestFriend]
            private protected ColumnOptionsBase(string outputColumnName, string inputColumnName,
                int maximumNumberOfKeys, KeyOrdinality keyOrdinality, bool addKeyValueAnnotationsAsText)
            {
                Contracts.CheckNonWhiteSpace(outputColumnName, nameof(outputColumnName));
                OutputColumnName = outputColumnName;
                InputColumnName = inputColumnName ?? outputColumnName;
                KeyOrdinality = keyOrdinality;
                MaximumNumberOfKeys = maximumNumberOfKeys;
                AddKeyValueAnnotationsAsText = addKeyValueAnnotationsAsText;
            }
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        [BestFriend]
        internal sealed class ColumnOptions : ColumnOptionsBase
        {
            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
            /// <param name="maximumNumberOfKeys">Maximum number of keys to keep per column when auto-training.</param>
            /// <param name="keyOrdinality">How items should be ordered when vectorized. If <see cref="KeyOrdinality.ByOccurrence"/> choosen they will be in the order encountered.
            /// If <see cref="KeyOrdinality.ByValue"/>, items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').</param>
            /// <param name="addKeyValueAnnotationsAsText">Whether key value annotations should be text, regardless of the actual input type.</param>
            public ColumnOptions(string outputColumnName, string inputColumnName = null,
                int maximumNumberOfKeys = Defaults.MaximumNumberOfKeys,
                KeyOrdinality keyOrdinality = Defaults.Ordinality,
                bool addKeyValueAnnotationsAsText = false)
                : base(outputColumnName, inputColumnName, maximumNumberOfKeys, keyOrdinality, addKeyValueAnnotationsAsText)
            {
            }
        }

        private readonly IHost _host;
        private readonly ColumnOptionsBase[] _columns;
        private readonly IDataView _keyData;

        /// <summary>
        /// Initializes a new instance of <see cref="ValueToKeyMappingEstimator"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="maximumNumberOfKeys">Maximum number of keys to keep per column when auto-training.</param>
        /// <param name="keyOrdinality">How items should be ordered when vectorized. If <see cref="KeyOrdinality.ByOccurrence"/> choosen they will be in the order encountered.
        /// If <see cref="KeyOrdinality.ByValue"/>, items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').</param>
        internal ValueToKeyMappingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null, int maximumNumberOfKeys = Defaults.MaximumNumberOfKeys, KeyOrdinality keyOrdinality = Defaults.Ordinality) :
           this(env, new [] { new ColumnOptions(outputColumnName, inputColumnName ?? outputColumnName, maximumNumberOfKeys, keyOrdinality) })
        {
        }

        internal ValueToKeyMappingEstimator(IHostEnvironment env, ColumnOptionsBase[] columns, IDataView keyData = null)
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
                if (!col.IsKey || !col.Annotations.TryFindColumn(AnnotationUtils.Kinds.KeyValues, out var kv) || kv.Kind != SchemaShape.Column.VectorKind.Vector)
                {
                    kv = new SchemaShape.Column(AnnotationUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector,
                        colInfo.AddKeyValueAnnotationsAsText ? TextDataViewType.Instance : col.ItemType, col.IsKey);
                }
                Contracts.Assert(kv.IsValid);

                if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.SlotNames, out var slotMeta))
                    metadata = new SchemaShape(new[] { slotMeta, kv });
                else
                    metadata = new SchemaShape(new[] { kv });
                result[colInfo.OutputColumnName] = new SchemaShape.Column(colInfo.OutputColumnName, col.Kind, NumberDataViewType.UInt32, true, metadata);
            }

            return new SchemaShape(result.Values);
        }
    }
}
