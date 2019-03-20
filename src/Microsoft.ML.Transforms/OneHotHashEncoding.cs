// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(OneHotHashEncodingTransformer.Summary, typeof(IDataTransform), typeof(OneHotHashEncodingTransformer), typeof(OneHotHashEncodingTransformer.Options), typeof(SignatureDataTransform),
    OneHotHashEncodingTransformer.UserName, "CategoricalHashTransform", "CatHashTransform", "CategoricalHash", "CatHash")]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Produces a column of indicator vectors. The mapping between a value and a corresponding index is done through hashing.
    /// </summary>
    public sealed class OneHotHashEncodingTransformer : ITransformer
    {
        internal sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "The number of bits to hash into. Must be between 1 and 30, inclusive.",
                ShortName = "bits")]
            public int? NumberOfBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint? Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each term should be included in the hash", ShortName = "ord")]
            public bool? Ordered;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int? MaximumNumberOfInverts;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output kind: Bag (multi-set vector), Ind (indicator vector), or Key (index)",
                ShortName = "kind", SortOrder = 102)]
            public OneHotEncodingEstimator.OutputKind? OutputKind;

            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            private protected override bool TryParse(string str)
            {
                Contracts.AssertNonEmpty(str);

                // We accept N:B:S where N is the new column name, B is the number of bits,
                // and S is source column names.
                if (!TryParse(str, out string extra))
                    return false;
                if (extra == null)
                    return true;
                if (!int.TryParse(extra, out int bits))
                    return false;
                NumberOfBits = bits;
                return true;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (Seed != null || Ordered != null || MaximumNumberOfInverts != null)
                    return false;
                if (NumberOfBits == null)
                    return TryUnparseCore(sb);
                string extra = NumberOfBits.Value.ToString();
                return TryUnparseCore(sb, extra);
            }
        }

        /// <summary>
        /// This class is a merger of <see cref="HashingTransformer.Options"/> and <see cref="KeyToVectorMappingTransformer.Options"/>
        /// with join option removed
        /// </summary>
        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:numberOfBits:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bits to hash into. Must be between 1 and 30, inclusive.",
                ShortName = "bits", SortOrder = 2)]
            public int NumberOfBits = OneHotHashEncodingEstimator.Defaults.NumberOfBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint Seed = OneHotHashEncodingEstimator.Defaults.Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each term should be included in the hash", ShortName = "ord")]
            public bool Ordered = OneHotHashEncodingEstimator.Defaults.UseOrderedHashing;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int MaximumNumberOfInverts = OneHotHashEncodingEstimator.Defaults.MaximumNumberOfInverts;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output kind: Bag (multi-set vector), Ind (indicator vector), or Key (index)",
                ShortName = "kind", SortOrder = 102)]
            public OneHotEncodingEstimator.OutputKind OutputKind = OneHotHashEncodingEstimator.Defaults.OutputKind;
        }

        internal const string Summary = "Converts the categorical value into an indicator array by hashing the value and using the hash as an index in the "
            + "bag. If the input column is a vector, a single indicator bag is returned for it.";

        internal const string UserName = "Categorical Hash Transform";

        /// <summary>
        /// A helper method to create <see cref="OneHotHashEncodingTransformer"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be transformed. If this is null '<paramref name="name"/>' will be used.</param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="maximumNumberOfInverts">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        /// <param name="outputKind">The type of output expected.</param>
        private static IDataView Create(IHostEnvironment env,
            IDataView input,
            string name,
            string source = null,
            int numberOfBits = OneHotHashEncodingEstimator.Defaults.NumberOfBits,
            int maximumNumberOfInverts = OneHotHashEncodingEstimator.Defaults.MaximumNumberOfInverts,
            OneHotEncodingEstimator.OutputKind outputKind = OneHotHashEncodingEstimator.Defaults.OutputKind)
        {
            return new OneHotHashEncodingEstimator(env, name, source, numberOfBits, maximumNumberOfInverts, outputKind).Fit(input).Transform(input) as IDataView;
        }

        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("Categorical");
            h.CheckValue(options, nameof(options));
            h.CheckValue(input, nameof(input));
            h.CheckUserArg(Utils.Size(options.Columns) > 0, nameof(options.Columns));

            var columns = new List<OneHotHashEncodingEstimator.ColumnOptions>();
            foreach (var column in options.Columns)
            {
                var col = new OneHotHashEncodingEstimator.ColumnOptions(
                    column.Name,
                    column.Source ?? column.Name,
                    column.OutputKind ?? options.OutputKind,
                    column.NumberOfBits ?? options.NumberOfBits,
                    column.Seed ?? options.Seed,
                    column.Ordered ?? options.Ordered,
                    column.MaximumNumberOfInverts ?? options.MaximumNumberOfInverts);
                columns.Add(col);
            }
            return new OneHotHashEncodingEstimator(env, columns.ToArray()).Fit(input).Transform(input) as IDataTransform;
        }

        private readonly TransformerChain<ITransformer> _transformer;

        internal OneHotHashEncodingTransformer(HashingEstimator hash, IEstimator<ITransformer> keyToVector, IDataView input)
        {
            if (keyToVector != null)
                _transformer = hash.Append(keyToVector).Fit(input);
            else
                _transformer = new TransformerChain<ITransformer>(hash.Fit(input));
        }
        /// <summary>
        /// Schema propagation for transformers. Returns the output schema of the data, if
        /// the input schema is like the one provided.
        /// </summary>
        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => _transformer.GetOutputSchema(inputSchema);

        /// <summary>
        /// Take the data in, make transformations, output the data. Note that <see cref="IDataView"/>
        /// are lazy, so no actual transformations happen here, just schema validation.
        /// </summary>
        public IDataView Transform(IDataView input) => _transformer.Transform(input);

        void ICanSaveModel.Save(ModelSaveContext ctx) => (_transformer as ICanSaveModel).Save(ctx);

        /// <summary>
        /// Whether a call to <see cref="ITransformer.GetRowToRowMapper"/> should succeed, on an appropriate schema.
        /// </summary>
        bool ITransformer.IsRowToRowMapper => ((ITransformer)_transformer).IsRowToRowMapper;

        /// <summary>
        /// Constructs a row-to-row mapper based on an input schema.
        /// </summary>
        IRowToRowMapper ITransformer.GetRowToRowMapper(DataViewSchema inputSchema) => ((ITransformer)_transformer).GetRowToRowMapper(inputSchema);
    }

    /// <summary>
    /// Estimator that produces a column of indicator vectors. The mapping between a value and a corresponding index is done through hashing.
    /// </summary>
    public sealed class OneHotHashEncodingEstimator : IEstimator<OneHotHashEncodingTransformer>
    {
        [BestFriend]
        internal static class Defaults
        {
            public const int NumberOfBits = 16;
            public const uint Seed = 314489979;
            public const bool UseOrderedHashing = true;
            public const int MaximumNumberOfInverts = 0;
            public const OneHotEncodingEstimator.OutputKind OutputKind = OneHotEncodingEstimator.OutputKind.Bag;
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        [BestFriend]
        internal sealed class ColumnOptions
        {
            public readonly HashingEstimator.ColumnOptions HashingOptions;
            public readonly OneHotEncodingEstimator.OutputKind OutputKind;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="outputKind">Kind of output: bag, indicator vector etc.</param>
            /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
            /// <param name="seed">Hashing seed.</param>
            /// <param name="useOrderedHashing">Whether the position of each term should be included in the hash.</param>
            /// <param name="maximumNumberOfInverts">During hashing we constuct mappings between original values and the produced hash values.
            /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
            /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
            /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
            public ColumnOptions(string name, string inputColumnName = null,
                OneHotEncodingEstimator.OutputKind outputKind = Defaults.OutputKind,
                int numberOfBits = Defaults.NumberOfBits,
                uint seed = Defaults.Seed,
                bool useOrderedHashing = Defaults.UseOrderedHashing,
                int maximumNumberOfInverts = Defaults.MaximumNumberOfInverts)
            {
                HashingOptions = new HashingEstimator.ColumnOptions(name, inputColumnName ?? name, numberOfBits, seed, useOrderedHashing, maximumNumberOfInverts);
                OutputKind = outputKind;
            }
        }

        private readonly IHost _host;
        private readonly IEstimator<ITransformer> _toSomething;
        private HashingEstimator _hash;

        /// <summary>
        /// Instantiates a new instance of <see cref="OneHotHashEncodingEstimator"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform.
        /// If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="maximumNumberOfInverts">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        /// <param name="outputKind">The type of output expected.</param>
        internal OneHotHashEncodingEstimator(IHostEnvironment env,
            string outputColumnName,
            string inputColumnName = null,
            int numberOfBits = Defaults.NumberOfBits,
            int maximumNumberOfInverts = Defaults.MaximumNumberOfInverts,
            OneHotEncodingEstimator.OutputKind outputKind = Defaults.OutputKind)
            : this(env, new ColumnOptions(outputColumnName, inputColumnName ?? outputColumnName, outputKind, numberOfBits, maximumNumberOfInverts: maximumNumberOfInverts))
        {
        }

        internal OneHotHashEncodingEstimator(IHostEnvironment env, params ColumnOptions[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(ValueToKeyMappingEstimator));
            _hash = new HashingEstimator(_host, columns.Select(x => x.HashingOptions).ToArray());
            using (var ch = _host.Start(nameof(OneHotHashEncodingEstimator)))
            {
                var binaryCols = new List<(string outputColumnName, string inputColumnName)>();
                var cols = new List<(string outputColumnName, string inputColumnName, bool bag)>();
                for (int i = 0; i < columns.Length; i++)
                {
                    var column = columns[i];
                    OneHotEncodingEstimator.OutputKind kind = columns[i].OutputKind;
                    switch (kind)
                    {
                        default:
                            throw _host.ExceptUserArg(nameof(column.OutputKind));
                        case OneHotEncodingEstimator.OutputKind.Key:
                            continue;
                        case OneHotEncodingEstimator.OutputKind.Binary:
                            if ((column.HashingOptions.MaximumNumberOfInverts) != 0)
                                ch.Warning("Invert hashing is being used with binary encoding.");
                            binaryCols.Add((column.HashingOptions.Name, column.HashingOptions.Name));
                            break;
                        case OneHotEncodingEstimator.OutputKind.Indicator:
                            cols.Add((column.HashingOptions.Name, column.HashingOptions.Name, false));
                            break;
                        case OneHotEncodingEstimator.OutputKind.Bag:
                            cols.Add((column.HashingOptions.Name, column.HashingOptions.Name, true));
                            break;
                    }
                }
                IEstimator<ITransformer> toBinVector = null;
                IEstimator<ITransformer> toVector = null;
                if (binaryCols.Count > 0)
                    toBinVector = new KeyToBinaryVectorMappingEstimator(_host, binaryCols.Select(x => (x.outputColumnName, x.inputColumnName)).ToArray());
                if (cols.Count > 0)
                    toVector = new KeyToVectorMappingEstimator(_host, cols.Select(x => new KeyToVectorMappingEstimator.ColumnOptions(x.outputColumnName, x.inputColumnName, x.bag)).ToArray());

                if (toBinVector != null && toVector != null)
                    _toSomething = toVector.Append(toBinVector);
                else
                {
                    if (toBinVector != null)
                        _toSomething = toBinVector;
                    else
                        _toSomething = toVector;
                }
            }
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            if (_toSomething != null)
                return _hash.Append(_toSomething).GetOutputSchema(inputSchema);
            else
                return _hash.GetOutputSchema(inputSchema);
        }

        /// <summary>
        /// Trains and returns a <see cref="OneHotHashEncodingTransformer"/>.
        /// </summary>
        public OneHotHashEncodingTransformer Fit(IDataView input) => new OneHotHashEncodingTransformer(_hash, _toSomething, input);
    }
}
