// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Conversions;
using System.Collections.Generic;
using System.Linq;
using System.Text;

[assembly: LoadableClass(CategoricalHashTransform.Summary, typeof(IDataTransform), typeof(CategoricalHashTransform), typeof(CategoricalHashTransform.Arguments), typeof(SignatureDataTransform),
    CategoricalHashTransform.UserName, "CategoricalHashTransform", "CatHashTransform", "CategoricalHash", "CatHash")]

namespace Microsoft.ML.Transforms.Categorical
{
    public sealed class CategoricalHashTransform : ITransformer, ICanSaveModel
    {
        public sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "The number of bits to hash into. Must be between 1 and 30, inclusive.",
                ShortName = "bits")]
            public int? HashBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint? Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each term should be included in the hash", ShortName = "ord")]
            public bool? Ordered;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int? InvertHash;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output kind: Bag (multi-set vector), Ind (indicator vector), or Key (index)",
                ShortName = "kind", SortOrder = 102)]
            public CategoricalTransform.OutputKind? OutputKind;

            public static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            protected override bool TryParse(string str)
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
                HashBits = bits;
                return true;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (Seed != null || Ordered != null || InvertHash != null)
                    return false;
                if (HashBits == null)
                    return TryUnparseCore(sb);
                string extra = HashBits.Value.ToString();
                return TryUnparseCore(sb, extra);
            }
        }

        private static class Defaults
        {
            public const int HashBits = 16;
            public const uint Seed = 314489979;
            public const bool Ordered = true;
            public const int InvertHash = 0;
            public const CategoricalTransform.OutputKind OutputKind = CategoricalTransform.OutputKind.Bag;
        }

        /// <summary>
        /// This class is a merger of <see cref="HashTransformer.Arguments"/> and <see cref="KeyToVectorTransform.Arguments"/>
        /// with join option removed
        /// </summary>
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:hashBits:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bits to hash into. Must be between 1 and 30, inclusive.",
                ShortName = "bits", SortOrder = 2)]
            public int HashBits = Defaults.HashBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint Seed = Defaults.Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each term should be included in the hash", ShortName = "ord")]
            public bool Ordered = Defaults.Ordered;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int InvertHash = Defaults.InvertHash;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output kind: Bag (multi-set vector), Ind (indicator vector), or Key (index)",
                ShortName = "kind", SortOrder = 102)]
            public CategoricalTransform.OutputKind OutputKind = Defaults.OutputKind;
        }

        internal const string Summary = "Converts the categorical value into an indicator array by hashing the value and using the hash as an index in the "
            + "bag. If the input column is a vector, a single indicator bag is returned for it.";

        public const string UserName = "Categorical Hash Transform";

        /// <summary>
        /// A helper method to create <see cref="CategoricalHashTransform"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be transformed. If this is null '<paramref name="name"/>' will be used.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        /// <param name="outputKind">The type of output expected.</param>
        public static IDataView Create(IHostEnvironment env,
            IDataView input,
            string name,
            string source = null,
            int hashBits = OneHotHashEncodingEstimator.Defaults.HashBits,
            int invertHash = OneHotHashEncodingEstimator.Defaults.InvertHash,
            CategoricalTransform.OutputKind outputKind = OneHotHashEncodingEstimator.Defaults.OutputKind)
        {
            return new OneHotHashEncodingEstimator(env, name, source, hashBits, invertHash, outputKind).Fit(input).Transform(input) as IDataView;
        }

        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("Categorical");
            h.CheckValue(args, nameof(args));
            h.CheckValue(input, nameof(input));
            h.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column));

            var columns = new List<OneHotHashEncodingEstimator.ColumnInfo>();
            foreach (var column in args.Column)
            {
                var col = new OneHotHashEncodingEstimator.ColumnInfo(
                    column.Source ?? column.Name,
                    column.Name,
                    column.OutputKind ?? args.OutputKind,
                    column.HashBits ?? args.HashBits,
                    column.Seed ?? args.Seed,
                    column.Ordered ?? args.Ordered,
                    column.InvertHash ?? args.InvertHash);
                columns.Add(col);
            }
            return new OneHotHashEncodingEstimator(env, columns.ToArray()).Fit(input).Transform(input) as IDataTransform;
        }

        private readonly TransformerChain<ITransformer> _transformer;

        internal CategoricalHashTransform(HashingEstimator hash, IEstimator<ITransformer> keyToVector, IDataView input)
        {
            var chain = hash.Append(keyToVector);
            _transformer = chain.Fit(input);
        }

        public Schema GetOutputSchema(Schema inputSchema) => _transformer.GetOutputSchema(inputSchema);

        public IDataView Transform(IDataView input) => _transformer.Transform(input);

        public void Save(ModelSaveContext ctx) => _transformer.Save(ctx);

        public bool IsRowToRowMapper => _transformer.IsRowToRowMapper;

        public IRowToRowMapper GetRowToRowMapper(Schema inputSchema) => _transformer.GetRowToRowMapper(inputSchema);
    }

    /// <summary>
    /// Estimator which takes set of columns and produce for each column indicator array. Use hashing to determine indicator position.
    /// </summary>
    public sealed class OneHotHashEncodingEstimator : IEstimator<CategoricalHashTransform>
    {
        internal static class Defaults
        {
            public const int HashBits = 16;
            public const uint Seed = 314489979;
            public const bool Ordered = true;
            public const int InvertHash = 0;
            public const CategoricalTransform.OutputKind OutputKind = CategoricalTransform.OutputKind.Bag;
        }

        public sealed class ColumnInfo
        {
            public readonly HashTransformer.ColumnInfo HashInfo;
            public readonly CategoricalTransform.OutputKind OutputKind;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="input">Name of input column.</param>
            /// <param name="output">Name of output column.</param>
            /// <param name="outputKind">Kind of output: bag, indicator vector etc.</param>
            /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 31, inclusive.</param>
            /// <param name="seed">Hashing seed.</param>
            /// <param name="ordered">Whether the position of each term should be included in the hash.</param>
            /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
            public ColumnInfo(string input, string output,
                CategoricalTransform.OutputKind outputKind = Defaults.OutputKind,
                int hashBits = Defaults.HashBits,
                uint seed = Defaults.Seed,
                bool ordered = Defaults.Ordered,
                int invertHash = Defaults.InvertHash)
            {
                HashInfo = new HashTransformer.ColumnInfo(input, output, hashBits, seed, ordered, invertHash);
                OutputKind = outputKind;
            }
        }

        private readonly IHost _host;
        private readonly IEstimator<ITransformer> _toSomething;
        private HashingEstimator _hash;

        /// A helper method to create <see cref="OneHotHashEncodingEstimator"/> for public facing API.
        /// <param name="env">Host Environment.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the output column. If this is null '<paramref name="inputColumn"/>' will be used.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        /// <param name="outputKind">The type of output expected.</param>
        public OneHotHashEncodingEstimator(IHostEnvironment env,
            string inputColumn,
            string outputColumn,
            int hashBits = OneHotHashEncodingEstimator.Defaults.HashBits,
            int invertHash = OneHotHashEncodingEstimator.Defaults.InvertHash,
            CategoricalTransform.OutputKind outputKind = OneHotHashEncodingEstimator.Defaults.OutputKind)
            : this(env, new ColumnInfo(inputColumn, outputColumn ?? inputColumn, outputKind, hashBits, invertHash: invertHash))
        {
        }

        public OneHotHashEncodingEstimator(IHostEnvironment env, params ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(ValueToKeyMappingEstimator));
            _hash = new HashingEstimator(_host, columns.Select(x => x.HashInfo).ToArray());
            using (var ch = _host.Start(nameof(OneHotHashEncodingEstimator)))
            {
                var binaryCols = new List<(string input, string output)>();
                var cols = new List<(string input, string output, bool bag)>();
                for (int i = 0; i < columns.Length; i++)
                {
                    var column = columns[i];
                    CategoricalTransform.OutputKind kind = columns[i].OutputKind;
                    switch (kind)
                    {
                        default:
                            throw _host.ExceptUserArg(nameof(column.OutputKind));
                        case CategoricalTransform.OutputKind.Key:
                            continue;
                        case CategoricalTransform.OutputKind.Bin:
                            if ((column.HashInfo.InvertHash) != 0)
                                ch.Warning("Invert hashing is being used with binary encoding.");
                            binaryCols.Add((column.HashInfo.Output, column.HashInfo.Output));
                            break;
                        case CategoricalTransform.OutputKind.Ind:
                            cols.Add((column.HashInfo.Output, column.HashInfo.Output, false));
                            break;
                        case CategoricalTransform.OutputKind.Bag:
                            cols.Add((column.HashInfo.Output, column.HashInfo.Output, true));
                            break;
                    }
                }
                IEstimator<ITransformer> toBinVector = null;
                IEstimator<ITransformer> toVector = null;
                if (binaryCols.Count > 0)
                    toBinVector = new KeyToBinaryVectorMappingEstimator(_host, binaryCols.Select(x => new KeyToBinaryVectorTransform.ColumnInfo(x.input, x.output)).ToArray());
                if (cols.Count > 0)
                    toVector = new KeyToVectorMappingEstimator(_host, cols.Select(x => new KeyToVectorTransform.ColumnInfo(x.input, x.output, x.bag)).ToArray());

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

        public SchemaShape GetOutputSchema(SchemaShape inputSchema) => _hash.Append(_toSomething).GetOutputSchema(inputSchema);

        public CategoricalHashTransform Fit(IDataView input) => new CategoricalHashTransform(_hash, _toSomething, input);
    }

    public static class CategoricalHashStaticExtensions
    {
        public enum OneHotHashVectorOutputKind : byte
        {
            /// <summary>
            /// Output is a bag (multi-set) vector
            /// </summary>
            Bag = 1,

            /// <summary>
            /// Output is an indicator vector
            /// </summary>
            Ind = 2,

            /// <summary>
            /// Output is binary encoded
            /// </summary>
            Bin = 4,
        }

        public enum OneHotHashScalarOutputKind : byte
        {
            /// <summary>
            /// Output is an indicator vector
            /// </summary>
            Ind = 2,

            /// <summary>
            /// Output is binary encoded
            /// </summary>
            Bin = 4,
        }

        private const OneHotHashVectorOutputKind DefOut = (OneHotHashVectorOutputKind)OneHotHashEncodingEstimator.Defaults.OutputKind;
        private const int DefHashBits = OneHotHashEncodingEstimator.Defaults.HashBits;
        private const uint DefSeed = OneHotHashEncodingEstimator.Defaults.Seed;
        private const bool DefOrdered = OneHotHashEncodingEstimator.Defaults.Ordered;
        private const int DefInvertHash = OneHotHashEncodingEstimator.Defaults.InvertHash;

        private readonly struct Config
        {
            public readonly int HashBits;
            public readonly uint Seed;
            public readonly bool Ordered;
            public readonly int InvertHash;
            public readonly OneHotHashVectorOutputKind OutputKind;

            public Config(OneHotHashVectorOutputKind outputKind, int hashBits, uint seed, bool ordered, int invertHash)
            {
                OutputKind = outputKind;
                HashBits = hashBits;
                Seed = seed;
                Ordered = ordered;
                InvertHash = invertHash;
            }
        }

        private interface ICategoricalCol
        {
            PipelineColumn Input { get; }
            Config Config { get; }
        }

        private sealed class ImplScalar<T> : Vector<float>, ICategoricalCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplScalar(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class ImplVector<T> : Vector<float>, ICategoricalCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplVector(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class Rec : EstimatorReconciler
        {
            public static readonly Rec Inst = new Rec();

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env, PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames, IReadOnlyDictionary<PipelineColumn, string> outputNames, IReadOnlyCollection<string> usedNames)
            {
                var infos = new OneHotHashEncodingEstimator.ColumnInfo[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var tcol = (ICategoricalCol)toOutput[i];
                    infos[i] = new OneHotHashEncodingEstimator.ColumnInfo(inputNames[tcol.Input], outputNames[toOutput[i]], (CategoricalTransform.OutputKind)tcol.Config.OutputKind,
                        tcol.Config.HashBits, tcol.Config.Seed, tcol.Config.Ordered, tcol.Config.InvertHash);
                }
                return new OneHotHashEncodingEstimator(env, infos);
            }
        }

        /// <summary>
        /// Converts the categorical value into an indicator array by hashing categories into certain value and using that value as the index in the array.
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="outputKind">Specify the output type of indicator array: array or binary encoded data.</param>
        /// <param name="hashBits">Amount of bits to use for hashing.</param>
        /// <param name="seed">Seed value used for hashing.</param>
        /// <param name="ordered">Whether the position of each term should be included in the hash.</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        public static Vector<float> OneHotHashEncoding(this Scalar<string> input, OneHotHashScalarOutputKind outputKind = (OneHotHashScalarOutputKind)DefOut,
            int hashBits = DefHashBits, uint seed = DefSeed, bool ordered = DefOrdered, int invertHash = DefInvertHash)
        {
            Contracts.CheckValue(input, nameof(input));
            return new ImplScalar<string>(input, new Config((OneHotHashVectorOutputKind)outputKind, hashBits, seed, ordered, invertHash));
        }

        /// <summary>
        /// Converts the categorical value into an indicator array by building a dictionary of categories based on the data and using the id in the dictionary as the index in the array
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="outputKind">Specify the output type of indicator array: array or binary encoded data.</param>
        /// <param name="hashBits">Amount of bits to use for hashing.</param>
        /// <param name="seed">Seed value used for hashing.</param>
        /// <param name="ordered">Whether the position of each term should be included in the hash.</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        public static Vector<float> OneHotHashEncoding(this Vector<string> input, OneHotHashVectorOutputKind outputKind = DefOut,
            int hashBits = DefHashBits, uint seed = DefSeed, bool ordered = DefOrdered, int invertHash = DefInvertHash)
        {
            Contracts.CheckValue(input, nameof(input));
            return new ImplVector<string>(input, new Config(outputKind, hashBits, seed, ordered, invertHash));
        }

    }
}
