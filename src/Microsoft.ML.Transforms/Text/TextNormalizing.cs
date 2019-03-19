// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Text;

[assembly: LoadableClass(TextNormalizingTransformer.Summary, typeof(IDataTransform), typeof(TextNormalizingTransformer), typeof(TextNormalizingTransformer.Options), typeof(SignatureDataTransform),
    "Text Normalizer Transform", "TextNormalizerTransform", "TextNormalizer", "TextNorm")]

[assembly: LoadableClass(TextNormalizingTransformer.Summary, typeof(IDataTransform), typeof(TextNormalizingTransformer), null, typeof(SignatureLoadDataTransform),
    "Text Normalizer Transform", TextNormalizingTransformer.LoaderSignature)]

[assembly: LoadableClass(TextNormalizingTransformer.Summary, typeof(TextNormalizingTransformer), null, typeof(SignatureLoadModel),
     "Text Normalizer Transform", TextNormalizingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(TextNormalizingTransformer), null, typeof(SignatureLoadRowMapper),
   "Text Normalizer Transform", TextNormalizingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.Text
{
    /// <summary>
    /// A text normalization transform that allows normalizing text case, removing diacritical marks, punctuation marks and/or numbers.
    /// The transform operates on text input as well as vector of tokens/text (vector of ReadOnlyMemory).
    /// </summary>
    public sealed class TextNormalizingTransformer : OneToOneTransformerBase
    {
        internal sealed class Column : OneToOneColumn
        {
            internal static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        internal sealed class Options
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Casing text using the rules of the invariant culture.", ShortName = "case", SortOrder = 1)]
            public TextNormalizingEstimator.CaseMode TextCase = TextNormalizingEstimator.Defaults.Mode;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep diacritical marks or remove them.",
                ShortName = "diac", SortOrder = 1)]
            public bool KeepDiacritics = TextNormalizingEstimator.Defaults.KeepDiacritics;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep punctuation marks or remove them.", ShortName = "punc", SortOrder = 2)]
            public bool KeepPunctuations = TextNormalizingEstimator.Defaults.KeepPunctuations;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep numbers or remove them.", ShortName = "num", SortOrder = 2)]
            public bool KeepNumbers = TextNormalizingEstimator.Defaults.KeepNumbers;
        }

        internal const string Summary = "A text normalization transform that allows normalizing text case, removing diacritical marks, punctuation marks and/or numbers." +
            " The transform operates on text input as well as vector of tokens/text (vector of ReadOnlyMemory).";

        internal const string LoaderSignature = "TextNormalizerTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TEXTNORM",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TextNormalizingTransformer).Assembly.FullName);
        }

        private const string RegistrationName = "TextNormalizer";

        /// <summary>
        /// The names of the output and input column pairs on which the transformation is applied.
        /// </summary>
        internal IReadOnlyCollection<(string outputColumnName, string inputColumnName)> Columns => ColumnPairs.AsReadOnly();

        private readonly TextNormalizingEstimator.CaseMode _caseMode;
        private readonly bool _keepDiacritics;
        private readonly bool _keepPunctuations;
        private readonly bool _keepNumbers;

        internal TextNormalizingTransformer(IHostEnvironment env,
            TextNormalizingEstimator.CaseMode caseMode = TextNormalizingEstimator.Defaults.Mode,
            bool keepDiacritics = TextNormalizingEstimator.Defaults.KeepDiacritics,
            bool keepPunctuations = TextNormalizingEstimator.Defaults.KeepPunctuations,
            bool keepNumbers = TextNormalizingEstimator.Defaults.KeepNumbers,
            params (string outputColumnName, string inputColumnName)[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), columns)
        {
            _caseMode = caseMode;
            _keepDiacritics = keepDiacritics;
            _keepPunctuations = keepPunctuations;
            _keepNumbers = keepNumbers;

        }

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema[srcCol].Type;
            if (!TextNormalizingEstimator.IsColumnTypeValid(type))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, TextNormalizingEstimator.ExpectedColumnType, type.ToString());
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // byte: case
            // bool: whether to keep diacritics
            // bool: whether to keep punctuations
            // bool: whether to keep numbers
            SaveColumns(ctx);

            ctx.Writer.Write((byte)_caseMode);
            ctx.Writer.WriteBoolByte(_keepDiacritics);
            ctx.Writer.WriteBoolByte(_keepPunctuations);
            ctx.Writer.WriteBoolByte(_keepNumbers);
        }

        // Factory method for SignatureLoadModel.
        private static TextNormalizingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new TextNormalizingTransformer(host, ctx);
        }

        private TextNormalizingTransformer(IHost host, ModelLoadContext ctx)
          : base(host, ctx)
        {
            var columnsLength = ColumnPairs.Length;
            // *** Binary format ***
            // <base>
            // byte: case
            // bool: whether to keep diacritics
            // bool: whether to keep punctuations
            // bool: whether to keep numbers
            _caseMode = (TextNormalizingEstimator.CaseMode)ctx.Reader.ReadByte();
            host.CheckDecode(Enum.IsDefined(typeof(TextNormalizingEstimator.CaseMode), _caseMode));

            _keepDiacritics = ctx.Reader.ReadBoolByte();
            _keepPunctuations = ctx.Reader.ReadBoolByte();
            _keepNumbers = ctx.Reader.ReadBoolByte();
        }

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Columns, nameof(options.Columns));
            var cols = new (string outputColumnName, string inputColumnName)[options.Columns.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = options.Columns[i];
                cols[i] = (item.Name, item.Source ?? item.Name);
            }
            return new TextNormalizingTransformer(env, options.TextCase, options.KeepDiacritics, options.KeepPunctuations, options.KeepNumbers, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly DataViewType[] _types;
            private readonly TextNormalizingTransformer _parent;

            public Mapper(TextNormalizingTransformer parent, DataViewSchema inputSchema)
              : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new DataViewType[_parent.ColumnPairs.Length];
                for (int i = 0; i < _types.Length; i++)
                {
                    inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int srcCol);
                    var srcType = inputSchema[srcCol].Type;
                    _types[i] = srcType is VectorType ? new VectorType(TextDataViewType.Instance) : srcType;
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, _types[i], null);
                }
                return result;
            }

            // A map where keys are letters combined with diacritics and values are the letters without diacritics.
            private static volatile Dictionary<char, char> _combinedDiacriticsMap;

            // List of pairs of (letters combined with diacritics, the letters without diacritics) from Office NL team.
            private static readonly string[] _combinedDiacriticsPairs =
            {
                // Latin letters combined with diacritics:
                "ÀA", "ÁA", "ÂA", "ÃA", "ÄA", "ÅA", "ÇC", "ÈE", "ÉE", "ÊE", "ËE", "ÌI", "ÍI", "ÎI", "ÏI", "ÑN",
                "ÒO", "ÓO", "ÔO", "ÕO", "ÖO", "ÙU", "ÚU", "ÛU", "ÜU", "ÝY", "àa", "áa", "âa", "ãa", "äa", "åa",
                "çc", "èe", "ée", "êe", "ëe", "ìi", "íi", "îi", "ïi", "ñn", "òo", "óo", "ôo", "õo", "öo", "ùu",
                "úu", "ûu", "üu", "ýy", "ÿy", "ĀA", "āa", "ĂA", "ăa", "ĄA", "ąa", "ĆC", "ćc", "ĈC", "ĉc", "ĊC",
                "ċc", "ČC", "čc", "ĎD", "ďd", "ĒE", "ēe", "ĔE", "ĕe", "ĖE", "ėe", "ĘE", "ęe", "ĚE", "ěe", "ĜG",
                "ĝg", "ĞG", "ğg", "ĠG", "ġg", "ĢG", "ģg", "ĤH", "ĥh", "ĨI", "ĩi", "ĪI", "īi", "ĬI", "ĭi", "ĮI",
                "įi", "İI", "ĴJ", "ĵj", "ĶK", "ķk", "ĹL", "ĺl", "ĻL", "ļl", "ĽL", "ľl", "ŃN", "ńn", "ŅN", "ņn",
                "ŇN", "ňn", "ŌO", "ōo", "ŎO", "ŏo", "ŐO", "őo", "ŔR", "ŕr", "ŖR", "ŗr", "ŘR", "řr", "ŚS", "śs",
                "ŜS", "ŝs", "ŞS", "şs", "ŠS", "šs", "ŢT", "ţt", "ŤT", "ťt", "ŨU", "ũu", "ŪU", "ūu", "ŬU", "ŭu",
                "ŮU", "ůu", "ŰU", "űu", "ŲU", "ųu", "ŴW", "ŵw", "ŶY", "ŷy", "ŸY", "ŹZ", "źz", "ŻZ", "żz", "ŽZ",
                "žz", "ƠO", "ơo", "ƯU", "ưu", "ǍA", "ǎa", "ǏI", "ǐi", "ǑO", "ǒo", "ǓU", "ǔu", "ǕU", "ǖu", "ǗU",
                "ǘu", "ǙU", "ǚu", "ǛU", "ǜu", "ǞA", "ǟa", "ǠA", "ǡa", "ǢÆ", "ǣæ", "ǦG", "ǧg", "ǨK", "ǩk", "ǪO",
                "ǫo", "ǬO", "ǭo", "ǮƷ", "ǯʒ", "ǰj", "ǴG", "ǵg", "ǸN", "ǹn", "ǺA", "ǻa", "ǼÆ", "ǽæ", "ǾØ", "ǿø",
                "ȀA", "ȁa", "ȂA", "ȃa", "ȄE", "ȅe", "ȆE", "ȇe", "ȈI", "ȉi", "ȊI", "ȋi", "ȌO", "ȍo", "ȎO", "ȏo",
                "ȐR", "ȑr", "ȒR", "ȓr", "ȔU", "ȕu", "ȖU", "ȗu", "ȘS", "șs", "ȚT", "țt", "ȞH", "ȟh", "ȦA", "ȧa",
                "ȨE", "ȩe", "ȪO", "ȫo", "ȬO", "ȭo", "ȮO", "ȯo", "ȰO", "ȱo", "ȲY", "ȳy",

                // Greek letters combined with diacritics:
                "ΆΑ", "ΈΕ", "ΉΗ", "ΊΙ", "ΌΟ", "ΎΥ", "ΏΩ", "ΐι", "ΪΙ", "ΫΥ", "άα", "έε", "ήη", "ίι", "ΰυ", "ϊι",
                "ϋυ", "όο", "ύυ", "ώω", "ϓϒ", "ϔϒ",

                // Cyrillic letters combined with diacritics:
                "ЀЕ", "ЁЕ", "ЃГ", "ЇІ", "ЌК", "ЍИ", "ЎУ", "ЙИ", "йи", "ѐе", "ёе", "ѓг", "їі", "ќк", "ѝи", "ўу",
                "ѶѴ", "ѷѵ", "ӁЖ", "ӂж", "ӐА", "ӑа", "ӒА", "ӓа", "ӖЕ", "ӗе", "ӚӘ", "ӛә", "ӜЖ", "ӝж", "ӞЗ", "ӟз",
                "ӢИ", "ӣи", "ӤИ", "ӥи", "ӦО", "ӧо", "ӪӨ", "ӫө", "ӬЭ", "ӭэ", "ӮУ", "ӯу", "ӰУ", "ӱу", "ӲУ", "ӳу",
                "ӴЧ", "ӵч", "ӸЫ", "ӹы"
            };

            private static Dictionary<char, char> CombinedDiacriticsMap
            {
                get
                {
                    Dictionary<char, char> result = _combinedDiacriticsMap;
                    if (result == null)
                    {
                        var combinedDiacriticsMap = new Dictionary<char, char>();
                        for (int i = 0; i < _combinedDiacriticsPairs.Length; i++)
                        {
                            Contracts.Assert(_combinedDiacriticsPairs[i].Length == 2);
                            combinedDiacriticsMap.Add(_combinedDiacriticsPairs[i][0], _combinedDiacriticsPairs[i][1]);
                        }

                        Interlocked.CompareExchange(ref _combinedDiacriticsMap, combinedDiacriticsMap, null);
                        result = _combinedDiacriticsMap;
                    }

                    return result;
                }
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                var srcType = input.Schema[_parent.ColumnPairs[iinfo].inputColumnName].Type;
                Host.Assert(srcType.GetItemType() is TextDataViewType);

                if (srcType is VectorType vectorType)
                {
                    Host.Assert(vectorType.Size >= 0);
                    return MakeGetterVec(input, iinfo);
                }

                return MakeGetterOne(input, iinfo);
            }

            private ValueGetter<ReadOnlyMemory<char>> MakeGetterOne(DataViewRow input, int iinfo)
            {
                var getSrc = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[ColMapNewToOld[iinfo]]);
                Host.AssertValue(getSrc);
                var src = default(ReadOnlyMemory<char>);
                var buffer = new StringBuilder();
                return
                    (ref ReadOnlyMemory<char> dst) =>
                    {
                        getSrc(ref src);
                        NormalizeSrc(in src, ref dst, buffer);
                    };
            }

            private ValueGetter<VBuffer<ReadOnlyMemory<char>>> MakeGetterVec(DataViewRow input, int iinfo)
            {
                var getSrc = input.GetGetter<VBuffer<ReadOnlyMemory<char>>>(input.Schema[ColMapNewToOld[iinfo]]);
                Host.AssertValue(getSrc);
                var src = default(VBuffer<ReadOnlyMemory<char>>);
                var buffer = new StringBuilder();
                var list = new List<ReadOnlyMemory<char>>();
                var temp = default(ReadOnlyMemory<char>);
                return
                    (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                    {
                        getSrc(ref src);
                        list.Clear();
                        var srcValues = src.GetValues();
                        for (int i = 0; i < srcValues.Length; i++)
                        {
                            NormalizeSrc(in srcValues[i], ref temp, buffer);
                            if (!temp.IsEmpty)
                                list.Add(temp);
                        }

                        VBufferUtils.Copy(list, ref dst, list.Count);
                    };
            }

            private void NormalizeSrc(in ReadOnlyMemory<char> src, ref ReadOnlyMemory<char> dst, StringBuilder buffer)
            {
                Host.AssertValue(buffer);

                if (src.IsEmpty)
                {
                    dst = src;
                    return;
                }

                buffer.Clear();

                int i = 0;
                int min = 0;
                var span = src.Span;
                while (i < src.Length)
                {
                    char ch = span[i];
                    if (!_parent._keepPunctuations && char.IsPunctuation(ch) || !_parent._keepNumbers && char.IsNumber(ch))
                    {
                        // Append everything before ch and ignore ch.
                        buffer.AppendSpan(span.Slice(min, i - min));
                        min = i + 1;
                        i++;
                        continue;
                    }

                    if (!_parent._keepDiacritics)
                    {
                        if (IsCombiningDiacritic(ch))
                        {
                            buffer.AppendSpan(span.Slice(min, i - min));
                            min = i + 1;
                            i++;
                            continue;
                        }

                        if (CombinedDiacriticsMap.ContainsKey(ch))
                            ch = CombinedDiacriticsMap[ch];
                    }

                    if (_parent._caseMode == TextNormalizingEstimator.CaseMode.Lower)
                        ch = CharUtils.ToLowerInvariant(ch);
                    else if (_parent._caseMode == TextNormalizingEstimator.CaseMode.Upper)
                        ch = CharUtils.ToUpperInvariant(ch);

                    if (ch != src.Span[i])
                    {
                        buffer.AppendSpan(span.Slice(min, i - min)).Append(ch);
                        min = i + 1;
                    }

                    i++;
                }

                Host.Assert(i == src.Length);
                int len = i - min;
                if (min == 0)
                {
                    Host.Assert(src.Length == len);
                    dst = src;
                }
                else
                {
                    buffer.AppendSpan(span.Slice(min, len));
                    dst = buffer.ToString().AsMemory();
                }
            }

            /// <summary>
            /// Whether a character is a combining diacritic character or not.
            /// Combining diacritic characters are the set of diacritics intended to modify other characters.
            /// The list is provided by Office NL team.
            /// </summary>
            private bool IsCombiningDiacritic(char ch)
            {
                if (ch < 0x0300 || ch > 0x0670)
                    return false;

                // Basic combining diacritics
                return ch >= 0x0300 && ch <= 0x036F ||

                    // Hebrew combining diacritics
                    ch >= 0x0591 && ch <= 0x05BD || ch == 0x05C1 || ch == 0x05C2 || ch == 0x05C4 ||
                    ch == 0x05C5 || ch == 0x05C7 ||

                    // Arabic combining diacritics
                    ch >= 0x0610 && ch <= 0x0615 || ch >= 0x064C && ch <= 0x065E || ch == 0x0670;
            }
        }
    }

    public sealed class TextNormalizingEstimator : TrivialEstimator<TextNormalizingTransformer>
    {
        /// <summary>
        /// Case normalization mode of text. This enumeration is serialized.
        /// </summary>
        public enum CaseMode
        {
            /// <summary>
            /// Make the output characters lowercased.
            /// </summary>
            Lower = 0,
            /// <summary>
            /// Make the output characters uppercased.
            /// </summary>
            Upper = 1,
            /// <summary>
            /// Do not change the case of output characters.
            /// </summary>
            None = 2
        }

        internal static class Defaults
        {
            public const CaseMode Mode = CaseMode.Lower;
            public const bool KeepDiacritics = false;
            public const bool KeepPunctuations = true;
            public const bool KeepNumbers = true;
        }

        internal static bool IsColumnTypeValid(DataViewType type) => (type.GetItemType() is TextDataViewType);

        internal const string ExpectedColumnType = "Text or vector of text.";

        /// <summary>
        /// Normalizes incoming text in <paramref name="inputColumnName"/> by changing case, removing diacritical marks, punctuation marks and/or numbers
        /// and outputs new text as <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="caseMode">Casing text using the rules of the invariant culture.</param>
        /// <param name="keepDiacritics">Whether to keep diacritical marks or remove them.</param>
        /// <param name="keepPunctuations">Whether to keep punctuation marks or remove them.</param>
        /// <param name="keepNumbers">Whether to keep numbers or remove them.</param>
        internal TextNormalizingEstimator(IHostEnvironment env,
            string outputColumnName,
            string inputColumnName = null,
            CaseMode caseMode = Defaults.Mode,
            bool keepDiacritics = Defaults.KeepDiacritics,
            bool keepPunctuations = Defaults.KeepPunctuations,
            bool keepNumbers = Defaults.KeepNumbers)
            : this(env, caseMode, keepDiacritics, keepPunctuations, keepNumbers, (outputColumnName, inputColumnName ?? outputColumnName))
        {
        }

        /// <summary>
        /// Normalizes incoming text in input columns by changing case, removing diacritical marks, punctuation marks and/or numbers
        /// and outputs new text as output columns.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="caseMode">Casing text using the rules of the invariant culture.</param>
        /// <param name="keepDiacritics">Whether to keep diacritical marks or remove them.</param>
        /// <param name="keepPunctuations">Whether to keep punctuation marks or remove them.</param>
        /// <param name="keepNumbers">Whether to keep numbers or remove them.</param>
        /// <param name="columns">Pairs of columns to run the text normalization on.</param>
        internal TextNormalizingEstimator(IHostEnvironment env,
            CaseMode caseMode = Defaults.Mode,
            bool keepDiacritics = Defaults.KeepDiacritics,
            bool keepPunctuations = Defaults.KeepPunctuations,
            bool keepNumbers = Defaults.KeepNumbers,
            params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TextNormalizingEstimator)),
                  new TextNormalizingTransformer(env, caseMode, keepDiacritics, keepPunctuations, keepNumbers, columns))
        {
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.inputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.inputColumnName);
                if (!IsColumnTypeValid(col.ItemType))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.inputColumnName, TextNormalizingEstimator.ExpectedColumnType, col.ItemType.ToString());
                result[colInfo.outputColumnName] = new SchemaShape.Column(colInfo.outputColumnName, col.Kind == SchemaShape.Column.VectorKind.Vector ? SchemaShape.Column.VectorKind.VariableVector : SchemaShape.Column.VectorKind.Scalar, col.ItemType, false);
            }
            return new SchemaShape(result.Values);
        }
    }
}
