// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

[assembly: LoadableClass(TextNormalizerTransform.Summary, typeof(IDataTransform), typeof(TextNormalizerTransform), typeof(TextNormalizerTransform.Arguments), typeof(SignatureDataTransform),
    "Text Normalizer Transform", "TextNormalizerTransform", "TextNormalizer", "TextNorm")]

[assembly: LoadableClass(TextNormalizerTransform.Summary, typeof(IDataTransform), typeof(TextNormalizerTransform), null, typeof(SignatureLoadDataTransform),
    "Text Normalizer Transform", TextNormalizerTransform.LoaderSignature)]

[assembly: LoadableClass(TextNormalizerTransform.Summary, typeof(TextNormalizerTransform), null, typeof(SignatureLoadModel),
     "Text Normalizer Transform", TextNormalizerTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(TextNormalizerTransform), null, typeof(SignatureLoadRowMapper),
   "Text Normalizer Transform", TextNormalizerTransform.LoaderSignature)]

namespace Microsoft.ML.Transforms.Text
{
    /// <summary>
    /// A text normalization transform that allows normalizing text case, removing diacritical marks, punctuation marks and/or numbers.
    /// The transform operates on text input as well as vector of tokens/text (vector of ReadOnlyMemory).
    /// </summary>
    public sealed class TextNormalizerTransform : OneToOneTransformerBase
    {
        public sealed class Column : OneToOneColumn
        {
            public static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Casing text using the rules of the invariant culture.", ShortName = "case", SortOrder = 1)]
            public TextNormalizerEstimator.CaseNormalizationMode TextCase = TextNormalizerEstimator.Defaults.TextCase;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep diacritical marks or remove them.",
                ShortName = "diac", SortOrder = 1)]
            public bool KeepDiacritics = TextNormalizerEstimator.Defaults.KeepDiacritics;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep punctuation marks or remove them.", ShortName = "punc", SortOrder = 2)]
            public bool KeepPunctuations = TextNormalizerEstimator.Defaults.KeepPunctuations;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep numbers or remove them.", ShortName = "num", SortOrder = 2)]
            public bool KeepNumbers = TextNormalizerEstimator.Defaults.KeepNumbers;
        }

        internal const string Summary = "A text normalization transform that allows normalizing text case, removing diacritical marks, punctuation marks and/or numbers." +
            " The transform operates on text input as well as vector of tokens/text (vector of ReadOnlyMemory).";

        internal const string LoaderSignature = nameof(TextNormalizerTransform);

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TEXTNORM",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TextNormalizerTransform).Assembly.FullName);
        }

        private const string RegistrationName = "TextNormalizer";
        public IReadOnlyCollection<(string input, string output)> Columns => ColumnPairs.AsReadOnly();

        private readonly TextNormalizerEstimator.CaseNormalizationMode _textCase;
        private readonly bool _keepDiacritics;
        private readonly bool _keepPunctuations;
        private readonly bool _keepNumbers;

        public TextNormalizerTransform(IHostEnvironment env,
            TextNormalizerEstimator.CaseNormalizationMode textCase = TextNormalizerEstimator.Defaults.TextCase,
            bool keepDiacritics = TextNormalizerEstimator.Defaults.KeepDiacritics,
            bool keepPunctuations = TextNormalizerEstimator.Defaults.KeepPunctuations,
            bool keepNumbers = TextNormalizerEstimator.Defaults.KeepNumbers,
            params (string input, string output)[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), columns)
        {
            _textCase = textCase;
            _keepDiacritics = keepDiacritics;
            _keepPunctuations = keepPunctuations;
            _keepNumbers = keepNumbers;

        }

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema.GetColumnType(srcCol);
            if (!TextNormalizerEstimator.IsColumnTypeValid(type))
                throw Host.ExceptParam(nameof(inputSchema), TextNormalizerEstimator.ExpectedColumnType);
        }

        public override void Save(ModelSaveContext ctx)
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

            ctx.Writer.Write((byte)_textCase);
            ctx.Writer.WriteBoolByte(_keepDiacritics);
            ctx.Writer.WriteBoolByte(_keepPunctuations);
            ctx.Writer.WriteBoolByte(_keepNumbers);
        }

        // Factory method for SignatureLoadModel.
        private static TextNormalizerTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new TextNormalizerTransform(host, ctx);
        }

        private TextNormalizerTransform(IHost host, ModelLoadContext ctx)
          : base(host, ctx)
        {
            var columnsLength = ColumnPairs.Length;
            // *** Binary format ***
            // <base>
            // byte: case
            // bool: whether to keep diacritics
            // bool: whether to keep punctuations
            // bool: whether to keep numbers
            _textCase = (TextNormalizerEstimator.CaseNormalizationMode)ctx.Reader.ReadByte();
            host.CheckDecode(Enum.IsDefined(typeof(TextNormalizerEstimator.CaseNormalizationMode), _textCase));

            _keepDiacritics = ctx.Reader.ReadBoolByte();
            _keepPunctuations = ctx.Reader.ReadBoolByte();
            _keepNumbers = ctx.Reader.ReadBoolByte();
        }

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Column, nameof(args.Column));
            var cols = new (string input, string output)[args.Column.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = args.Column[i];
                cols[i] = (item.Source ?? item.Name, item.Name);
            }
            return new TextNormalizerTransform(env,  args.TextCase, args.KeepDiacritics, args.KeepPunctuations, args.KeepNumbers, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        protected override IRowMapper MakeRowMapper(ISchema schema) => new Mapper(this, Schema.Create(schema));

        private sealed class Mapper : MapperBase
        {
            private readonly ColumnType[] _types;
            private readonly TextNormalizerTransform _parent;

            public Mapper(TextNormalizerTransform parent, Schema inputSchema)
              : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new ColumnType[_parent.ColumnPairs.Length];
                for (int i = 0; i < _types.Length; i++)
                {
                    inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int srcCol);
                    var srcType = inputSchema.GetColumnType(srcCol);
                    _types[i] = srcType.IsVector ? new VectorType(TextType.Instance) : srcType;
                }
            }

            public override Schema.Column[] GetOutputColumns()
            {
                var result = new Schema.Column[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    result[i] = new Schema.Column(_parent.ColumnPairs[i].output, _types[i], null);
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
                    if (_combinedDiacriticsMap == null)
                    {
                        var combinedDiacriticsMap = new Dictionary<char, char>();
                        for (int i = 0; i < _combinedDiacriticsPairs.Length; i++)
                        {
                            Contracts.Assert(_combinedDiacriticsPairs[i].Length == 2);
                            combinedDiacriticsMap.Add(_combinedDiacriticsPairs[i][0], _combinedDiacriticsPairs[i][1]);
                        }

                        Interlocked.CompareExchange(ref _combinedDiacriticsMap, combinedDiacriticsMap, null);
                    }

                    return _combinedDiacriticsMap;
                }
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                input.Schema.TryGetColumnIndex(_parent.ColumnPairs[iinfo].input, out int srcCol);
                var srcType = input.Schema.GetColumnType(srcCol);
                Host.Assert(srcType.ItemType.IsText);

                if (srcType.IsVector)
                {
                    Host.Assert(srcType.VectorSize >= 0);
                    return MakeGetterVec(input, iinfo);
                }

                Host.Assert(!srcType.IsVector);
                return MakeGetterOne(input, iinfo);
            }

            private ValueGetter<ReadOnlyMemory<char>> MakeGetterOne(IRow input, int iinfo)
            {
                var getSrc = input.GetGetter<ReadOnlyMemory<char>>(ColMapNewToOld[iinfo]);
                Host.AssertValue(getSrc);
                var src = default(ReadOnlyMemory<char>);
                var buffer = new StringBuilder();
                return
                    (ref ReadOnlyMemory<char> dst) =>
                    {
                        getSrc(ref src);
                        NormalizeSrc(ref src, ref dst, buffer);
                    };
            }

            private ValueGetter<VBuffer<ReadOnlyMemory<char>>> MakeGetterVec(IRow input, int iinfo)
            {
                var getSrc = input.GetGetter<VBuffer<ReadOnlyMemory<char>>>(ColMapNewToOld[iinfo]);
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
                        for (int i = 0; i < src.Count; i++)
                        {
                            NormalizeSrc(ref src.Values[i], ref temp, buffer);
                            if (!temp.IsEmpty)
                                list.Add(temp);
                        }

                        VBufferUtils.Copy(list, ref dst, list.Count);
                    };
            }

            private void NormalizeSrc(ref ReadOnlyMemory<char> src, ref ReadOnlyMemory<char> dst, StringBuilder buffer)
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

                    if (_parent._textCase == TextNormalizerEstimator.CaseNormalizationMode.Lower)
                        ch = CharUtils.ToLowerInvariant(ch);
                    else if (_parent._textCase == TextNormalizerEstimator.CaseNormalizationMode.Upper)
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

    public sealed class TextNormalizerEstimator : TrivialEstimator<TextNormalizerTransform>
    {
        /// <summary>
        /// Case normalization mode of text. This enumeration is serialized.
        /// </summary>
        public enum CaseNormalizationMode
        {
            Lower = 0,
            Upper = 1,
            None = 2
        }

        internal static class Defaults
        {
            public const CaseNormalizationMode TextCase = CaseNormalizationMode.Lower;
            public const bool KeepDiacritics = false;
            public const bool KeepPunctuations = true;
            public const bool KeepNumbers = true;

        }

        public static bool IsColumnTypeValid(ColumnType type) => (type.ItemType.IsText);

        internal const string ExpectedColumnType = "Expected Text item type";

        /// <summary>
        /// Normalizes incoming text in <paramref name="inputColumn"/> by changing case, removing diacritical marks, punctuation marks and/or numbers
        /// and outputs new text as <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to normalize.</param>
        /// <param name="outputColumn">The column containing output tokens. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="textCase">Casing text using the rules of the invariant culture.</param>
        /// <param name="keepDiacritics">Whether to keep diacritical marks or remove them.</param>
        /// <param name="keepPunctuations">Whether to keep punctuation marks or remove them.</param>
        /// <param name="keepNumbers">Whether to keep numbers or remove them.</param>
        public TextNormalizerEstimator(IHostEnvironment env,
            string inputColumn,
            string outputColumn = null,
            CaseNormalizationMode textCase = Defaults.TextCase,
            bool keepDiacritics = Defaults.KeepDiacritics,
            bool keepPunctuations = Defaults.KeepPunctuations,
            bool keepNumbers = Defaults.KeepNumbers)
            : this(env, textCase, keepDiacritics, keepPunctuations, keepNumbers, (inputColumn, outputColumn ?? inputColumn))
        {
        }

        /// <summary>
        /// Normalizes incoming text in input columns by changing case, removing diacritical marks, punctuation marks and/or numbers
        /// and outputs new text as output columns.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="textCase">Casing text using the rules of the invariant culture.</param>
        /// <param name="keepDiacritics">Whether to keep diacritical marks or remove them.</param>
        /// <param name="keepPunctuations">Whether to keep punctuation marks or remove them.</param>
        /// <param name="keepNumbers">Whether to keep numbers or remove them.</param>
        /// <param name="columns">Pairs of columns to run the text normalization on.</param>
        public TextNormalizerEstimator(IHostEnvironment env,
            CaseNormalizationMode textCase = Defaults.TextCase,
            bool keepDiacritics = Defaults.KeepDiacritics,
            bool keepPunctuations = Defaults.KeepPunctuations,
            bool keepNumbers = Defaults.KeepNumbers,
            params (string input, string output)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TextNormalizerEstimator)), new TextNormalizerTransform(env, textCase, keepDiacritics, keepPunctuations, keepNumbers, columns))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.input, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.input);
                if (!IsColumnTypeValid(col.ItemType))
                    throw Host.ExceptParam(nameof(inputSchema), ExpectedColumnType);
                result[colInfo.output] = new SchemaShape.Column(colInfo.output, col.Kind == SchemaShape.Column.VectorKind.Vector ? SchemaShape.Column.VectorKind.VariableVector : SchemaShape.Column.VectorKind.Scalar, col.ItemType, false);
            }
            return new SchemaShape(result.Values);
        }
    }
}
