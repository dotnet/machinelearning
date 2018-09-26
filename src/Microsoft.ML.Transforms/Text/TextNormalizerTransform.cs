// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TextAnalytics;

[assembly: LoadableClass(TextNormalizerTransform.Summary, typeof(TextNormalizerTransform), typeof(TextNormalizerTransform.Arguments), typeof(SignatureDataTransform),
    "Text Normalizer Transform", "TextNormalizerTransform", "TextNormalizer", "TextNorm")]

[assembly: LoadableClass(TextNormalizerTransform.Summary, typeof(TextNormalizerTransform), null, typeof(SignatureLoadDataTransform),
    "Text Normalizer Transform", TextNormalizerTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.TextAnalytics
{
    /// <summary>
    /// A text normalization transform that allows normalizing text case, removing diacritical marks, punctuation marks and/or numbers.
    /// The transform operates on text input as well as vector of tokens/text (vector of ReadOnlyMemory).
    /// </summary>
    public sealed class TextNormalizerTransform : OneToOneTransformBase
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
            public CaseNormalizationMode TextCase = CaseNormalizationMode.Lower;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep diacritical marks or remove them.",
                ShortName = "diac", SortOrder = 1)]
            public bool KeepDiacritics = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep punctuation marks or remove them.", ShortName = "punc", SortOrder = 2)]
            public bool KeepPunctuations = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep numbers or remove them.", ShortName = "num", SortOrder = 2)]
            public bool KeepNumbers = true;
        }

        internal const string Summary = "A text normalization transform that allows normalizing text case, removing diacritical marks, punctuation marks and/or numbers." +
            " The transform operates on text input as well as vector of tokens/text (vector of ReadOnlyMemory).";

        public const string LoaderSignature = "TextNormalizerTransform";
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

        // Arguments
        private readonly CaseNormalizationMode _case;
        private readonly bool _keepDiacritics;
        private readonly bool _keepPunctuations;
        private readonly bool _keepNumbers;

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

        public TextNormalizerTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Column, input, TestIsTextItem)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            using (var ch = Host.Start("Construction"))
            {
                ch.CheckUserArg(Enum.IsDefined(typeof(CaseNormalizationMode), args.TextCase),
                    nameof(args.TextCase), "Invalid case normalization mode");

                _case = args.TextCase;
                _keepDiacritics = args.KeepDiacritics;
                _keepPunctuations = args.KeepPunctuations;
                _keepNumbers = args.KeepNumbers;

                ch.Done();
            }
            Metadata.Seal();
        }

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

        private TextNormalizerTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestIsTextItem)
        {
            Host.AssertValue(ctx);

            using (var ch = Host.Start("Deserialization"))
            {
                // *** Binary format ***
                // <base>
                //   byte: case
                //   bool: whether to keep diacritics
                //   bool: whether to keep punctuations
                //   bool: whether to keep numbers
                ch.AssertNonEmpty(Infos);

                _case = (CaseNormalizationMode)ctx.Reader.ReadByte();
                ch.CheckDecode(Enum.IsDefined(typeof(CaseNormalizationMode), _case));

                _keepDiacritics = ctx.Reader.ReadBoolByte();
                _keepPunctuations = ctx.Reader.ReadBoolByte();
                _keepNumbers = ctx.Reader.ReadBoolByte();

                ch.Done();
            }
            Metadata.Seal();
        }

        public static TextNormalizerTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new TextNormalizerTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            //   byte: case
            //   bool: whether to keep diacritics
            //   bool: whether to keep punctuations
            //   bool: whether to keep numbers
            SaveBase(ctx);

            ctx.Writer.Write((byte)_case);
            ctx.Writer.WriteBoolByte(_keepDiacritics);
            ctx.Writer.WriteBoolByte(_keepPunctuations);
            ctx.Writer.WriteBoolByte(_keepNumbers);
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < Infos.Length);
            return Infos[iinfo].TypeSrc.IsVector ? new VectorType(TextType.Instance) : Infos[iinfo].TypeSrc;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            var typeSrc = Infos[iinfo].TypeSrc;
            Host.Assert(typeSrc.ItemType.IsText);

            if (typeSrc.IsVector)
            {
                Host.Assert(typeSrc.VectorSize >= 0);
                return MakeGetterVec(input, iinfo);
            }

            Host.Assert(!typeSrc.IsVector);
            return MakeGetterOne(input, iinfo);
        }

        private ValueGetter<ReadOnlyMemory<char>> MakeGetterOne(IRow input, int iinfo)
        {
            Contracts.Assert(Infos[iinfo].TypeSrc.IsText);
            var getSrc = GetSrcGetter<ReadOnlyMemory<char>>(input, iinfo);
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
            var getSrc = GetSrcGetter<VBuffer<ReadOnlyMemory<char>>>(input, iinfo);
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
                if (!_keepPunctuations && char.IsPunctuation(ch) || !_keepNumbers && char.IsNumber(ch))
                {
                    // Append everything before ch and ignore ch.
                    buffer.AppendSpan(span.Slice(min, i - min));
                    min = i + 1;
                    i++;
                    continue;
                }

                if (!_keepDiacritics)
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

                if (_case == CaseNormalizationMode.Lower)
                    ch = CharUtils.ToLowerInvariant(ch);
                else if (_case == CaseNormalizationMode.Upper)
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
