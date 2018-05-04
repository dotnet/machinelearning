// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.TextAnalytics;

[assembly: LoadableClass(TextTransform.Summary, typeof(IDataTransform), typeof(TextTransform), typeof(TextTransform.Arguments), typeof(SignatureDataTransform),
    TextTransform.UserName, "TextTransform", TextTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    using StopWordsArgs = StopWordsRemoverTransform.Arguments;
    using TextNormalizerArgs = TextNormalizerTransform.Arguments;
    using StopWordsCol = StopWordsRemoverTransform.Column;
    using TextNormalizerCol = TextNormalizerTransform.Column;
    using StopWordsLang = StopWordsRemoverTransform.Language;
    using CaseNormalizationMode = TextNormalizerTransform.CaseNormalizationMode;

    /// <summary>
    /// A transform that turns a collection of text documents into numerical feature vectors. The feature vectors are counts 
    /// of (word or character) ngrams in a given text. It offers ngram hashing (finding the ngram token string name to feature
    /// integer index mapping through hashing) as an option.
    /// </summary>
    public static class TextTransform
    {
        /// <summary>
        /// Text language. This enumeration is serialized.
        /// </summary>
        public enum Language
        {
            English = 1,
            French = 2,
            German = 3,
            Dutch = 4,
            Italian = 5,
            Spanish = 6,
            Japanese = 7
        }

        /// <summary>
        /// Text vector normalizer kind.
        /// </summary>
        public enum TextNormKind
        {
            None = 0,
            L1 = 1,
            L2 = 2,
            LInf = 3
        }

        public sealed class Column : ManyToOneColumn
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

        /// <summary>
        /// This class exposes <see cref="NgramExtractorTransform"/>/<see cref="NgramHashExtractorTransform"/> arguments.
        /// </summary>
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "New column definition (optional form: name:srcs).", ShortName = "col", SortOrder = 1)]
            public Column Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Dataset language or 'AutoDetect' to detect language per row.", ShortName = "lang", SortOrder = 3)]
            public Language Language = DefaultLanguage;

            [Argument(ArgumentType.Multiple, HelpText = "Stopwords remover.", ShortName = "remover", NullName = "<None>", SortOrder = 4)]
            public IStopWordsRemoverFactory StopWordsRemover;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Casing text using the rules of the invariant culture.", ShortName = "case", SortOrder = 5)]
            public CaseNormalizationMode TextCase = CaseNormalizationMode.Lower;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep diacritical marks or remove them.", ShortName = "diac", SortOrder = 6)]
            public bool KeepDiacritics;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep punctuation marks or remove them.", ShortName = "punc", SortOrder = 7)]
            public bool KeepPunctuations = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep numbers or remove them.", ShortName = "num", SortOrder = 8)]
            public bool KeepNumbers = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to output the transformed text tokens as an additional column.", ShortName = "tokens,showtext,showTransformedText", SortOrder = 9)]
            public bool OutputTokens;

            [Argument(ArgumentType.Multiple, HelpText = "A dictionary of whitelisted terms.", ShortName = "dict", NullName = "<None>", SortOrder = 10, Hide = true)]
            public TermLoaderArguments Dictionary;

            [TGUI(Label = "Word Gram Extractor")]
            [Argument(ArgumentType.Multiple, HelpText = "Ngram feature extractor to use for words (WordBag/WordHashBag).", ShortName = "wordExtractor", NullName = "<None>", SortOrder = 11)]
            public INgramExtractorFactoryFactory WordFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments();

            [TGUI(Label = "Char Gram Extractor")]
            [Argument(ArgumentType.Multiple, HelpText = "Ngram feature extractor to use for characters (WordBag/WordHashBag).", ShortName = "charExtractor", NullName = "<None>", SortOrder = 12)]
            public INgramExtractorFactoryFactory CharFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 3, AllLengths = false };

            [Argument(ArgumentType.AtMostOnce, HelpText = "Normalize vectors (rows) individually by rescaling them to unit norm.", ShortName = "norm", SortOrder = 13)]
            public TextNormKind VectorNormalizer = TextNormKind.L2;
        }

        /// <summary>
        /// A distilled version of the TextTransform Arguments, with all fields marked readonly and
        /// only the exact set of information needed to construct the transforms preserved.
        /// </summary>
        private sealed class TransformApplierParams
        {
            public readonly INgramExtractorFactory WordExtractorFactory;
            public readonly INgramExtractorFactory CharExtractorFactory;

            public readonly TextNormKind VectorNormalizer;
            public readonly Language Language;
            public readonly IStopWordsRemoverFactory StopWordsRemover;
            public readonly CaseNormalizationMode TextCase;
            public readonly bool KeepDiacritics;
            public readonly bool KeepPunctuations;
            public readonly bool KeepNumbers;
            public readonly bool OutputTextTokens;
            public readonly TermLoaderArguments Dictionary;

            public StopWordsRemoverTransform.Language StopwordsLanguage
            {
                get
                {
                    return (StopWordsRemoverTransform.Language)
                        Enum.Parse(typeof(StopWordsRemoverTransform.Language), Language.ToString());
                }
            }

            public LpNormNormalizerTransform.NormalizerKind LpNormalizerKind
            {
                get
                {
                    switch (VectorNormalizer)
                    {
                        case TextNormKind.L1:
                            return LpNormNormalizerTransform.NormalizerKind.L1Norm;
                        case TextNormKind.L2:
                            return LpNormNormalizerTransform.NormalizerKind.L2Norm;
                        case TextNormKind.LInf:
                            return LpNormNormalizerTransform.NormalizerKind.LInf;
                        default:
                            Contracts.Assert(false, "Unexpected normalizer type");
                            return LpNormNormalizerTransform.NormalizerKind.L2Norm;
                    }
                }
            }

            // These properties encode the logic needed to determine which transforms to apply.
            #region NeededTransforms
            public bool NeedsWordTokenizationTransform { get { return WordExtractorFactory != null || NeedsRemoveStopwordsTransform || OutputTextTokens; } }

            public bool NeedsRemoveStopwordsTransform { get { return StopWordsRemover != null; } }

            public bool NeedsNormalizeTransform
            {
                get
                {
                    return
                        TextCase != CaseNormalizationMode.None ||
                        !KeepDiacritics ||
                        !KeepPunctuations ||
                        !KeepNumbers;
                }
            }

            private bool UsesHashExtractors
            {
                get
                {
                    return
                        (WordExtractorFactory == null ? true : WordExtractorFactory.UseHashingTrick) &&
                        (CharExtractorFactory == null ? true : CharExtractorFactory.UseHashingTrick);
                }
            }

            // If we're performing language auto detection, or either of our extractors aren't hashing then
            // we need all the input text concatenated into a single Vect<DvText>, for the LanguageDetectionTransform
            // to operate on the entire text vector, and for the Dictionary feature extractor to build its bound dictionary
            // correctly.
            public bool NeedInitialSourceColumnConcatTransform
            {
                get
                {
                    return !UsesHashExtractors;
                }
            }
            #endregion

            public TransformApplierParams(IHost host, Arguments args)
            {
                Contracts.AssertValue(host);
                host.CheckUserArg(args.Column != null, nameof(args.Column), "Columns must be specified");
                host.CheckUserArg(args.WordFeatureExtractor != null || args.CharFeatureExtractor != null || args.OutputTokens,
                    nameof(args.WordFeatureExtractor), "At least one feature extractor or OutputTokens must be specified.");
                host.Check(Enum.IsDefined(typeof(Language), args.Language));
                host.Check(Enum.IsDefined(typeof(CaseNormalizationMode), args.TextCase));
                WordExtractorFactory = args.WordFeatureExtractor?.CreateComponent(host, args.Dictionary);
                CharExtractorFactory = args.CharFeatureExtractor?.CreateComponent(host, args.Dictionary);
                VectorNormalizer = args.VectorNormalizer;
                Language = args.Language;
                StopWordsRemover = args.StopWordsRemover;
                TextCase = args.TextCase;
                KeepDiacritics = args.KeepDiacritics;
                KeepPunctuations = args.KeepPunctuations;
                KeepNumbers = args.KeepNumbers;
                OutputTextTokens = args.OutputTokens;
                Dictionary = args.Dictionary;
            }
        }

        internal const string Summary = "A transform that turns a collection of text documents into numerical feature vectors. " +
            "The feature vectors are normalized counts of (word and/or character) ngrams in a given tokenized text.";

        internal const string UserName = "Text Transform";
        internal const string LoaderSignature = "Text";

        public const Language DefaultLanguage = Language.English;

        private const string TransformedTextColFormat = "{0}_TransformedText";

        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(LoaderSignature);
            h.CheckValue(args, nameof(args));
            h.CheckValue(input, nameof(input));

            var tparams = new TransformApplierParams(h, args);
            string[] textCols = args.Column.Source;
            string[] wordTokCols = null;
            string[] charTokCols = null;
            string wordFeatureCol = null;
            string charFeatureCol = null;
            List<string> tempCols = new List<string>();
            IDataView view = input;

            if (tparams.NeedInitialSourceColumnConcatTransform && textCols.Length > 1)
            {
                var xfCols = new ConcatTransform.Column[] { new ConcatTransform.Column() };
                xfCols[0].Source = textCols;
                textCols = new[] { GenerateColumnName(input.Schema, args.Column.Name, "InitialConcat") };
                xfCols[0].Name = textCols[0];
                tempCols.Add(textCols[0]);
                view = new ConcatTransform(h, new ConcatTransform.Arguments() { Column = xfCols }, view);
            }

            if (tparams.NeedsWordTokenizationTransform)
            {
                var xfCols = new DelimitedTokenizeTransform.Column[textCols.Length];
                wordTokCols = new string[textCols.Length];
                for (int i = 0; i < textCols.Length; i++)
                {
                    var col = new DelimitedTokenizeTransform.Column();
                    col.Source = textCols[i];
                    col.Name = GenerateColumnName(view.Schema, textCols[i], "WordTokenizer");
                    
                    xfCols[i] = col;

                    wordTokCols[i] = col.Name;
                    tempCols.Add(col.Name);
                }

                view = new DelimitedTokenizeTransform(h, new DelimitedTokenizeTransform.Arguments() { Column = xfCols }, view);
            }

            if (tparams.NeedsNormalizeTransform)
            {
                string[] srcCols = wordTokCols == null ? textCols : wordTokCols;
                var xfCols = new TextNormalizerCol[srcCols.Length];
                string[] dstCols = new string[srcCols.Length];
                for (int i = 0; i < srcCols.Length; i++)
                {
                    dstCols[i] = GenerateColumnName(view.Schema, srcCols[i], "TextNormalizer");
                    tempCols.Add(dstCols[i]);
                    xfCols[i] = new TextNormalizerCol() { Source = srcCols[i], Name = dstCols[i] };
                }

                view = new TextNormalizerTransform(h,
                    new TextNormalizerArgs()
                    {
                        Column = xfCols,
                        KeepDiacritics = tparams.KeepDiacritics,
                        KeepNumbers = tparams.KeepNumbers,
                        KeepPunctuations = tparams.KeepPunctuations,
                        TextCase = tparams.TextCase
                    }, view);

                if (wordTokCols != null)
                    wordTokCols = dstCols;
                else
                    textCols = dstCols;
            }

            if (tparams.NeedsRemoveStopwordsTransform)
            {
                Contracts.Assert(wordTokCols != null, "StopWords transform requires that word tokenization has been applied to the input text.");
                var xfCols = new StopWordsCol[wordTokCols.Length];
                var dstCols = new string[wordTokCols.Length];
                for (int i = 0; i < wordTokCols.Length; i++)
                {
                    var col = new StopWordsCol();
                    col.Source = wordTokCols[i];
                    col.Name = GenerateColumnName(view.Schema, wordTokCols[i], "StopWordsRemoverTransform");
                    dstCols[i] = col.Name;
                    tempCols.Add(col.Name);
                    col.Language = tparams.StopwordsLanguage;

                    xfCols[i] = col;
                }
                view = tparams.StopWordsRemover.CreateComponent(h, view, xfCols);
                wordTokCols = dstCols;
            }

            if (tparams.WordExtractorFactory != null)
            {
                var dstCol = GenerateColumnName(view.Schema, args.Column.Name, "WordExtractor");
                tempCols.Add(dstCol);
                view = tparams.WordExtractorFactory.Create(h, view, new[] {
                    new ExtractorColumn()
                    {
                        Name = dstCol,
                        Source = wordTokCols,
                        FriendlyNames = args.Column.Source
                    }});
                wordFeatureCol = dstCol;
            }

            if (tparams.OutputTextTokens)
            {
                string[] srcCols = wordTokCols ?? textCols;
                view = new ConcatTransform(h,
                    new ConcatTransform.Arguments()
                    {
                        Column = new[] { new ConcatTransform.Column()
                        {
                            Name = string.Format(TransformedTextColFormat, args.Column.Name),
                            Source = srcCols
                        }}
                    }, view);
            }

            if (tparams.CharExtractorFactory != null)
            {
                {
                    var srcCols = wordTokCols ?? textCols;
                    charTokCols = new string[srcCols.Length];
                    var xfCols = new CharTokenizeTransform.Column[srcCols.Length];
                    for (int i = 0; i < srcCols.Length; i++)
                    {
                        var col = new CharTokenizeTransform.Column();
                        col.Source = srcCols[i];
                        col.Name = GenerateColumnName(view.Schema, srcCols[i], "CharTokenizer");
                        tempCols.Add(col.Name);
                        charTokCols[i] = col.Name;
                        xfCols[i] = col;
                    }
                    view = new CharTokenizeTransform(h, new CharTokenizeTransform.Arguments() { Column = xfCols }, view);
                }

                {
                    charFeatureCol = GenerateColumnName(view.Schema, args.Column.Name, "CharExtractor");
                    tempCols.Add(charFeatureCol);
                    view = tparams.CharExtractorFactory.Create(h, view, new[] {
                        new ExtractorColumn()
                        {
                            Source = charTokCols,
                            FriendlyNames = args.Column.Source,
                            Name = charFeatureCol
                        }});
                }
            }

            if (tparams.VectorNormalizer != TextNormKind.None)
            {
                var xfCols = new List<LpNormNormalizerTransform.Column>(2);
                if (charFeatureCol != null)
                {
                    var dstCol = GenerateColumnName(view.Schema, charFeatureCol, "LpCharNorm");
                    tempCols.Add(dstCol);
                    xfCols.Add(new LpNormNormalizerTransform.Column()
                    {
                        Source = charFeatureCol,
                        Name = dstCol
                    });
                    charFeatureCol = dstCol;
                }

                if (wordFeatureCol != null)
                {
                    var dstCol = GenerateColumnName(view.Schema, wordFeatureCol, "LpWordNorm");
                    tempCols.Add(dstCol);
                    xfCols.Add(new LpNormNormalizerTransform.Column()
                    {
                        Source = wordFeatureCol,
                        Name = dstCol
                    });
                    wordFeatureCol = dstCol;
                }
                if (xfCols.Count > 0)
                    view = new LpNormNormalizerTransform(h, new LpNormNormalizerTransform.Arguments()
                    {
                        NormKind = tparams.LpNormalizerKind,
                        Column = xfCols.ToArray()
                    }, view);
            }

            {
                var srcTaggedCols = new List<KeyValuePair<string, string>>(2);
                if (charFeatureCol != null && wordFeatureCol != null)
                {
                    // If we're producing both char and word grams, then we need to disambiguate
                    // between them (e.g. the word 'a' vs. the char gram 'a').
                    srcTaggedCols.Add(new KeyValuePair<string, string>("Char", charFeatureCol));
                    srcTaggedCols.Add(new KeyValuePair<string, string>("Word", wordFeatureCol));
                }
                else
                {
                    // Otherwise, simply use the slot names, omitting the original source column names
                    // entirely. For the Concat transform setting the Key == Value of the TaggedColumn
                    // KVP signals this intent.
                    Contracts.Assert(charFeatureCol != null || wordFeatureCol != null || tparams.OutputTextTokens);
                    if (charFeatureCol != null)
                        srcTaggedCols.Add(new KeyValuePair<string, string>(charFeatureCol, charFeatureCol));
                    else if (wordFeatureCol != null)
                        srcTaggedCols.Add(new KeyValuePair<string, string>(wordFeatureCol, wordFeatureCol));
                }
                if (srcTaggedCols.Count > 0)
                    view = new ConcatTransform(h, new ConcatTransform.TaggedArguments()
                    {
                        Column = new[] { new ConcatTransform.TaggedColumn() {
                        Name = args.Column.Name,
                        Source = srcTaggedCols.ToArray()
                    }}
                    }, view);
            }

            return new DropColumnsTransform(h,
                new DropColumnsTransform.Arguments() { Column = tempCols.ToArray() }, view);
        }

        private static string GenerateColumnName(ISchema schema, string srcName, string xfTag)
        {
            return schema.GetTempColumnName(string.Format("{0}_{1}", srcName, xfTag));
        }
    }
}
