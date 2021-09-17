// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Text;

[assembly: LoadableClass(TextFeaturizingEstimator.Summary, typeof(IDataTransform), typeof(TextFeaturizingEstimator), typeof(TextFeaturizingEstimator.Options), typeof(SignatureDataTransform),
    TextFeaturizingEstimator.UserName, "TextTransform", TextFeaturizingEstimator.LoaderSignature)]

[assembly: LoadableClass(TextFeaturizingEstimator.Summary, typeof(ITransformer), typeof(TextFeaturizingEstimator), null, typeof(SignatureLoadModel),
    TextFeaturizingEstimator.UserName, "TextTransform", TextFeaturizingEstimator.LoaderSignature)]

namespace Microsoft.ML.Transforms.Text
{
    using CaseMode = TextNormalizingEstimator.CaseMode;
    using StopWordsCol = StopWordsRemovingTransformer.Column;

    /// <summary>
    /// Defines the different type of stop words remover supported.
    /// </summary>
    public interface IStopWordsRemoverOptions { }

    /// <summary>
    ///  An estimator that turns a collection of text documents into numerical feature vectors.
    ///  The feature vectors are normalized counts of word and/or character n-grams (based on the options supplied).
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | Yes. |
    /// | Input column data type | [text](xref:Microsoft.ML.Data.TextDataViewType) |
    /// | Output column data type | Vector of <xref:System.Single> |
    /// | Exportable to ONNX | No |
    ///
    /// This estimator gives the user one-stop solution for doing:
    /// * Language Detection
    /// * [Tokenization](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization)
    /// * [Text normalization](https://en.wikipedia.org/wiki/Text_normalization)
    /// * [Predefined and custom stopwords removal](https://en.wikipedia.org/wiki/Stop_words)
    /// * [Word-based or character-based Ngram extraction and SkipGram extraction (through the advanced [options](xref:Microsoft.ML.Transforms.TextFeaturizingEstimator.Options.WordFeatureExtractor))](https://en.wikipedia.org/wiki/N-gram)
    /// * [TF, IDF or TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
    /// * [L-p vector normalization](xref: Microsoft.ML.Transforms.LpNormNormalizingTransformer)
    ///
    /// By default the features are made of (word/character) n-grams/skip-grams​ and the number of features are equal to the vocabulary size found by analyzing the data.
    /// To output an additional column with the tokens generated, use [OutputTokensColumnName](xref:Microsoft.ML.Transforms.Text.TextFeaturizingEstimator.Options.OutputTokensColumnName).
    /// The number of features can also be specified by selecting the maximum number of n-gram to keep in the <xref:Microsoft.ML.Transforms.Text.TextFeaturizingEstimator.Options>, where the estimator can be further tuned.
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]></format>
    /// </remarks>
    /// <seealso cref="TextCatalog.FeaturizeText(TransformsCatalog.TextTransforms, string, Options, string[])"/>
    /// <seealso cref="TextCatalog.FeaturizeText(TransformsCatalog.TextTransforms, string, string)"/>
    public sealed class TextFeaturizingEstimator : IEstimator<ITransformer>
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
        public enum NormFunction
        {
            /// <summary>
            /// Use this to disable normalization.
            /// </summary>
            None = 0,
            /// <summary>
            /// L1-norm.
            /// </summary>
            L1 = 1,
            /// <summary>
            /// L2-norm.
            /// </summary>
            L2 = 2,
            /// <summary>
            /// Infinity-norm.
            /// </summary>
            Infinity = 3
        }

        internal sealed class Column : ManyToOneColumn
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

        /// <summary>
        /// Advanced options for the <see cref="TextFeaturizingEstimator"/>.
        /// </summary>
        public sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "New column definition (optional form: name:srcs).", Name = "Column", ShortName = "col", SortOrder = 1)]
            internal Column Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Dataset language or 'AutoDetect' to detect language per row.", ShortName = "lang", SortOrder = 3)]
            internal Language Language = DefaultLanguage;

            [Argument(ArgumentType.Multiple, Name = "StopWordsRemover", HelpText = "Stopwords remover.", ShortName = "remover", NullName = "<None>", SortOrder = 4)]
            internal IStopWordsRemoverFactory StopWordsRemover;

            /// <summary>
            /// The underlying state of <see cref="StopWordsRemover"/> and <see cref="StopWordsRemoverOptions"/>.
            /// </summary>
            private IStopWordsRemoverOptions _stopWordsRemoverOptions;

            /// <summary>
            /// Option to set type of stop word remover to use.
            /// The following options are available
            /// <list type="bullet">
            ///     <item>
            ///         <description>The <see cref="StopWordsRemovingEstimator.Options"/> removes the language specific list of stop words from the input.</description>
            ///     </item>
            ///     <item>
            ///        <description>The <see cref="CustomStopWordsRemovingEstimator.Options"/> uses user provided list of stop words.</description>
            ///     </item>
            /// </list>
            /// Setting this to 'null' does not remove stop words from the input.
            /// </summary>
            public IStopWordsRemoverOptions StopWordsRemoverOptions
            {
                get { return _stopWordsRemoverOptions; }
                set
                {
                    _stopWordsRemoverOptions = value;
                    IStopWordsRemoverFactory options = null;
                    if (_stopWordsRemoverOptions != null)
                    {
                        if (_stopWordsRemoverOptions is StopWordsRemovingEstimator.Options)
                        {
                            options = new PredefinedStopWordsRemoverFactory();
                            Language = (_stopWordsRemoverOptions as StopWordsRemovingEstimator.Options).Language;
                        }
                        else if (_stopWordsRemoverOptions is CustomStopWordsRemovingEstimator.Options)
                        {
                            var stopwords = (_stopWordsRemoverOptions as CustomStopWordsRemovingEstimator.Options).StopWords;
                            options = new CustomStopWordsRemovingTransformer.LoaderArguments()
                            {
                                Stopwords = stopwords,
                                Stopword = string.Join(",", stopwords)
                            };
                        }
                    }
                    StopWordsRemover = options;
                }
            }

            [Argument(ArgumentType.AtMostOnce, HelpText = "Casing text using the rules of the invariant culture.", Name = "TextCase", ShortName = "case", SortOrder = 5)]
            public CaseMode CaseMode = TextNormalizingEstimator.Defaults.Mode;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep diacritical marks or remove them.", ShortName = "diac", SortOrder = 6)]
            public bool KeepDiacritics = TextNormalizingEstimator.Defaults.KeepDiacritics;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep punctuation marks or remove them.", ShortName = "punc", SortOrder = 7)]
            public bool KeepPunctuations = TextNormalizingEstimator.Defaults.KeepPunctuations;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep numbers or remove them.", ShortName = "num", SortOrder = 8)]
            public bool KeepNumbers = TextNormalizingEstimator.Defaults.KeepNumbers;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Column containing the transformed text tokens.", ShortName = "tokens,showtext,showTransformedText", SortOrder = 9)]
            public string OutputTokensColumnName;

            [Argument(ArgumentType.Multiple, HelpText = "A dictionary of allowed terms.", ShortName = "dict", NullName = "<None>", SortOrder = 10, Hide = true)]
            internal TermLoaderArguments Dictionary;

            [TGUI(Label = "Word Gram Extractor")]
            [Argument(ArgumentType.Multiple, Name = "WordFeatureExtractor", HelpText = "Ngram feature extractor to use for words (WordBag/WordHashBag).", ShortName = "wordExtractor", NullName = "<None>", SortOrder = 11)]
            internal INgramExtractorFactoryFactory WordFeatureExtractorFactory;

            /// <summary>
            /// The underlying state of <see cref="WordFeatureExtractorFactory"/> and <see cref="WordFeatureExtractor"/>.
            /// </summary>
            private WordBagEstimator.Options _wordFeatureExtractor;

            /// <summary>
            /// Norm of the output vector. It will be normalized to one.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Normalize vectors (rows) individually by rescaling them to unit norm.", Name = "VectorNormalizer", ShortName = "norm", SortOrder = 13)]
            public NormFunction Norm = NormFunction.L2;

            /// <summary>
            /// Ngram feature extractor to use for words (WordBag/WordHashBag).
            /// Set to <see langword="null" /> to turn off n-gram generation for words.
            /// </summary>
            public WordBagEstimator.Options WordFeatureExtractor
            {
                get { return _wordFeatureExtractor; }
                set
                {
                    _wordFeatureExtractor = value;
                    NgramExtractorTransform.NgramExtractorArguments extractor = null;
                    if (_wordFeatureExtractor != null)
                    {
                        extractor = new NgramExtractorTransform.NgramExtractorArguments();
                        extractor.NgramLength = _wordFeatureExtractor.NgramLength;
                        extractor.SkipLength = _wordFeatureExtractor.SkipLength;
                        extractor.UseAllLengths = _wordFeatureExtractor.UseAllLengths;
                        extractor.MaxNumTerms = _wordFeatureExtractor.MaximumNgramsCount;
                        extractor.Weighting = _wordFeatureExtractor.Weighting;
                    }
                    WordFeatureExtractorFactory = extractor;
                }
            }

            [TGUI(Label = "Char Gram Extractor")]
            [Argument(ArgumentType.Multiple, Name = "CharFeatureExtractor", HelpText = "Ngram feature extractor to use for characters (WordBag/WordHashBag).", ShortName = "charExtractor", NullName = "<None>", SortOrder = 12)]
            internal INgramExtractorFactoryFactory CharFeatureExtractorFactory;

            /// <summary>
            /// The underlying state of <see cref="CharFeatureExtractorFactory"/> and <see cref="CharFeatureExtractor"/>
            /// </summary>
            private WordBagEstimator.Options _charFeatureExtractor;

            /// <summary>
            /// Ngram feature extractor to use for characters (WordBag/WordHashBag).
            /// Set to <see langword="null" /> to turn off n-gram generation for characters.
            /// </summary>
            public WordBagEstimator.Options CharFeatureExtractor
            {
                get { return _charFeatureExtractor; }
                set
                {
                    _charFeatureExtractor = value;
                    NgramExtractorTransform.NgramExtractorArguments extractor = null;
                    if (_charFeatureExtractor != null)
                    {
                        extractor = new NgramExtractorTransform.NgramExtractorArguments();
                        extractor.NgramLength = _charFeatureExtractor.NgramLength;
                        extractor.SkipLength = _charFeatureExtractor.SkipLength;
                        extractor.UseAllLengths = _charFeatureExtractor.UseAllLengths;
                        extractor.MaxNumTerms = _charFeatureExtractor.MaximumNgramsCount;
                        extractor.Weighting = _charFeatureExtractor.Weighting;
                    }
                    CharFeatureExtractorFactory = extractor;
                }
            }

            public Options()
            {
                WordFeatureExtractor = new WordBagEstimator.Options();
                CharFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 3, UseAllLengths = false };
            }
        }

        internal readonly string OutputColumn;
        private readonly string[] _inputColumns;
        private IReadOnlyCollection<string> InputColumns => _inputColumns.AsReadOnly();
        internal Options OptionalSettings { get; }

        // These parameters are hardcoded for now.
        // REVIEW: expose them once sub-transforms are estimators.
        private IStopWordsRemoverFactory _stopWordsRemover;
        private TermLoaderArguments _dictionary;
        private INgramExtractorFactoryFactory _wordFeatureExtractor;
        private INgramExtractorFactoryFactory _charFeatureExtractor;

        private readonly IHost _host;

        /// <summary>
        /// A distilled version of the TextFeaturizingEstimator  Arguments, with all fields marked readonly and
        /// only the exact set of information needed to construct the transforms preserved.
        /// </summary>
        private sealed class TransformApplierParams
        {
            public readonly INgramExtractorFactory WordExtractorFactory;
            public readonly INgramExtractorFactory CharExtractorFactory;

            public readonly NormFunction Norm;
            public readonly Language Language;
            public readonly IStopWordsRemoverFactory StopWordsRemover;
            public readonly CaseMode TextCase;
            public readonly bool KeepDiacritics;
            public readonly bool KeepPunctuations;
            public readonly bool KeepNumbers;
            public readonly string OutputTextTokensColumnName;
            public readonly TermLoaderArguments Dictionary;

            public StopWordsRemovingEstimator.Language StopwordsLanguage
                => (StopWordsRemovingEstimator.Language)Enum.Parse(typeof(StopWordsRemovingEstimator.Language), Language.ToString());

            internal LpNormNormalizingEstimatorBase.NormFunction LpNorm
            {
                get
                {
                    switch (Norm)
                    {
                        case NormFunction.L1:
                            return LpNormNormalizingEstimatorBase.NormFunction.L1;
                        case NormFunction.L2:
                            return LpNormNormalizingEstimatorBase.NormFunction.L2;
                        case NormFunction.Infinity:
                            return LpNormNormalizingEstimatorBase.NormFunction.Infinity;
                        default:
                            Contracts.Assert(false, "Unexpected normalizer type");
                            return LpNormNormalizingEstimatorBase.NormFunction.L2;
                    }
                }
            }

            // These properties encode the logic needed to determine which transforms to apply.
            #region NeededTransforms
            public bool NeedsWordTokenizationTransform { get { return WordExtractorFactory != null || NeedsRemoveStopwordsTransform || !string.IsNullOrEmpty(OutputTextTokensColumnName); } }

            public bool NeedsRemoveStopwordsTransform { get { return StopWordsRemover != null; } }

            public bool NeedsNormalizeTransform
            {
                get
                {
                    return
                        TextCase != CaseMode.None ||
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
            // we need all the input text concatenated into a single ReadOnlyMemory, for the LanguageDetectionTransform
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

            public TransformApplierParams(TextFeaturizingEstimator parent)
            {
                var host = parent._host;
                host.Check(Enum.IsDefined(typeof(Language), parent.OptionalSettings.Language));
                host.Check(Enum.IsDefined(typeof(CaseMode), parent.OptionalSettings.CaseMode));
                WordExtractorFactory = parent._wordFeatureExtractor?.CreateComponent(host, parent._dictionary);
                CharExtractorFactory = parent._charFeatureExtractor?.CreateComponent(host, parent._dictionary);
                Norm = parent.OptionalSettings.Norm;
                Language = parent.OptionalSettings.Language;
                StopWordsRemover = parent._stopWordsRemover;
                TextCase = parent.OptionalSettings.CaseMode;
                KeepDiacritics = parent.OptionalSettings.KeepDiacritics;
                KeepPunctuations = parent.OptionalSettings.KeepPunctuations;
                KeepNumbers = parent.OptionalSettings.KeepNumbers;
                OutputTextTokensColumnName = parent.OptionalSettings.OutputTokensColumnName;
                Dictionary = parent._dictionary;
            }
        }

        internal const string Summary = "A transform that turns a collection of text documents into numerical feature vectors. " +
            "The feature vectors are normalized counts of (word and/or character) n-grams in a given tokenized text.";

        internal const string UserName = "Text Transform";
        internal const string LoaderSignature = "Text";

        internal const Language DefaultLanguage = Language.English;

        internal TextFeaturizingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null)
            : this(env, outputColumnName, new[] { inputColumnName ?? outputColumnName })
        {
        }

        internal TextFeaturizingEstimator(IHostEnvironment env, string name, IEnumerable<string> source, Options options = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(TextFeaturizingEstimator));
            _host.CheckValue(source, nameof(source));
            _host.CheckParam(source.Any(), nameof(source));
            _host.CheckParam(!source.Any(string.IsNullOrWhiteSpace), nameof(source));
            _host.CheckNonEmpty(name, nameof(name));
            _host.CheckValueOrNull(options);

            _inputColumns = source.ToArray();
            OutputColumn = name;

            OptionalSettings = new Options();
            if (options != null)
                OptionalSettings = options;

            _stopWordsRemover = OptionalSettings.StopWordsRemover;
            _dictionary = null;
            _wordFeatureExtractor = OptionalSettings.WordFeatureExtractorFactory;
            _charFeatureExtractor = OptionalSettings.CharFeatureExtractorFactory;

        }

        /// <summary>
        /// Trains and returns a <see cref="ITransformer"/>.
        /// </summary>
        public ITransformer Fit(IDataView input)
        {
            var h = _host;
            h.CheckValue(input, nameof(input));

            var tparams = new TransformApplierParams(this);
            string[] textCols = _inputColumns;
            string[] wordTokCols = null;
            string[] charTokCols = null;
            string wordFeatureCol = null;
            string charFeatureCol = null;
            List<string> tempCols = new List<string>();
            IDataView view = input;
            TransformerChain<ITransformer> chain = new TransformerChain<ITransformer>();

            if (tparams.NeedInitialSourceColumnConcatTransform && textCols.Length > 1)
            {
                var srcCols = textCols;
                textCols = new[] { GenerateColumnName(input.Schema, OutputColumn, "InitialConcat") };
                tempCols.Add(textCols[0]);
                chain = AddToChainAndTransform(chain, new ColumnConcatenatingTransformer(h, textCols[0], srcCols), ref view);
            }

            if (tparams.NeedsNormalizeTransform)
            {
                var xfCols = new (string outputColumnName, string inputColumnName)[textCols.Length];
                string[] dstCols = new string[textCols.Length];
                for (int i = 0; i < textCols.Length; i++)
                {
                    dstCols[i] = GenerateColumnName(view.Schema, textCols[i], "TextNormalizer");
                    tempCols.Add(dstCols[i]);
                    xfCols[i] = (dstCols[i], textCols[i]);
                }
                chain = AddToChainAndTransform(chain,
                    new TextNormalizingEstimator(h, tparams.TextCase, tparams.KeepDiacritics, tparams.KeepPunctuations,
                    tparams.KeepNumbers, xfCols).Fit(view), ref view);

                textCols = dstCols;
            }

            if (tparams.NeedsWordTokenizationTransform)
            {
                var xfCols = new WordTokenizingEstimator.ColumnOptions[textCols.Length];
                wordTokCols = new string[textCols.Length];
                for (int i = 0; i < textCols.Length; i++)
                {
                    var col = new WordTokenizingEstimator.ColumnOptions(GenerateColumnName(view.Schema, textCols[i], "WordTokenizer"), textCols[i]);
                    xfCols[i] = col;
                    wordTokCols[i] = col.Name;
                    tempCols.Add(col.Name);
                }

                chain = AddToChainAndTransform(chain, new WordTokenizingEstimator(h, xfCols).Fit(view), ref view);
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
                chain = AddToChainAndTransform(chain, tparams.StopWordsRemover.CreateComponent(h, view, xfCols), ref view);
                wordTokCols = dstCols;
            }

            if (tparams.WordExtractorFactory != null)
            {
                var dstCol = GenerateColumnName(view.Schema, OutputColumn, "WordExtractor");
                tempCols.Add(dstCol);
                chain = AddToChainAndTransform(chain, tparams.WordExtractorFactory.Create(h, view, new[] {
                    new ExtractorColumn()
                    {
                        Name = dstCol,
                        Source = wordTokCols,
                        FriendlyNames = _inputColumns
                    }}), ref view);
                wordFeatureCol = dstCol;
            }

            if (!string.IsNullOrEmpty(tparams.OutputTextTokensColumnName))
            {
                string[] srcCols = wordTokCols ?? textCols;
                chain = AddToChainAndTransform(chain, new ColumnConcatenatingTransformer(h, tparams.OutputTextTokensColumnName, srcCols), ref view);
            }

            if (tparams.CharExtractorFactory != null)
            {
                var srcCols = tparams.NeedsRemoveStopwordsTransform ? wordTokCols : textCols;
                charTokCols = new string[srcCols.Length];
                var xfCols = new (string outputColumnName, string inputColumnName)[srcCols.Length];
                for (int i = 0; i < srcCols.Length; i++)
                {
                    xfCols[i] = (GenerateColumnName(view.Schema, srcCols[i], "CharTokenizer"), srcCols[i]);
                    tempCols.Add(xfCols[i].outputColumnName);
                    charTokCols[i] = xfCols[i].outputColumnName;
                }
                chain = AddToChainAndTransform(chain, new TokenizingByCharactersTransformer(h, columns: xfCols), ref view);

                charFeatureCol = GenerateColumnName(view.Schema, OutputColumn, "CharExtractor");
                tempCols.Add(charFeatureCol);
                chain = AddToChainAndTransform(chain, tparams.CharExtractorFactory.Create(h, view, new[] {
                    new ExtractorColumn()
                    {
                        Source = charTokCols,
                        FriendlyNames = _inputColumns,
                        Name = charFeatureCol
                    } }), ref view);
            }

            if (tparams.Norm != NormFunction.None)
            {
                var xfCols = new List<LpNormNormalizingEstimator.ColumnOptions>(2);

                if (charFeatureCol != null)
                {
                    var dstCol = GenerateColumnName(view.Schema, charFeatureCol, "LpCharNorm");
                    tempCols.Add(dstCol);
                    xfCols.Add(new LpNormNormalizingEstimator.ColumnOptions(dstCol, charFeatureCol, norm: tparams.LpNorm));
                    charFeatureCol = dstCol;
                }

                if (wordFeatureCol != null)
                {
                    var dstCol = GenerateColumnName(view.Schema, wordFeatureCol, "LpWordNorm");
                    tempCols.Add(dstCol);
                    xfCols.Add(new LpNormNormalizingEstimator.ColumnOptions(dstCol, wordFeatureCol, norm: tparams.LpNorm));
                    wordFeatureCol = dstCol;
                }

                if (xfCols.Count > 0)
                    chain = AddToChainAndTransform(chain, new LpNormNormalizingTransformer(h, xfCols.ToArray()), ref view);
            }

            {
                var srcTaggedCols = new List<KeyValuePair<string, string>>(2);
                if (charFeatureCol != null && wordFeatureCol != null)
                {
                    // If we're producing both char and word grams, then we need to disambiguate
                    // between them (for example, the word 'a' vs. the char gram 'a').
                    srcTaggedCols.Add(new KeyValuePair<string, string>("Char", charFeatureCol));
                    srcTaggedCols.Add(new KeyValuePair<string, string>("Word", wordFeatureCol));
                }
                else
                {
                    // Otherwise, simply use the slot names, omitting the original source column names
                    // entirely. For the Concat transform setting the Key == Value of the TaggedColumn
                    // KVP signals this intent.
                    Contracts.Assert(charFeatureCol != null || wordFeatureCol != null || !string.IsNullOrEmpty(tparams.OutputTextTokensColumnName));
                    if (charFeatureCol != null)
                        srcTaggedCols.Add(new KeyValuePair<string, string>(charFeatureCol, charFeatureCol));
                    else if (wordFeatureCol != null)
                        srcTaggedCols.Add(new KeyValuePair<string, string>(wordFeatureCol, wordFeatureCol));
                }
                if (srcTaggedCols.Count > 0)
                {
                    chain = AddToChainAndTransform(chain, new ColumnConcatenatingTransformer(h, new ColumnConcatenatingTransformer.ColumnOptions(OutputColumn,
                        srcTaggedCols.Select(kvp => (kvp.Value, kvp.Key)))), ref view);
                }
            }

            chain = AddToChainAndTransform(chain, new ColumnSelectingTransformer(h, null, tempCols.ToArray()), ref view);
            return new Transformer(_host, chain);
        }

        private static TransformerChain<ITransformer> AddToChainAndTransform(TransformerChain<ITransformer> chain, ITransformer transformer, ref IDataView view)
        {
            Contracts.AssertValue(chain);
            Contracts.AssertValue(transformer);
            Contracts.AssertValue(view);

            view = transformer.Transform(view);
            return chain.Append(transformer);
        }

        private static ITransformer Create(IHostEnvironment env, ModelLoadContext ctx)
            => new Transformer(env, ctx);

        private static string GenerateColumnName(DataViewSchema schema, string srcName, string xfTag)
        {
            return schema.GetTempColumnName(string.Format("{0}_{1}", srcName, xfTag));
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var srcName in _inputColumns)
            {
                if (!inputSchema.TryFindColumn(srcName, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", srcName);
                if (!(col.ItemType is TextDataViewType))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", srcName, "scalar or vector of String", col.GetTypeString());
            }

            var metadata = new List<SchemaShape.Column>(2);
            metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextDataViewType.Instance, false));
            if (OptionalSettings.Norm != NormFunction.None)
                metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false));

            result[OutputColumn] = new SchemaShape.Column(OutputColumn, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false,
                new SchemaShape(metadata));

            if (!string.IsNullOrEmpty(OptionalSettings.OutputTokensColumnName))
            {
                string name = OptionalSettings.OutputTokensColumnName;
                result[name] = new SchemaShape.Column(name, SchemaShape.Column.VectorKind.VariableVector, TextDataViewType.Instance, false);
            }

            return new SchemaShape(result.Values);
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options args, IDataView data)
        {
            var estimator = new TextFeaturizingEstimator(env, args.Columns.Name, args.Columns.Source ?? new[] { args.Columns.Name }, args);
            estimator._stopWordsRemover = args.StopWordsRemover;
            estimator._dictionary = args.Dictionary;
            // Review: I don't think the following two lines are needed.
            estimator._wordFeatureExtractor = args.WordFeatureExtractorFactory;
            estimator._charFeatureExtractor = args.CharFeatureExtractorFactory;
            return estimator.Fit(data).Transform(data) as IDataTransform;
        }

        private sealed class Transformer : ITransformer
        {
            private const string TransformDirTemplate = "Step_{0:000}";
            private const uint VerIDataTransform = 0x00010001;

            private readonly IHost _host;
            private readonly TransformerChain<ITransformer> _chain;

            internal Transformer(IHostEnvironment env, TransformerChain<ITransformer> chain)
            {
                Contracts.AssertValue(env);
                env.AssertValue(chain);
                _host = env.Register(nameof(Transformer));
                _chain = chain;
            }

            public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
            {
                _host.CheckValue(inputSchema, nameof(inputSchema));
                return _chain.GetOutputSchema(inputSchema);
            }

            public IDataView Transform(IDataView input)
            {
                _host.CheckValue(input, nameof(input));
                return _chain.Transform(input);
            }

            bool ITransformer.IsRowToRowMapper => true;

            IRowToRowMapper ITransformer.GetRowToRowMapper(DataViewSchema inputSchema)
            {
                _host.CheckValue(inputSchema, nameof(inputSchema));
                return (_chain as ITransformer).GetRowToRowMapper(inputSchema);
            }

            void ICanSaveModel.Save(ModelSaveContext ctx)
            {
                _host.CheckValue(ctx, nameof(ctx));
                ctx.CheckAtModel();
                ctx.SetVersionInfo(GetVersionInfo());

                ctx.SaveModel(_chain, "Chain");
            }

            public Transformer(IHostEnvironment env, ModelLoadContext ctx)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register(nameof(Transformer));
                _host.CheckValue(ctx, nameof(ctx));

                ctx.CheckAtModel(GetVersionInfo());

                if (ctx.Header.ModelVerReadable == VerIDataTransform)
                {
                    int n = ctx.Reader.ReadInt32();
                    _chain = new TransformerChain<ITransformer>();
                    ctx.LoadModel<ILegacyDataLoader, SignatureLoadDataLoader>(env, out var loader, "Loader", new MultiFileSource(null));
                    IDataView data = loader;
                    for (int i = 0; i < n; i++)
                    {
                        var dirName = string.Format(TransformDirTemplate, i);
                        ITransformer transformer;
                        // Try to load as an ITransformer.
                        try
                        {
                            ctx.LoadModelOrNull<ITransformer, SignatureLoadModel>(env, out transformer, dirName);
                        }
                        catch (FormatException)
                        {
                            transformer = null;
                        }

                        // If that didn't work, this should be a RowToRowMapperTransform with a "Mapper" folder in it containing an ITransformer.
                        var mapperDirName = Path.Combine(dirName, "Mapper");
                        if (transformer == null && ctx.ContainsModel(mapperDirName))
                            ctx.LoadModelOrNull<ITransformer, SignatureLoadModel>(env, out transformer, mapperDirName);

                        if (transformer != null)
                            data = transformer.Transform(data);
                        else
                        {
                            ctx.LoadModel<IDataTransform, SignatureLoadDataTransform>(env, out var xf, dirName, data);
                            data = xf;
                            transformer = new TransformWrapper(_host, xf);
                        }
                        _chain = _chain.Append(transformer);
                    }
                }
                else
                    ctx.LoadModel<TransformerChain<ITransformer>, SignatureLoadModel>(env, out _chain, "Chain");
            }

            private static VersionInfo GetVersionInfo()
            {
                return new VersionInfo(
                    modelSignature: "TEXT XFR",
                    //verWrittenCur: 0x00010001, // Initial
                    verWrittenCur: 0x00010002, // Save as TransformerChain instead of an array of IDataTransform
                    verReadableCur: 0x00010002,
                    verWeCanReadBack: 0x00010001,
                    loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(Transformer).Assembly.FullName);
            }
        }
    }
}
