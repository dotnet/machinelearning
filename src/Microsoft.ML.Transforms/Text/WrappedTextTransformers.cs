// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.TextAnalytics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static Microsoft.ML.Runtime.TextAnalytics.StopWordsRemoverTransform;
using static Microsoft.ML.Runtime.TextAnalytics.TextNormalizerTransform;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Word tokenizer splits text into tokens using the delimiter.
    /// For each text input, the output column is a variable vector of text.
    /// </summary>
    public sealed class WordTokenizer : TrivialWrapperEstimator
    {
        /// <summary>
        /// Tokenize incoming text in <paramref name="inputColumn"/> and output the tokens as <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to tokenize.</param>
        /// <param name="outputColumn">The column containing output tokens. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="separators">The separators to use (comma separated).</param>
        public WordTokenizer(IHostEnvironment env, string inputColumn, string outputColumn = null, string separators = "space")
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, separators)
        {
        }

        /// <summary>
        /// Tokenize incoming text in input columns and output the tokens as output columns.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        /// <param name="separators">The separators to use (comma separated).</param>
        public WordTokenizer(IHostEnvironment env, (string input, string output)[] columns, string separators = "space")
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(WordTokenizer)), MakeTransformer(env, columns, separators))
        {
        }

        private static TransformWrapper MakeTransformer(IHostEnvironment env, (string input, string output)[] columns, string separators)
        {
            Contracts.AssertValue(env);
            env.CheckNonEmpty(columns, nameof(columns));
            foreach (var (input, output) in columns)
            {
                env.CheckValue(input, nameof(input));
                env.CheckValue(output, nameof(input));
            }

            // Create arguments.
            // REVIEW: enable multiple separators via something other than parsing strings.
            var args = new DelimitedTokenizeTransform.Arguments
            {
                Column = columns.Select(x => new DelimitedTokenizeTransform.Column { Source = x.input, Name = x.output }).ToArray(),
                TermSeparators = separators
            };

            // Create a valid instance of data.
            var schema = new SimpleSchema(env, columns.Select(x => new KeyValuePair<string, ColumnType>(x.input, TextType.Instance)).ToArray());
            var emptyData = new EmptyDataView(env, schema);

            return new TransformWrapper(env, new DelimitedTokenizeTransform(env, args, emptyData));
        }
    }

    /// <summary>
    /// Character tokenizer splits text into sequences of characters using a sliding window.
    /// </summary>
    public sealed class CharacterTokenizer : TrivialWrapperEstimator
    {
        /// <summary>
        /// Tokenize incoming text in <paramref name="inputColumn"/> and output the tokens as <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to tokenize.</param>
        /// <param name="outputColumn">The column containing output tokens. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="useMarkerCharacters">Whether to use marker characters to separate words.</param>
        public CharacterTokenizer(IHostEnvironment env, string inputColumn, string outputColumn = null, bool useMarkerCharacters = true)
            : this (env, new[] { (inputColumn, outputColumn ?? inputColumn) }, useMarkerCharacters)
        {
        }

        /// <summary>
        /// Tokenize incoming text in input columns and output the tokens as output columns.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        /// <param name="useMarkerCharacters">Whether to use marker characters to separate words.</param>
        public CharacterTokenizer(IHostEnvironment env, (string input, string output)[] columns, bool useMarkerCharacters = true)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(CharacterTokenizer)), MakeTransformer(env, columns, useMarkerCharacters))
        {
        }

        private static TransformWrapper MakeTransformer(IHostEnvironment env, (string input, string output)[] columns, bool useMarkerChars)
        {
            Contracts.AssertValue(env);
            env.CheckNonEmpty(columns, nameof(columns));
            foreach (var (input, output) in columns)
            {
                env.CheckValue(input, nameof(input));
                env.CheckValue(output, nameof(input));
            }

            // Create arguments.
            var args = new CharTokenizeTransform.Arguments
            {
                Column = columns.Select(x => new CharTokenizeTransform.Column { Source = x.input, Name = x.output }).ToArray(),
                UseMarkerChars = useMarkerChars
            };

            // Create a valid instance of data.
            var schema = new SimpleSchema(env, columns.Select(x => new KeyValuePair<string, ColumnType>(x.input, TextType.Instance)).ToArray());
            var emptyData = new EmptyDataView(env, schema);

            return new TransformWrapper(env, new CharTokenizeTransform(env, args, emptyData));
        }
    }

    /// <summary>
    /// Stopword remover removes language-specific lists of stop words (most common words)
    /// This is usually applied after tokenizing text, so it compares individual tokens
    /// (case-insensitive comparison) to the stopwords.
    /// </summary>
    public sealed class StopwordRemover : TrivialWrapperEstimator
    {
        /// <summary>
        /// Removes stop words from incoming token streams in <paramref name="inputColumn"/>
        /// and outputs the token streams without stopwords as <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to remove stop words on.</param>
        /// <param name="outputColumn">The column containing output text. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="language">Langauge of the input text column <paramref name="inputColumn"/>.</param>
        public StopwordRemover(IHostEnvironment env, string inputColumn, string outputColumn = null, Language language = Language.English)
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, language)
        {
        }

        /// <summary>
        /// Removes stop words from incoming token streams in input columns
        /// and outputs the token streams without stop words as output columns.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to remove stop words on.</param>
        /// <param name="language">Langauge of the input text columns <paramref name="columns"/>.</param>
        public StopwordRemover(IHostEnvironment env, (string input, string output)[] columns, Language language = Language.English)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(StopwordRemover)), MakeTransformer(env, columns, language))
        {
        }

        private static TransformWrapper MakeTransformer(IHostEnvironment env, (string input, string output)[] columns, Language language)
        {
            Contracts.AssertValue(env);
            env.CheckNonEmpty(columns, nameof(columns));
            foreach (var (input, output) in columns)
            {
                env.CheckValue(input, nameof(input));
                env.CheckValue(output, nameof(input));
            }

            // Create arguments.
            var args = new StopWordsRemoverTransform.Arguments
            {
                Column = columns.Select(x => new StopWordsRemoverTransform.Column { Source = x.input, Name = x.output }).ToArray(),
                Language = language
            };

            // Create a valid instance of data.
            var schema = new SimpleSchema(env, columns.Select(x => new KeyValuePair<string, ColumnType>(x.input, new VectorType(TextType.Instance))).ToArray());
            var emptyData = new EmptyDataView(env, schema);

            return new TransformWrapper(env, new StopWordsRemoverTransform(env, args, emptyData));
        }
    }

    /// <summary>
    /// Text normalizer allows normalizing text by changing case (Upper/Lower case), removing diacritical marks, punctuation marks and/or numbers.
    /// </summary>
    public sealed class TextNormalizer : TrivialWrapperEstimator
    {
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
        public TextNormalizer(IHostEnvironment env,
            string inputColumn,
            string outputColumn = null,
            CaseNormalizationMode textCase = CaseNormalizationMode.Lower,
            bool keepDiacritics = false,
            bool keepPunctuations = true,
            bool keepNumbers = true)
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, textCase, keepDiacritics, keepPunctuations, keepNumbers)
        {
        }

        /// <summary>
        /// Normalizes incoming text in input columns by changing case, removing diacritical marks, punctuation marks and/or numbers
        /// and outputs new text as output columns.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the text normalization on.</param>
        /// <param name="textCase">Casing text using the rules of the invariant culture.</param>
        /// <param name="keepDiacritics">Whether to keep diacritical marks or remove them.</param>
        /// <param name="keepPunctuations">Whether to keep punctuation marks or remove them.</param>
        /// <param name="keepNumbers">Whether to keep numbers or remove them.</param>
        public TextNormalizer(IHostEnvironment env,
            (string input, string output)[] columns,
            CaseNormalizationMode textCase = CaseNormalizationMode.Lower,
            bool keepDiacritics = false,
            bool keepPunctuations = true,
            bool keepNumbers = true)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TextNormalizer)),
                  MakeTransformer(env, columns, textCase, keepDiacritics, keepPunctuations, keepNumbers))
        {
        }

        private static TransformWrapper MakeTransformer(IHostEnvironment env,
            (string input, string output)[] columns,
            CaseNormalizationMode textCase,
            bool keepDiacritics,
            bool keepPunctuations,
            bool keepNumbers)
        {
            Contracts.AssertValue(env);
            env.CheckNonEmpty(columns, nameof(columns));
            foreach (var (input, output) in columns)
            {
                env.CheckValue(input, nameof(input));
                env.CheckValue(output, nameof(input));
            }

            // Create arguments.
            var args = new TextNormalizerTransform.Arguments
            {
                Column = columns.Select(x => new TextNormalizerTransform.Column { Source = x.input, Name = x.output }).ToArray(),
                TextCase = textCase,
                KeepDiacritics = keepDiacritics,
                KeepPunctuations = keepPunctuations,
                KeepNumbers = keepNumbers
            };

            // Create a valid instance of data.
            var schema = new SimpleSchema(env, columns.Select(x => new KeyValuePair<string, ColumnType>(x.input, TextType.Instance)).ToArray());
            var emptyData = new EmptyDataView(env, schema);

            return new TransformWrapper(env, new TextNormalizerTransform(env, args, emptyData));
        }
    }

    /// <summary>
    /// Produces a bag of counts of ngrams (sequences of consecutive words ) in a given text.
    /// It does so by building a dictionary of ngrams and using the id in the dictionary as the index in the bag.
    /// </summary>
    public sealed class WordBagEstimator : TrainedWrapperEstimatorBase
    {
        private readonly (string[] inputs, string output)[] _columns;
        private readonly int _ngramLength;
        private readonly int _skipLength;
        private readonly bool _allLengths;
        private readonly int _maxNumTerms;
        private readonly NgramTransform.WeightingCriteria _weighting;

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words ) in <paramref name="inputColumn"/>
        /// and outputs bag of word vector as <paramref name="outputColumn"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to compute bag of word vector.</param>
        /// <param name="outputColumn">The column containing bag of word vector. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public WordBagEstimator(IHostEnvironment env,
            string inputColumn,
            string outputColumn = null,
            int ngramLength = 1,
            int skipLength=0,
            bool allLengths = true,
            int maxNumTerms = 10000000,
            NgramTransform.WeightingCriteria weighting = NgramTransform.WeightingCriteria.Tf)
            : this(env, new[] { ( new[] { inputColumn }, outputColumn ?? inputColumn) }, ngramLength, skipLength, allLengths, maxNumTerms, weighting)
        {
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words ) in <paramref name="inputColumns"/>
        /// and outputs bag of word vector as <paramref name="outputColumn"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumns">The columns containing text to compute bag of word vector.</param>
        /// <param name="outputColumn">The column containing output tokens.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public WordBagEstimator(IHostEnvironment env,
            string[] inputColumns,
            string outputColumn,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            int maxNumTerms = 10000000,
            NgramTransform.WeightingCriteria weighting = NgramTransform.WeightingCriteria.Tf)
            : this(env, new[] { (inputColumns, outputColumn) }, ngramLength, skipLength, allLengths, maxNumTerms, weighting)
        {
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words ) in <paramref name="columns.inputs"/>
        /// and outputs bag of word vector for each output in <paramref name="columns.output"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to compute bag of word vector.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public WordBagEstimator(IHostEnvironment env,
            (string[] inputs, string output)[] columns,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            int maxNumTerms = 10000000,
            NgramTransform.WeightingCriteria weighting = NgramTransform.WeightingCriteria.Tf)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(WordBagEstimator)))
        {
            foreach (var (input, output) in columns)
            {
                Host.CheckUserArg(Utils.Size(input) > 0, nameof(input));
                Host.CheckValue(output, nameof(input));
            }

            _columns = columns;
            _ngramLength = ngramLength;
            _skipLength = skipLength;
            _allLengths = allLengths;
            _maxNumTerms = maxNumTerms;
            _weighting = weighting;
        }

        public override TransformWrapper Fit(IDataView input)
        {
            // Create arguments.
            var args = new WordBagTransform.Arguments
            {
                Column = _columns.Select(x => new WordBagTransform.Column { Source = x.inputs, Name = x.output }).ToArray(),
                NgramLength = _ngramLength,
                SkipLength = _skipLength,
                AllLengths = _allLengths,
                MaxNumTerms = new[] { _maxNumTerms },
                Weighting = _weighting
            };

            return new TransformWrapper(Host, WordBagTransform.Create(Host, args, input));
        }
    }

    /// <summary>
    /// Produces a bag of counts of ngrams (sequences of consecutive words of length 1-n) in a given text.
    /// It does so by hashing each ngram and using the hash value as the index in the bag.
    /// </summary>
    public sealed class WordHashBagEstimator : TrainedWrapperEstimatorBase
    {
        private readonly (string[] inputs, string output)[] _columns;
        private readonly int _hashBits;
        private readonly int _ngramLength;
        private readonly int _skipLength;
        private readonly bool _allLengths;
        private readonly uint _seed;
        private readonly bool _ordered;
        private readonly int _invertHash;

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="inputColumn"/>
        /// and outputs bag of word vector as <paramref name="outputColumn"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to compute bag of word vector.</param>
        /// <param name="outputColumn">The column containing bag of word vector. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        public WordHashBagEstimator(IHostEnvironment env,
            string inputColumn,
            string outputColumn = null,
            int hashBits = 16,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            uint seed = 314489979,
            bool ordered = true,
            int invertHash = 0)
            : this(env, new[] { (new[] { inputColumn }, outputColumn ?? inputColumn) }, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash)
        {
        }

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="inputColumns"/>
        /// and outputs bag of word vector as <paramref name="outputColumn"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumns">The columns containing text to compute bag of word vector.</param>
        /// <param name="outputColumn">The column containing output tokens.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        public WordHashBagEstimator(IHostEnvironment env,
            string[] inputColumns,
            string outputColumn,
            int hashBits = 16,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            uint seed = 314489979,
            bool ordered = true,
            int invertHash = 0)
            : this(env, new[] { (inputColumns, outputColumn) }, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash)
        {
        }

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="columns.inputs"/>
        /// and outputs bag of word vector for each output in <paramref name="columns.output"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to compute bag of word vector.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        public WordHashBagEstimator(IHostEnvironment env,
            (string[] inputs, string output)[] columns,
            int hashBits = 16,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            uint seed = 314489979,
            bool ordered = true,
            int invertHash = 0)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(WordBagEstimator)))
        {
            foreach (var (input, output) in columns)
            {
                Host.CheckUserArg(Utils.Size(input) > 0, nameof(input));
                Host.CheckValue(output, nameof(input));
            }

            _columns = columns;
            _hashBits = hashBits;
            _ngramLength = ngramLength;
            _skipLength = skipLength;
            _allLengths = allLengths;
            _seed = seed;
            _ordered = ordered;
            _invertHash = invertHash;
        }

        public override TransformWrapper Fit(IDataView input)
        {
            // Create arguments.
            var args = new WordHashBagTransform.Arguments
            {
                Column = _columns.Select(x => new WordHashBagTransform.Column { Source = x.inputs, Name = x.output }).ToArray(),
                HashBits = _hashBits,
                NgramLength = _ngramLength,
                SkipLength = _skipLength,
                AllLengths = _allLengths,
                Seed = _seed,
                Ordered = _ordered,
                InvertHash = _invertHash
            };

            return new TransformWrapper(Host, WordHashBagTransform.Create(Host, args, input));
        }
    }

    /// <summary>
    /// Produces a bag of counts of ngrams(sequences of consecutive values of length 1-n) in a given vector of keys.
    /// It does so by building a dictionary of ngrams and using the id in the dictionary as the index in the bag.
    /// </summary>
    public sealed class NgramEstimator : TrainedWrapperEstimatorBase
    {
        private readonly (string inputs, string output)[] _columns;
        private readonly int _ngramLength;
        private readonly int _skipLength;
        private readonly bool _allLengths;
        private readonly int _maxNumTerms;
        private readonly NgramTransform.WeightingCriteria _weighting;

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words ) in <paramref name="inputColumn"/>
        /// and outputs bag of word vector as <paramref name="outputColumn"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to compute bag of word vector.</param>
        /// <param name="outputColumn">The column containing bag of word vector. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public NgramEstimator(IHostEnvironment env,
            string inputColumn,
            string outputColumn = null,
            int ngramLength = 2,
            int skipLength = 0,
            bool allLengths = true,
            int maxNumTerms = 10000000,
            NgramTransform.WeightingCriteria weighting = NgramTransform.WeightingCriteria.Tf)
            : this(env, new[] { ( inputColumn, outputColumn ?? inputColumn) }, ngramLength, skipLength, allLengths, maxNumTerms, weighting)
        {
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words ) in <paramref name="columns.inputs"/>
        /// and outputs bag of word vector for each output in <paramref name="columns.output"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to compute bag of word vector.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public NgramEstimator(IHostEnvironment env,
            (string inputs, string output)[] columns,
            int ngramLength = 2,
            int skipLength = 0,
            bool allLengths = true,
            int maxNumTerms = 10000000,
            NgramTransform.WeightingCriteria weighting = NgramTransform.WeightingCriteria.Tf)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(WordBagEstimator)))
        {
            foreach (var (input, output) in columns)
            {
                Host.CheckUserArg(Utils.Size(input) > 0, nameof(input));
                Host.CheckValue(output, nameof(input));
            }

            _columns = columns;
            _ngramLength = ngramLength;
            _skipLength = skipLength;
            _allLengths = allLengths;
            _maxNumTerms = maxNumTerms;
            _weighting = weighting;
        }

        public override TransformWrapper Fit(IDataView input)
        {
            // Create arguments.
            var args = new NgramTransform.Arguments
            {
                Column = _columns.Select(x => new NgramTransform.Column { Source = x.inputs, Name = x.output }).ToArray(),
                NgramLength = _ngramLength,
                SkipLength = _skipLength,
                AllLengths = _allLengths,
                MaxNumTerms = new[] { _maxNumTerms },
                Weighting = _weighting
            };

            return new TransformWrapper(Host, new NgramTransform(Host, args, input));
        }
    }

    /// <summary>
    /// Produces a bag of counts of ngrams (sequences of consecutive words of length 1-n) in a given text.
    /// It does so by hashing each ngram and using the hash value as the index in the bag.
    ///
    /// <see cref="NgramHashEstimator"/> is different from <see cref="WordHashBagEstimator"/> in a way that <see cref="NgramHashEstimator"/>
    /// takes tokenized text as input while <see cref="WordHashBagEstimator"/> tokenizes text internally.
    /// </summary>
    public sealed class NgramHashEstimator : TrainedWrapperEstimatorBase
    {
        private readonly (string[] inputs, string output)[] _columns;
        private readonly int _hashBits;
        private readonly int _ngramLength;
        private readonly int _skipLength;
        private readonly bool _allLengths;
        private readonly uint _seed;
        private readonly bool _ordered;
        private readonly int _invertHash;

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="inputColumn"/>
        /// and outputs ngram vector as <paramref name="outputColumn"/>
        ///
        /// <see cref="NgramHashEstimator"/> is different from <see cref="WordHashBagEstimator"/> in a way that <see cref="NgramHashEstimator"/>
        /// takes tokenized text as input while <see cref="WordHashBagEstimator"/> tokenizes text internally.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to compute bag of word vector.</param>
        /// <param name="outputColumn">The column containing bag of word vector. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        public NgramHashEstimator(IHostEnvironment env,
            string inputColumn,
            string outputColumn = null,
            int hashBits = 16,
            int ngramLength = 2,
            int skipLength = 0,
            bool allLengths = true,
            uint seed = 314489979,
            bool ordered = true,
            int invertHash = 0)
            : this(env, new[] { (new[] { inputColumn }, outputColumn ?? inputColumn) }, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash)
        {
        }

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="inputColumns"/>
        /// and outputs ngram vector as <paramref name="outputColumn"/>
        ///
        /// <see cref="NgramHashEstimator"/> is different from <see cref="WordHashBagEstimator"/> in a way that <see cref="NgramHashEstimator"/>
        /// takes tokenized text as input while <see cref="WordHashBagEstimator"/> tokenizes text internally.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumns">The columns containing text to compute bag of word vector.</param>
        /// <param name="outputColumn">The column containing output tokens.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        public NgramHashEstimator(IHostEnvironment env,
            string[] inputColumns,
            string outputColumn,
            int hashBits = 16,
            int ngramLength = 2,
            int skipLength = 0,
            bool allLengths = true,
            uint seed = 314489979,
            bool ordered = true,
            int invertHash = 0)
            : this(env, new[] { (inputColumns, outputColumn) }, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash)
        {
        }

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="columns.inputs"/>
        /// and outputs ngram vector for each output in <paramref name="columns.output"/>
        ///
        /// <see cref="NgramHashEstimator"/> is different from <see cref="WordHashBagEstimator"/> in a way that <see cref="NgramHashEstimator"/>
        /// takes tokenized text as input while <see cref="WordHashBagEstimator"/> tokenizes text internally.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to compute bag of word vector.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        public NgramHashEstimator(IHostEnvironment env,
            (string[] inputs, string output)[] columns,
            int hashBits = 16,
            int ngramLength = 2,
            int skipLength = 0,
            bool allLengths = true,
            uint seed = 314489979,
            bool ordered = true,
            int invertHash = 0)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(WordBagEstimator)))
        {
            foreach (var (input, output) in columns)
            {
                Host.CheckUserArg(Utils.Size(input) > 0, nameof(input));
                Host.CheckValue(output, nameof(input));
            }

            _columns = columns;
            _hashBits = hashBits;
            _ngramLength = ngramLength;
            _skipLength = skipLength;
            _allLengths = allLengths;
            _seed = seed;
            _ordered = ordered;
            _invertHash = invertHash;
        }

        public override TransformWrapper Fit(IDataView input)
        {
            // Create arguments.
            var args = new NgramHashTransform.Arguments
            {
                Column = _columns.Select(x => new NgramHashTransform.Column { Source = x.inputs, Name = x.output }).ToArray(),
                HashBits = _hashBits,
                NgramLength = _ngramLength,
                SkipLength = _skipLength,
                AllLengths = _allLengths,
                Seed = _seed,
                Ordered = _ordered,
                InvertHash = _invertHash
            };

            return new TransformWrapper(Host, new NgramHashTransform(Host, args, input));
        }
    }
}
