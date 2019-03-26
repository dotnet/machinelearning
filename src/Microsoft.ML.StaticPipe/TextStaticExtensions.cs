// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Text;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// Extensions for statically typed word tokenizer.
    /// </summary>
    public static class WordTokenizerStaticExtensions
    {
        private sealed class OutPipelineColumn : VarVector<string>
        {
            public readonly Scalar<string> Input;

            public OutPipelineColumn(Scalar<string> input, char[] separators)
                : base(new Reconciler(separators), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly char[] _separators;

            public Reconciler(char[] separators)
            {
                _separators = separators;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string outputColumnName, string inputColumnName)>();
                foreach (var outCol in toOutput)
                    pairs.Add((outputNames[outCol], inputNames[((OutPipelineColumn)outCol).Input]));

                return new WordTokenizingEstimator(env, pairs.ToArray(), _separators);
            }
        }

        /// <summary>
        /// Tokenize incoming text using <paramref name="separators"/> and output the tokens.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="separators">The separators to use (uses space character by default).</param>
        public static VarVector<string> TokenizeIntoWords(this Scalar<string> input, char[] separators = null) => new OutPipelineColumn(input, separators);
    }

    /// <summary>
    /// Extensions for statically typed character tokenizer.
    /// </summary>
    public static class CharacterTokenizerStaticExtensions
    {
        private sealed class OutPipelineColumn : VarVector<Key<ushort, string>>
        {
            public readonly Scalar<string> Input;

            public OutPipelineColumn(Scalar<string> input, bool useMarkerChars)
                : base(new Reconciler(useMarkerChars), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler, IEquatable<Reconciler>
        {
            private readonly bool _useMarker;

            public Reconciler(bool useMarkerChars)
            {
                _useMarker = useMarkerChars;
            }

            public bool Equals(Reconciler other)
            {
                return _useMarker == other._useMarker;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string outputColumnName, string inputColumnName)>();
                foreach (var outCol in toOutput)
                    pairs.Add((outputNames[outCol], inputNames[((OutPipelineColumn)outCol).Input]));

                return new TokenizingByCharactersEstimator(env, _useMarker, pairs.ToArray());
            }
        }

        /// <summary>
        /// Tokenize incoming text into a sequence of characters.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="useMarkerCharacters">Whether to use marker characters to separate words.</param>
        public static VarVector<Key<ushort, string>> TokenizeIntoCharactersAsKeys(this Scalar<string> input, bool useMarkerCharacters = true) => new OutPipelineColumn(input, useMarkerCharacters);
    }

    /// <summary>
    /// Extensions for statically typed stop word remover.
    /// </summary>
    public static class StopwordRemoverStaticExtensions
    {
        private sealed class OutPipelineColumn : VarVector<string>
        {
            public readonly VarVector<string> Input;

            public OutPipelineColumn(VarVector<string> input, StopWordsRemovingEstimator.Language language)
                : base(new Reconciler(language), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler, IEquatable<Reconciler>
        {
            private readonly StopWordsRemovingEstimator.Language _language;

            public Reconciler(StopWordsRemovingEstimator.Language language)
            {
                _language = language;
            }

            public bool Equals(Reconciler other)
            {
                return _language == other._language;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var columns = new List<StopWordsRemovingEstimator.ColumnOptions>();
                foreach (var outCol in toOutput)
                    columns.Add(new StopWordsRemovingEstimator.ColumnOptions(outputNames[outCol], inputNames[((OutPipelineColumn)outCol).Input], _language));

                return new StopWordsRemovingEstimator(env, columns.ToArray());
            }
        }

        /// <summary>
        /// Remove stop words from incoming text.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="language">Langauge of the input text. It will be used to retrieve a built-in stopword list.</param>
        public static VarVector<string> RemoveDefaultStopWords(this VarVector<string> input,
            StopWordsRemovingEstimator.Language language = StopWordsRemovingEstimator.Language.English) => new OutPipelineColumn(input, language);
    }

    /// <summary>
    /// Extensions for statically typed text normalizer.
    /// </summary>
    public static class TextNormalizerStaticExtensions
    {
        private sealed class OutPipelineColumn : Scalar<string>
        {
            public readonly Scalar<string> Input;

            public OutPipelineColumn(Scalar<string> input, TextNormalizingEstimator.CaseMode textCase, bool keepDiacritics, bool keepPunctuations, bool keepNumbers)
                : base(new Reconciler(textCase, keepDiacritics, keepPunctuations, keepNumbers), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler, IEquatable<Reconciler>
        {
            private readonly TextNormalizingEstimator.CaseMode _textCase;
            private readonly bool _keepDiacritics;
            private readonly bool _keepPunctuations;
            private readonly bool _keepNumbers;

            public Reconciler(TextNormalizingEstimator.CaseMode textCase, bool keepDiacritics, bool keepPunctuations, bool keepNumbers)
            {
                _textCase = textCase;
                _keepDiacritics = keepDiacritics;
                _keepPunctuations = keepPunctuations;
                _keepNumbers = keepNumbers;

            }

            public bool Equals(Reconciler other)
            {
                return _textCase == other._textCase &&
                _keepDiacritics == other._keepDiacritics &&
                _keepPunctuations == other._keepPunctuations &&
                _keepNumbers == other._keepNumbers;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string outputColumnName, string inputColumnName)>();
                foreach (var outCol in toOutput)
                    pairs.Add((outputNames[outCol], inputNames[((OutPipelineColumn)outCol).Input]));

                return new TextNormalizingEstimator(env, _textCase, _keepDiacritics, _keepPunctuations, _keepNumbers, pairs.ToArray());
            }
        }

        /// <summary>
        /// Normalizes input text by changing case, removing diacritical marks, punctuation marks and/or numbers.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="caseMode">Casing text using the rules of the invariant culture.</param>
        /// <param name="keepDiacritics">Whether to keep diacritical marks or remove them.</param>
        /// <param name="keepPunctuations">Whether to keep punctuation marks or remove them.</param>
        /// <param name="keepNumbers">Whether to keep numbers or remove them.</param>
        public static Scalar<string> NormalizeText(this Scalar<string> input,
            TextNormalizingEstimator.CaseMode caseMode = TextNormalizingEstimator.CaseMode.Lower,
            bool keepDiacritics = false,
            bool keepPunctuations = true,
            bool keepNumbers = true) => new OutPipelineColumn(input, caseMode, keepDiacritics, keepPunctuations, keepNumbers);
    }

    /// <summary>
    /// Extensions for statically typed bag of word converter.
    /// </summary>
    public static class WordBagEstimatorStaticExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Scalar<string> Input;

            public OutPipelineColumn(Scalar<string> input,
                int ngramLength,
                int skipLength,
                bool allLengths,
                int maxNumTerms,
                NgramExtractingEstimator.WeightingCriteria weighting)
                : base(new Reconciler(ngramLength, skipLength, allLengths, maxNumTerms, weighting), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler, IEquatable<Reconciler>
        {
            private readonly int _ngramLength;
            private readonly int _skipLength;
            private readonly bool _useAllLengths;
            private readonly int _maxNumTerms;
            private readonly NgramExtractingEstimator.WeightingCriteria _weighting;

            public Reconciler(int ngramLength, int skipLength, bool allLengths, int maxNumTerms, NgramExtractingEstimator.WeightingCriteria weighting)
            {
                _ngramLength = ngramLength;
                _skipLength = skipLength;
                _useAllLengths = allLengths;
                _maxNumTerms = maxNumTerms;
                _weighting = weighting;

            }

            public bool Equals(Reconciler other)
            {
                return _ngramLength == other._ngramLength &&
                _skipLength == other._skipLength &&
                _useAllLengths == other._useAllLengths &&
                _maxNumTerms == other._maxNumTerms &&
                _weighting == other._weighting;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string names, string[] sources)>();
                foreach (var outCol in toOutput)
                    pairs.Add((outputNames[outCol], new[] { inputNames[((OutPipelineColumn)outCol).Input] }));

                return new WordBagEstimator(env, pairs.ToArray(), _ngramLength, _skipLength, _useAllLengths, _maxNumTerms, _weighting);
            }
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words ) in a given text.
        /// It does so by building a dictionary of ngrams and using the id in the dictionary as the index in the bag.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="useAllLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maximumNgramsCount">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public static Vector<float> ProduceWordBags(this Scalar<string> input,
            int ngramLength = 1,
            int skipLength = 0,
            bool useAllLengths = true,
            int maximumNgramsCount = 10000000,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.WeightingCriteria.Tf)
                => new OutPipelineColumn(input, ngramLength, skipLength, useAllLengths, maximumNgramsCount, weighting);
    }

    /// <summary>
    /// Extensions for statically typed bag of wordhash converter.
    /// </summary>
    public static class WordHashBagEstimatorStaticExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Scalar<string> Input;

            public OutPipelineColumn(Scalar<string> input,
                int numberOfBits,
                int ngramLength,
                int skipLength,
                bool useAllLengths,
                uint seed,
                bool useOrderedHashing,
                int maximumNumberOfInverts)
                : base(new Reconciler(numberOfBits, ngramLength, skipLength, useAllLengths, seed, useOrderedHashing, maximumNumberOfInverts), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler, IEquatable<Reconciler>
        {
            private readonly int _numberOfBits;
            private readonly int _ngramLength;
            private readonly int _skipLength;
            private readonly bool _useAllLengths;
            private readonly uint _seed;
            private readonly bool _useOrderedHashing;
            private readonly int _maximumNumberOfInverts;

            public Reconciler(int numberOfBits, int ngramLength, int skipLength, bool useAllLengths, uint seed, bool useOrderedHashing, int maximumNumberOfInverts)
            {
                _numberOfBits = numberOfBits;
                _ngramLength = ngramLength;
                _skipLength = skipLength;
                _useAllLengths = useAllLengths;
                _seed = seed;
                _useOrderedHashing = useOrderedHashing;
                _maximumNumberOfInverts = maximumNumberOfInverts;
            }

            public bool Equals(Reconciler other)
            {
                return _numberOfBits == other._numberOfBits &&
                    _ngramLength == other._ngramLength &&
                    _skipLength == other._skipLength &&
                    _useAllLengths == other._useAllLengths &&
                    _seed == other._seed &&
                    _useOrderedHashing == other._useOrderedHashing &&
                    _maximumNumberOfInverts == other._maximumNumberOfInverts;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string name, string[] sources)>();
                foreach (var outCol in toOutput)
                    pairs.Add((outputNames[outCol], new[] { inputNames[((OutPipelineColumn)outCol).Input] }));

                return new WordHashBagEstimator(env, pairs.ToArray(), _numberOfBits, _ngramLength, _skipLength, _useAllLengths, _seed, _useOrderedHashing, _maximumNumberOfInverts);
            }
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words of length 1-n) in a given text.
        /// It does so by hashing each ngram and using the hash value as the index in the bag.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="useAllLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="useOrderedHashing">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="maximumNumberOfInverts">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public static Vector<float> ProduceHashedWordBags(this Scalar<string> input,
            int numberOfBits = 16,
            int ngramLength = 1,
            int skipLength = 0,
            bool useAllLengths = true,
            uint seed = 314489979,
            bool useOrderedHashing = true,
            int maximumNumberOfInverts = 0) => new OutPipelineColumn(input, numberOfBits, ngramLength, skipLength, useAllLengths, seed, useOrderedHashing, maximumNumberOfInverts);
    }

    /// <summary>
    /// Extensions for statically typed ngram estimator.
    /// </summary>
    public static class NgramEstimatorStaticExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly PipelineColumn Input;

            public OutPipelineColumn(PipelineColumn input,
                int ngramLength,
                int skipLength,
                bool useAllLengths,
                int maxNumTerms,
                NgramExtractingEstimator.WeightingCriteria weighting)
                : base(new Reconciler(ngramLength, skipLength, useAllLengths, maxNumTerms, weighting), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler, IEquatable<Reconciler>
        {
            private readonly int _ngramLength;
            private readonly int _skipLength;
            private readonly bool _useAllLengths;
            private readonly int _maxNgramsCount;
            private readonly NgramExtractingEstimator.WeightingCriteria _weighting;

            public Reconciler(int ngramLength, int skipLength, bool useAllLengths, int maxNumTerms, NgramExtractingEstimator.WeightingCriteria weighting)
            {
                _ngramLength = ngramLength;
                _skipLength = skipLength;
                _useAllLengths = useAllLengths;
                _maxNgramsCount = maxNumTerms;
                _weighting = weighting;

            }

            public bool Equals(Reconciler other)
            {
                return _ngramLength == other._ngramLength &&
                _skipLength == other._skipLength &&
                _useAllLengths == other._useAllLengths &&
                _maxNgramsCount == other._maxNgramsCount &&
                _weighting == other._weighting;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string inputs, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((outputNames[outCol], inputNames[((OutPipelineColumn)outCol).Input]));

                return new NgramExtractingEstimator(env, pairs.ToArray(), _ngramLength, _skipLength, _useAllLengths, _maxNgramsCount, _weighting);
            }
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words ) in a given tokenized text.
        /// It does so by building a dictionary of ngrams and using the id in the dictionary as the index in the bag.
        ///
        /// /// <see cref="ProduceNgrams"/> is different from <see cref="WordBagEstimatorStaticExtensions.ProduceWordBags"/>
        /// in a way that <see cref="ProduceNgrams"/> takes tokenized text as input while <see cref="WordBagEstimatorStaticExtensions.ProduceWordBags"/> tokenizes text internally.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="useAllLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maximumNgramsCount">Maximum number of n-grams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public static Vector<float> ProduceNgrams<TKey>(this VarVector<Key<TKey, string>> input,
            int ngramLength = 1,
            int skipLength = 0,
            bool useAllLengths = true,
            int maximumNgramsCount = 10000000,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.WeightingCriteria.Tf)
                => new OutPipelineColumn(input, ngramLength, skipLength, useAllLengths, maximumNgramsCount, weighting);
    }

    /// <summary>
    /// Extensions for statically typed ngram hash estimator.
    /// </summary>
    public static class NgramHashEstimatorStaticExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly VarVector<Key<uint, string>> Input;

            public OutPipelineColumn(VarVector<Key<uint, string>> input, int numberOfBits, int ngramLength, int skipLength, bool useAllLengths, uint seed, bool useOrderedHashing, int maximumNumberOfInverts)
                : base(new Reconciler(numberOfBits, ngramLength, skipLength, useAllLengths, seed, useOrderedHashing, maximumNumberOfInverts), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler, IEquatable<Reconciler>
        {
            private readonly int _numberOfBits;
            private readonly int _ngramLength;
            private readonly int _skipLength;
            private readonly bool _useAllLengths;
            private readonly uint _seed;
            private readonly bool _useOrderedHashing;
            private readonly int _maximumNumberOfInverts;

            public Reconciler(int numberOfBits, int ngramLength, int skipLength, bool useAllLengths, uint seed, bool useOrderedHashing, int maximumNumberOfInverts)
            {
                _numberOfBits = numberOfBits;
                _ngramLength = ngramLength;
                _skipLength = skipLength;
                _useAllLengths = useAllLengths;
                _seed = seed;
                _useOrderedHashing = useOrderedHashing;
                _maximumNumberOfInverts = maximumNumberOfInverts;
            }

            public bool Equals(Reconciler other)
            {
                return _numberOfBits == other._numberOfBits &&
                    _ngramLength == other._ngramLength &&
                    _skipLength == other._skipLength &&
                    _useAllLengths == other._useAllLengths &&
                    _seed == other._seed &&
                    _useOrderedHashing == other._useOrderedHashing &&
                    _maximumNumberOfInverts == other._maximumNumberOfInverts;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);
                var columns = new List<NgramHashingEstimator.ColumnOptions>();
                foreach (var outCol in toOutput)
                    columns.Add(new NgramHashingEstimator.ColumnOptions(outputNames[outCol], new[] { inputNames[((OutPipelineColumn)outCol).Input] },
                          _ngramLength, _skipLength, _useAllLengths, _numberOfBits, _seed, _useOrderedHashing, _maximumNumberOfInverts));

                return new NgramHashingEstimator(env, columns.ToArray());
            }
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words of length 1-n) in a given tokenized text.
        /// It does so by hashing each ngram and using the hash value as the index in the bag.
        ///
        /// <see cref="ProduceHashedNgrams"/> is different from <see cref="WordHashBagEstimatorStaticExtensions.ProduceHashedWordBags"/>
        /// in a way that <see cref="ProduceHashedNgrams"/> takes tokenized text as input while <see cref="WordHashBagEstimatorStaticExtensions.ProduceHashedWordBags"/> tokenizes text internally.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="numberOfBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="useAllLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="useOrderedHashing">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="maximumNumberOfInverts">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="maximumNumberOfInverts"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public static Vector<float> ProduceHashedNgrams(this VarVector<Key<uint, string>> input,
            int numberOfBits = 16,
            int ngramLength = 2,
            int skipLength = 0,
            bool useAllLengths = true,
            uint seed = 314489979,
            bool useOrderedHashing = true,
            int maximumNumberOfInverts = 0) => new OutPipelineColumn(input, numberOfBits, ngramLength, skipLength, useAllLengths, seed, useOrderedHashing, maximumNumberOfInverts);
    }
}
