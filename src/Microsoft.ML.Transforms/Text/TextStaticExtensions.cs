// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using System;
using System.Collections.Generic;
using static Microsoft.ML.Transforms.Text.StopWordsRemovingTransformer;

namespace Microsoft.ML.Transforms.Text
{
    /// <summary>
    /// Extensions for statically typed word tokenizer.
    /// </summary>
    public static class WordTokenizerExtensions
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

                var pairs = new List<(string input, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumn)outCol).Input], outputNames[outCol]));

                return new WordTokenizingEstimator(env, pairs.ToArray(), _separators);
            }
        }

        /// <summary>
        /// Tokenize incoming text using <paramref name="separators"/> and output the tokens.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="separators">The separators to use (uses space character by default).</param>
        public static VarVector<string> TokenizeText(this Scalar<string> input, char[] separators = null) => new OutPipelineColumn(input, separators);
    }

    /// <summary>
    /// Extensions for statically typed character tokenizer.
    /// </summary>
    public static class CharacterTokenizerExtensions
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

                var pairs = new List<(string input, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumn)outCol).Input], outputNames[outCol]));

                return new CharacterTokenizingEstimator(env, _useMarker, pairs.ToArray());
            }
        }

        /// <summary>
        /// Tokenize incoming text into a sequence of characters.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="useMarkerCharacters">Whether to use marker characters to separate words.</param>
        public static VarVector<Key<ushort, string>> TokenizeIntoCharacters(this Scalar<string> input, bool useMarkerCharacters = true) => new OutPipelineColumn(input, useMarkerCharacters);
    }

    /// <summary>
    /// Extensions for statically typed stop word remover.
    /// </summary>
    public static class StopwordRemoverExtensions
    {
        private sealed class OutPipelineColumn : VarVector<string>
        {
            public readonly VarVector<string> Input;

            public OutPipelineColumn(VarVector<string> input, Language language)
                : base(new Reconciler(language), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler, IEquatable<Reconciler>
        {
            private readonly Language _language;

            public Reconciler(Language language)
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

                var pairs = new List<(string input, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumn)outCol).Input], outputNames[outCol]));

                return new StopwordRemover(env, pairs.ToArray(), _language);
            }
        }

        /// <summary>
        /// Remove stop words from incoming text.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="language">Langauge of the input text.</param>
        public static VarVector<string> RemoveStopwords(this VarVector<string> input,
            Language language = Language.English) => new OutPipelineColumn(input, language);
    }

    /// <summary>
    /// Extensions for statically typed text normalizer.
    /// </summary>
    public static class TextNormalizerExtensions
    {
        private sealed class OutPipelineColumn : Scalar<string>
        {
            public readonly Scalar<string> Input;

            public OutPipelineColumn(Scalar<string> input, TextNormalizingEstimator.CaseNormalizationMode textCase, bool keepDiacritics, bool keepPunctuations, bool keepNumbers)
                : base(new Reconciler(textCase, keepDiacritics, keepPunctuations, keepNumbers), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler, IEquatable<Reconciler>
        {
            private readonly TextNormalizingEstimator.CaseNormalizationMode _textCase;
            private readonly bool _keepDiacritics;
            private readonly bool _keepPunctuations;
            private readonly bool _keepNumbers;

            public Reconciler(TextNormalizingEstimator.CaseNormalizationMode textCase, bool keepDiacritics, bool keepPunctuations, bool keepNumbers)
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

                var pairs = new List<(string input, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumn)outCol).Input], outputNames[outCol]));

                return new TextNormalizingEstimator(env, _textCase, _keepDiacritics, _keepPunctuations, _keepNumbers, pairs.ToArray());
            }
        }

        /// <summary>
        /// Normalizes input text by changing case, removing diacritical marks, punctuation marks and/or numbers.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="textCase">Casing text using the rules of the invariant culture.</param>
        /// <param name="keepDiacritics">Whether to keep diacritical marks or remove them.</param>
        /// <param name="keepPunctuations">Whether to keep punctuation marks or remove them.</param>
        /// <param name="keepNumbers">Whether to keep numbers or remove them.</param>
        public static Scalar<string> NormalizeText(this Scalar<string> input,
            TextNormalizingEstimator.CaseNormalizationMode textCase = TextNormalizingEstimator.CaseNormalizationMode.Lower,
            bool keepDiacritics = false,
            bool keepPunctuations = true,
            bool keepNumbers = true) => new OutPipelineColumn(input, textCase, keepDiacritics, keepPunctuations, keepNumbers);
    }

    /// <summary>
    /// Extensions for statically typed bag of word converter.
    /// </summary>
    public static class WordBagEstimatorExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Scalar<string> Input;

            public OutPipelineColumn(Scalar<string> input,
                int ngramLength,
                int skipLength,
                bool allLengths,
                int maxNumTerms,
                NgramTokenizingTransformer.WeightingCriteria weighting)
                : base(new Reconciler(ngramLength, skipLength, allLengths, maxNumTerms, weighting), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler, IEquatable<Reconciler>
        {
            private readonly int _ngramLength;
            private readonly int _skipLength;
            private readonly bool _allLengths;
            private readonly int _maxNumTerms;
            private readonly NgramTokenizingTransformer.WeightingCriteria _weighting;

            public Reconciler(int ngramLength, int skipLength, bool allLengths, int maxNumTerms, NgramTokenizingTransformer.WeightingCriteria weighting)
            {
                _ngramLength = ngramLength;
                _skipLength = skipLength;
                _allLengths = allLengths;
                _maxNumTerms = maxNumTerms;
                _weighting = weighting;

            }

            public bool Equals(Reconciler other)
            {
                return _ngramLength == other._ngramLength &&
                _skipLength == other._skipLength &&
                _allLengths == other._allLengths &&
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

                var pairs = new List<(string[] inputs, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((new[] { inputNames[((OutPipelineColumn)outCol).Input] }, outputNames[outCol]));

                return new WordBagEstimator(env, pairs.ToArray(), _ngramLength, _skipLength, _allLengths, _maxNumTerms, _weighting);
            }
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words ) in a given text.
        /// It does so by building a dictionary of ngrams and using the id in the dictionary as the index in the bag.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public static Vector<float> ToBagofWords(this Scalar<string> input,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            int maxNumTerms = 10000000,
            NgramTokenizingTransformer.WeightingCriteria weighting = NgramTokenizingTransformer.WeightingCriteria.Tf)
                => new OutPipelineColumn(input, ngramLength, skipLength, allLengths, maxNumTerms, weighting);
    }

    /// <summary>
    /// Extensions for statically typed bag of wordhash converter.
    /// </summary>
    public static class WordHashBagEstimatorExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Scalar<string> Input;

            public OutPipelineColumn(Scalar<string> input,
                int hashBits,
                int ngramLength,
                int skipLength,
                bool allLengths,
                uint seed,
                bool ordered,
                int invertHash)
                : base(new Reconciler(hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler, IEquatable<Reconciler>
        {
            private readonly int _hashBits;
            private readonly int _ngramLength;
            private readonly int _skipLength;
            private readonly bool _allLengths;
            private readonly uint _seed;
            private readonly bool _ordered;
            private readonly int _invertHash;

            public Reconciler(int hashBits, int ngramLength, int skipLength, bool allLengths, uint seed, bool ordered, int invertHash)
            {
                _hashBits = hashBits;
                _ngramLength = ngramLength;
                _skipLength = skipLength;
                _allLengths = allLengths;
                _seed = seed;
                _ordered = ordered;
                _invertHash = invertHash;
            }

            public bool Equals(Reconciler other)
            {
                return _hashBits == other._hashBits &&
                    _ngramLength == other._ngramLength &&
                    _skipLength == other._skipLength &&
                    _allLengths == other._allLengths &&
                    _seed == other._seed &&
                    _ordered == other._ordered &&
                    _invertHash == other._invertHash;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string[] inputs, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((new[] { inputNames[((OutPipelineColumn)outCol).Input] }, outputNames[outCol]));

                return new WordHashBagEstimator(env, pairs.ToArray(), _hashBits, _ngramLength, _skipLength, _allLengths, _seed, _ordered, _invertHash);
            }
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words of length 1-n) in a given text.
        /// It does so by hashing each ngram and using the hash value as the index in the bag.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        public static Vector<float> ToBagofHashedWords(this Scalar<string> input,
            int hashBits = 16,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            uint seed = 314489979,
            bool ordered = true,
            int invertHash = 0) => new OutPipelineColumn(input, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash);
    }

    /// <summary>
    /// Extensions for statically typed ngram estimator.
    /// </summary>
    public static class NgramEstimatorExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly PipelineColumn Input;

            public OutPipelineColumn(PipelineColumn input,
                int ngramLength,
                int skipLength,
                bool allLengths,
                int maxNumTerms,
                NgramTokenizingTransformer.WeightingCriteria weighting)
                : base(new Reconciler(ngramLength, skipLength, allLengths, maxNumTerms, weighting), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler, IEquatable<Reconciler>
        {
            private readonly int _ngramLength;
            private readonly int _skipLength;
            private readonly bool _allLengths;
            private readonly int _maxNumTerms;
            private readonly NgramTokenizingTransformer.WeightingCriteria _weighting;

            public Reconciler(int ngramLength, int skipLength, bool allLengths, int maxNumTerms, NgramTokenizingTransformer.WeightingCriteria weighting)
            {
                _ngramLength = ngramLength;
                _skipLength = skipLength;
                _allLengths = allLengths;
                _maxNumTerms = maxNumTerms;
                _weighting = weighting;

            }

            public bool Equals(Reconciler other)
            {
                return _ngramLength == other._ngramLength &&
                _skipLength == other._skipLength &&
                _allLengths == other._allLengths &&
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

                var pairs = new List<(string inputs, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumn)outCol).Input], outputNames[outCol]));

                return new NgramEstimator(env, pairs.ToArray(), _ngramLength, _skipLength, _allLengths, _maxNumTerms, _weighting);
            }
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words ) in a given tokenized text.
        /// It does so by building a dictionary of ngrams and using the id in the dictionary as the index in the bag.
        ///
        /// /// <see cref="ToNgrams"/> is different from <see cref="WordBagEstimatorExtensions.ToBagofWords"/>
        /// in a way that <see cref="ToNgrams"/> takes tokenized text as input while <see cref="WordBagEstimatorExtensions.ToBagofWords"/> tokenizes text internally.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public static Vector<float> ToNgrams<TKey>(this VarVector<Key<TKey, string>> input,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            int maxNumTerms = 10000000,
            NgramTokenizingTransformer.WeightingCriteria weighting = NgramTokenizingTransformer.WeightingCriteria.Tf)
                => new OutPipelineColumn(input, ngramLength, skipLength, allLengths, maxNumTerms, weighting);
    }

    /// <summary>
    /// Extensions for statically typed ngram hash estimator.
    /// </summary>
    public static class NgramHashEstimatorExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly VarVector<Key<uint, string>> Input;

            public OutPipelineColumn(VarVector<Key<uint, string>> input, int hashBits, int ngramLength, int skipLength, bool allLengths, uint seed, bool ordered, int invertHash)
                : base(new Reconciler(hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler, IEquatable<Reconciler>
        {
            private readonly int _hashBits;
            private readonly int _ngramLength;
            private readonly int _skipLength;
            private readonly bool _allLengths;
            private readonly uint _seed;
            private readonly bool _ordered;
            private readonly int _invertHash;

            public Reconciler(int hashBits, int ngramLength, int skipLength, bool allLengths, uint seed, bool ordered, int invertHash)
            {
                _hashBits = hashBits;
                _ngramLength = ngramLength;
                _skipLength = skipLength;
                _allLengths = allLengths;
                _seed = seed;
                _ordered = ordered;
                _invertHash = invertHash;
            }

            public bool Equals(Reconciler other)
            {
                return _hashBits == other._hashBits &&
                    _ngramLength == other._ngramLength &&
                    _skipLength == other._skipLength &&
                    _allLengths == other._allLengths &&
                    _seed == other._seed &&
                    _ordered == other._ordered &&
                    _invertHash == other._invertHash;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string[] inputs, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((new[] { inputNames[((OutPipelineColumn)outCol).Input] }, outputNames[outCol]));

                return new NgramHashEstimator(env, pairs.ToArray(), _hashBits, _ngramLength, _skipLength, _allLengths, _seed, _ordered, _invertHash);
            }
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words of length 1-n) in a given tokenized text.
        /// It does so by hashing each ngram and using the hash value as the index in the bag.
        ///
        /// <see cref="ToNgramsHash"/> is different from <see cref="WordHashBagEstimatorExtensions.ToBagofHashedWords"/>
        /// in a way that <see cref="ToNgramsHash"/> takes tokenized text as input while <see cref="WordHashBagEstimatorExtensions.ToBagofHashedWords"/> tokenizes text internally.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.</param>
        public static Vector<float> ToNgramsHash(this VarVector<Key<uint, string>> input,
            int hashBits = 16,
            int ngramLength = 2,
            int skipLength = 0,
            bool allLengths = true,
            uint seed = 314489979,
            bool ordered = true,
            int invertHash = 0) => new OutPipelineColumn(input, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash);
    }

    /// <summary>
    /// Extensions for statically typed <see cref="LdaEstimator"/>.
    /// </summary>
    public static class LdaEstimatorExtensions
    {
        private sealed class OutPipelineColumn : Vector<float>
        {
            public readonly Vector<float> Input;

            public OutPipelineColumn(Vector<float> input, int numTopic, Action<LdaTransform.Arguments> advancedSettings)
                : base(new Reconciler(numTopic, advancedSettings), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly int _numTopic;
            private readonly Action<LdaTransform.Arguments> _advancedSettings;

            public Reconciler(int numTopic, Action<LdaTransform.Arguments> advancedSettings)
            {
                _numTopic = numTopic;
                _advancedSettings = advancedSettings;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string input, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumn)outCol).Input], outputNames[outCol]));

                return new LdaEstimator(env, pairs.ToArray(), _numTopic, _advancedSettings);
            }
        }

        /// <include file='doc.xml' path='doc/members/member[@name="LightLDA"]/*' />
        /// <param name="input">The column to apply to.</param>
        /// <param name="numTopic">The number of topics in the LDA.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public static Vector<float> ToLdaTopicVector(this Vector<float> input,
            int numTopic = 100,
            Action<LdaTransform.Arguments> advancedSettings = null) => new OutPipelineColumn(input, numTopic, advancedSettings);
    }
}
