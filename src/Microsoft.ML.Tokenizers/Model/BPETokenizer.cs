// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Represent the Byte Pair Encoding model.
    /// </summary>
    public sealed class BpeTokenizer : Tokenizer
    {
        /// A [Byte Pair Encoding](https://www.aclweb.org/anthology/P16-1162/) model.

        private const int MaxWordLengthToCache = 15;
        private string? _unknownToken;
        private int? _unknownTokenId;
        private readonly PreTokenizer? _preTokenizer;
        private readonly Normalizer? _normalizer;
        private readonly Dictionary<StringSpanOrdinalKey, (int, string)>? _specialTokens;
        private readonly Dictionary<int, string>? _specialTokensReverse;
        private const int BufferLength = 128;

        /// <summary>
        /// Gets the special tokens.
        /// </summary>
        public IReadOnlyDictionary<string, int>? SpecialTokens { get; }

        /// <summary>
        /// Gets or Sets unknown token. The unknown token to be used when we encounter an unknown char
        /// </summary>
        public string? UnknownToken
        {
            get
            {
                return _unknownToken;
            }

            private set
            {
                if (value is null)
                {
                    _unknownToken = value;
                    _unknownTokenId = null;
                    return;
                }

                if (!_vocab.TryGetValue(value, out int id))
                {
                    throw new InvalidOperationException($"Unknown Token '{value}' was not present in '{nameof(Vocabulary)}'.");
                }

                _unknownTokenId = id;
                _unknownToken = value;
            }
        }

        /// <summary>
        /// A prefix to be used for every subword that is not a beginning-of-word
        /// </summary>
        public string? ContinuingSubwordPrefix { get; }

        /// <summary>
        /// An optional suffix to characterize the end-of-word and sub-word
        /// </summary>
        public string? EndOfWordSuffix { get; }

        /// <summary>
        /// Gets a value indicating whether to merge the sequence of the unknown tokens together.
        /// </summary>
        public bool FuseUnknownTokens { get; }

        /// <summary>
        /// Create a new Bpe tokenizer object to use for text encoding.
        /// </summary>
        /// <param name="vocabFile">The JSON file path containing the dictionary of string keys and their ids.</param>
        /// <param name="mergesFile">The file path containing the tokens's pairs list.</param>
        /// <remarks>
        /// When creating the tokenizer, ensure that the vocabulary file is sourced from a trusted provider.
        /// </remarks>
        public static BpeTokenizer Create(string vocabFile, string? mergesFile)
            => Create(vocabFile, mergesFile, preTokenizer: PreTokenizer.CreateWordOrNonWord(), normalizer: null, unknownToken: null, continuingSubwordPrefix: null, endOfWordSuffix: null, fuseUnknownTokens: false);

        /// <summary>
        /// Create a new Bpe tokenizer object to use for text encoding.
        /// </summary>
        /// <param name="vocabFile">The JSON file path containing the dictionary of string keys and their ids.</param>
        /// <param name="mergesFile">The file path containing the tokens's pairs list.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="specialTokens">The dictionary mapping special tokens to Ids.</param>
        /// <param name="unknownToken"> The unknown token to be used by the model.</param>
        /// <param name="continuingSubwordPrefix">The prefix to attach to sub-word units that don’t represent a beginning of word.</param>
        /// <param name="endOfWordSuffix">The suffix to attach to sub-word units that represent an end of word.</param>
        /// <param name="fuseUnknownTokens">Indicate whether allowing multiple unknown tokens get fused.</param>
        /// <remarks>
        /// When creating the tokenizer, ensure that the vocabulary file is sourced from a trusted provider.
        /// </remarks>
        public static BpeTokenizer Create(
                                string vocabFile,
                                string? mergesFile,
                                PreTokenizer? preTokenizer = null,
                                Normalizer? normalizer = null,
                                IReadOnlyDictionary<string, int>? specialTokens = null,
                                string? unknownToken = null,
                                string? continuingSubwordPrefix = null,
                                string? endOfWordSuffix = null,
                                bool fuseUnknownTokens = false)
        {
            if (vocabFile is null)
            {
                throw new ArgumentNullException(nameof(vocabFile));
            }

            using Stream vocabStream = File.OpenRead(vocabFile);
            using Stream? mergesStream = mergesFile is null ? null : File.OpenRead(mergesFile);

            (Dictionary<StringSpanOrdinalKey, int>? vocab, Vec<(string, string)> merges) result = ReadModelDataAsync(vocabStream, mergesStream, useAsync: false).GetAwaiter().GetResult();

            return new BpeTokenizer(result.vocab, result.merges, preTokenizer, normalizer, specialTokens, unknownToken, continuingSubwordPrefix, endOfWordSuffix, fuseUnknownTokens);
        }

        public static BpeTokenizer Create(BpeOptions options)
        {
            if (options is null)
            {
                throw new ArgumentNullException(nameof(options));
            }

            if (options.Vocabulary is null)
            {
                throw new ArgumentNullException(nameof(options.Vocabulary), "The vocabulary cannot be null.");
            }

            Dictionary<StringSpanOrdinalKey, int> vocab = new Dictionary<StringSpanOrdinalKey, int>(1000);

            foreach ((string token, int id) in options.Vocabulary)
            {
                vocab.Add(new StringSpanOrdinalKey(token), id);
            }

            if (vocab.Count == 0)
            {
                throw new InvalidOperationException("The vocabulary cannot be empty.");
            }

            Vec<(string, string)> merges = default;
            if (options.Merges is not null)
            {
                merges = new Vec<(string, string)>(1000);

                foreach (string merge in options.Merges)
                {
                    if (merge is null)
                    {
                        throw new InvalidOperationException("The merge entries cannot be null.");
                    }

                    int index = merge.IndexOf(' ');
                    if (index < 0 || index == merge.Length - 1 || merge.IndexOf(' ', index + 1) >= 0)
                    {
                        throw new InvalidOperationException($"Invalid merger file format");
                    }

                    merges.Push((merge.Substring(0, index), merge.Substring(index + 1)));
                }
            }

            return new BpeTokenizer(
                            vocab, merges,
                            options.PreTokenizer,
                            options.Normalizer,
                            options.SpecialTokens,
                            options.UnknownToken,
                            options.ContinuingSubwordPrefix,
                            options.EndOfWordSuffix,
                            options.FuseUnknownTokens,
                            options.ByteLevel,
                            options.BeginningOfSentenceToken,
                            options.EndOfSentenceToken);
        }

        /// <summary>
        /// Create a new Bpe tokenizer object to use for text encoding.
        /// </summary>
        /// <param name="vocabStream">The JSON stream containing the dictionary of string keys and their ids.</param>
        /// <param name="mergesStream">The stream containing the tokens's pairs list.</param>
        /// <remarks>
        /// When creating the tokenizer, ensure that the vocabulary stream is sourced from a trusted provider.
        /// </remarks>
        public static BpeTokenizer Create(Stream vocabStream, Stream? mergesStream)
            => Create(vocabStream, mergesStream, preTokenizer: PreTokenizer.CreateWordOrNonWord(), normalizer: null, specialTokens: null, unknownToken: null, continuingSubwordPrefix: null, endOfWordSuffix: null, fuseUnknownTokens: false);

        /// <summary>
        /// Create a new Bpe tokenizer object to use for text encoding.
        /// </summary>
        /// <param name="vocabStream">The JSON stream containing the dictionary of string keys and their ids.</param>
        /// <param name="mergesStream">The stream containing the tokens's pairs list.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="specialTokens">The dictionary mapping special tokens to Ids.</param>
        /// <param name="unknownToken"> The unknown token to be used by the model.</param>
        /// <param name="continuingSubwordPrefix">The prefix to attach to sub-word units that don’t represent a beginning of word.</param>
        /// <param name="endOfWordSuffix">The suffix to attach to sub-word units that represent an end of word.</param>
        /// <param name="fuseUnknownTokens">Indicate whether allowing multiple unknown tokens get fused.</param>
        /// <remarks>
        /// When creating the tokenizer, ensure that the vocabulary stream is sourced from a trusted provider.
        /// </remarks>
        public static BpeTokenizer Create(
                                Stream vocabStream,
                                Stream? mergesStream,
                                PreTokenizer? preTokenizer = null,
                                Normalizer? normalizer = null,
                                IReadOnlyDictionary<string, int>? specialTokens = null,
                                string? unknownToken = null,
                                string? continuingSubwordPrefix = null,
                                string? endOfWordSuffix = null,
                                bool fuseUnknownTokens = false)
        {
            if (vocabStream is null)
            {
                throw new ArgumentNullException(nameof(vocabStream));
            }

            (Dictionary<StringSpanOrdinalKey, int>? vocab, Vec<(string, string)> merges) result = ReadModelDataAsync(vocabStream, mergesStream, useAsync: false).GetAwaiter().GetResult();

            return new BpeTokenizer(result.vocab, result.merges, preTokenizer, normalizer, specialTokens, unknownToken, continuingSubwordPrefix, endOfWordSuffix, fuseUnknownTokens);
        }

        /// <summary>
        /// Create a new Bpe tokenizer object asynchronously to use for text encoding.
        /// </summary>
        /// <param name="vocabStream">The JSON stream containing the dictionary of string keys and their ids.</param>
        /// <param name="mergesStream">The stream containing the tokens's pairs list.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="specialTokens">The dictionary mapping special tokens to Ids.</param>
        /// <param name="unknownToken"> The unknown token to be used by the model.</param>
        /// <param name="continuingSubwordPrefix">The prefix to attach to sub-word units that don’t represent a beginning of word.</param>
        /// <param name="endOfWordSuffix">The suffix to attach to sub-word units that represent an end of word.</param>
        /// <param name="fuseUnknownTokens">Indicate whether allowing multiple unknown tokens get fused.</param>
        /// <remarks>
        /// When creating the tokenizer, ensure that the vocabulary stream is sourced from a trusted provider.
        /// </remarks>
        public static async Task<BpeTokenizer> CreateAsync(
                                Stream vocabStream,
                                Stream? mergesStream,
                                PreTokenizer? preTokenizer = null,
                                Normalizer? normalizer = null,
                                IReadOnlyDictionary<string, int>? specialTokens = null,
                                string? unknownToken = null,
                                string? continuingSubwordPrefix = null,
                                string? endOfWordSuffix = null,
                                bool fuseUnknownTokens = false)
        {
            if (vocabStream is null)
            {
                throw new ArgumentNullException(nameof(vocabStream));
            }

            (Dictionary<StringSpanOrdinalKey, int>? vocab, Vec<(string, string)> merges) result = await ReadModelDataAsync(vocabStream, mergesStream, useAsync: true).ConfigureAwait(false);

            return new BpeTokenizer(result.vocab, result.merges, preTokenizer, normalizer, specialTokens, unknownToken, continuingSubwordPrefix, endOfWordSuffix, fuseUnknownTokens);
        }

        /// <summary>
        /// Construct a new Bpe model object to use for text encoding.
        /// </summary>
        /// <param name="vocab">The dictionary vocabulary mapping token string to ids.</param>
        /// <param name="merges">The pairs list help in merging tokens during the encoding process.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="specialTokens">The dictionary mapping special tokens to Ids.</param>
        /// <param name="unknownToken"> The unknown token to be used by the model.</param>
        /// <param name="continuingSubwordPrefix">The prefix to attach to sub-word units that don’t represent a beginning of word.</param>
        /// <param name="endOfWordSuffix">The suffix to attach to sub-word units that represent an end of word.</param>
        /// <param name="fuseUnknownTokens">Indicate whether allowing multiple unknown tokens get fused.</param>
        /// <param name="byteLevel">Indicate whether to handle the input text in byte level.</param>
        /// <param name="beginningOfSentenceToken">The beginning of sentence token.</param>
        /// <param name="endOfSentenceToken">The end of sentence token.</param>
        private BpeTokenizer(
                    Dictionary<StringSpanOrdinalKey, int>? vocab,
                    Vec<(string, string)> merges,
                    PreTokenizer? preTokenizer,
                    Normalizer? normalizer,
                    IReadOnlyDictionary<string, int>? specialTokens,
                    string? unknownToken,
                    string? continuingSubwordPrefix,
                    string? endOfWordSuffix,
                    bool fuseUnknownTokens,
                    bool byteLevel = false,
                    string? beginningOfSentenceToken = null,
                    string? endOfSentenceToken = null)
        {
            FuseUnknownTokens = fuseUnknownTokens;
            ContinuingSubwordPrefix = continuingSubwordPrefix;
            EndOfWordSuffix = endOfWordSuffix;
            ByteLevel = byteLevel;
            _preTokenizer = preTokenizer ?? PreTokenizer.CreateWordOrNonWord(); // Default to WordOrNonWord pre-tokenizer
            _normalizer = normalizer;

            _vocab = vocab ?? new Dictionary<StringSpanOrdinalKey, int>();

            if (beginningOfSentenceToken is not null)
            {
                if (!_vocab.TryGetValue(beginningOfSentenceToken, out int aId))
                {
                    throw new InvalidOperationException($"The beginning of sentence token '{beginningOfSentenceToken}' was not present in the vocabulary.");
                }

                BeginningOfSentenceId = aId;
                BeginningOfSentenceToken = beginningOfSentenceToken;
            }

            if (endOfSentenceToken is not null)
            {
                if (!_vocab.TryGetValue(endOfSentenceToken, out int aId))
                {
                    throw new InvalidOperationException($"The end of sentence token '{endOfSentenceToken}' was not present in the vocabulary.");
                }

                EndOfSentenceId = aId;
                EndOfSentenceToken = endOfSentenceToken;
            }

            Cache = new StringSpanOrdinalKeyCache<Word>();

            VocabReverse = new();

            foreach (KeyValuePair<StringSpanOrdinalKey, int> kvp in _vocab)
            {
                VocabReverse.Add(kvp.Value, kvp.Key.Data!);
            }

            if (specialTokens is not null)
            {
                SpecialTokens = specialTokens;
                _specialTokens = specialTokens.ToDictionary(kvp => new StringSpanOrdinalKey(kvp.Key), kvp => (kvp.Value, kvp.Key));
                _specialTokensReverse = specialTokens.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
            }

            UnknownToken = unknownToken;

            int prefixLen = ContinuingSubwordPrefix is null ? 0 : ContinuingSubwordPrefix.Length;

            Merges = new();
            for (int i = 0; i < merges.Count; i++)
            {
                (string a, string b) mergeValues = merges[i];

                if (!_vocab.TryGetValue(mergeValues.a, out int aId))
                {
                    throw new InvalidOperationException($"Trying to merge a token '{mergeValues.a}' which not exist in the vocabulary.");
                }

                if (!_vocab.TryGetValue(mergeValues.b, out int bId))
                {
                    throw new InvalidOperationException($"Trying to merge a token '{mergeValues.b}' which not exist in the vocabulary.");
                }

                if (mergeValues.b.Length <= prefixLen)
                {
                    throw new InvalidOperationException($"The merge value '{mergeValues.b}' is too short to be merged with a prefix of length {prefixLen}. This implies that the merge file is either damaged or missing the prefix in its entries.");
                }

                string newToken = $"{mergeValues.a}{mergeValues.b.Substring(prefixLen)}";
                if (!_vocab.TryGetValue(newToken, out int newId))
                {
                    throw new InvalidOperationException($"Trying to merge a token '{newToken}' which not exist in the vocabulary.");
                }

                Merges.Add(new Pair<int>(aId, bId), (i, newId));
            }
        }

        /// <summary>
        /// Gets a value indicating whether to handle the input text in byte level.
        /// if true, the input text will be converted to UTF-8 bytes before encoding it.
        /// Additionally, some ASCII characters will be transformed to another characters (e.g Space character will be transformed to 'Ġ' character).
        /// </summary>
        public bool ByteLevel { get; }

        /// <summary>
        /// Gets the optional beginning of sentence token.
        /// </summary>
        internal string? BeginningOfSentenceToken { get; }

        /// <summary>
        /// The id of the beginning of sentence token.
        /// </summary>
        internal int BeginningOfSentenceId { get; }

        /// <summary>
        /// Gets the optional end of sentence token.
        /// </summary>
        internal string? EndOfSentenceToken { get; }

        /// <summary>
        /// The id of the end of sentence token.
        /// </summary>
        internal int EndOfSentenceId { get; }

        /// <summary>
        /// Gets the PreTokenizer used by the Tokenizer.
        /// </summary>
        public override PreTokenizer? PreTokenizer => _preTokenizer;

        /// <summary>
        /// Gets the Normalizer in use by the Tokenizer.
        /// </summary>
        public override Normalizer? Normalizer => _normalizer;

        /// <summary>
        /// Encodes input text to a list of <see cref="EncodedToken" />s.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        protected override EncodeResults<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
        {
            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                return new EncodeResults<EncodedToken> { Tokens = [], NormalizedText = null, CharsConsumed = 0 };
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(
                                                                text,
                                                                textSpan,
                                                                settings.ConsiderPreTokenization,
                                                                settings.ConsiderNormalization,
                                                                _normalizer,
                                                                _preTokenizer,
                                                                out string? normalizedText,
                                                                out ReadOnlySpan<char> textSpanToEncode,
                                                                out int charsConsumed);

            List<EncodedToken> tokens = new();
            if (BeginningOfSentenceToken is not null)
            {
                tokens.Add(new EncodedToken(BeginningOfSentenceId, BeginningOfSentenceToken, new Range(0, 0)));
            }

            PriorityQueue<Merge>? priorityQueue = null;

            if (splits is not null)
            {
                foreach ((int Offset, int Length) split in splits)
                {
                    EncodeWithCache(textSpanToEncode.Slice(split.Offset, split.Length), tokens, split.Offset, ref priorityQueue);
                }
            }
            else
            {
                EncodeWithCache(textSpanToEncode, tokens, 0, ref priorityQueue);
            }

            if (EndOfSentenceToken is not null)
            {
                tokens.Add(new EncodedToken(EndOfSentenceId, EndOfSentenceToken, new Range(charsConsumed, charsConsumed)));
            }

            return new EncodeResults<EncodedToken> { Tokens = tokens, NormalizedText = normalizedText, CharsConsumed = charsConsumed };
        }

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        /// <returns>The encoded results containing the list of encoded Ids.</returns>
        protected override EncodeResults<int> EncodeToIds(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
        {
            int maxTokenCount = settings.MaxTokenCount;
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(settings.MaxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                return new EncodeResults<int> { Tokens = [], NormalizedText = null, CharsConsumed = 0 };
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(
                                                                text,
                                                                textSpan,
                                                                settings.ConsiderPreTokenization,
                                                                settings.ConsiderNormalization,
                                                                _normalizer,
                                                                _preTokenizer,
                                                                out string? normalizedText,
                                                                out ReadOnlySpan<char> textSpanToEncode,
                                                                out _);

            List<int> ids = new();

            if (BeginningOfSentenceToken is not null)
            {
                ids.Add(BeginningOfSentenceId);
            }

            PriorityQueue<Merge>? priorityQueue = null;

            int charsConsumed = 0;
            if (splits is not null)
            {
                foreach ((int Offset, int Length) split in splits)
                {
                    EncodeToIdsWithCache(textSpanToEncode.Slice(split.Offset, split.Length), ids, maxTokenCount - ids.Count, out int length, ref priorityQueue);
                    charsConsumed = split.Offset + length;

                    if (length < split.Length || ids.Count >= maxTokenCount)
                    {
                        break;
                    }
                }
            }
            else
            {
                EncodeToIdsWithCache(textSpanToEncode, ids, maxTokenCount, out charsConsumed, ref priorityQueue);
            }

            if (EndOfSentenceToken is not null && ids.Count < maxTokenCount)
            {
                ids.Add(EndOfSentenceId);
            }

            return new EncodeResults<int> { Tokens = ids, NormalizedText = normalizedText, CharsConsumed = charsConsumed };
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        protected override int CountTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
        {
            int maxTokenCount = settings.MaxTokenCount;
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(settings.MaxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                return 0;
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(
                                                                text,
                                                                textSpan,
                                                                settings.ConsiderPreTokenization,
                                                                settings.ConsiderNormalization,
                                                                _normalizer,
                                                                _preTokenizer,
                                                                out string? normalizedText,
                                                                out ReadOnlySpan<char> textSpanToEncode,
                                                                out _);

            PriorityQueue<Merge>? priorityQueue = null;
            int count = BeginningOfSentenceToken is null ? 0 : 1;
            int textLength = 0;

            if (splits is not null)
            {
                foreach ((int Offset, int Length) split in splits)
                {
                    count += EncodeToIdsWithCache(textSpanToEncode.Slice(split.Offset, split.Length), null, maxTokenCount - count, out int length, ref priorityQueue);
                    textLength = split.Offset + length;

                    if (length < split.Length || count >= maxTokenCount)
                    {
                        break;
                    }
                }
            }
            else
            {
                count = EncodeToIdsWithCache(textSpanToEncode, null, maxTokenCount, out textLength, ref priorityQueue);
            }

            if (EndOfSentenceToken is not null && count < maxTokenCount)
            {
                count++;
            }

            return count;
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        /// <param name="fromEnd">Indicate whether to find the index from the end of the text.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="settings" /> has <see cref="EncodeSettings.ConsiderNormalization"/> is <see langword="false"/>, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to <see langword="null"/>.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// If <paramRef name="fromEnd" /> is <see langword="false"/>, it represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the input text or the <paramref name="normalizedText"/> if the normalization is enabled.
        /// If <paramRef name="fromEnd" /> is <see langword="true"/>, it represents the index of the first character to be included. In cases where no tokens fit, the result will be the text length; conversely,
        /// if all tokens fit, the result will be zero.
        /// </returns>
        protected override int GetIndexByTokenCount(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings, bool fromEnd, out string? normalizedText, out int tokenCount)
        {
            if (fromEnd)
            {
                return LastIndexOf(text, textSpan, settings.MaxTokenCount, settings.ConsiderPreTokenization, settings.ConsiderNormalization, out normalizedText, out tokenCount);
            }

            tokenCount = CountTokens(text, textSpan, settings.ConsiderPreTokenization, settings.ConsiderNormalization, out normalizedText, out int charsConsumed, settings.MaxTokenCount);
            return charsConsumed;
        }

        private int CountTokens(string? text, ReadOnlySpan<char> textSpan, bool considerPreTokenization, bool considerNormalization, out string? normalizedText, out int charsConsumed, int maxTokenCount = int.MaxValue)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            charsConsumed = 0;
            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedText = null;
                return 0;
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(
                                                                text,
                                                                textSpan,
                                                                considerPreTokenization,
                                                                considerNormalization,
                                                                _normalizer,
                                                                _preTokenizer,
                                                                out normalizedText,
                                                                out ReadOnlySpan<char> textSpanToEncode,
                                                                out _);

            PriorityQueue<Merge>? priorityQueue = null;
            int count = BeginningOfSentenceToken is null ? 0 : 1;
            if (splits is not null)
            {
                foreach ((int Offset, int Length) split in splits)
                {
                    count += EncodeToIdsWithCache(textSpanToEncode.Slice(split.Offset, split.Length), null, maxTokenCount - count, out int length, ref priorityQueue);
                    charsConsumed = split.Offset + length;

                    if (length < split.Length || count >= maxTokenCount)
                    {
                        break;
                    }
                }
            }
            else
            {
                count = EncodeToIdsWithCache(textSpanToEncode, null, maxTokenCount, out charsConsumed, ref priorityQueue);
            }

            if (EndOfSentenceToken is not null && count < maxTokenCount)
            {
                count++;
            }

            return count;
        }

        private int LastIndexOf(string? text, ReadOnlySpan<char> textSpan, int maxTokenCount, bool considerPreTokenization, bool considerNormalization, out string? normalizedText, out int tokenCount)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The max token count must be greater than 0.");
            }

            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedText = null;
                tokenCount = 0;
                return 0;
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(
                                                                text,
                                                                textSpan,
                                                                considerPreTokenization,
                                                                considerNormalization,
                                                                _normalizer,
                                                                _preTokenizer,
                                                                out normalizedText,
                                                                out ReadOnlySpan<char> textSpanToEncode,
                                                                out _);

            PriorityQueue<Merge>? priorityQueue = null;

            if (splits is not null)
            {
                tokenCount = EndOfSentenceToken is null ? 0 : 1;
                foreach ((int Offset, int Length) split in splits.Reverse())
                {
                    tokenCount += EncodeToIdsFromEndWithCache(textSpanToEncode.Slice(split.Offset, split.Length), null, maxTokenCount - tokenCount, out int textIndex, ref priorityQueue);
                    if (textIndex > 0 || tokenCount >= maxTokenCount)
                    {
                        return split.Offset + textIndex;
                    }
                }

                tokenCount += tokenCount < maxTokenCount && BeginningOfSentenceToken is not null ? 1 : 0;
            }
            else
            {
                tokenCount = EncodeToIdsFromEndWithCache(textSpanToEncode, null, maxTokenCount - (EndOfSentenceToken is null ? 0 : 1), out int charsConsumed, ref priorityQueue);
                tokenCount += tokenCount < maxTokenCount && BeginningOfSentenceToken is not null ? 1 : 0;
                return charsConsumed;
            }

            return 0;
        }

        /// <summary>
        /// Map the token to encoded Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <returns>The mapped Id of the token.</returns>
        private int? MapTokenToId(ReadOnlySpan<char> token) => _vocab.TryGetValue(token, out int value) ? value : null;

        /// <summary>
        /// Map the encoded Id to the token.
        /// </summary>
        /// <param name="id">The Id to map to the token.</param>
        /// <returns>The mapped token of the Id.</returns>
        private string? MapIdToToken(int id)
        {
            if (VocabReverse.TryGetValue(id, out string? value))
            {
                return value;
            }

            return null;
        }

        /// <summary>
        /// Gets the dictionary mapping tokens to Ids.
        /// </summary>
        public IReadOnlyDictionary<string, int> Vocabulary => _vocabOriginal ??= new ReadOnlyDictionary<string, int>(_vocab.ToDictionary(kvp => kvp.Key.Data!, kvp => kvp.Value));

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <returns>The decoded string.</returns>
        public override string Decode(IEnumerable<int> ids) => Decode(ids, considerSpecialTokens: true);

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="considerSpecialTokens">Indicate whether to consider special tokens or not.</param>
        /// <returns>The decoded string.</returns>
        public string Decode(IEnumerable<int> ids, bool considerSpecialTokens)
        {
            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
            }

            if (ByteLevel)
            {
                return DecodeByteLevel(ids, considerSpecialTokens);
            }

            ValueStringBuilder sb = new ValueStringBuilder();

            bool decodeUnknownToken = _unknownTokenId.HasValue && considerSpecialTokens;

            if (decodeUnknownToken)
            {
                foreach (int id in ids)
                {
                    if (MapIdToToken(id) is string s)
                    {
                        sb.Append(s);
                    }
                }
            }
            else
            {
                foreach (int id in ids)
                {
                    if (id == _unknownTokenId)
                    {
                        continue;
                    }

                    if (MapIdToToken(id) is string s)
                    {
                        sb.Append(s);
                    }
                }
            }

            if (EndOfWordSuffix is not null)
            {
                sb.RemoveSuffix(EndOfWordSuffix);

                sb.Replace(EndOfWordSuffix, " ");
            }

            if (ContinuingSubwordPrefix is not null)
            {
                sb.Replace(ContinuingSubwordPrefix, string.Empty);
            }

            return sb.ToString();
        }

        private string DecodeByteLevel(IEnumerable<int> ids, bool considerSpecialTokens)
        {
            int bytesIndex = 0;
            byte[] bytes = ArrayPool<byte>.Shared.Rent(BufferLength << 1);

            foreach (int id in ids)
            {
                if (_specialTokensReverse?.TryGetValue(id, out string? token) is true)
                {
                    if (!considerSpecialTokens)
                    {
                        continue;
                    }

                    Helpers.AppendToBytesArray(token.AsSpan(), ref bytes, ref bytesIndex);
                    continue;
                }

                if (MapIdToToken(id) is string s)
                {
                    ReadOnlySpan<char> span = s.AsSpan();
                    if (EndOfWordSuffix is not null && span.EndsWith(EndOfWordSuffix.AsSpan(), StringComparison.Ordinal))
                    {
                        span = span.Slice(0, span.Length - EndOfWordSuffix.Length);
                    }

                    if (ContinuingSubwordPrefix is not null && span.StartsWith(ContinuingSubwordPrefix.AsSpan(), StringComparison.Ordinal))
                    {
                        span = span.Slice(ContinuingSubwordPrefix.Length);
                    }

                    Helpers.AppendToBytesArray(span, ref bytes, ref bytesIndex);
                }
            }

            string decodedString = Helpers.GetString(bytes.AsSpan(0, bytesIndex));

            ArrayPool<byte>.Shared.Return(bytes);

            return decodedString;
        }

        /// <summary>
        /// Decode the given ids back to text and store the result in the <paramref name="destination"/> span.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="destination">The span to store the decoded text.</param>
        /// <param name="idsConsumed">The number of ids consumed during the decoding.</param>
        /// <param name="charsWritten">The number of characters written to the destination span.</param>
        /// <returns>The operation status indicates whether all IDs were successfully decoded or if the <paramref name="destination"/> is too small to contain the entire decoded result.</returns>
        public override OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, out int idsConsumed, out int charsWritten)
            => Decode(ids, destination, considerSpecialTokens: true, out idsConsumed, out charsWritten);

        /// <summary>
        /// Decode the given ids back to text and store the result in the <paramref name="destination"/> span.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="destination">The span to store the decoded text.</param>
        /// <param name="considerSpecialTokens">Indicate whether to consider special tokens or not.</param>
        /// <param name="idsConsumed">The number of ids consumed during the decoding.</param>
        /// <param name="charsWritten">The number of characters written to the destination span.</param>
        /// <returns>The operation status indicates whether all IDs were successfully decoded or if the <paramref name="destination"/> is too small to contain the entire decoded result.</returns>
        public OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, bool considerSpecialTokens, out int idsConsumed, out int charsWritten)
        {
            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
            }

            if (ByteLevel)
            {
                return DecodeByteLevel(ids, destination, considerSpecialTokens, out idsConsumed, out charsWritten);
            }

            idsConsumed = 0;
            charsWritten = 0;

            Span<char> buffer = destination;

            bool skipUnknownToken = !_unknownTokenId.HasValue || !considerSpecialTokens;

            bool addSpace = false;
            bool continuingSubwordPrefix = ContinuingSubwordPrefix is not null && ContinuingSubwordPrefix.Length > 0;
            bool endOfWordSuffix = EndOfWordSuffix is not null && EndOfWordSuffix.Length > 0;
            int previousCharsWritten = 0;
            int previousIdsConsumed = 0;

            foreach (int id in ids)
            {
                if (skipUnknownToken && id == _unknownTokenId)
                {
                    idsConsumed++;
                    continue;
                }

                if (addSpace)
                {
                    if (buffer.Length == 0)
                    {
                        charsWritten = previousCharsWritten;
                        return OperationStatus.DestinationTooSmall;
                    }

                    buffer[0] = ' ';
                    buffer = buffer.Slice(1);
                    charsWritten++;
                }

                if (MapIdToToken(id) is string s)
                {
                    ReadOnlySpan<char> sSpan = s.AsSpan();

                    if (continuingSubwordPrefix && sSpan.StartsWith(ContinuingSubwordPrefix.AsSpan(), StringComparison.Ordinal))
                    {
                        sSpan = sSpan.Slice(ContinuingSubwordPrefix!.Length);
                    }

                    addSpace = false;
                    if (endOfWordSuffix && sSpan.EndsWith(EndOfWordSuffix!.AsSpan(), StringComparison.Ordinal))
                    {
                        sSpan = sSpan.Slice(0, sSpan.Length - EndOfWordSuffix!.Length);

                        addSpace = true;
                        previousIdsConsumed = idsConsumed;
                        previousCharsWritten = charsWritten;
                    }

                    if (sSpan.Length > buffer.Length)
                    {
                        return OperationStatus.DestinationTooSmall;
                    }

                    sSpan.CopyTo(buffer);

                    charsWritten += sSpan.Length;
                    buffer = buffer.Slice(sSpan.Length);
                }
                idsConsumed++;
            }

            return OperationStatus.Done;
        }

        private OperationStatus DecodeByteLevel(IEnumerable<int> ids, Span<char> destination, bool considerSpecialTokens, out int idsConsumed, out int charsWritten)
        {
            idsConsumed = 0;
            charsWritten = 0;

            // Enough buffer to carry one converted vocabulary to UTF-16 form
            Span<char> vocabBuffer = stackalloc char[128];

            // Enough buffer to carry one UTF-8 vocabulary
            Span<byte> utf8bytes = stackalloc byte[256];

            int incompleteUtf8BytesInBuffer = 0;
            int incompleteUtf8BytesInBufferIndex = 0;
            int utf16CharsInBuffer = 0;
            int idsHangingCount = 0;

            ByteToUnicodeEncoding byteToUnicodeEncoding = ByteToUnicodeEncoding.Instance;

            Span<char> buffer = destination;

            foreach (int id in ids)
            {
                if (_specialTokensReverse?.TryGetValue(id, out string? specialToken) is true)
                {
                    if (!considerSpecialTokens)
                    {
                        idsConsumed++;
                        continue;
                    }

                    if (incompleteUtf8BytesInBuffer > 0)
                    {
                        return OperationStatus.InvalidData; // unexpected case
                    }

                    ReadOnlySpan<char> specialTokenSpan = specialToken.AsSpan();

                    if (specialTokenSpan.Length > buffer.Length)
                    {
                        return OperationStatus.DestinationTooSmall;
                    }

                    specialTokenSpan.CopyTo(buffer);
                    buffer = buffer.Slice(specialTokenSpan.Length);
                    charsWritten += specialTokenSpan.Length;
                    idsConsumed++;
                    continue;
                }

                // vocabularies are stored in UTF-8 form with escaping the control characters.
                // Need to convert the vocabulary to the original UTF-16 form.
                if (MapIdToToken(id) is string s)
                {
                    ReadOnlySpan<char> span = s.AsSpan();
                    if (EndOfWordSuffix is not null && span.EndsWith(EndOfWordSuffix.AsSpan(), StringComparison.Ordinal))
                    {
                        span = span.Slice(0, span.Length - EndOfWordSuffix.Length);
                    }

                    if (ContinuingSubwordPrefix is not null && span.StartsWith(ContinuingSubwordPrefix.AsSpan(), StringComparison.Ordinal))
                    {
                        span = span.Slice(ContinuingSubwordPrefix.Length);
                    }

                    Span<byte> current = utf8bytes.Slice(incompleteUtf8BytesInBufferIndex + incompleteUtf8BytesInBuffer);

                    if (current.Length < span.Length)
                    {
                        return OperationStatus.InvalidData; // unexpected case
                    }

                    for (int i = 0; i < span.Length; i++)
                    {
                        current[i] = (byte)byteToUnicodeEncoding.UnicodeToByte[span[i]];
                    }

                    current = utf8bytes.Slice(incompleteUtf8BytesInBufferIndex, incompleteUtf8BytesInBuffer + span.Length);
                    if (!Helpers.ConvertUtf8ToUtf16(current, vocabBuffer.Slice(utf16CharsInBuffer), out int utf8BytesConsumed, out int utf16CharsWritten))
                    {
                        return OperationStatus.InvalidData; // unexpected case of malformed utf8 sequence
                    }

                    if (current.Length == utf8BytesConsumed) // encoding is complete
                    {
                        int completeCharsWritten = utf16CharsInBuffer + utf16CharsWritten;
                        if (completeCharsWritten > buffer.Length)
                        {
                            return OperationStatus.DestinationTooSmall;
                        }

                        vocabBuffer.Slice(0, completeCharsWritten).CopyTo(buffer);
                        buffer = buffer.Slice(completeCharsWritten);
                        charsWritten += completeCharsWritten;

                        incompleteUtf8BytesInBuffer = 0;
                        incompleteUtf8BytesInBufferIndex = 0;
                        utf16CharsInBuffer = 0;
                        idsConsumed += idsHangingCount + 1;
                        idsHangingCount = 0;
                    }
                    else
                    {
                        // Incomplete utf8 sequence, complete it in the next iteration
                        incompleteUtf8BytesInBuffer = current.Length - utf8BytesConsumed;
                        incompleteUtf8BytesInBufferIndex += utf8BytesConsumed;
                        utf16CharsInBuffer += utf16CharsWritten;
                        idsHangingCount++;
                    }

                    continue;
                }

                return OperationStatus.InvalidData; // encountered unknown id
            }

            return OperationStatus.Done;
        }

        /// Read the given files to extract the vocab and merges
        internal static async ValueTask<(Dictionary<StringSpanOrdinalKey, int>?, Vec<(string, string)>)> ReadModelDataAsync(Stream vocab, Stream? merges, bool useAsync, CancellationToken cancellationToken = default)
        {
            Dictionary<StringSpanOrdinalKey, int>? dic = useAsync
                                                         ? await JsonSerializer.DeserializeAsync(vocab, ModelSourceGenerationContext.Default.DictionaryStringSpanOrdinalKeyInt32, cancellationToken).ConfigureAwait(false)
                                                         : JsonSerializer.Deserialize(vocab, ModelSourceGenerationContext.Default.DictionaryStringSpanOrdinalKeyInt32);

            var m = useAsync ?
                    await ConvertMergesToHashmapAsync(merges, useAsync, cancellationToken).ConfigureAwait(false) :
                    ConvertMergesToHashmapAsync(merges, useAsync).GetAwaiter().GetResult();

            return (dic, m);
        }

        /// The vocabulary assigns a number to each token.
        private readonly Dictionary<StringSpanOrdinalKey, int> _vocab;

        private IReadOnlyDictionary<string, int>? _vocabOriginal;

        /// Contains the mapping between Pairs and their (rank, newId).
        internal Dictionary<Pair<int>, (int, int)> Merges { get; }

        /// Contains the cache for optimizing the encoding step.
        internal StringSpanOrdinalKeyCache<Word>? Cache { get; }

        internal static readonly int DefaultCacheCapacity = 10_000;

        /// Reversed vocabulary, to rebuild the text.
        internal SortedDictionary<int, string> VocabReverse { get; }

        /// Dropout probability for merges. 0 = no dropout is the default. At 1.0, tokenization will
        /// perform no merges, so the result will just be characters.
        internal float? Dropout { get; }

        /// Converts the merges strings (for example from `merges.txt` file) with the format
        /// "{pair_a} {pair_b}" into the format expected by the BPE struct
        internal static async ValueTask<Vec<(string, string)>> ConvertMergesToHashmapAsync(Stream? mergesStream, bool useAsync = false, CancellationToken cancellationToken = default)
        {
            if (mergesStream is null)
            {
                return new Vec<(string, string)>();
            }

            // Don't dispose the reader as it will dispose the underlying stream mergesStream. The caller is responsible for disposing the stream.
            StreamReader reader = new StreamReader(mergesStream);

            Vec<(string, string)> merges = new(1000);

            int lineNumber = 0;
            while (true)
            {
                string? line = useAsync ?
                    await Helpers.ReadLineAsync(reader, cancellationToken).ConfigureAwait(false) :
                    reader.ReadLine();

                if (line is null)
                {
                    break;
                }

                lineNumber++;
                if (line.StartsWith("#version", StringComparison.Ordinal) || line.Length == 0)
                {
                    continue;
                }

                int index = line.IndexOf(' ');
                if (index < 0 || index == line.Length - 1 || line.IndexOf(' ', index + 1) >= 0)
                {
                    throw new InvalidOperationException($"Invalid merger file format at line: {lineNumber}");
                }
                merges.Push((line.Substring(0, index), line.Substring(index + 1)));
            }

            return merges;
        }

        private readonly Dictionary<char, string> _charToString = new Dictionary<char, string>();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal string CharToString(char c)
        {
            if (_charToString.TryGetValue(c, out string? v))
            {
                return v;
            }

            string s = c.ToString();
            _charToString[c] = s;
            return s;
        }

        internal Word MergeWord(ReadOnlySpan<char> w, ref PriorityQueue<Merge>? priorityQueue)
        {
            Word word = Word.WithCapacity(w.Length);
            (int Id, int Len)? unk = null;
            int i = 0;

            Span<char> buffer = stackalloc char[256];
            scoped ReadOnlySpan<char> s;

            while (i < w.Length)
            {
                int length;

                if (Char.IsHighSurrogate(w[i]) && i < w.Length - 1 && Char.IsLowSurrogate(w[i + 1]))
                {
                    length = 2;
                    s = w.Slice(i, 2);
                }
                else
                {
                    length = 1;
                    s = w.Slice(i, 1);
                }

                // Add the `continuing_subword_prefix` if relevant
                if (i > 0 && ContinuingSubwordPrefix is not null)
                {
                    if (ContinuingSubwordPrefix.Length + s.Length <= buffer.Length)
                    {
                        ContinuingSubwordPrefix.AsSpan().CopyTo(buffer);
                        s.CopyTo(buffer.Slice(ContinuingSubwordPrefix.Length));
                        s = buffer.Slice(0, ContinuingSubwordPrefix.Length + s.Length);
                    }
                    else
                    {
#if NETCOREAPP
                        s = $"{ContinuingSubwordPrefix}{s}".AsSpan();
#else
                        string s1 = s.Length == 1 ? CharToString(s[0]) : s.ToString();
                        s = $"{ContinuingSubwordPrefix}{s1}".AsSpan();
#endif
                    }
                }

                // Add the `end_of_word_suffix` if relevant
                if (i + length >= w.Length && EndOfWordSuffix is not null)
                {
                    if (s.Length + EndOfWordSuffix.Length <= buffer.Length)
                    {
                        s.CopyTo(buffer);
                        EndOfWordSuffix.AsSpan().CopyTo(buffer.Slice(s.Length));
                        s = buffer.Slice(0, s.Length + EndOfWordSuffix.Length);
                    }
                    else
                    {
#if NETCOREAPP
                        s = $"{s}{EndOfWordSuffix}".AsSpan();
#else
                        string s1 = s.Length == 1 ? CharToString(s[0]) : s.ToString();
                        s = $"{s1}{EndOfWordSuffix}".AsSpan();
#endif
                    }
                }

                if (_vocab.TryGetValue(s, out int id))
                {
                    if (unk.HasValue)
                    {
                        word.Add(unk.Value.Id, unk.Value.Len);
                        unk = null;
                    }
                    word.Add(id, length);
                }
                else if (UnknownToken is not null)
                {
                    if (unk.HasValue)
                    {
                        if (FuseUnknownTokens)
                        {
                            // Fuse unk
                            unk = (unk.Value.Id, unk.Value.Len + length);
                        }
                        else
                        {
                            // Do not fuse unk, add the previous one
                            word.Add(unk.Value.Id, unk.Value.Len);
                            if (!_vocab.TryGetValue(UnknownToken, out int value))
                            {
                                throw new InvalidOperationException($"Unknown Token Out Of Vocabulary.");
                            }
                            unk = (value, length);
                        }
                    }
                    else
                    {
                        if (!_vocab.TryGetValue(UnknownToken, out int value))
                        {
                            throw new InvalidOperationException($"Unknown Token Out Of Vocabulary.");
                        }
                        unk = (value, length);
                    }
                }

                i += length;
            }

            if (unk.HasValue)
            {
                word.Add(unk.Value.Id, unk.Value.Len);
            }

            word.MergeAll(Merges, Dropout, ref priorityQueue);
            return word;
        }

        internal void WordToTokens(ref Word word, List<EncodedToken> tokens, int offset, ReadOnlySpan<int> mapping) => word.ToTokens(VocabReverse, tokens, offset, mapping);

        internal void EncodeWithCache(ReadOnlySpan<char> text, List<EncodedToken> tokens, int offset, ref PriorityQueue<Merge>? priorityQueue)
        {
            if (_specialTokens?.TryGetValue(text, out (int specialTokenId, string specialToken) value) is true)
            {
                tokens.Add(new EncodedToken(value.specialTokenId, value.specialToken, new Range(offset, offset + text.Length)));
                return;
            }

            scoped ReadOnlySpan<char> textSpan = text;
            scoped Span<char> token;
            scoped Span<int> mapping = Span<int>.Empty;
            char[]? tokenBuffer = null;
            int[]? mappingBuffer = null;

            if (ByteLevel)
            {
                int destinationMaxSize = Encoding.UTF8.GetMaxByteCount(text.Length);
                if (destinationMaxSize > BufferLength)
                {
                    tokenBuffer = ArrayPool<char>.Shared.Rent(destinationMaxSize);
                    token = tokenBuffer;

                    mappingBuffer = ArrayPool<int>.Shared.Rent(destinationMaxSize);
                    mapping = mappingBuffer;
                }
                else
                {
                    token = stackalloc char[destinationMaxSize];
                    mapping = stackalloc int[destinationMaxSize];
                }

                int encodedLength = Helpers.EncodeToUtf8AndTransform(text, token, mapping);
                textSpan = token.Slice(0, encodedLength);
                mapping = mapping.Slice(0, encodedLength);
            }

            Word word;
            if (Cache is not null)
            {
                if (Cache.TryGetValue(textSpan, out word))
                {
                    WordToTokens(ref word, tokens, offset, mapping);
                    return;
                }

                word = MergeWord(textSpan, ref priorityQueue);

                if (textSpan.Length <= MaxWordLengthToCache)
                {
                    Cache.Set(textSpan.ToString(), word);
                }
            }
            else
            {
                word = MergeWord(text, ref priorityQueue);
            }

            WordToTokens(ref word, tokens, offset, mapping);

            if (tokenBuffer is not null)
            {
                ArrayPool<char>.Shared.Return(tokenBuffer);
                Debug.Assert(mappingBuffer is not null);
                ArrayPool<int>.Shared.Return(mappingBuffer);
            }
        }

        internal int WordToIds(ref Word word, IList<int>? accumulatedIds, out int charsConsumed, int fullTextLength, int maxTokens)
        {
            if (word.SymbolsCount <= maxTokens)
            {
                charsConsumed = fullTextLength;
                if (accumulatedIds is not null)
                {
                    word.PopulateIds(accumulatedIds);
                }

                return word.SymbolsCount;
            }

            if (accumulatedIds is not null)
            {
                return word.PopulateIdsUpToMax(accumulatedIds, maxTokens, out charsConsumed);
            }

            return word.CountIdsUpToMax(maxTokens, out charsConsumed);
        }

        internal int WordToIdsFromEnd(ref Word word, IList<int>? accumulatedIds, out int textIndex, int fullTextLength, int maxTokens)
        {
            if (word.SymbolsCount <= maxTokens)
            {
                textIndex = 0;
                if (accumulatedIds is not null)
                {
                    word.PopulateIds(accumulatedIds);
                }

                return word.SymbolsCount;
            }

            if (accumulatedIds is not null)
            {
                return word.PopulateIdsUpToMaxFromEnd(accumulatedIds, maxTokens, fullTextLength, out textIndex);
            }

            return word.CountIdsUpToMaxFromEnd(maxTokens, fullTextLength, out textIndex);
        }

        private int EncodeToIdsWithCache(ReadOnlySpan<char> text, List<int>? accumulatedIds, int maxTokens, out int charsConsumed, ref PriorityQueue<Merge>? priorityQueue)
        {
            if (_specialTokens?.TryGetValue(text, out (int specialTokenId, string specialToken) value) is true && maxTokens > 0)
            {
                accumulatedIds?.Add(value.specialTokenId);
                charsConsumed = text.Length;
                return 1;
            }

            Word word;

            scoped ReadOnlySpan<char> textSpan = text;

            scoped Span<char> token;
            scoped Span<int> mapping = Span<int>.Empty;
            char[]? tokenBuffer = null;
            int[]? mappingBuffer = null;

            if (ByteLevel)
            {
                int destinationMaxSize = Encoding.UTF8.GetMaxByteCount(text.Length);
                if (destinationMaxSize > BufferLength)
                {
                    tokenBuffer = ArrayPool<char>.Shared.Rent(destinationMaxSize);
                    token = tokenBuffer;

                    mappingBuffer = ArrayPool<int>.Shared.Rent(destinationMaxSize);
                    mapping = mappingBuffer;
                }
                else
                {
                    token = stackalloc char[destinationMaxSize];
                    mapping = stackalloc int[destinationMaxSize];
                }

                int encodedLength = Helpers.EncodeToUtf8AndTransform(text, token, mapping);
                textSpan = token.Slice(0, encodedLength);
            }

            if (Cache is not null)
            {
                if (Cache.TryGetValue(textSpan, out Word hit))
                {
                    int res = WordToIds(ref hit, accumulatedIds, out charsConsumed, textSpan.Length, maxTokens);
                    if (ByteLevel)
                    {
                        charsConsumed = charsConsumed >= textSpan.Length ? text.Length : mapping[charsConsumed];
                    }

                    return res;
                }

                word = MergeWord(textSpan, ref priorityQueue);

                if (textSpan.Length <= MaxWordLengthToCache)
                {
                    Cache.Set(textSpan.ToString(), word);
                }
            }
            else
            {
                word = MergeWord(textSpan, ref priorityQueue);
            }

            int result = WordToIds(ref word, accumulatedIds, out charsConsumed, textSpan.Length, maxTokens);
            if (ByteLevel)
            {
                charsConsumed = charsConsumed >= textSpan.Length ? text.Length : mapping[charsConsumed];
            }

            if (tokenBuffer is not null)
            {
                ArrayPool<char>.Shared.Return(tokenBuffer);
                Debug.Assert(mappingBuffer is not null);
                ArrayPool<int>.Shared.Return(mappingBuffer);
            }

            return result;
        }

        internal int EncodeToIdsFromEndWithCache(ReadOnlySpan<char> text, IList<int>? accumulatedIds, int maxTokens, out int textIndex, ref PriorityQueue<Merge>? priorityQueue)
        {
            Word word;

            if (_specialTokens?.TryGetValue(text, out (int specialTokenId, string specialToken) value) is true && maxTokens > 0)
            {
                accumulatedIds?.Add(value.specialTokenId);
                textIndex = 0;
                return 1;
            }

            scoped ReadOnlySpan<char> textSpan = text;

            scoped Span<char> token;
            scoped Span<int> mapping = Span<int>.Empty;
            char[]? tokenBuffer = null;
            int[]? mappingBuffer = null;

            if (ByteLevel)
            {
                int destinationMaxSize = Encoding.UTF8.GetMaxByteCount(text.Length);
                if (destinationMaxSize > BufferLength)
                {
                    tokenBuffer = ArrayPool<char>.Shared.Rent(destinationMaxSize);
                    token = tokenBuffer;

                    mappingBuffer = ArrayPool<int>.Shared.Rent(destinationMaxSize);
                    mapping = mappingBuffer;
                }
                else
                {
                    token = stackalloc char[destinationMaxSize];
                    mapping = stackalloc int[destinationMaxSize];
                }

                int encodedLength = Helpers.EncodeToUtf8AndTransform(text, token, mapping);
                textSpan = token.Slice(0, encodedLength);
            }

            if (Cache is not null)
            {
                if (Cache.TryGetValue(textSpan, out Word hit))
                {
                    int res = WordToIdsFromEnd(ref hit, accumulatedIds, out textIndex, textSpan.Length, maxTokens);

                    if (ByteLevel)
                    {
                        textIndex = textIndex >= textSpan.Length ? text.Length : mapping[textIndex];
                    }

                    return res;
                }

                word = MergeWord(textSpan, ref priorityQueue);

                if (textSpan.Length <= MaxWordLengthToCache)
                {
                    Cache.Set(textSpan.ToString(), word);
                }
            }
            else
            {
                word = MergeWord(textSpan, ref priorityQueue);
            }

            int result = WordToIdsFromEnd(ref word, accumulatedIds, out textIndex, textSpan.Length, maxTokens);

            if (ByteLevel)
            {
                textIndex = textIndex >= textSpan.Length ? text.Length : mapping[textIndex];
            }

            if (tokenBuffer is not null)
            {
                ArrayPool<char>.Shared.Return(tokenBuffer);
                Debug.Assert(mappingBuffer is not null);
                ArrayPool<int>.Shared.Return(mappingBuffer);
            }

            return result;
        }
    }
}
