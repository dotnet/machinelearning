// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Represent the Byte Pair Encoding model.
    /// </summary>
    public sealed class Bpe : Tokenizer
    {
        /// A [Byte Pair Encoding](https://www.aclweb.org/anthology/P16-1162/) model.

        private const int MaxWordLengthToCache = 15;
        private string? _unknownToken;
        private int? _unknownTokenId;
        private readonly PreTokenizer? _preTokenizer;
        private readonly Normalizer? _normalizer;

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
                    throw new InvalidOperationException($"Unknown Token '{value}' was not present in '{nameof(Vocab)}'.");
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
        /// An optional suffix to characterize and end-of-word sub-word
        /// </summary>
        public string? EndOfWordSuffix { get; }

        /// <summary>
        /// Gets or sets whether allowing multiple unknown tokens get fused
        /// </summary>
        public bool FuseUnknownTokens { get; }

        /// <summary>
        /// Construct a new Bpe model object to use for text encoding.
        /// </summary>
        /// <param name="vocabFile">The JSON file path containing the dictionary of string keys and their ids.</param>
        /// <param name="mergesFile">The file path containing the tokens's pairs list.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="unknownToken"> The unknown token to be used by the model.</param>
        /// <param name="continuingSubwordPrefix">The prefix to attach to sub-word units that don’t represent a beginning of word.</param>
        /// <param name="endOfWordSuffix">The suffix to attach to sub-word units that represent an end of word.</param>
        /// <param name="fuseUnknownTokens">Indicate whether allowing multiple unknown tokens get fused.</param>
        public Bpe(string vocabFile, string? mergesFile, PreTokenizer? preTokenizer = null, Normalizer? normalizer = null, string? unknownToken = null, string? continuingSubwordPrefix = null, string? endOfWordSuffix = null, bool fuseUnknownTokens = false) :
            this(vocabFile is null ? throw new ArgumentNullException(nameof(vocabFile)) : File.Open(vocabFile, FileMode.Open, FileAccess.Read),
                mergesFile is null ? null : File.Open(mergesFile, FileMode.Open, FileAccess.Read), preTokenizer, normalizer, unknownToken, continuingSubwordPrefix, endOfWordSuffix, fuseUnknownTokens, disposeStreams: true)
        {
        }

        /// <summary>
        /// Construct a new Bpe model object to use for text encoding.
        /// </summary>
        /// <param name="vocabStream">The JSON stream containing the dictionary of string keys and their ids.</param>
        /// <param name="mergesStream">The stream containing the tokens's pairs list.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="unknownToken"> The unknown token to be used by the model.</param>
        /// <param name="continuingSubwordPrefix">The prefix to attach to sub-word units that don’t represent a beginning of word.</param>
        /// <param name="endOfWordSuffix">The suffix to attach to sub-word units that represent an end of word.</param>
        /// <param name="fuseUnknownTokens">Indicate whether allowing multiple unknown tokens get fused.</param>
        public Bpe(Stream vocabStream, Stream? mergesStream, PreTokenizer? preTokenizer = null, Normalizer? normalizer = null, string? unknownToken = null, string? continuingSubwordPrefix = null, string? endOfWordSuffix = null, bool fuseUnknownTokens = false) :
                this(vocabStream, mergesStream, preTokenizer, normalizer, unknownToken, continuingSubwordPrefix, endOfWordSuffix, fuseUnknownTokens, disposeStreams: false)
        {
        }

        private Bpe(Stream vocabStream, Stream? mergesStream, PreTokenizer? preTokenizer, Normalizer? normalizer, string? unknownToken, string? continuingSubwordPrefix, string? endOfWordSuffix, bool fuseUnknownTokens, bool disposeStreams)
        {
            try
            {
                if (vocabStream is null)
                {
                    throw new ArgumentNullException(nameof(vocabStream));
                }

                FuseUnknownTokens = fuseUnknownTokens;
                ContinuingSubwordPrefix = continuingSubwordPrefix;
                EndOfWordSuffix = endOfWordSuffix;
                _preTokenizer = preTokenizer ?? WhiteSpace.Instance; // Default to WhiteSpace pre-tokenizer
                _normalizer = normalizer;

                (Dictionary<StringSpanOrdinalKey, int>? vocab1, Vec<(string, string)> merges) = ReadModelData(vocabStream, mergesStream);
                _vocab = vocab1 ?? new Dictionary<StringSpanOrdinalKey, int>();
                Cache = new StringSpanOrdinalKeyCache<Word>();

                VocabReverse = new();

                foreach (KeyValuePair<StringSpanOrdinalKey, int> kvp in _vocab)
                {
                    VocabReverse.Add(kvp.Value, kvp.Key.Data!);
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
            finally
            {
                if (disposeStreams)
                {
                    vocabStream.Dispose();
                    mergesStream?.Dispose();
                }
            }
        }

        /// <summary>
        /// Gets the PreTokenizer used by the Tokenizer.
        /// </summary>
        public override PreTokenizer? PreTokenizer => _preTokenizer;

        /// <summary>
        /// Gets the Normalizer in use by the Tokenizer.
        /// </summary>
        public override Normalizer? Normalizer => _normalizer;

        /// <summary>
        /// Encodes input text a list of <see cref="Token" />s with string value of the token, id, and offset.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The tokenization result includes a list of <see cref="Token" />s with string value of the token, id, and offset.</returns>
        public override IReadOnlyList<Token> Encode(string text, out string? normalizedString, bool considerPreTokenization = true, bool considerNormalization = true) => Encode(text, Span<char>.Empty, out normalizedString, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Encodes input text a list of <see cref="Token" />s with string value of the token, id, and offset.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The tokenization result includes a list of <see cref="Token" />s with string value of the token, id, and offset.</returns>
        public override IReadOnlyList<Token> Encode(ReadOnlySpan<char> text, out string? normalizedString, bool considerPreTokenization = true, bool considerNormalization = true) => Encode(null, text, out normalizedString, considerPreTokenization, considerNormalization);

        private IReadOnlyList<Token> Encode(string? text, ReadOnlySpan<char> textSpan, out string? normalizedString, bool considerPreTokenization, bool considerNormalization)
        {
            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedString = null;
                return [];
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(text, textSpan, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedString, out ReadOnlySpan<char> textSpanToEncode);

            List<Token> tokens = new();
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

            return tokens;
        }

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public override IReadOnlyList<int> EncodeToIds(string text, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(text, Span<char>.Empty, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public override IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(null, text, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Encodes input text to token Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public override IReadOnlyList<int> EncodeToIds(string text, int maxTokenCount, out string? normalizedString, out int textLength, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(text, Span<char>.Empty, considerPreTokenization, considerNormalization, out normalizedString, out textLength, maxTokenCount);

        /// <summary>
        /// Encodes input text to token Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public override IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedString, out int textLength, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(null, text, considerPreTokenization, considerNormalization, out normalizedString, out textLength, maxTokenCount);

        private IReadOnlyList<int> EncodeToIds(string? text, ReadOnlySpan<char> textSpan, bool considerPreTokenization, bool considerNormalization, out string? normalizedString, out int textLength, int maxTokenCount = int.MaxValue)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                textLength = 0;
                normalizedString = null;
                return [];
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(text, textSpan, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedString, out ReadOnlySpan<char> textSpanToEncode);

            List<int> ids = new();
            PriorityQueue<Merge>? priorityQueue = null;

            if (splits is not null)
            {
                textLength = 0;
                foreach ((int Offset, int Length) split in splits)
                {
                    EncodeToIdsWithCache(textSpanToEncode.Slice(split.Offset, split.Length), ids, maxTokenCount - ids.Count, out int length, ref priorityQueue);
                    textLength = split.Offset + length;

                    if (length < split.Length || ids.Count >= maxTokenCount)
                    {
                        break;
                    }
                }
            }
            else
            {
                EncodeToIdsWithCache(textSpanToEncode, ids, maxTokenCount, out textLength, ref priorityQueue);
            }

            return ids;
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        public override int CountTokens(string text, bool considerPreTokenization = true, bool considerNormalization = true)
            => CountTokens(text, Span<char>.Empty, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        public override int CountTokens(ReadOnlySpan<char> text, bool considerPreTokenization = true, bool considerNormalization = true)
            => CountTokens(null, text, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the start within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the text or the <paramref name="normalizedString"/> if the normalization is enabled.
        /// </returns>
        public override int IndexOfTokenCount(string text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            tokenCount = CountTokens(text, Span<char>.Empty, considerPreTokenization, considerNormalization, out normalizedString, out int textLength, maxTokenCount);
            return textLength;
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the start within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the text or the <paramref name="normalizedString"/> if the normalization is enabled.
        /// </returns>
        public override int IndexOfTokenCount(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            tokenCount = CountTokens(null, text, considerPreTokenization, considerNormalization, out normalizedString, out int textLength, maxTokenCount);
            return textLength;
        }

        private int CountTokens(string? text, ReadOnlySpan<char> textSpan, bool considerPreTokenization, bool considerNormalization, out string? normalizedString, out int textLength, int maxTokenCount = int.MaxValue)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            textLength = 0;
            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedString = null;
                return 0;
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(text, textSpan, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedString, out ReadOnlySpan<char> textSpanToEncode);

            PriorityQueue<Merge>? priorityQueue = null;
            int count = 0;
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

            return count;
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the text or the <paramref name="normalizedString"/> if normalization is enabled;
        /// conversely, if all tokens fit, the result will be 0.
        /// </returns>
        public override int LastIndexOfTokenCount(string text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
            => LastIndexOf(text, Span<char>.Empty, maxTokenCount, considerPreTokenization, considerNormalization, out normalizedString, out tokenCount);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the <paramref name="normalizedString"/>; conversely, if all tokens fit, the result will be 0.
        /// </returns>
        public override int LastIndexOfTokenCount(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
            => LastIndexOf(null, text, maxTokenCount, considerPreTokenization, considerNormalization, out normalizedString, out tokenCount);

        private int LastIndexOf(string? text, ReadOnlySpan<char> textSpan, int maxTokenCount, bool considerPreTokenization, bool considerNormalization, out string? normalizedString, out int tokenCount)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The max token count must be greater than 0.");
            }

            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedString = null;
                tokenCount = 0;
                return 0;
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(text, textSpan, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedString, out ReadOnlySpan<char> textSpanToEncode);

            PriorityQueue<Merge>? priorityQueue = null;

            if (splits is not null)
            {
                tokenCount = 0;
                foreach ((int Offset, int Length) split in splits.Reverse())
                {
                    tokenCount += EncodeToIdsFromEndWithCache(textSpanToEncode.Slice(split.Offset, split.Length), null, maxTokenCount - tokenCount, out int textIndex, ref priorityQueue);
                    if (textIndex > 0 || tokenCount >= maxTokenCount)
                    {
                        return split.Offset + textIndex;
                    }
                }
            }
            else
            {
                tokenCount = EncodeToIdsFromEndWithCache(textSpanToEncode, null, maxTokenCount, out int textLength, ref priorityQueue);
                return textLength;
            }

            return 0;
        }

        /// <summary>
        /// Map the token to encoded Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <returns>The mapped Id of the token.</returns>
        public override int? MapTokenToId(ReadOnlySpan<char> token) => _vocab.TryGetValue(token, out int value) ? value : null;

        /// <summary>
        /// Map the encoded Id to the token.
        /// </summary>
        /// <param name="id">The Id to map to the token.</param>
        /// <returns>The mapped token of the Id.</returns>
        public override string? MapIdToToken(int id)
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
        public IReadOnlyDictionary<string, int> Vocab => _vocabOriginal ??= _vocab.ToDictionary(kvp => kvp.Key.Data!, kvp => kvp.Value);

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <returns>The decoded string.</returns>
        public override string? Decode(IEnumerable<int> ids) => Decode(ids, considerSpecialTokens: true);

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="considerSpecialTokens">Indicate whether to consider special tokens or not.</param>
        /// <returns>The decoded string.</returns>
        public string? Decode(IEnumerable<int> ids, bool considerSpecialTokens)
        {
            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
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

        /// Read the given files to extract the vocab and merges
        internal static (Dictionary<StringSpanOrdinalKey, int>?, Vec<(string, string)>) ReadModelData(Stream vocab, Stream? merges)
        {
            JsonSerializerOptions options = new() { Converters = { StringSpanOrdinalKeyConverter.Instance } };
            Dictionary<StringSpanOrdinalKey, int>? dic = JsonSerializer.Deserialize<Dictionary<StringSpanOrdinalKey, int>>(vocab, options) as Dictionary<StringSpanOrdinalKey, int>;

            return (dic, ConvertMergesToHashmap(merges));
        }

        /// The vocabulary assigns a number to each token.
        private readonly Dictionary<StringSpanOrdinalKey, int> _vocab;

        private Dictionary<string, int>? _vocabOriginal;

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
        internal static Vec<(string, string)> ConvertMergesToHashmap(Stream? mergesStream)
        {
            if (mergesStream is null)
            {
                return new Vec<(string, string)>();
            }

            using StreamReader reader = new StreamReader(mergesStream);

            Vec<(string, string)> merges = new(1000);

            int lineNumber = 0;
            while (true)
            {
                string? line = reader.ReadLine();
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

        internal void WordToTokens(ref Word word, List<Token> tokens, int offset) => word.ToTokens(VocabReverse, tokens, offset);

        internal void EncodeWithCache(ReadOnlySpan<char> text, List<Token> tokens, int offset, ref PriorityQueue<Merge>? priorityQueue)
        {
            Word word;
            if (Cache is not null)
            {
                if (Cache.TryGetValue(text, out word))
                {
                    WordToTokens(ref word, tokens, offset);
                    return;
                }

                word = MergeWord(text, ref priorityQueue);

                if (text.Length <= MaxWordLengthToCache)
                {
                    Cache.Set(text.ToString(), word);
                }
            }
            else
            {
                word = MergeWord(text, ref priorityQueue);
            }

            WordToTokens(ref word, tokens, offset);
        }

        internal int WordToIds(ref Word word, IList<int>? accumulatedIds, out int textLength, int fullTextLength, int maxTokens)
        {
            if (word.SymbolsCount <= maxTokens)
            {
                textLength = fullTextLength;
                if (accumulatedIds is not null)
                {
                    word.PopulateIds(accumulatedIds);
                }

                return word.SymbolsCount;
            }

            if (accumulatedIds is not null)
            {
                return word.PopulateIdsUpToMax(accumulatedIds, maxTokens, out textLength);
            }

            return word.CountIdsUpToMax(maxTokens, out textLength);
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

        private int EncodeToIdsWithCache(ReadOnlySpan<char> text, List<int>? accumulatedIds, int maxTokens, out int textLength, ref PriorityQueue<Merge>? priorityQueue)
        {
            Word word;

            if (Cache is not null)
            {
                if (Cache.TryGetValue(text, out Word hit))
                {
                    return WordToIds(ref hit, accumulatedIds, out textLength, text.Length, maxTokens);
                }

                word = MergeWord(text, ref priorityQueue);

                if (text.Length <= MaxWordLengthToCache)
                {
                    Cache.Set(text.ToString(), word);
                }
            }
            else
            {
                word = MergeWord(text, ref priorityQueue);
            }

            return WordToIds(ref word, accumulatedIds, out textLength, text.Length, maxTokens);
        }

        internal int EncodeToIdsFromEndWithCache(ReadOnlySpan<char> text, IList<int>? accumulatedIds, int maxTokens, out int textIndex, ref PriorityQueue<Merge>? priorityQueue)
        {
            Word word;

            if (Cache is not null)
            {
                if (Cache.TryGetValue(text, out Word hit))
                {
                    return WordToIdsFromEnd(ref hit, accumulatedIds, out textIndex, text.Length, maxTokens);
                }

                word = MergeWord(text, ref priorityQueue);

                if (text.Length <= MaxWordLengthToCache)
                {
                    Cache.Set(text.ToString(), word);
                }
            }
            else
            {
                word = MergeWord(text, ref priorityQueue);
            }

            return WordToIdsFromEnd(ref word, accumulatedIds, out textIndex, text.Length, maxTokens);
        }
    }
}
