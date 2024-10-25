// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Represent the WordPiece tokenizer.
    /// </summary>
    /// <remarks>
    /// The WordPiece tokenizer is a sub-word tokenizer that is used in BERT and other transformer models.
    /// The implementation is based on the Hugging Face WordPiece tokenizer https://huggingface.co/docs/tokenizers/api/models#tokenizers.models.WordPiece.
    /// </remarks>
    public partial class WordPieceTokenizer : Tokenizer
    {
        private readonly PreTokenizer? _preTokenizer;
        private readonly Normalizer? _normalizer;
        private readonly Dictionary<StringSpanOrdinalKey, int> _vocab;
        private readonly Dictionary<int, string> _vocabReverse;

        internal const string DefaultContinuingSubwordPrefix = "##";
        internal const int DefaultMaxInputCharsPerWord = 100;

        internal WordPieceTokenizer(
                    Dictionary<StringSpanOrdinalKey, int> vocab,
                    Dictionary<int, string> vocabReverse,
                    PreTokenizer? preTokenizer,
                    Normalizer? normalizer,
                    IReadOnlyDictionary<string, int>? specialTokens,
                    string unknownToken,
                    string continuingSubwordPrefix = DefaultContinuingSubwordPrefix,
                    int maxInputCharsPerWord = DefaultMaxInputCharsPerWord)
        {
            Debug.Assert(vocab is not null);
            Debug.Assert(vocabReverse is not null);
            _vocab = vocab!;
            _vocabReverse = vocabReverse!;
            SpecialTokens = specialTokens;
            SpecialTokensReverse = specialTokens is not null ? specialTokens.ToDictionary(kvp => kvp.Value, kvp => kvp.Key) : null;

            if (unknownToken is null)
            {
                throw new ArgumentNullException(nameof(unknownToken));
            }

            if (continuingSubwordPrefix is null)
            {
                throw new ArgumentNullException(nameof(continuingSubwordPrefix));
            }

            if (maxInputCharsPerWord <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxInputCharsPerWord), "The maximum number of characters per word must be greater than zero.");
            }

            if (!vocab!.TryGetValue(unknownToken, out int id))
            {
                throw new ArgumentException($"The unknown token '{unknownToken}' is not in the vocabulary.");
            }

            UnknownToken = unknownToken;
            UnknownTokenId = id;
            ContinuingSubwordPrefix = continuingSubwordPrefix;
            MaxInputCharsPerWord = maxInputCharsPerWord;

            _preTokenizer = preTokenizer ?? PreTokenizer.CreateWhiteSpacePreTokenizer(specialTokens);
            _normalizer = normalizer;
        }

        /// <summary>
        /// Gets the unknown token ID.
        /// A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
        /// </summary>
        public int UnknownTokenId { get; }

        /// <summary>
        /// Gets the prefix to use for sub-words that are not the first part of a word.
        /// </summary>
        public string ContinuingSubwordPrefix { get; }

        /// <summary>
        /// Gets the maximum number of characters to authorize in a single word.
        /// </summary>
        public int MaxInputCharsPerWord { get; }

        internal static async ValueTask<(Dictionary<StringSpanOrdinalKey, int>, Dictionary<int, string>)> LoadVocabAsync(Stream vocabStream, bool useAsync, CancellationToken cancellationToken = default)
        {
            if (vocabStream is null)
            {
                throw new ArgumentNullException(nameof(vocabStream));
            }

            Dictionary<StringSpanOrdinalKey, int> vocab = new Dictionary<StringSpanOrdinalKey, int>();
            Dictionary<int, string> vocabReverse = new Dictionary<int, string>();

            StreamReader reader = new StreamReader(vocabStream);
            string? line = useAsync ? await Helpers.ReadLineAsync(reader, cancellationToken).ConfigureAwait(false) : reader.ReadLine();
            int lineNumber = 0;

            while (line is not null)
            {
                if (line.Length != 0)
                {
                    vocab.Add(new StringSpanOrdinalKey(line), lineNumber);
                    vocabReverse.Add(lineNumber, line);
                }

                lineNumber++;
                line = useAsync ? await Helpers.ReadLineAsync(reader, cancellationToken).ConfigureAwait(false) : reader.ReadLine();
            }

            return (vocab, vocabReverse);
        }

        /// <summary>
        /// Create a new instance of the <see cref="WordPieceTokenizer"/> class.
        /// </summary>
        /// <param name="vocabFilePath">The path to the WordPiece vocab file.</param>
        /// <param name="preTokenizer">The PreTokenizer to use.</param>
        /// <param name="normalizer">The Normalizer to use.</param>
        /// <param name="specialTokens">The dictionary containing the special tokens and their corresponding ids.</param>
        /// <param name="unknownToken">The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.</param>
        /// <param name="continuingSubwordPrefix">The prefix to use for sub-words that are not the first part of a word.</param>
        /// <param name="maxInputCharsPerWord">The maximum number of characters to authorize in a single word.</param>
        /// <returns>A new instance of the <see cref="WordPieceTokenizer"/> class.</returns>
        /// <remarks>
        /// If the <paramref name="preTokenizer"/> is null, the whitespace pre-tokenizer will be used.
        /// </remarks>
        public static WordPieceTokenizer Create(
                        string vocabFilePath,
                        PreTokenizer? preTokenizer = null,
                        Normalizer? normalizer = null,
                        IReadOnlyDictionary<string, int>? specialTokens = null,
                        string unknownToken = "[UNK]",
                        string continuingSubwordPrefix = DefaultContinuingSubwordPrefix,
                        int maxInputCharsPerWord = DefaultMaxInputCharsPerWord) =>
            Create(string.IsNullOrEmpty(vocabFilePath) ? throw new ArgumentNullException(nameof(vocabFilePath)) : File.OpenRead(vocabFilePath), preTokenizer, normalizer, specialTokens, unknownToken, continuingSubwordPrefix, maxInputCharsPerWord, disposeStream: true);

        /// <summary>
        /// Create a new instance of the <see cref="WordPieceTokenizer"/> class.
        /// </summary>
        /// <param name="vocabStream">The path to the WordPiece vocab file.</param>
        /// <param name="preTokenizer">The PreTokenizer to use.</param>
        /// <param name="normalizer">The Normalizer to use.</param>
        /// <param name="specialTokens">The dictionary containing the special tokens and their corresponding ids.</param>
        /// <param name="unknownToken">The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.</param>
        /// <param name="continuingSubwordPrefix">The prefix to use for sub-words that are not the first part of a word.</param>
        /// <param name="maxInputCharsPerWord">The maximum number of characters to authorize in a single word.</param>
        /// <returns>A new instance of the <see cref="WordPieceTokenizer"/> class.</returns>
        /// <remarks>
        /// If the <paramref name="preTokenizer"/> is null, the whitespace pre-tokenizer will be used.
        /// </remarks>
        public static WordPieceTokenizer Create(
                            Stream vocabStream,
                            PreTokenizer? preTokenizer = null,
                            Normalizer? normalizer = null,
                            IReadOnlyDictionary<string, int>? specialTokens = null,
                            string unknownToken = "[UNK]",
                            string continuingSubwordPrefix = DefaultContinuingSubwordPrefix,
                            int maxInputCharsPerWord = DefaultMaxInputCharsPerWord) => Create(vocabStream, preTokenizer, normalizer, specialTokens, unknownToken, continuingSubwordPrefix, maxInputCharsPerWord, disposeStream: false);

        private static WordPieceTokenizer Create(
                            Stream vocabStream,
                            PreTokenizer? preTokenizer,
                            Normalizer? normalizer,
                            IReadOnlyDictionary<string, int>? specialTokens,
                            string unknownToken,
                            string continuingSubwordPrefix,
                            int maxInputCharsPerWord,
                            bool disposeStream)
        {
            if (vocabStream is null)
            {
                throw new ArgumentNullException(nameof(vocabStream));
            }

            try
            {
                (Dictionary<StringSpanOrdinalKey, int> vocab, Dictionary<int, string> vocabReverse) = LoadVocabAsync(vocabStream, useAsync: false).GetAwaiter().GetResult();

                return new WordPieceTokenizer(vocab, vocabReverse, preTokenizer, normalizer, specialTokens, unknownToken, continuingSubwordPrefix, maxInputCharsPerWord);
            }
            finally
            {
                if (disposeStream)
                {
                    vocabStream.Dispose();
                }
            }
        }

        /// <summary>
        /// Create a new instance of the <see cref="WordPieceTokenizer"/> class asynchronously.
        /// </summary>
        /// <param name="vocabFilePath">The path to the WordPiece vocab file.</param>
        /// <param name="preTokenizer">The PreTokenizer to use.</param>
        /// <param name="normalizer">The Normalizer to use.</param>
        /// <param name="specialTokens">The dictionary containing the special tokens and their corresponding ids.</param>
        /// <param name="unknownToken">The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.</param>
        /// <param name="continuingSubwordPrefix">The prefix to use for sub-words that are not the first part of a word.</param>
        /// <param name="maxInputCharsPerWord">The maximum number of characters to authorize in a single word.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>A new instance of the <see cref="WordPieceTokenizer"/> class.</returns>
        /// <remarks>
        /// If the <paramref name="preTokenizer"/> is null, the whitespace pre-tokenizer will be used.
        /// </remarks>
        public static async Task<WordPieceTokenizer> CreateAsync(
                                string vocabFilePath,
                                PreTokenizer? preTokenizer = null,
                                Normalizer? normalizer = null,
                                IReadOnlyDictionary<string, int>? specialTokens = null,
                                string unknownToken = "[UNK]",
                                string continuingSubwordPrefix = DefaultContinuingSubwordPrefix,
                                int maxInputCharsPerWord = DefaultMaxInputCharsPerWord,
                                CancellationToken cancellationToken = default) =>
            await CreateAsync(
                    string.IsNullOrEmpty(vocabFilePath) ? throw new ArgumentNullException(nameof(vocabFilePath)) : File.OpenRead(vocabFilePath),
                    preTokenizer,
                    normalizer,
                    specialTokens,
                    unknownToken,
                    continuingSubwordPrefix,
                    maxInputCharsPerWord,
                    cancellationToken,
                    disposeStream: true).ConfigureAwait(false);

        /// <summary>
        /// Create a new instance of the <see cref="WordPieceTokenizer"/> class asynchronously.
        /// </summary>
        /// <param name="vocabStream">The path to the WordPiece vocab file.</param>
        /// <param name="preTokenizer">The PreTokenizer to use.</param>
        /// <param name="normalizer">The Normalizer to use.</param>
        /// <param name="specialTokens">The dictionary containing the special tokens and their corresponding ids.</param>
        /// <param name="unknownToken">The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.</param>
        /// <param name="continuingSubwordPrefix">The prefix to use for sub-words that are not the first part of a word.</param>
        /// <param name="maxInputCharsPerWord">The maximum number of characters to authorize in a single word.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>A new instance of the <see cref="WordPieceTokenizer"/> class.</returns>
        /// <remarks>
        /// If the <paramref name="preTokenizer"/> is null, the whitespace pre-tokenizer will be used.
        /// </remarks>
        public static async Task<WordPieceTokenizer> CreateAsync(
                                Stream vocabStream,
                                PreTokenizer? preTokenizer = null,
                                Normalizer? normalizer = null,
                                IReadOnlyDictionary<string, int>? specialTokens = null,
                                string unknownToken = "[UNK]",
                                string continuingSubwordPrefix = DefaultContinuingSubwordPrefix,
                                int maxInputCharsPerWord = DefaultMaxInputCharsPerWord,
                                CancellationToken cancellationToken = default) =>
            await CreateAsync(vocabStream, preTokenizer, normalizer, specialTokens, unknownToken, continuingSubwordPrefix, maxInputCharsPerWord, cancellationToken, disposeStream: false).ConfigureAwait(false);

        private static async Task<WordPieceTokenizer> CreateAsync(
                                Stream vocabStream,
                                PreTokenizer? preTokenizer,
                                Normalizer? normalizer,
                                IReadOnlyDictionary<string, int>? specialTokens,
                                string unknownToken,
                                string continuingSubwordPrefix,
                                int maxInputCharsPerWord,
                                CancellationToken cancellationToken,
                                bool disposeStream)
        {
            if (vocabStream is null)
            {
                throw new ArgumentNullException(nameof(vocabStream));
            }

            try
            {
                (Dictionary<StringSpanOrdinalKey, int> vocab, Dictionary<int, string> vocabReverse) = await LoadVocabAsync(vocabStream, useAsync: true, cancellationToken);

                return new WordPieceTokenizer(vocab, vocabReverse, preTokenizer, normalizer, specialTokens, unknownToken, continuingSubwordPrefix, maxInputCharsPerWord);
            }
            finally
            {
                if (disposeStream)
                {
                    vocabStream.Dispose();
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
        /// Gets the unknown token.
        /// A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
        /// </summary>
        public string UnknownToken { get; }

        /// <summary>
        /// Gets the special tokens and their corresponding ids.
        /// </summary>
        public IReadOnlyDictionary<string, int>? SpecialTokens { get; }

        /// <summary>
        /// Gets the Ids to tokens mapping for special tokens.
        /// </summary>
        internal IReadOnlyDictionary<int, string>? SpecialTokensReverse { get; }

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
                return new EncodeResults<EncodedToken> { NormalizedText = null, Tokens = [], CharsConsumed = 0 };
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(
                                                                text,
                                                                textSpan,
                                                                settings.ConsiderPreTokenization,
                                                                settings.ConsiderNormalization,
                                                                _normalizer,
                                                                _preTokenizer,
                                                                out string? normalizedString,
                                                                out ReadOnlySpan<char> textSpanToEncode,
                                                                out int charsConsumed);

            List<EncodedToken> tokens = new();

            if (splits is not null)
            {
                foreach ((int Offset, int Length) split in splits)
                {
                    EncodeToTokens(textSpanToEncode.Slice(split.Offset, split.Length), tokens, split.Offset);
                }
            }
            else
            {
                EncodeToTokens(textSpanToEncode, tokens, 0);
            }

            return new EncodeResults<EncodedToken> { NormalizedText = normalizedString, Tokens = tokens, CharsConsumed = charsConsumed };
        }

        /// <summary>
        /// Encode text to a list of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="tokens">The list of tokens to populate.</param>
        /// <param name="offset">The offset to start encoding from.</param>
        private void EncodeToTokens(ReadOnlySpan<char> text, List<EncodedToken> tokens, int offset)
        {
            Debug.Assert(!text.IsEmpty);

            if (text.Length > MaxInputCharsPerWord)
            {
                tokens.Add(new EncodedToken(UnknownTokenId, UnknownToken, new Range(offset, offset + text.Length)));
                return;
            }

            int maxLength = MaxInputCharsPerWord + ContinuingSubwordPrefix.Length;
            char[]? arrayPool = maxLength <= 250 ? null : ArrayPool<char>.Shared.Rent(maxLength);
            Span<char> buffer = arrayPool is null ? stackalloc char[maxLength] : arrayPool;
            ContinuingSubwordPrefix.AsSpan().CopyTo(buffer);

            int initialTokensCount = tokens.Count;
            int textLength = text.Length;
            bool isBad = false;

            int start = 0;

            while (start < textLength)
            {
                int end = textLength;
                EncodedToken curToken = default;

                while (start < end)
                {
                    scoped ReadOnlySpan<char> subStr = text.Slice(start, end - start);

                    if (start > 0)
                    {
                        subStr.CopyTo(buffer.Slice(ContinuingSubwordPrefix.Length));
                        subStr = buffer.Slice(0, ContinuingSubwordPrefix.Length + subStr.Length);
                    }

                    if (_vocab.TryGetValue(subStr, out int id))
                    {
                        Debug.Assert(_vocabReverse.ContainsKey(id));
                        curToken = new EncodedToken(id, _vocabReverse[id], new Range(offset + start, offset + end));
                        break;
                    }

                    end -= 1;
                }

                if (curToken.Value is null)
                {
                    isBad = true;
                    break;
                }

                tokens.Add(curToken);
                start = end;
            }

            if (isBad)
            {
                // remove previously added tokens and add the unknown token
                tokens.RemoveRange(initialTokensCount, tokens.Count - initialTokensCount);
                tokens.Add(new EncodedToken(UnknownTokenId, UnknownToken, new Range(offset, offset + textLength)));
            }

            if (arrayPool is not null)
            {
                ArrayPool<char>.Shared.Return(arrayPool);
            }
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
                return new EncodeResults<int> { NormalizedText = null, Tokens = [], CharsConsumed = 0 };
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(
                                                                text,
                                                                textSpan,
                                                                settings.ConsiderPreTokenization,
                                                                settings.ConsiderNormalization,
                                                                _normalizer,
                                                                _preTokenizer,
                                                                out string? normalizedString,
                                                                out ReadOnlySpan<char> textSpanToEncode,
                                                                out int charsConsumed);

            List<int> ids = new();

            if (splits is not null)
            {
                charsConsumed = 0;
                foreach ((int Offset, int Length) split in splits)
                {
                    EncodeToIds(textSpanToEncode.Slice(split.Offset, split.Length), ids, out int length, maxTokenCount - ids.Count);

                    if (length < split.Length || ids.Count >= maxTokenCount)
                    {
                        break;
                    }

                    charsConsumed = split.Offset + length;
                }
            }
            else
            {
                EncodeToIds(textSpanToEncode, ids, out charsConsumed);
            }

            return new EncodeResults<int> { NormalizedText = normalizedString, Tokens = ids, CharsConsumed = charsConsumed };
        }

        /// <summary>
        /// Encode text to a list of Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="accumulatedIds">The list of accumulated Ids.</param>
        /// <param name="charsConsumed">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        private int EncodeToIds(ReadOnlySpan<char> text, List<int>? accumulatedIds, out int charsConsumed, int maxTokenCount = int.MaxValue)
        {
            Debug.Assert(maxTokenCount > 0);

            if (text.IsEmpty)
            {
                charsConsumed = 0;
                return 0;
            }

            if (text.Length > MaxInputCharsPerWord)
            {
                accumulatedIds?.Add(UnknownTokenId);
                charsConsumed = text.Length;
                return 1;
            }

            int maxLength = MaxInputCharsPerWord + ContinuingSubwordPrefix.Length;
            char[]? arrayPool = maxLength <= 250 ? null : ArrayPool<char>.Shared.Rent(maxLength);
            Span<char> buffer = arrayPool is null ? stackalloc char[maxLength] : arrayPool;
            ContinuingSubwordPrefix.AsSpan().CopyTo(buffer);

            int addedIds = 0;
            int textLength = text.Length;
            bool isBad = false;

            int start = 0;

            while (start < textLength)
            {
                int end = textLength;
                int curId = 0;
                bool found = false;

                while (start < end)
                {
                    scoped ReadOnlySpan<char> subStr = text.Slice(start, end - start);

                    if (start > 0)
                    {
                        subStr.CopyTo(buffer.Slice(ContinuingSubwordPrefix.Length));
                        subStr = buffer.Slice(0, ContinuingSubwordPrefix.Length + subStr.Length);
                    }

                    if (_vocab.TryGetValue(subStr, out curId))
                    {
                        found = true;
                        break;
                    }

                    end -= 1;
                }

                if (!found)
                {
                    isBad = true;
                    break;
                }

                accumulatedIds?.Add(curId);
                addedIds++;
                start = end;
            }

            charsConsumed = textLength;
            if (addedIds > maxTokenCount)
            {
                // not enough space to hold added ids. Remove previously added ids
                accumulatedIds?.RemoveRange(accumulatedIds.Count - addedIds, addedIds);
                addedIds = 0;
                charsConsumed = 0;
            }
            else if (isBad)
            {
                // remove previously added ids and add the unknown token id
                accumulatedIds?.RemoveRange(accumulatedIds.Count - addedIds, addedIds);
                accumulatedIds?.Add(UnknownTokenId);
                addedIds = 1;
            }

            if (arrayPool is not null)
            {
                ArrayPool<char>.Shared.Return(arrayPool);
            }

            return addedIds;
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
                                                                out string? normalizedString,
                                                                out ReadOnlySpan<char> textSpanToEncode,
                                                                out int charsConsumed);

            int count = 0;
            if (splits is not null)
            {
                foreach ((int Offset, int Length) split in splits)
                {
                    count += EncodeToIds(textSpanToEncode.Slice(split.Offset, split.Length), accumulatedIds: null, out int length, maxTokenCount - count);

                    if (length < split.Length || count >= maxTokenCount)
                    {
                        break;
                    }
                }
            }
            else
            {
                count = EncodeToIds(textSpanToEncode, accumulatedIds: null, out charsConsumed, maxTokenCount);
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
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="settings" /> has <see cref="EncodeSettings.ConsiderNormalization"/> is <see langword="false"/>, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to <see langword="null"/>.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// If <paramRef name="fromEnd" /> is <see langword="false"/>, it represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the input text or the <paramref name="normalizedString"/> if the normalization is enabled.
        /// If <paramRef name="fromEnd" /> is <see langword="true"/>, it represents the index of the first character to be included. In cases where no tokens fit, the result will be the text length; conversely,
        /// if all tokens fit, the result will be zero.
        /// </returns>
        protected override int GetIndexByTokenCount(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings, bool fromEnd, out string? normalizedString, out int tokenCount)
        {
            if (settings.MaxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(settings.MaxTokenCount), "The max token count must be greater than 0.");
            }

            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedString = null;
                tokenCount = 0;
                return 0;
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(
                                                                text,
                                                                textSpan,
                                                                settings.ConsiderNormalization,
                                                                settings.ConsiderNormalization,
                                                                _normalizer,
                                                                _preTokenizer,
                                                                out normalizedString,
                                                                out ReadOnlySpan<char> textSpanToEncode,
                                                                out _);

            int charsConsumed;

            if (splits is null)
            {
                tokenCount = EncodeToIds(textSpanToEncode, accumulatedIds: null, out charsConsumed, settings.MaxTokenCount);
                if (charsConsumed != textSpanToEncode.Length)
                {
                    tokenCount = 0;
                    return fromEnd ? textSpanToEncode.Length : 0;
                }

                return fromEnd ? 0 : textSpanToEncode.Length;
            }

            if (fromEnd)
            {
                splits = splits.Reverse();
            }

            tokenCount = 0;
            foreach ((int Offset, int Length) split in splits)
            {
                int count = EncodeToIds(textSpanToEncode.Slice(split.Offset, split.Length), accumulatedIds: null, out charsConsumed, settings.MaxTokenCount - tokenCount);
                if (charsConsumed != split.Length)
                {
                    return fromEnd ? split.Offset + split.Length : split.Offset;
                }

                tokenCount += count;

                if (count >= settings.MaxTokenCount)
                {
                    return fromEnd ? split.Offset : split.Offset + split.Length;
                }
            }

            return fromEnd ? 0 : textSpanToEncode.Length;
        }

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <returns>The decoded string.</returns>
        public override string Decode(IEnumerable<int> ids) => Decode(ids, skipSpecialTokens: false);

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="skipSpecialTokens">Indicate whether to skip the special tokens during the decoding.</param>
        /// <returns>The decoded string.</returns>
        public string Decode(IEnumerable<int> ids, bool skipSpecialTokens)
        {
            ValueStringBuilder sb = new ValueStringBuilder();
            bool first = true;
            bool ignoreSpecialTokens = skipSpecialTokens && SpecialTokensReverse is not null;

            foreach (int id in ids)
            {
                if (ignoreSpecialTokens && SpecialTokensReverse!.TryGetValue(id, out _))
                {
                    continue;
                }

                if (_vocabReverse.TryGetValue(id, out string? token))
                {
                    if (token.StartsWith(ContinuingSubwordPrefix))
                    {
                        sb.Append(token.AsSpan().Slice(ContinuingSubwordPrefix.Length));
                    }
                    else
                    {
                        if (!first && token[0] is not ('.' or ',' or '!' or '?' or '\''))
                        {
                            sb.Append(' ');
                        }

                        sb.Append(token);
                    }
                }

                first = false;
            }

            return sb.ToString();
        }

        /// <summary>
        /// Decode the given ids back to text and store the result in the <paramref name="destination"/> span.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="destination">The span to store the decoded text.</param>
        /// <param name="idsConsumed">The number of ids consumed during the decoding.</param>
        /// <param name="charsWritten">The number of characters written to the destination span.</param>
        /// <returns>The operation status indicates whether all IDs were successfully decoded or if the <paramref name="destination"/> is too small to contain the entire decoded result.</returns>
        public override OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, out int idsConsumed, out int charsWritten) =>
            Decode(ids, destination, skipSpecialTokens: false, out idsConsumed, out charsWritten);

        /// <summary>
        /// Decode the given ids back to text and store the result in the <paramref name="destination"/> span.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="destination">The span to store the decoded text.</param>
        /// <param name="skipSpecialTokens">Indicate whether to skip the special tokens during the decoding.</param>
        /// <param name="idsConsumed">The number of ids consumed during the decoding.</param>
        /// <param name="charsWritten">The number of characters written to the destination span.</param>
        /// <returns>The operation status indicates whether all IDs were successfully decoded or if the <paramref name="destination"/> is too small to contain the entire decoded result.</returns>
        public OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, bool skipSpecialTokens, out int idsConsumed, out int charsWritten)
        {
            charsWritten = 0;
            idsConsumed = 0;
            Span<char> buffer = destination;

            bool first = true;
            bool ignoreSpecialTokens = SpecialTokensReverse is not null && skipSpecialTokens;

            foreach (int id in ids)
            {
                if (ignoreSpecialTokens && SpecialTokensReverse!.TryGetValue(id, out _))
                {
                    continue;
                }

                if (_vocabReverse.TryGetValue(id, out string? token))
                {
                    if (token.StartsWith(ContinuingSubwordPrefix, StringComparison.Ordinal))
                    {
                        if (token.Length - ContinuingSubwordPrefix.Length > buffer.Length)
                        {
                            return OperationStatus.DestinationTooSmall;
                        }
                        token.AsSpan().Slice(ContinuingSubwordPrefix.Length).CopyTo(buffer);
                        buffer = buffer.Slice(token.Length - ContinuingSubwordPrefix.Length);
                        charsWritten += token.Length - ContinuingSubwordPrefix.Length;
                    }
                    else
                    {
                        if (!first)
                        {
                            if (token.Length + 1 > buffer.Length)
                            {
                                return OperationStatus.DestinationTooSmall;
                            }

                            buffer[0] = ' ';
                            token.AsSpan().CopyTo(buffer.Slice(1));
                            buffer = buffer.Slice(token.Length + 1);
                            charsWritten += token.Length + 1;
                        }
                        else
                        {
                            if (token.Length > buffer.Length)
                            {
                                return OperationStatus.DestinationTooSmall;
                            }

                            token.AsSpan().CopyTo(buffer);
                            buffer = buffer.Slice(token.Length);
                            charsWritten += token.Length;
                        }
                    }

                    first = false;

                    idsConsumed++;
                }
                else
                {
                    return OperationStatus.InvalidData;
                }
            }

            return OperationStatus.Done;
        }
    }
}