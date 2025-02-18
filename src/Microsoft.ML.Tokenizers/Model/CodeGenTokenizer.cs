// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Represent the Byte Pair Encoding model.
    /// Implement the CodeGen tokenizer described in https://huggingface.co/docs/transformers/main/en/model_doc/codegen#overview
    /// </summary>
    public class CodeGenTokenizer : Tokenizer
    {
        // https://github.com/huggingface/transformers/blob/main/src/transformers/models/codegen/tokenization_codegen.py
        private readonly Dictionary<StringSpanOrdinalKey, (int Id, string Token)> _vocab;
        private IReadOnlyDictionary<string, int>? _vocabOriginal;
        private readonly IReadOnlyDictionary<int, string> _vocabReverse;
        private readonly Dictionary<StringSpanOrdinalKey, (int, string)>? _specialTokens;
        private readonly Dictionary<int, string>? _specialTokensReverse;
        private readonly Dictionary<StringSpanOrdinalKeyPair, int> _mergeRanks;
        private readonly StringSpanOrdinalKeyCache<List<EncodedToken>> _cache;
        private readonly PreTokenizer? _preTokenizer;
        private readonly Normalizer? _normalizer;
        private const int MaxTokenLengthToCache = 15;
        internal const string DefaultSpecialToken = "<|endoftext|>";
        private const int BufferLength = 128;

        /// <summary>
        /// Construct tokenizer's model object to use with the English Robert model.
        /// </summary>
        /// <param name="vocabularyPath">The JSON file path containing the dictionary of string keys and their ids.</param>
        /// <param name="mergePath">The file path containing the tokens's pairs list.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="specialTokens">The dictionary mapping special tokens to Ids.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="unknownToken">The unknown token.</param>
        /// <param name="beginningOfSentenceToken">The beginning of sentence token.</param>
        /// <param name="endOfSentenceToken">The end of sentence token.</param>
        internal CodeGenTokenizer(
                string vocabularyPath,
                string mergePath,
                PreTokenizer? preTokenizer = null,
                Normalizer? normalizer = null,
                IReadOnlyDictionary<string, int>? specialTokens = null,
                bool addPrefixSpace = false,
                bool addBeginningOfSentence = false,
                bool addEndOfSentence = false,
                string? unknownToken = DefaultSpecialToken,
                string? beginningOfSentenceToken = DefaultSpecialToken,
                string? endOfSentenceToken = DefaultSpecialToken) :
            this(vocabularyPath is null ? throw new ArgumentNullException(nameof(vocabularyPath)) : File.OpenRead(vocabularyPath),
                mergePath is null ? throw new ArgumentNullException(nameof(mergePath)) : File.OpenRead(mergePath),
                preTokenizer, normalizer, specialTokens, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, unknownToken, beginningOfSentenceToken, endOfSentenceToken, disposeStream: true)
        {
        }

        /// <summary>
        /// Construct tokenizer's model object to use with the English Robert model.
        /// </summary>
        /// <param name="vocabularyStream">The stream of a JSON file containing the dictionary of string keys and their ids.</param>
        /// <param name="mergeStream">The stream of a file containing the tokens's pairs list.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="specialTokens">The dictionary mapping special tokens to Ids.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="unknownToken">The unknown token.</param>
        /// <param name="beginningOfSentenceToken">The beginning of sentence token.</param>
        /// <param name="endOfSentenceToken">The end of sentence token.</param>
        internal CodeGenTokenizer(
                Stream vocabularyStream,
                Stream mergeStream,
                PreTokenizer? preTokenizer = null,
                Normalizer? normalizer = null,
                IReadOnlyDictionary<string, int>? specialTokens = null,
                bool addPrefixSpace = false,
                bool addBeginningOfSentence = false,
                bool addEndOfSentence = false,
                string? unknownToken = DefaultSpecialToken,
                string? beginningOfSentenceToken = DefaultSpecialToken,
                string? endOfSentenceToken = DefaultSpecialToken) :
            this(vocabularyStream, mergeStream, preTokenizer, normalizer, specialTokens, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, unknownToken, beginningOfSentenceToken, endOfSentenceToken, disposeStream: false)
        {
        }

        private CodeGenTokenizer(Stream vocabularyStream, Stream mergeStream, PreTokenizer? preTokenizer, Normalizer? normalizer, IReadOnlyDictionary<string, int>? specialTokens, bool addPrefixSpace,
                        bool addBeginningOfSentence, bool addEndOfSentence, string? unknownToken, string? beginningOfSentenceToken, string? endOfSentenceToken, bool disposeStream)
        {
            if (vocabularyStream is null)
            {
                throw new ArgumentNullException(nameof(vocabularyStream));
            }

            if (mergeStream is null)
            {
                throw new ArgumentNullException(nameof(mergeStream));
            }

            _preTokenizer = preTokenizer;
            _normalizer = normalizer;

            // Tokenizer data files can be found in codegen-350M-mono
            // https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/vocab.json?download=true
            // https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/merges.txt?download=true

            // Or in Phi-2
            // https://huggingface.co/microsoft/phi-2/resolve/main/vocab.json?download=true
            // https://huggingface.co/microsoft/phi-2/resolve/main/merges.txt?download=true

            _vocab = GetVocabulary(vocabularyStream);
            _vocabReverse = _vocab.ToDictionary(kvp => kvp.Value.Id, kvp => kvp.Value.Token);
            _mergeRanks = GetMergeRanks(mergeStream);
            _cache = new StringSpanOrdinalKeyCache<List<EncodedToken>>();

            try
            {
                if (specialTokens is not null)
                {
                    SpecialTokens = specialTokens;
                    _specialTokens = specialTokens.ToDictionary(kvp => new StringSpanOrdinalKey(kvp.Key), kvp => (kvp.Value, kvp.Key));
                    _specialTokensReverse = specialTokens.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
                }

                UnknownToken = unknownToken;
                BeginningOfSentenceToken = beginningOfSentenceToken;
                EndOfSentenceToken = endOfSentenceToken;

                AddPrefixSpace = addPrefixSpace;
                AddBeginningOfSentence = addBeginningOfSentence;
                AddEndOfSentence = addEndOfSentence;

                if (!string.IsNullOrEmpty(UnknownToken))
                {
                    if (!_vocab.TryGetValue(UnknownToken!, out (int unknownId, string token) value))
                    {
                        throw new ArgumentException($"The Unknown token '{UnknownToken}' is not found in the vocabulary.");
                    }

                    UnknownTokenId = value.unknownId;
                }

                if (!string.IsNullOrEmpty(BeginningOfSentenceToken))
                {
                    if (!_vocab.TryGetValue(BeginningOfSentenceToken!, out (int beggingOfSentenceId, string token) value))
                    {
                        throw new ArgumentException($"The beginning of sentence token '{BeginningOfSentenceToken}' is not found in the vocabulary.");
                    }

                    BeginningOfSentenceId = value.beggingOfSentenceId;
                }

                if (!string.IsNullOrEmpty(EndOfSentenceToken))
                {
                    if (!_vocab.TryGetValue(EndOfSentenceToken!, out (int endOfSentenceId, string token) value))
                    {
                        throw new ArgumentException($"The end of sentence token '{EndOfSentenceToken}' is not found in the vocabulary.");
                    }

                    EndOfSentenceId = value.endOfSentenceId;
                }

                if (AddBeginningOfSentence && string.IsNullOrEmpty(BeginningOfSentenceToken))
                {
                    throw new ArgumentException("The beginning of sentence token must be provided when the flag is set to include it in the encoding.");
                }

                if (AddEndOfSentence && string.IsNullOrEmpty(EndOfSentenceToken))
                {
                    throw new ArgumentException("The end of sentence token must be provided when the flag is set to include it in the encoding.");
                }
            }
            finally
            {
                if (disposeStream)
                {
                    vocabularyStream.Dispose();
                    mergeStream.Dispose();
                }
            }
        }

        /// <summary>
        /// Gets the added tokens.
        /// </summary>
        public IReadOnlyDictionary<string, int>? SpecialTokens { get; }

        /// <summary>
        /// The Unknown token.
        /// </summary>
        public string? UnknownToken { get; }

        /// <summary>
        /// Gets the Unknown token Id.
        /// </summary>
        public int? UnknownTokenId { get; }

        /// <summary>
        /// Gets the flag indicating whether to include the beginning of sentence token in the encoding.
        /// </summary>
        public bool AddBeginningOfSentence { get; }

        /// <summary>
        /// Gets the flag indicating whether to include the end of sentence token in the encoding.
        /// </summary>
        public bool AddEndOfSentence { get; }

        /// <summary>
        /// Gets the beginning of sentence token.
        /// </summary>
        public string? BeginningOfSentenceToken { get; }

        /// <summary>
        /// Gets the end of sentence token Id.
        /// </summary>
        public int? BeginningOfSentenceId { get; }

        /// <summary>
        /// Gets the end of sentence token Id.
        /// </summary>
        public int? EndOfSentenceId { get; }

        /// <summary>
        /// Gets the end of sentence token.
        /// </summary>
        public string? EndOfSentenceToken { get; }

        /// <summary>
        /// Gets the flag indicating whether to include a leading space before encoding the text.
        /// </summary>
        public bool AddPrefixSpace { get; }

        /// <summary>
        /// Gets the PreTokenizer used by the Tokenizer.
        /// </summary>
        public override PreTokenizer? PreTokenizer => _preTokenizer;

        /// <summary>
        /// Gets the Normalizer in use by the Tokenizer.
        /// </summary>
        public override Normalizer? Normalizer => _normalizer;

        /// <summary>
        /// Gets the dictionary mapping tokens to Ids.
        /// </summary>
        public IReadOnlyDictionary<string, int> Vocabulary
        {
            get
            {
                IReadOnlyDictionary<string, int>? publicVocab = Volatile.Read(ref _vocabOriginal);
                if (publicVocab is null)
                {
                    var vocab = new ReadOnlyDictionary<string, int>(_vocab.ToDictionary(kvp => kvp.Value.Token, kvp => kvp.Value.Id));
                    Interlocked.CompareExchange(ref _vocabOriginal, vocab, null);
                    publicVocab = _vocabOriginal;
                }

                return publicVocab;
            }
        }

        //
        // Public Model interfaces implementation
        //

        /// <summary>
        /// Encodes input text to a list of <see cref="EncodedToken" />s.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        protected override EncodeResults<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
            => EncodeToTokens(text, textSpan, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, settings.ConsiderPreTokenization, settings.ConsiderNormalization);

        /// <summary>
        /// Encodes input text to object has the tokens list, tokens Ids, tokens offset mapping.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will null.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The tokenization result includes the tokens list, tokens Ids, tokens offset mapping.</returns>
        public IReadOnlyList<EncodedToken> EncodeToTokens(string text, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, out string? normalizedText, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            EncodeResults<EncodedToken> result = EncodeToTokens(text, ReadOnlySpan<char>.Empty, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization);
            normalizedText = result.NormalizedText;
            return result.Tokens;
        }

        /// <summary>
        /// Encodes input text to object has the tokens list, tokens Ids, tokens offset mapping.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will null.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The tokenization result includes the tokens list, tokens Ids, tokens offset mapping.</returns>
        public IReadOnlyList<EncodedToken> EncodeToTokens(ReadOnlySpan<char> text, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, out string? normalizedText, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            EncodeResults<EncodedToken> result = EncodeToTokens(null, text, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization);
            normalizedText = result.NormalizedText;
            return result.Tokens;
        }

        private EncodeResults<EncodedToken> EncodeToTokens(string? text, scoped ReadOnlySpan<char> textSpan, bool addPrefixSpace, bool addBos, bool addEos, bool considerPreTokenization, bool considerNormalization)
        {
            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                return new EncodeResults<EncodedToken> { Tokens = [], NormalizedText = null, CharsConsumed = 0 };
            }

            char[]? mutatedInputText = null;
            try
            {
                Span<char> mutatedInputSpan = stackalloc char[BufferLength];
                scoped ReadOnlySpan<char> textSpanToEncode;
                IEnumerable<(int Offset, int Length)>? splits;
                string? normalizedText;

                if (addPrefixSpace)
                {
                    ReadOnlySpan<char> span = text is null ? textSpan : text.AsSpan();
                    if (span.Length + 1 > BufferLength)
                    {
                        mutatedInputText = ArrayPool<char>.Shared.Rent(span.Length + 1);
                        mutatedInputSpan = mutatedInputText;
                    }
                    mutatedInputSpan[0] = ' ';
                    span.CopyTo(mutatedInputSpan.Slice(1));
                    span = mutatedInputSpan.Slice(0, span.Length + 1);

                    splits = InitializeForEncoding(
                                null,
                                span,
                                considerPreTokenization,
                                considerNormalization,
                                _normalizer,
                                _preTokenizer,
                                out normalizedText,
                                out textSpanToEncode,
                                out _);
                }
                else
                {
                    splits = InitializeForEncoding(
                                text,
                                textSpan,
                                considerPreTokenization,
                                considerNormalization,
                                _normalizer,
                                _preTokenizer,
                                out normalizedText,
                                out textSpanToEncode,
                                out _);
                }

                List<EncodedToken> tokens = new();
                if (addBos && BeginningOfSentenceId.HasValue)
                {
                    tokens.Add(new EncodedToken(BeginningOfSentenceId.Value, BeginningOfSentenceToken!, new Range(0, 0)));
                }

                PriorityQueue<SymbolPair> agenda = new(textSpanToEncode.Length);

                if (splits is not null)
                {
                    foreach ((int Offset, int Length) split in splits)
                    {
                        EncodeInternal(null, textSpanToEncode.Slice(split.Offset, split.Length), tokens, addPrefixSpace, split.Offset, agenda);
                    }
                }
                else
                {
                    EncodeInternal(addPrefixSpace ? null : (normalizedText ?? text), textSpanToEncode, tokens, addPrefixSpace, 0, agenda);
                }

                if (addEos && EndOfSentenceId.HasValue)
                {
                    int index = addPrefixSpace ? Math.Max(0, textSpanToEncode.Length - 1) : textSpanToEncode.Length;
                    tokens.Add(new EncodedToken(EndOfSentenceId.Value, EndOfSentenceToken!, new Range(index, index)));
                }

                return new EncodeResults<EncodedToken> { Tokens = tokens, NormalizedText = normalizedText, CharsConsumed = textSpanToEncode.Length };
            }
            finally
            {
                if (mutatedInputText is not null)
                {
                    ArrayPool<char>.Shared.Return(mutatedInputText);
                }
            }
        }

        /// <summary>
        /// Encode a text string to a list of tokens.
        /// </summary>
        /// <param name="text">The text in form of string to encode if it is available.</param>
        /// <param name="textSpan">The text in form of span to encode.</param>
        /// <param name="tokens">The tokens to include in the newly encoded sequence.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="offset">The offset to adjust the token's offset.</param>
        /// <param name="agenda">The priority queue to use for encoding.</param>
        private void EncodeInternal(string? text, scoped ReadOnlySpan<char> textSpan, List<EncodedToken> tokens, bool addPrefixSpace, int offset, PriorityQueue<SymbolPair> agenda)
        {
            if (textSpan.IsEmpty)
            {
                return;
            }

            if (_specialTokens is not null && _specialTokens.TryGetValue(textSpan, out (int specialTokenId, string specialToken) value))
            {
                int index = (addPrefixSpace && offset > 0) ? offset - 1 : offset;
                tokens.Add(new EncodedToken(value.specialTokenId, value.specialToken, new Range(index, index + ((addPrefixSpace && offset == 0) ? textSpan.Length - 1 : textSpan.Length))));
                return;
            }

            if (_cache.TryGetValue(textSpan, out List<EncodedToken>? hit))
            {
                AppendTokenWithOffsetAdjusting(hit, tokens, offset, addPrefixSpace);
                return;
            }

            Span<char> token = stackalloc char[BufferLength];
            Span<int> mapping = stackalloc int[BufferLength];
            char[]? tokenBuffer = null;
            int[]? mappingBuffer = null;

            try
            {
                int destinationMaxSize = Encoding.UTF8.GetMaxByteCount(textSpan.Length);
                if (destinationMaxSize > BufferLength)
                {
                    tokenBuffer = ArrayPool<char>.Shared.Rent(destinationMaxSize);
                    token = tokenBuffer;

                    mappingBuffer = ArrayPool<int>.Shared.Rent(destinationMaxSize);
                    mapping = mappingBuffer;
                }

                int encodedLength = Helpers.EncodeToUtf8AndTransform(textSpan, token, mapping);

                List<EncodedToken> result = EncodeToTokens(token.Slice(0, encodedLength), mapping.Slice(0, encodedLength), textSpan, agenda);

                if (textSpan.Length <= MaxTokenLengthToCache)
                {
                    _cache.Set(text is null ? textSpan.ToString() : text, result);
                }

                AppendTokenWithOffsetAdjusting(result, tokens, offset, addPrefixSpace);
            }
            finally
            {
                if (tokenBuffer is not null)
                {
                    ArrayPool<char>.Shared.Return(tokenBuffer);
                    Debug.Assert(mappingBuffer is not null);
                    ArrayPool<int>.Shared.Return(mappingBuffer);
                }
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
            return new EncodeResults<int>
            {
                Tokens = EncodeToIds(text, textSpan, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, settings.ConsiderPreTokenization, settings.ConsiderNormalization,
                                    out string? normalizedText, out int charsConsumed, settings.MaxTokenCount),
                NormalizedText = normalizedText,
                CharsConsumed = charsConsumed
            };
        }

        /// <summary>
        /// Encodes input text to tokens Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(string text, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            return EncodeToIds(text, ReadOnlySpan<char>.Empty, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out _, out _);
        }

        /// <summary>
        /// Encodes input text to tokens Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            return EncodeToIds(null, text, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out _, out _);
        }

        /// <summary>
        /// Encodes input text to tokens Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="charsConsumed">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(string text, int maxTokenCount, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, out string? normalizedText, out int charsConsumed, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            return EncodeToIds(text, ReadOnlySpan<char>.Empty, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out normalizedText, out charsConsumed, maxTokenCount);
        }

        /// <summary>
        /// Encodes input text to tokens Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="charsConsumed">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, int maxTokenCount, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, out string? normalizedText, out int charsConsumed, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            return EncodeToIds(null, text, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out normalizedText, out charsConsumed, maxTokenCount);
        }

        private IReadOnlyList<int> EncodeToIds(
                                    string? text,
                                    scoped ReadOnlySpan<char> textSpan,
                                    bool addPrefixSpace,
                                    bool addBeginningOfSentence,
                                    bool addEndOfSentence,
                                    bool considerPreTokenization,
                                    bool considerNormalization,
                                    out string? normalizedText,
                                    out int charsConsumed,
                                    int maxTokenCount = int.MaxValue)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                charsConsumed = 0;
                normalizedText = null;
                return [];
            }

            char[]? mutatedInputText = null;

            try
            {
                Span<char> mutatedInputSpan = stackalloc char[BufferLength];
                scoped ReadOnlySpan<char> textSpanToEncode;
                IEnumerable<(int Offset, int Length)>? splits;
                if (addPrefixSpace)
                {
                    ReadOnlySpan<char> span = text is null ? textSpan : text.AsSpan();
                    if (span.Length + 1 > BufferLength)
                    {
                        mutatedInputText = ArrayPool<char>.Shared.Rent(span.Length + 1);
                        mutatedInputSpan = mutatedInputText;
                    }
                    mutatedInputSpan[0] = ' ';
                    span.CopyTo(mutatedInputSpan.Slice(1));
                    span = mutatedInputSpan.Slice(0, span.Length + 1);

                    splits = InitializeForEncoding(null, span, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedText, out textSpanToEncode, out _);
                }
                else
                {
                    splits = InitializeForEncoding(text, textSpan, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedText, out textSpanToEncode, out _);
                }

                List<int> ids = new();

                if (addBeginningOfSentence && BeginningOfSentenceId.HasValue)
                {
                    ids.Add(BeginningOfSentenceId.Value);
                }

                PriorityQueue<SymbolPair> agenda = new(textSpanToEncode.Length);

                if (splits is not null)
                {
                    charsConsumed = 0;
                    foreach ((int Offset, int Length) split in splits)
                    {
                        EncodeToIdsInternal(null, textSpanToEncode.Slice(split.Offset, split.Length), ids, agenda, out int length, maxTokenCount - ids.Count);
                        charsConsumed = split.Offset + length;

                        if (length < split.Length || ids.Count >= maxTokenCount)
                        {
                            break;
                        }
                    }
                }
                else
                {
                    EncodeToIdsInternal(addPrefixSpace ? null : (normalizedText ?? text), textSpanToEncode, ids, agenda, out charsConsumed, maxTokenCount - ids.Count);
                }

                if (addEndOfSentence && EndOfSentenceId.HasValue && ids.Count < maxTokenCount)
                {
                    ids.Add(EndOfSentenceId.Value);
                }

                if (addPrefixSpace && charsConsumed > 0)
                {
                    charsConsumed--;
                }

                return ids;
            }
            finally
            {
                if (mutatedInputText is not null)
                {
                    ArrayPool<char>.Shared.Return(mutatedInputText);
                }
            }
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        protected override int CountTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
            => CountTokens(text, textSpan, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, settings.ConsiderPreTokenization, settings.ConsiderNormalization, out _, out _, settings.MaxTokenCount);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of tokens Ids that the input text will be encoded to.</returns>
        public int CountTokens(string text, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => CountTokens(text, Span<char>.Empty, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of tokens Ids that the input text will be encoded to.</returns>
        public int CountTokens(ReadOnlySpan<char> text, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => CountTokens(null, text, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out _, out _);

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
                return LastIndexOf(text, textSpan, settings.MaxTokenCount, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, settings.ConsiderPreTokenization,
                            settings.ConsiderNormalization, out normalizedText, out tokenCount);
            }

            tokenCount = CountTokens(text, textSpan, AddPrefixSpace, AddBeginningOfSentence, AddEndOfSentence, settings.ConsiderPreTokenization, settings.ConsiderNormalization, out normalizedText, out int charsConsumed, settings.MaxTokenCount);
            return charsConsumed;
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the start within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the text or the <paramref name="normalizedText"/> if the normalization is enabled.
        /// </returns>
        public int GetIndexByTokenCount(string text, int maxTokenCount, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, out string? normalizedText, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            tokenCount = CountTokens(text, Span<char>.Empty, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out normalizedText, out int charsConsumed, maxTokenCount);
            return charsConsumed;
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the start within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the text or the <paramref name="normalizedText"/> if the normalization is enabled.
        /// </returns>
        public int GetIndexByTokenCount(
                    ReadOnlySpan<char> text,
                    int maxTokenCount,
                    bool addPrefixSpace,
                    bool addBeginningOfSentence,
                    bool addEndOfSentence,
                    out string? normalizedText,
                    out int tokenCount,
                    bool considerPreTokenization = true,
                    bool considerNormalization = true)
        {
            tokenCount = CountTokens(null, text, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out normalizedText, out int charsConsumed, maxTokenCount);
            return charsConsumed;
        }

        private int CountTokens(
                        string? text,
                        scoped ReadOnlySpan<char> textSpan,
                        bool addPrefixSpace,
                        bool addBeginningOfSentence,
                        bool addEndOfSentence,
                        bool considerPreTokenization,
                        bool considerNormalization,
                        out string? normalizedText,
                        out int charsConsumed,
                        int maxTokenCount = int.MaxValue)
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

            char[]? mutatedInputText = null;

            try
            {

                Span<char> mutatedInputSpan = stackalloc char[BufferLength];
                scoped ReadOnlySpan<char> textSpanToEncode;
                IEnumerable<(int Offset, int Length)>? splits;
                if (addPrefixSpace)
                {
                    ReadOnlySpan<char> span = text is null ? textSpan : text.AsSpan();
                    if (span.Length + 1 > BufferLength)
                    {
                        mutatedInputText = ArrayPool<char>.Shared.Rent(span.Length + 1);
                        mutatedInputSpan = mutatedInputText;
                    }
                    mutatedInputSpan[0] = ' ';
                    span.CopyTo(mutatedInputSpan.Slice(1));
                    span = mutatedInputSpan.Slice(0, span.Length + 1);

                    splits = InitializeForEncoding(null, span, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedText, out textSpanToEncode, out _);
                }
                else
                {
                    splits = InitializeForEncoding(text, textSpan, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedText, out textSpanToEncode, out _);
                }

                PriorityQueue<SymbolPair> agenda = new(textSpanToEncode.Length);

                int count = (addBeginningOfSentence && BeginningOfSentenceId.HasValue) ? 1 : 0;
                if (splits is not null)
                {
                    foreach ((int Offset, int Length) split in splits)
                    {
                        count += EncodeToIdsInternal(null, textSpanToEncode.Slice(split.Offset, split.Length), null, agenda, out int length, maxTokenCount - count);
                        charsConsumed = split.Offset + length;

                        if (length < split.Length || count >= maxTokenCount)
                        {
                            break;
                        }
                    }
                }
                else
                {
                    count = EncodeToIdsInternal(addPrefixSpace ? null : text, textSpanToEncode, null, agenda, out charsConsumed, maxTokenCount - count);
                }

                if (addEndOfSentence && EndOfSentenceId.HasValue && count < maxTokenCount)
                {
                    count++;
                }

                if (addPrefixSpace && charsConsumed > 0)
                {
                    charsConsumed--;
                }

                return count;
            }
            finally
            {
                if (mutatedInputText is not null)
                {
                    ArrayPool<char>.Shared.Return(mutatedInputText);
                }
            }
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the text or the <paramref name="normalizedText"/> if normalization is enabled;
        /// conversely, if all tokens fit, the result will be 0.
        /// </returns>
        /// <remarks>
        /// If the whole text can be encoded within the token limit, the returned index will be 0.
        /// </remarks>
        public int GetIndexByTokenCountFromEnd(string text, int maxTokenCount, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, out string? normalizedText, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
            => LastIndexOf(text, Span<char>.Empty, maxTokenCount, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out normalizedText, out tokenCount);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will be null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the <paramref name="normalizedText"/>; conversely, if all tokens fit, the result will be 0.
        /// </returns>
        /// <remarks>
        /// If the whole text can be encoded within the token limit, the returned index will be 0.
        /// </remarks>
        public int GetIndexByTokenCountFromEnd(ReadOnlySpan<char> text, int maxTokenCount, bool addPrefixSpace, bool addBeginningOfSentence, bool addEndOfSentence, out string? normalizedText, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
            => LastIndexOf(null, text, maxTokenCount, addPrefixSpace, addBeginningOfSentence, addEndOfSentence, considerPreTokenization, considerNormalization, out normalizedText, out tokenCount);

        private int LastIndexOf(
                        string? text,
                        scoped ReadOnlySpan<char> textSpan,
                        int maxTokenCount,
                        bool addPrefixSpace,
                        bool addBeginningOfSentence,
                        bool addEndOfSentence,
                        bool considerPreTokenization,
                        bool considerNormalization,
                        out string? normalizedText,
                        out int tokenCount)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedText = null;
                tokenCount = 0;
                return 0;
            }

            char[]? mutatedInputText = null;

            try
            {
                Span<char> mutatedInputSpan = stackalloc char[BufferLength];
                scoped ReadOnlySpan<char> textSpanToEncode;
                IEnumerable<(int Offset, int Length)>? splits;
                if (addPrefixSpace)
                {
                    ReadOnlySpan<char> span = text is null ? textSpan : text.AsSpan();
                    if (span.Length + 1 > BufferLength)
                    {
                        mutatedInputText = ArrayPool<char>.Shared.Rent(span.Length + 1);
                        mutatedInputSpan = mutatedInputText;
                    }
                    mutatedInputSpan[0] = ' ';
                    span.CopyTo(mutatedInputSpan.Slice(1));
                    span = mutatedInputSpan.Slice(0, span.Length + 1);

                    splits = InitializeForEncoding(null, span, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedText, out textSpanToEncode, out _);
                }
                else
                {
                    splits = InitializeForEncoding(text, textSpan, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedText, out textSpanToEncode, out _);
                }

                PriorityQueue<SymbolPair> agenda = new(textSpanToEncode.Length);

                tokenCount = (addEndOfSentence && EndOfSentenceId.HasValue) ? 1 : 0;

                int retIndex = -1;
                if (splits is not null)
                {
                    foreach ((int Offset, int Length) split in splits.Reverse())
                    {
                        tokenCount += EncodeToIdsFromEndInternal(null, textSpanToEncode.Slice(split.Offset, split.Length), null, agenda, out int textIndex, maxTokenCount - tokenCount);
                        if (textIndex > 0 || tokenCount >= maxTokenCount)
                        {
                            retIndex = addPrefixSpace ? split.Offset + textIndex - 1 : split.Offset + textIndex;
                            break;
                        }
                    }
                }
                else
                {
                    tokenCount = EncodeToIdsFromEndInternal(addPrefixSpace ? null : text, textSpanToEncode, null, agenda, out int charsConsumed, maxTokenCount - tokenCount);
                    retIndex = addPrefixSpace ? Math.Max(0, charsConsumed - 1) : charsConsumed;
                }

                if (addBeginningOfSentence && BeginningOfSentenceId.HasValue && tokenCount < maxTokenCount)
                {
                    tokenCount++;
                }

                return Math.Max(retIndex, 0);
            }
            finally
            {
                if (mutatedInputText is not null)
                {
                    ArrayPool<char>.Shared.Return(mutatedInputText);
                }
            }
        }

        private int EncodeToIdsResult(List<EncodedToken> tokens, IList<int>? accumulatedIds, int maxTokens, int fullTextLength, out int charsConsumed)
        {
            charsConsumed = 0;

            if (tokens.Count <= maxTokens)
            {
                if (accumulatedIds is not null)
                {
                    foreach (var t in tokens)
                    {
                        accumulatedIds.Add(t.Id);
                    }
                }

                charsConsumed = fullTextLength;
                return tokens.Count;
            }

            int tokenCount;
            for (tokenCount = 0; tokenCount < maxTokens; tokenCount++)
            {
                // maxTokens is less than tokens.Count, so it is safe to index maxTokens.
                if (tokens[tokenCount].Offset.Start.Value == tokens[tokenCount + 1].Offset.Start.Value)
                {
                    // Ensure we'll not break the text in the middle of a code-point
                    int j = tokenCount + 2;
                    while (j < tokens.Count && tokens[j].Offset.Start.Value == tokens[tokenCount].Offset.Start.Value)
                    {
                        j++;
                    }

                    if (j <= maxTokens)
                    {
                        // append encountered tokens to the accumulatedIds
                        for (int k = tokenCount; k < j; k++)
                        {
                            accumulatedIds?.Add(tokens[k].Id);
                            charsConsumed += tokens[k].Offset.End.Value - tokens[k].Offset.Start.Value;
                        }
                        tokenCount = j - 1;
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    accumulatedIds?.Add(tokens[tokenCount].Id);
                    charsConsumed += tokens[tokenCount].Offset.End.Value - tokens[tokenCount].Offset.Start.Value;
                }
            }

            return tokenCount;
        }

        private int EncodeToIdsFromEndResult(List<EncodedToken> tokens, IList<int>? accumulatedIds, int maxTokens, int fullTextLength, out int textIndex)
        {
            textIndex = fullTextLength;

            if (tokens.Count <= maxTokens)
            {
                if (accumulatedIds is not null)
                {
                    foreach (var t in tokens)
                    {
                        accumulatedIds.Add(t.Id);
                    }
                }

                textIndex = 0;
                return tokens.Count;
            }

            int index = tokens.Count - maxTokens;

            // avoid breaking the text in the middle of a code-point
            while (index < tokens.Count && tokens[index].Offset.Start.Value == tokens[index - 1].Offset.Start.Value)
            {
                index++;
            }

            for (int i = index; i < tokens.Count; i++)
            {
                accumulatedIds?.Add(tokens[i].Id);
                textIndex -= tokens[i].Offset.End.Value - tokens[i].Offset.Start.Value;
            }

            return tokens.Count - index;
        }

        private int EncodeToIdsInternal(string? text, scoped ReadOnlySpan<char> textSpan, IList<int>? accumulatedIds, PriorityQueue<SymbolPair> agenda, out int charsConsumed, int maxTokens)
        {
            if (textSpan.IsEmpty)
            {
                charsConsumed = 0;
                return 0;
            }

            if (_specialTokens is not null && _specialTokens.TryGetValue(textSpan, out (int specialTokenId, string specialToken) value) && maxTokens > 0)
            {
                if (accumulatedIds is not null)
                {
                    accumulatedIds.Add(value.specialTokenId);
                }

                charsConsumed = textSpan.Length;
                return 1;
            }

            if (_cache.TryGetValue(textSpan, out List<EncodedToken>? hit))
            {
                return EncodeToIdsResult(hit, accumulatedIds, maxTokens, textSpan.Length, out charsConsumed);
            }

            Span<char> token = stackalloc char[BufferLength];
            Span<int> mapping = stackalloc int[BufferLength];
            char[]? tokenBuffer = null;
            int[]? mappingBuffer = null;

            try
            {
                int destinationMaxSize = Encoding.UTF8.GetMaxByteCount(textSpan.Length);
                if (destinationMaxSize > token.Length)
                {
                    tokenBuffer = ArrayPool<char>.Shared.Rent(destinationMaxSize);
                    token = tokenBuffer;

                    mappingBuffer = ArrayPool<int>.Shared.Rent(destinationMaxSize);
                    mapping = mappingBuffer;
                }

                int encodedLength = Helpers.EncodeToUtf8AndTransform(textSpan, token, mapping);

                List<EncodedToken> result = EncodeToTokens(token.Slice(0, encodedLength), mapping.Slice(0, encodedLength), textSpan, agenda);

                int length = text is not null ? text.Length : textSpan.Length;
                if (length <= MaxTokenLengthToCache)
                {
                    _cache.Set(text ?? textSpan.ToString(), result);
                }

                return EncodeToIdsResult(result, accumulatedIds, maxTokens, textSpan.Length, out charsConsumed);
            }
            finally
            {
                if (tokenBuffer is not null)
                {
                    ArrayPool<char>.Shared.Return(tokenBuffer);
                    Debug.Assert(mappingBuffer is not null);
                    ArrayPool<int>.Shared.Return(mappingBuffer);
                }
            }
        }

        private int EncodeToIdsFromEndInternal(string? text, scoped ReadOnlySpan<char> textSpan, IList<int>? accumulatedIds, PriorityQueue<SymbolPair> agenda, out int textIndex, int maxTokens)
        {
            if (textSpan.IsEmpty)
            {
                textIndex = textSpan.Length;
                return 0;
            }

            if (_specialTokens is not null && _specialTokens.TryGetValue(textSpan, out (int specialTokenId, string specialToken) value) && maxTokens > 0)
            {
                if (accumulatedIds is not null)
                {
                    accumulatedIds.Add(value.specialTokenId);
                }

                textIndex = 0;
                return 1;
            }

            if (_cache.TryGetValue(textSpan, out List<EncodedToken>? hit))
            {
                return EncodeToIdsFromEndResult(hit, accumulatedIds, maxTokens, textSpan.Length, out textIndex);
            }

            Span<char> token = stackalloc char[100];
            Span<int> mapping = stackalloc int[100];
            char[]? tokenBuffer = null;
            int[]? mappingBuffer = null;

            try
            {
                int destinationMaxSize = Encoding.UTF8.GetMaxByteCount(textSpan.Length);
                if (destinationMaxSize > token.Length)
                {
                    tokenBuffer = ArrayPool<char>.Shared.Rent(destinationMaxSize);
                    token = tokenBuffer;

                    mappingBuffer = ArrayPool<int>.Shared.Rent(destinationMaxSize);
                    mapping = mappingBuffer;
                }

                int encodedLength = Helpers.EncodeToUtf8AndTransform(textSpan, token, mapping);

                List<EncodedToken> result = EncodeToTokens(token.Slice(0, encodedLength), mapping.Slice(0, encodedLength), textSpan, agenda);

                int length = text is not null ? text.Length : textSpan.Length;
                if (length <= MaxTokenLengthToCache)
                {
                    _cache.Set(text ?? textSpan.ToString(), result);
                }

                return EncodeToIdsFromEndResult(result, accumulatedIds, maxTokens, textSpan.Length, out textIndex);
            }
            finally
            {
                if (tokenBuffer is not null)
                {
                    ArrayPool<char>.Shared.Return(tokenBuffer);
                    Debug.Assert(mappingBuffer is not null);
                    ArrayPool<int>.Shared.Return(mappingBuffer);
                }
            }
        }

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <returns>The decoded string.</returns>
        public override string Decode(IEnumerable<int> ids) => Decode(ids, hasPrefixSpace: AddPrefixSpace, considerSpecialTokens: false);

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="hasPrefixSpace">Indicate whether the encoded string has a leading space.</param>
        /// <param name="considerSpecialTokens">Indicate whether to consider special tokens during decoding.</param>
        /// <returns>The decoded string.</returns>
        public string Decode(IEnumerable<int> ids, bool hasPrefixSpace, bool considerSpecialTokens)
        {
            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
            }

            byte[] bytes = ArrayPool<byte>.Shared.Rent(128);

            try
            {
                int bytesIndex = 0;
                bool firstToken = true;

                foreach (int id in ids)
                {
                    if (BeginningOfSentenceId.HasValue && id == BeginningOfSentenceId.Value)
                    {
                        if (considerSpecialTokens)
                        {
                            AppendToBytesArray(BeginningOfSentenceToken!.AsSpan(), ref bytes, ref bytesIndex);
                        }
                        continue;
                    }

                    if (EndOfSentenceId.HasValue && id == EndOfSentenceId.Value)
                    {
                        if (considerSpecialTokens)
                        {
                            AppendToBytesArray(EndOfSentenceToken!.AsSpan(), ref bytes, ref bytesIndex);
                        }
                        continue;
                    }

                    if (UnknownTokenId.HasValue && id == UnknownTokenId.Value)
                    {
                        if (considerSpecialTokens)
                        {
                            AppendToBytesArray(UnknownToken!.AsSpan(), ref bytes, ref bytesIndex);
                        }
                        continue;
                    }

                    if (_specialTokensReverse is not null && _specialTokensReverse.TryGetValue(id, out string? specialToken))
                    {
                        int bytesCountToEncode = Encoding.UTF8.GetMaxByteCount(specialToken.Length);
                        if (bytes.Length - bytesIndex < bytesCountToEncode)
                        {
                            Helpers.ArrayPoolGrow(ref bytes, (bytes.Length + bytesCountToEncode) * 2);
                        }

                        bool removePrefixSpace = firstToken && hasPrefixSpace && specialToken.Length > 0 && specialToken[0] == ' ';
                        bytesIndex += Helpers.GetUtf8Bytes(removePrefixSpace ? specialToken.AsSpan().Slice(1) : specialToken.AsSpan(), bytes.AsSpan().Slice(bytesIndex));
                        firstToken = false;
                        continue;
                    }

                    // vocabularies are stored in UTF-8 form with escaping the control characters.
                    // Need to convert the vocabulary to the original UTF-16 form.
                    if (MapIdToToken(id) is string s)
                    {
                        ReadOnlySpan<char> span = firstToken && hasPrefixSpace && s.Length > 0 && s[0] == _transformedSpace ? s.AsSpan(1) : s.AsSpan();
                        firstToken = false;
                        AppendToBytesArray(span, ref bytes, ref bytesIndex);
                    }
                }

                string result = Encoding.UTF8.GetString(bytes, 0, bytesIndex);
                return result;
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(bytes);
            }
        }

        /// <summary>
        /// Decode the given ids back to text and store the result in the <paramref name="destination"/> span.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="destination">The span to store the decoded text.</param>
        ///
        /// <param name="idsConsumed">The number of ids consumed during the decoding.</param>
        /// <param name="charsWritten">The number of characters written to the destination span.</param>
        /// <returns>The operation status indicates whether all IDs were successfully decoded or if the <paramref name="destination"/> is too small to contain the entire decoded result.</returns>
        public override OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, out int idsConsumed, out int charsWritten)
            => Decode(ids, destination, hasPrefixSpace: AddPrefixSpace, considerSpecialTokens: false, out idsConsumed, out charsWritten);

        /// <summary>
        /// Decode the given ids back to text and store the result in the <paramref name="destination"/> span.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="destination">The span to store the decoded text.</param>
        /// <param name="hasPrefixSpace">Indicate whether the encoded string has a leading space.</param>
        /// <param name="considerSpecialTokens">Indicate whether to consider special tokens during decoding.</param>
        /// <param name="idsConsumed">The number of ids consumed during the decoding.</param>
        /// <param name="charsWritten">The number of characters written to the destination span.</param>
        /// <returns>The operation status indicates whether all IDs were successfully decoded or if the <paramref name="destination"/> is too small to contain the entire decoded result.</returns>
        public OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, bool hasPrefixSpace, bool considerSpecialTokens, out int idsConsumed, out int charsWritten)
        {
            idsConsumed = 0;
            charsWritten = 0;

            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
            }

            // Enough buffer to carry one converted vocabulary to UTF-16 form
            Span<char> vocabBuffer = stackalloc char[256];
            // Enough buffer to carry one UTF-8 vocabulary
            Span<byte> utf8bytes = stackalloc byte[256];
            int incompleteUtf8BytesInBuffer = 0;
            int incompleteUtf8BytesInBufferIndex = 0;
            int utf16CharsInBuffer = 0;
            int idsHangingCount = 0;

            ByteToUnicodeEncoding byteToUnicodeEncoding = ByteToUnicodeEncoding.Instance;

            Span<char> buffer = destination;
            bool firstToken = true;

            foreach (int id in ids)
            {
                if (BeginningOfSentenceId.HasValue && id == BeginningOfSentenceId.Value)
                {
                    if (incompleteUtf8BytesInBuffer > 0)
                    {
                        return OperationStatus.InvalidData; // unexpected case
                    }

                    if (considerSpecialTokens)
                    {
                        if (BeginningOfSentenceToken!.Length > buffer.Length)
                        {
                            return OperationStatus.DestinationTooSmall;
                        }

                        BeginningOfSentenceToken.AsSpan().CopyTo(buffer);
                        buffer = buffer.Slice(BeginningOfSentenceToken.Length);
                        charsWritten += BeginningOfSentenceToken.Length;
                    }

                    idsConsumed++;
                    continue;
                }

                if (EndOfSentenceId.HasValue && id == EndOfSentenceId.Value)
                {
                    if (incompleteUtf8BytesInBuffer > 0)
                    {
                        return OperationStatus.InvalidData; // unexpected case
                    }

                    if (considerSpecialTokens)
                    {
                        if (EndOfSentenceToken!.Length > buffer.Length)
                        {
                            return OperationStatus.DestinationTooSmall;
                        }

                        EndOfSentenceToken.AsSpan().CopyTo(buffer);
                        buffer = buffer.Slice(EndOfSentenceToken.Length);
                        charsWritten += EndOfSentenceToken.Length;
                    }

                    idsConsumed++;
                    continue;
                }

                if (UnknownTokenId.HasValue && id == UnknownTokenId.Value)
                {
                    if (incompleteUtf8BytesInBuffer > 0)
                    {
                        return OperationStatus.InvalidData; // unexpected case
                    }

                    if (considerSpecialTokens)
                    {
                        if (UnknownToken!.Length > buffer.Length)
                        {
                            return OperationStatus.DestinationTooSmall;
                        }

                        UnknownToken.AsSpan().CopyTo(buffer);
                        buffer = buffer.Slice(UnknownToken.Length);
                        charsWritten += UnknownToken.Length;
                    }

                    idsConsumed++;
                    continue;
                }

                if (_specialTokensReverse is not null && _specialTokensReverse.TryGetValue(id, out string? specialToken))
                {
                    if (incompleteUtf8BytesInBuffer > 0)
                    {
                        return OperationStatus.InvalidData; // unexpected case
                    }

                    ReadOnlySpan<char> specialTokenSpan = specialToken.AsSpan();
                    if (firstToken && hasPrefixSpace && specialToken.Length > 0 && specialToken[0] == ' ')
                    {
                        specialTokenSpan = specialTokenSpan.Slice(1);
                    }

                    if (specialTokenSpan.Length > buffer.Length)
                    {
                        return OperationStatus.DestinationTooSmall;
                    }

                    specialTokenSpan.CopyTo(buffer);
                    buffer = buffer.Slice(specialTokenSpan.Length);
                    charsWritten += specialTokenSpan.Length;
                    firstToken = false;
                    idsConsumed++;
                    continue;
                }

                // vocabularies are stored in UTF-8 form with escaping the control characters.
                // Need to convert the vocabulary to the original UTF-16 form.
                if (_vocabReverse.TryGetValue(id, out string? s))
                {
                    ReadOnlySpan<char> span = firstToken && hasPrefixSpace && s.Length > 0 && s[0] == _transformedSpace ? s.AsSpan(1) : s.AsSpan();
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

                    firstToken = false;
                    continue;
                }

                return OperationStatus.InvalidData; // encountered unknown id
            }

            return OperationStatus.Done;
        }

        private static readonly char _transformedSpace = ByteToUnicodeEncoding.Instance.ByteToUnicode[' '];

        /// <summary>
        /// Map the encoded Id to the token.
        /// </summary>
        /// <param name="id">The Id to map to the string.</param>
        /// <returns>The mapped token of the Id.</returns>
        private string? MapIdToToken(int id)
        {
            if (_vocabReverse.TryGetValue(id, out var value))
            {
                return value;
            }

            if (_specialTokensReverse is not null && _specialTokensReverse.TryGetValue(id, out value))
            {
                return value;
            }

            return null;
        }

        /// <summary>
        /// Map the token to encoded Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <returns>The mapped Id of the token.</returns>
        private int? MapTokenToId(ReadOnlySpan<char> token)
        {
            if (_vocab.TryGetValue(token, out (int Id, string Token) value))
            {
                return value.Id;
            }

            if (_specialTokens is not null && _specialTokens.TryGetValue(token, out (int Id, string Token) specialToken))
            {
                return specialToken.Id;
            }

            return null;
        }

        private void AppendToBytesArray(ReadOnlySpan<char> text, ref byte[] bytes, ref int bytesIndex)
        {
            IReadOnlyDictionary<char, char> unicodeToByte = ByteToUnicodeEncoding.Instance.UnicodeToByte;
            for (int i = 0; i < text.Length; i++)
            {
                if (unicodeToByte.TryGetValue(text[i], out char c))
                {
                    if (bytesIndex >= bytes.Length)
                    {
                        Helpers.ArrayPoolGrow<byte>(ref bytes, bytes.Length * 2);
                    }

                    bytes[bytesIndex++] = (byte)c;
                    continue;
                }

                // rare cases
                i += Helpers.EncodeCodePointToUtf8(text, i, ref bytes, ref bytesIndex) - 1;
            }
        }

        //
        // Private & Internal methods
        //

        private static void AppendTokenWithOffsetAdjusting(IReadOnlyList<EncodedToken> tokensToAdd, List<EncodedToken> tokens, int offset, bool addPrefixSpace)
        {
            if (addPrefixSpace)
            {
                if (tokensToAdd.Count > 0)
                {
                    (int s, int e) r = offset == 0 ? (tokensToAdd[0].Offset.Start.Value, tokensToAdd[0].Offset.End.Value - 1) : (tokensToAdd[0].Offset.Start.Value + offset - 1, tokensToAdd[0].Offset.End.Value + offset - 1);
                    tokens.Add(new EncodedToken(tokensToAdd[0].Id, tokensToAdd[0].Value, new Range(r.s, r.e)));

                    for (int i = 1; i < tokensToAdd.Count; i++)
                    {
                        tokens.Add(new EncodedToken(tokensToAdd[i].Id, tokensToAdd[i].Value, new Range(tokensToAdd[i].Offset.Start.Value + offset - 1, tokensToAdd[i].Offset.End.Value + offset - 1)));
                    }
                }
            }
            else
            {
                foreach (EncodedToken t in tokensToAdd)
                {
                    tokens.Add(new EncodedToken(t.Id, t.Value, new Range(t.Offset.Start.Value + offset, t.Offset.End.Value + offset)));
                }
            }
        }

        /// <summary>
        /// Encode a token into BPE-ed sub-tokens. E.g., "playing" into ["play", "ing"].
        /// </summary>
        private List<EncodedToken> EncodeToTokens(Span<char> text, Span<int> mapping, ReadOnlySpan<char> originalText, PriorityQueue<SymbolPair> agenda)
        {
            if (text.Length == 0)
            {
                return [];
            }

            if (text.Length == 1)
            {
                char c = text[0];
                string[] charToString = ByteToUnicodeEncoding.Instance.CharToString;
                string tokenValue = (uint)c < charToString.Length ? charToString[c] : c.ToString();
                return new List<EncodedToken> { new EncodedToken(_vocab[new StringSpanOrdinalKey(tokenValue)].Id, tokenValue, new Range(mapping[0], mapping[0] + 1)) };
            }

            BpeSymbol[] symbols = ArrayPool<BpeSymbol>.Shared.Rent(text.Length);

            try
            {
                for (int i = 0; i < text.Length; i++)
                {
                    symbols[i] = new BpeSymbol(
                                        prev: i == 0 ? -1 : i - 1,
                                        next: i == text.Length - 1 ? -1 : i + 1,
                                        pieceSpan: (i, 1));

                }

                agenda.Clear();
                for (int i = 1; i < text.Length; i++)
                {
                    TryMerge(i - 1, i, text);
                }

                while (agenda.Count > 0)
                {
                    SymbolPair top = agenda.Dequeue();

                    if (symbols[top.Left].pieceSpan.Length == 0 || symbols[top.Right].pieceSpan.Length == 0 ||
                        symbols[top.Left].pieceSpan.Length + symbols[top.Right].pieceSpan.Length != top.Length)
                    {
                        continue;
                    }

                    // Replaces symbols with `top` rule.
                    symbols[top.Left].pieceSpan = (symbols[top.Left].pieceSpan.Index, symbols[top.Left].pieceSpan.Length + symbols[top.Right].pieceSpan.Length);

                    // Updates prev/next pointers.
                    symbols[top.Left].next = symbols[top.Right].next;

                    if (symbols[top.Right].next >= 0)
                    {
                        symbols[symbols[top.Right].next].prev = top.Left;
                    }
                    symbols[top.Right].pieceSpan = (0, 0);

                    // Adds new symbol pairs which are newly added after symbol replacement.
                    TryMerge(symbols[top.Left].prev, top.Left, text);
                    TryMerge(top.Left, symbols[top.Left].next, text);
                }

                List<EncodedToken> result = new List<EncodedToken>(text.Length);

                for (int index = 0; (uint)index < (uint)text.Length; index = symbols[index].next)
                {
                    if (_vocab.TryGetValue(text.Slice(symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length), out (int Id, string Token) value))
                    {
                        result.Add(GetToken(value.Id, value.Token, symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length, originalText, mapping));
                    }
                    else if (UnknownTokenId.HasValue)
                    {
                        result.Add(GetToken(UnknownTokenId.Value, UnknownToken!, symbols[index].pieceSpan.Index, symbols[index].pieceSpan.Length, originalText, mapping));
                    }
                }

                return result;
            }
            finally
            {
                ArrayPool<BpeSymbol>.Shared.Return(symbols);
            }

            static EncodedToken GetToken(int id, string token, int index, int length, ReadOnlySpan<char> originalText, Span<int> mapping)
            {
                int endIndex = index + length < mapping.Length ? mapping[index + length] : originalText.Length;
                return new EncodedToken(id, token, new Range(mapping[index], endIndex));
            }

            void TryMerge(int left, int right, ReadOnlySpan<char> textSpan)
            {
                if (left == -1 || right == -1)
                {
                    return;
                }

                if (!_mergeRanks.TryGetValue(textSpan.Slice(symbols[left].pieceSpan.Index, symbols[left].pieceSpan.Length), textSpan.Slice(symbols[right].pieceSpan.Index, symbols[right].pieceSpan.Length), out int rank))
                {
                    return;
                }

                SymbolPair pair = new(left, right, rank, symbols[left].pieceSpan.Length + symbols[right].pieceSpan.Length);
                agenda.Enqueue(pair);
            }
        }

        // Added Tokens from https://huggingface.co/Salesforce/codegen-350M-mono/raw/main/tokenizer.json
        internal static readonly Dictionary<string, int> CodeGenSpecialTokens = new()
        {
            { "<|endoftext|>",                      50256 },
            { "                               ",    50257 },
            { "                              ",     50258 },
            { "                             ",      50259 },
            { "                            ",       50260 },
            { "                           ",        50261 },
            { "                          ",         50262 },
            { "                         ",          50263 },
            { "                        ",           50264 },
            { "                       ",            50265 },
            { "                      ",             50266 },
            { "                     ",              50267 },
            { "                    ",               50268 },
            { "                   ",                50269 },
            { "                  ",                 50270 },
            { "                 ",                  50271 },
            { "                ",                   50272 },
            { "               ",                    50273 },
            { "              ",                     50274 },
            { "             ",                      50275 },
            { "            ",                       50276 },
            { "           ",                        50277 },
            { "          ",                         50278 },
            { "         ",                          50279 },
            { "        ",                           50280 },
            { "       ",                            50281 },
            { "      ",                             50282 },
            { "     ",                              50283 },
            { "    ",                               50284 },
            { "   ",                                50285 },
            { "  ",                                 50286 },
            { "\t\t\t\t\t\t\t\t\t",                 50287 },
            { "\t\t\t\t\t\t\t\t",                   50288 },
            { "\t\t\t\t\t\t\t",                     50289 },
            { "\t\t\t\t\t\t",                       50290 },
            { "\t\t\t\t\t",                         50291 },
            { "\t\t\t\t",                           50292 },
            { "\t\t\t",                             50293 },
            { "\t\t",                               50294 },
        };

        private static Dictionary<StringSpanOrdinalKey, (int, string)> GetVocabulary(Stream vocabularyStream)
        {
            Vocabulary? vocab;
            try
            {
                vocab = JsonSerializer.Deserialize(vocabularyStream, ModelSourceGenerationContext.Default.Vocabulary);
            }
            catch (Exception e)
            {
                throw new ArgumentException($"Problems met when parsing JSON vocabulary object.{Environment.NewLine}Error message: {e.Message}");
            }

            if (vocab is null)
            {
                throw new ArgumentException($"Failed to read the vocabulary file.");
            }

            return vocab;
        }

        internal static Dictionary<StringSpanOrdinalKeyPair, int> GetMergeRanks(Stream mergeStream)
        {
            var mergeRanks = new Dictionary<StringSpanOrdinalKeyPair, int>();
            try
            {
                using StreamReader reader = new StreamReader(mergeStream);

                // We ignore the first and last line in the file
                if (reader.Peek() >= 0)
                {
                    string ignored = reader.ReadLine()!;
                }

                int rank = 1;
                while (reader.Peek() >= 0)
                {
                    string line = reader.ReadLine()!;
                    int index = line.IndexOf(' ');
                    if (index < 1 || index == line.Length - 1 || line.IndexOf(' ', index + 1) != -1)
                    {
                        throw new FormatException($"Invalid format of merge file at line: \"{line}\"");
                    }

                    mergeRanks.Add(new StringSpanOrdinalKeyPair(line.Substring(0, index), line.Substring(index + 1)), rank++);
                }
            }
            catch (Exception e)
            {
                // Report any issues encountered while consuming a data file as IOExceptions.
                throw new IOException($"Cannot read the file Merge file.{Environment.NewLine}Error message: {e.Message}", e);
            }

            return mergeRanks;
        }

        private struct SymbolPair : IEquatable<SymbolPair>, IComparable<SymbolPair>
        {
            public int Left { get; set; }
            public int Right { get; set; }
            public int Length { get; set; }
            public int Score { get; set; }

            public SymbolPair(int left, int right, int score, int length)
            {
                Left = left;
                Right = right;
                Score = score;
                Length = length;
            }

            public int CompareTo(SymbolPair other)
            {
                if (Score != other.Score)
                {
                    return Score.CompareTo(other.Score);
                }

                return Left.CompareTo(other.Left);
            }

            public override int GetHashCode()
            {
                int hashCode = 23;
                hashCode = (hashCode * 37) + Score.GetHashCode();
                hashCode = (hashCode * 37) + Left.GetHashCode();
                return hashCode;
            }

            // If the Left is identical, comparing the Score alone is sufficient.
            // This is because the Left cannot merge into a different Right and yield the same Score.
            public bool Equals(SymbolPair other) => Left == other.Left && Score == other.Score;
        }

        private record struct BpeSymbol(int prev, int next, (int Index, int Length) pieceSpan);

        /// <summary>
        /// Create a CodeGen tokenizer from the given vocab and merges streams.
        /// </summary>
        /// <param name="vocabStream">The stream containing the vocab file.</param>
        /// <param name="mergesStream">The stream containing the merges file.</param>
        /// <param name="addPrefixSpace">Indicate whether to add a space before the token.</param>
        /// <param name="addBeginOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <returns>The CodeGen tokenizer object.</returns>
        /// <remarks>
        /// The tokenizer will be created according to the configuration specified in https://huggingface.co/Salesforce/codegen-350M-mono/raw/main/tokenizer.json.
        /// It is important to provide the similar vocab and merges files to the ones used in the training of the model.
        /// The vocab and merges files can be downloaded from the following links:
        ///     https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/vocab.json?download=true
        ///     https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/merges.txt?download=true
        /// When creating the tokenizer, ensure that the vocabulary stream is sourced from a trusted provider.
        /// </remarks>
        public static CodeGenTokenizer Create(
            Stream vocabStream,
            Stream mergesStream,
            bool addPrefixSpace = false,
            bool addBeginOfSentence = false,
            bool addEndOfSentence = false)
        {
            if (vocabStream is null)
            {
                throw new ArgumentNullException(nameof(vocabStream));
            }

            if (mergesStream is null)
            {
                throw new ArgumentNullException(nameof(mergesStream));
            }

            return new CodeGenTokenizer(
                        vocabStream,
                        mergesStream,
                        new RegexPreTokenizer(TiktokenTokenizer.P50kBaseRegex(), CodeGenTokenizer.CodeGenSpecialTokens),
                        normalizer: null,
                        CodeGenTokenizer.CodeGenSpecialTokens,
                        addPrefixSpace: addPrefixSpace,
                        addBeginningOfSentence: addBeginOfSentence,
                        addEndOfSentence: addEndOfSentence);
        }
    }
}
