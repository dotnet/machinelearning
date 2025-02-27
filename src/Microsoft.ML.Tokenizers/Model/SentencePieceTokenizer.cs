// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Sentencepiece;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;

namespace Microsoft.ML.Tokenizers
{
    // SentencePieceBpe is implementing the BPE algorithm based on the SentencePiece https://github.com/google/sentencepiece.
    // SentencePiece is under the Apache License 2.0 https://github.com/google/sentencepiece/blob/master/LICENSE

    /// <summary>
    /// SentencePieceBpe is a tokenizer that splits the input into tokens using the SentencePiece Bpe model.
    /// </summary>
    public class SentencePieceTokenizer : Tokenizer
    {
        private readonly SentencePieceBaseModel _model;

        internal SentencePieceTokenizer(ModelProto modelProto, bool addBos, bool addEos, IReadOnlyDictionary<string, int>? specialTokens = null)
        {
            _model = modelProto.TrainerSpec.ModelType switch
            {
                TrainerSpec.Types.ModelType.Bpe => new SentencePieceBpeModel(modelProto, addBos, addEos, specialTokens),
                TrainerSpec.Types.ModelType.Unigram => new SentencePieceUnigramModel(modelProto, addBos, addEos, specialTokens),
                _ => throw new ArgumentException($"The model type '{modelProto.TrainerSpec.ModelType}' is not supported.", nameof(modelProto))
            };
        }

        internal SentencePieceTokenizer(SentencePieceOptions options)
        {
            _model = options.ModelType switch
            {
                SentencePieceModelType.Bpe => new SentencePieceBpeModel(options),
                SentencePieceModelType.Unigram => new SentencePieceUnigramModel(options),
                _ => throw new ArgumentException($"The model type '{options.ModelType}' is not supported.", nameof(options.ModelType))
            };
        }

        /// <summary>
        /// The special tokens.
        /// </summary>
        public IReadOnlyDictionary<string, int>? SpecialTokens => _model.SpecialTokens;

        /// <summary>
        /// Specifies whether the model will do a byte fallback when it encounters unknown tokens during the encoding process.
        /// </summary>
        public bool ByteFallback => _model.ByteFallback;

        /// <summary>
        /// Indicate emitting the prefix character U+2581 at the beginning of sentence token during the normalization and encoding.
        /// </summary>
        public bool AddDummyPrefix => _model.AddDummyPrefix;

        /// <summary>
        /// Indicate if the spaces should be replaced with character U+2581 during the normalization and encoding.
        /// </summary>
        public bool EscapeWhiteSpaces => _model.EscapeWhiteSpaces;

        /// <summary>
        /// Indicate emitting the character U+2581 at the end of the last sentence token instead beginning of sentence token during the normalization and encoding.
        /// </summary>
        public bool TreatWhitespaceAsSuffix { get => _model.TreatWhitespaceAsSuffix; private set => _model.TreatWhitespaceAsSuffix = value; }

        /// <summary>
        /// Indicate emitting the beginning of sentence token during the encoding.
        /// </summary>
        public bool AddBeginningOfSentence => _model.AddBeginningOfSentence;

        /// <summary>
        /// Indicate emitting the end of sentence token during the encoding.
        /// </summary>
        public bool AddEndOfSentence => _model.AddEndOfSentence;

        /// <summary>
        /// The beginning of sentence token.
        /// </summary>
        public string BeginningOfSentenceToken => _model.BeginningOfSentenceToken;

        /// <summary>
        /// The end of sentence token.
        /// </summary>
        public string EndOfSentenceToken => _model.EndOfSentenceToken;

        /// <summary>
        /// The unknown token.
        /// </summary>
        public string UnknownToken => _model.UnknownToken;

        /// <summary>
        /// The id of the beginning of sentence token.
        /// </summary>
        public int BeginningOfSentenceId => _model.BeginningOfSentenceId;

        /// <summary>
        /// The id of the end of sentence token.
        /// </summary>
        public int EndOfSentenceId => _model.EndOfSentenceId;

        /// <summary>
        /// The id of the unknown token.
        /// </summary>
        public int UnknownId => _model.UnknownId;

        /// <summary>
        /// Gets the PreTokenizer used by the Tokenizer.
        /// </summary>
        public override PreTokenizer? PreTokenizer => null;

        /// <summary>
        /// Gets the Normalizer in use by the Tokenizer.
        /// </summary>
        public override Normalizer? Normalizer => _model.Normalizer;

        /// <summary>
        /// The vocabulary of the model.
        /// </summary>
        public IReadOnlyDictionary<string, int> Vocabulary => _model.Vocabulary;

        /// <summary>
        /// Encodes input text to a list of <see cref="EncodedToken" />s.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        protected override EncodeResults<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
        {
            return new EncodeResults<EncodedToken>
            {
                Tokens = _model.EncodeToTokens(text, textSpan, out string? normalizedText, AddBeginningOfSentence, AddEndOfSentence, settings.ConsiderNormalization),
                NormalizedText = normalizedText,
                CharsConsumed = normalizedText?.Length ?? text?.Length ?? textSpan.Length
            };
        }

        /// <summary>
        /// Encodes input text a list of <see cref="EncodedToken" />s with string value of the token, id, and offset.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The tokenization result includes a list of <see cref="EncodedToken" />s with string value of the token, id, and offset.</returns>
        public IReadOnlyList<EncodedToken> EncodeToTokens(string text, out string? normalizedText, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => _model.EncodeToTokens(text, Span<char>.Empty, out normalizedText, addBeginningOfSentence, addEndOfSentence, considerNormalization);

        /// <summary>
        /// Encodes input text a list of <see cref="EncodedToken" />s with string value of the token, id, and offset.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The tokenization result includes a list of <see cref="EncodedToken" />s with string value of the token, id, and offset.</returns>
        public IReadOnlyList<EncodedToken> EncodeToTokens(ReadOnlySpan<char> text, out string? normalizedText, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => _model.EncodeToTokens(null, text, out normalizedText, addBeginningOfSentence, addEndOfSentence, considerNormalization);


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
                Tokens = _model.EncodeToIds(text, textSpan, AddBeginningOfSentence, AddEndOfSentence, settings.ConsiderNormalization, out string? normalizedText, out int charsConsumed, settings.MaxTokenCount),
                NormalizedText = normalizedText,
                CharsConsumed = charsConsumed
            };
        }

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(string text, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => _model.EncodeToIds(text, Span<char>.Empty, addBeginningOfSentence, addEndOfSentence, considerNormalization, out _, out _);

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => _model.EncodeToIds(null, text, addBeginningOfSentence, addEndOfSentence, considerNormalization, out _, out _);

        /// <summary>
        /// Encodes input text to token Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="charsConsumed">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(string text, bool addBeginningOfSentence, bool addEndOfSentence, int maxTokenCount, out string? normalizedText, out int charsConsumed, bool considerPreTokenization = true, bool considerNormalization = true)
            => _model.EncodeToIds(text, Span<char>.Empty, addBeginningOfSentence, addEndOfSentence, considerNormalization, out normalizedText, out charsConsumed, maxTokenCount);

        /// <summary>
        /// Encodes input text to token Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="charsConsumed">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, bool addBeginningOfSentence, bool addEndOfSentence, int maxTokenCount, out string? normalizedText, out int charsConsumed, bool considerPreTokenization = true, bool considerNormalization = true)
            => _model.EncodeToIds(null, text, addBeginningOfSentence, addEndOfSentence, considerNormalization, out normalizedText, out charsConsumed, maxTokenCount);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        protected override int CountTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
            => _model.CountTokens(text, textSpan, AddBeginningOfSentence, AddEndOfSentence, settings.ConsiderNormalization, out _, out _, settings.MaxTokenCount);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        public int CountTokens(string text, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => _model.CountTokens(text, ReadOnlySpan<char>.Empty, addBeginningOfSentence, addEndOfSentence, considerNormalization, out _, out _, int.MaxValue);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        public int CountTokens(ReadOnlySpan<char> text, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization = true, bool considerNormalization = true)
            => _model.CountTokens(null, text, addBeginningOfSentence, addEndOfSentence, considerNormalization, out _, out _, int.MaxValue);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="charsConsumed">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        public int CountTokens(string text, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization, bool considerNormalization, out string? normalizedText, out int charsConsumed, int maxTokenCount = int.MaxValue)
            => _model.CountTokens(text, ReadOnlySpan<char>.Empty, addBeginningOfSentence, addEndOfSentence, considerNormalization, out normalizedText, out charsConsumed, maxTokenCount);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="charsConsumed">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        public int CountTokens(ReadOnlySpan<char> text, bool addBeginningOfSentence, bool addEndOfSentence, bool considerPreTokenization, bool considerNormalization, out string? normalizedText, out int charsConsumed, int maxTokenCount = int.MaxValue)
            => _model.CountTokens(null, text, addBeginningOfSentence, addEndOfSentence, considerNormalization, out normalizedText, out charsConsumed, maxTokenCount);

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
                return _model.GetIndexByTokenCountFromEnd(text, textSpan, AddBeginningOfSentence, AddEndOfSentence, settings.MaxTokenCount, settings.ConsiderNormalization, out normalizedText, out tokenCount);
            }

            tokenCount = _model.CountTokens(text, textSpan, AddBeginningOfSentence, AddEndOfSentence, settings.ConsiderNormalization, out normalizedText, out int charsConsumed, settings.MaxTokenCount);
            return charsConsumed;
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the start within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the text or the <paramref name="normalizedText"/> if the normalization is enabled.
        /// </returns>
        public int GetIndexByTokenCount(string text, bool addBeginningOfSentence, bool addEndOfSentence, int maxTokenCount, out string? normalizedText, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            tokenCount = _model.CountTokens(text, ReadOnlySpan<char>.Empty, addBeginningOfSentence, addEndOfSentence, considerNormalization, out normalizedText, out int charsConsumed, maxTokenCount);
            return charsConsumed;
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the start within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the text or the <paramref name="normalizedText"/> if the normalization is enabled.
        /// </returns>
        public int GetIndexByTokenCount(ReadOnlySpan<char> text, bool addBeginningOfSentence, bool addEndOfSentence, int maxTokenCount, out string? normalizedText, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            tokenCount = _model.CountTokens(null, text, addBeginningOfSentence, addEndOfSentence, considerNormalization, out normalizedText, out int charsConsumed, maxTokenCount);
            return charsConsumed;
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the <paramref name="normalizedText"/>; conversely, if all tokens fit, the result will be 0.
        /// </returns>
        public int GetIndexByTokenCountFromEnd(string text, bool addBeginningOfSentence, bool addEndOfSentence, int maxTokenCount, bool considerNormalization, out string? normalizedText, out int tokenCount)
            => _model.GetIndexByTokenCountFromEnd(text, ReadOnlySpan<char>.Empty, addBeginningOfSentence, addEndOfSentence, maxTokenCount, considerNormalization, out normalizedText, out tokenCount);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the <paramref name="normalizedText"/>; conversely, if all tokens fit, the result will be 0.
        /// </returns>
        public int GetIndexByTokenCountFromEnd(ReadOnlySpan<char> text, bool addBeginningOfSentence, bool addEndOfSentence, int maxTokenCount, bool considerNormalization, out string? normalizedText, out int tokenCount)
            => _model.GetIndexByTokenCountFromEnd(null, text, addBeginningOfSentence, addEndOfSentence, maxTokenCount, considerNormalization, out normalizedText, out tokenCount);

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <returns>The decoded string.</returns>
        public override string Decode(IEnumerable<int> ids) => _model.Decode(ids, considerSpecialTokens: false);

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="considerSpecialTokens">Indicate whether to consider special tokens during decoding.</param>
        /// <returns>The decoded string.</returns>
        public string Decode(IEnumerable<int> ids, bool considerSpecialTokens) => _model.Decode(ids, considerSpecialTokens);

        /// <summary>
        /// Decode the given ids back to text and store the result in the <paramref name="destination"/> span.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="destination">The span to store the decoded text.</param>
        /// <param name="idsConsumed">The number of ids consumed during the decoding.</param>
        /// <param name="charsWritten">The number of characters written to the destination span.</param>
        /// <returns>The operation status indicates whether all IDs were successfully decoded or if the <paramref name="destination"/> is too small to contain the entire decoded result.</returns>
        public override OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, out int idsConsumed, out int charsWritten)
            => _model.Decode(ids, destination, considerSpecialTokens: false, out idsConsumed, out charsWritten);

        /// <summary>
        /// Decode the given ids back to text and store the result in the <paramref name="destination"/> span.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="destination">The span to store the decoded text.</param>
        /// /// <param name="considerSpecialTokens">Indicate whether to consider special tokens during decoding.</param>
        /// <param name="idsConsumed">The number of ids consumed during the decoding.</param>
        /// <param name="charsWritten">The number of characters written to the destination span.</param>
        /// <returns>The operation status indicates whether all IDs were successfully decoded or if the <paramref name="destination"/> is too small to contain the entire decoded result.</returns>
        public OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, bool considerSpecialTokens, out int idsConsumed, out int charsWritten)
            => _model.Decode(ids, destination, considerSpecialTokens, out idsConsumed, out charsWritten);

        /// <summary>
        /// Creates an instance of SentencePieceTokenizer. The model stream should contain a SentencePiece model as specified in the following documentation:
        /// https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto.
        /// </summary>
        /// <param name="modelStream">The stream containing the SentencePiece Bpe or Unigram model.</param>
        /// <param name="addBeginOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="specialTokens">The additional tokens to add to the vocabulary.</param>
        /// <remarks>
        /// When creating the tokenizer, ensure that the vocabulary stream is sourced from a trusted provider.
        /// </remarks>
        public static SentencePieceTokenizer Create(
            Stream modelStream,
            bool addBeginOfSentence = true,
            bool addEndOfSentence = false,
            IReadOnlyDictionary<string, int>? specialTokens = null)
        {
            ModelProto modelProto = ModelProto.Parser.ParseFrom(modelStream);

            if (modelProto is null)
            {
                throw new ArgumentNullException(nameof(modelProto));
            }

            return new SentencePieceTokenizer(modelProto, addBeginOfSentence, addEndOfSentence, specialTokens);
        }

        /// <summary>
        /// Creates an instance of SentencePieceTokenizer.
        /// </summary>
        /// <param name="options">The options to use for the sentence piece tokenizer.</param>
        public static SentencePieceTokenizer Create(SentencePieceOptions options)
        {
            if (options is null)
            {
                throw new ArgumentNullException(nameof(options));
            }

            return new SentencePieceTokenizer(options);
        }
    }
}
