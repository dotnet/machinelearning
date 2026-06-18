// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Sentencepiece;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

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

        private SentencePieceTokenizer(SentencePieceBaseModel model)
        {
            _model = model;
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
        /// <param name="addBeginningOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <param name="specialTokens">The additional tokens to add to the vocabulary.</param>
        /// <remarks>
        /// When creating the tokenizer, ensure that the vocabulary stream is sourced from a trusted provider.
        /// </remarks>
        public static SentencePieceTokenizer Create(
            Stream modelStream,
            bool addBeginningOfSentence = true,
            bool addEndOfSentence = false,
            IReadOnlyDictionary<string, int>? specialTokens = null)
        {
            ModelProto modelProto = ModelProto.Parser.ParseFrom(modelStream);

            if (modelProto is null)
            {
                throw new ArgumentNullException(nameof(modelProto));
            }

            return new SentencePieceTokenizer(modelProto, addBeginningOfSentence, addEndOfSentence, specialTokens);
        }

        /// <summary>
        /// Creates a Unigram <see cref="SentencePieceTokenizer"/> from an in-memory vocabulary of (piece, score) pairs.
        /// </summary>
        /// <param name="vocab">
        /// The vocabulary as an ordered sequence of (piece, score) pairs. The position of each pair
        /// in the sequence determines its token ID.
        /// </param>
        /// <param name="unkId">The index (token ID) of the unknown token in <paramref name="vocab"/>.</param>
        /// <param name="addBeginningOfSentence">Whether to emit the beginning-of-sentence token during encoding.</param>
        /// <param name="addEndOfSentence">Whether to emit the end-of-sentence token during encoding.</param>
        /// <param name="precompiledCharsMap">
        /// Optional precompiled character normalization map (as found in the SentencePiece <c>normalizer_spec.precompiled_charsmap</c>
        /// field or in the Hugging Face <c>tokenizer.json</c> <c>normalizer.precompiled_charsmap</c> property).
        /// Pass <see langword="default"/> to skip precompiled normalization.
        /// </param>
        /// <param name="addDummyPrefix">Whether to prepend the dummy whitespace prefix character (U+2581) at the start of the input.</param>
        /// <param name="escapeWhiteSpaces">Whether to replace spaces with the dummy whitespace character (U+2581) during normalization.</param>
        /// <param name="treatWhitespaceAsSuffix">Whether to emit the U+2581 character at the end of the last token rather than the beginning of the first token.</param>
        /// <param name="byteFallback">Whether unknown characters are decomposed into UTF-8 byte pieces (<c>&lt;0x00&gt;</c>..<c>&lt;0xFF&gt;</c>) instead of the unknown token.</param>
        /// <param name="specialTokens">Additional special tokens to recognize, supplied as a mapping of token string to token ID.</param>
        /// <returns>A new <see cref="SentencePieceTokenizer"/> instance.</returns>
        /// <remarks>
        /// The beginning-of-sentence and end-of-sentence token IDs are auto-detected by looking for pieces
        /// named <c>&lt;s&gt;</c> and <c>&lt;/s&gt;</c> in <paramref name="vocab"/>. If a piece is not found it is
        /// treated as absent; requesting <paramref name="addBeginningOfSentence"/> or <paramref name="addEndOfSentence"/>
        /// when the corresponding piece is absent throws an <see cref="ArgumentException"/>. A <c>&lt;pad&gt;</c> piece
        /// is likewise detected automatically when present.
        /// <para>
        /// When creating the tokenizer, ensure that the vocabulary is sourced from a trusted provider.
        /// </para>
        /// </remarks>
        public static SentencePieceTokenizer Create(
            IEnumerable<(string Piece, float Score)> vocab,
            int unkId,
            bool addBeginningOfSentence = true,
            bool addEndOfSentence = false,
            ReadOnlySpan<byte> precompiledCharsMap = default,
            bool addDummyPrefix = true,
            bool escapeWhiteSpaces = true,
            bool treatWhitespaceAsSuffix = false,
            bool byteFallback = false,
            IReadOnlyDictionary<string, int>? specialTokens = null)
        {
            if (vocab is null)
            {
                throw new ArgumentNullException(nameof(vocab));
            }

            IReadOnlyList<(string Piece, float Score)> pieces = vocab as IReadOnlyList<(string Piece, float Score)>
                ?? new List<(string Piece, float Score)>(vocab);

            SentencePieceUnigramModel model = new SentencePieceUnigramModel(
                pieces, unkId, addBeginningOfSentence, addEndOfSentence,
                precompiledCharsMap, addDummyPrefix, escapeWhiteSpaces,
                treatWhitespaceAsSuffix, removeExtraWhitespaces: true, byteFallback, specialTokens);

            return new SentencePieceTokenizer(model);
        }

        /// <summary>
        /// Creates a Unigram <see cref="SentencePieceTokenizer"/> by parsing a Hugging Face <c>tokenizer.json</c>
        /// that contains a Unigram model (<c>model.type == "Unigram"</c>).
        /// </summary>
        /// <param name="tokenizerJsonStream">A stream containing the UTF-8-encoded <c>tokenizer.json</c> content.</param>
        /// <param name="addBeginningOfSentence">Whether to emit the beginning-of-sentence token during encoding.</param>
        /// <param name="addEndOfSentence">Whether to emit the end-of-sentence token during encoding.</param>
        /// <param name="specialTokens">Additional special tokens to recognize, supplied as a mapping of token string to token ID.</param>
        /// <returns>A new <see cref="SentencePieceTokenizer"/> instance.</returns>
        /// <remarks>
        /// The following fields are read from the JSON:
        /// <list type="bullet">
        ///   <item><description><c>model.vocab</c> — array of <c>[piece, score]</c> pairs (required).</description></item>
        ///   <item><description><c>model.unk_id</c> — index of the unknown token (required).</description></item>
        ///   <item><description><c>model.byte_fallback</c> — whether unknown characters fall back to UTF-8 byte pieces.</description></item>
        ///   <item><description><c>added_tokens</c> — special tokens (those with <c>"special": true</c>) and their IDs.</description></item>
        ///   <item><description><c>normalizer.precompiled_charsmap</c> (base64) — normalization map; also searched inside a <c>Sequence</c> normalizer.</description></item>
        ///   <item><description><c>pre_tokenizer</c> of type <c>Metaspace</c> — <c>add_prefix_space</c> and <c>replacement</c>; also searched inside a <c>Sequence</c> pre-tokenizer.</description></item>
        ///   <item><description><c>post_processor</c> (<c>TemplateProcessing</c>, <c>RobertaProcessing</c>, <c>BertProcessing</c>, or a <c>Sequence</c> of these) — the special tokens that wrap a single sequence, gated by <paramref name="addBeginningOfSentence"/> (prefix) and <paramref name="addEndOfSentence"/> (suffix).</description></item>
        /// </list>
        /// <para>
        /// <c>remove_extra_whitespaces</c> has no direct representation in <c>tokenizer.json</c>; it is deduced from
        /// the normalizer's whitespace-collapsing steps (a right-<c>Strip</c> plus a runs-of-spaces <c>Replace</c>) and
        /// from a <c>WhitespaceSplit</c> pre-tokenizer, defaulting to <see langword="false"/> when none are present, to
        /// match the Hugging Face fast-tokenizer runtime. Normalizers with content-modifying steps (per-character
        /// <c>Replace</c>, <c>Lowercase</c>, <c>StripAccents</c>, Unicode normalization, <c>Nmt</c>, <c>Prepend</c>) are
        /// applied in full before encoding. Pair-sequence templates and per-token <c>type_id</c>s are not applied.
        /// Templates that place a special token in the middle of the sequence are rejected with
        /// <see cref="NotSupportedException"/>.
        /// </para>
        /// <para>
        /// When creating the tokenizer, ensure that the JSON stream is sourced from a trusted provider.
        /// </para>
        /// </remarks>
        public static SentencePieceTokenizer CreateFromTokenizerJson(
            Stream tokenizerJsonStream,
            bool addBeginningOfSentence = true,
            bool addEndOfSentence = false,
            IReadOnlyDictionary<string, int>? specialTokens = null)
        {
            if (tokenizerJsonStream is null)
            {
                throw new ArgumentNullException(nameof(tokenizerJsonStream));
            }

            using JsonDocument doc = JsonDocument.Parse(tokenizerJsonStream);
            JsonElement root = doc.RootElement;

            // Validate model type
            if (!root.TryGetProperty("model", out JsonElement modelElement))
            {
                throw new InvalidDataException("The tokenizer.json does not contain a 'model' property.");
            }

            if (modelElement.ValueKind != JsonValueKind.Object)
            {
                throw new InvalidDataException("The tokenizer.json 'model' property must be a JSON object.");
            }

            // Validate the model is Unigram. Older tokenizer.json files (e.g. xlm-roberta-base, albert) omit the
            // model "type" entirely; treat a model that has a "vocab" but no BPE "merges" as Unigram, which matches
            // how the Hugging Face loaders disambiguate these files.
            if (modelElement.TryGetProperty("type", out JsonElement modelTypeElement) &&
                modelTypeElement.ValueKind == JsonValueKind.String)
            {
                if (!string.Equals(modelTypeElement.GetString(), "Unigram", StringComparison.OrdinalIgnoreCase))
                {
                    throw new InvalidDataException($"Expected model type 'Unigram' but found '{modelTypeElement.GetString()}'.");
                }
            }
            else if (modelElement.TryGetProperty("merges", out _))
            {
                throw new InvalidDataException("The tokenizer.json 'model' has no 'type' and contains 'merges'; this factory only supports 'Unigram' models.");
            }

            if (!modelElement.TryGetProperty("unk_id", out JsonElement unkIdElement))
            {
                throw new InvalidDataException("The tokenizer.json model does not contain an 'unk_id' property.");
            }

            int unkId = unkIdElement.GetInt32();

            bool byteFallback = modelElement.TryGetProperty("byte_fallback", out JsonElement byteFallbackElement) &&
                                byteFallbackElement.ValueKind == JsonValueKind.True;

            if (!modelElement.TryGetProperty("vocab", out JsonElement vocabElement) ||
                vocabElement.ValueKind != JsonValueKind.Array)
            {
                throw new InvalidDataException("The tokenizer.json model does not contain a valid 'vocab' array.");
            }

            List<(string Piece, float Score)> vocab = new List<(string Piece, float Score)>(vocabElement.GetArrayLength());
            foreach (JsonElement entry in vocabElement.EnumerateArray())
            {
                if (entry.ValueKind != JsonValueKind.Array || entry.GetArrayLength() < 2)
                {
                    throw new InvalidDataException("Each entry in 'model.vocab' must be a [piece, score] array.");
                }

                string? piece = entry[0].GetString();
                if (piece is null)
                {
                    throw new InvalidDataException("A piece string in 'model.vocab' is null.");
                }

                vocab.Add((piece, entry[1].GetSingle()));
            }

            // Extract normalizer settings
            byte[]? precompiledCharsMap = null;
            bool addDummyPrefix = true;
            // HF tokenizer.json has no remove_extra_whitespaces flag; SpmConverter encodes that behavior as
            // explicit normalizer steps (a right-Strip plus a Replace collapsing runs of spaces). Deduce it from
            // those steps, defaulting to false when absent to match the HF fast-tokenizer runtime.
            bool removeExtraWhitespaces = false;
            // When the normalizer has content-modifying steps that the charsmap + removeExtraWhitespaces
            // approximation cannot represent (per-character Replace, Lowercase, Unicode normalization, ...),
            // apply the full normalizer chain (charsmap included) before the Metaspace pass instead.
            SentencePieceNormalizationStep? chainNormalizer = null;
            if (root.TryGetProperty("normalizer", out JsonElement normalizerElement) &&
                normalizerElement.ValueKind == JsonValueKind.Object)
            {
                if (SentencePieceNormalizationStep.HasRichSteps(normalizerElement))
                {
                    chainNormalizer = SentencePieceNormalizationStep.Build(normalizerElement);
                    // The chain owns the charsmap and any strip/collapse, so the model's own normalizer must do
                    // only the Metaspace escaping to avoid applying either transformation twice.
                    precompiledCharsMap = null;
                    removeExtraWhitespaces = false;
                }
                else
                {
                    precompiledCharsMap = ExtractPrecompiledCharsMap(normalizerElement);
                    removeExtraWhitespaces = NormalizerCollapsesWhitespace(normalizerElement);
                }
            }

            // Extract pre_tokenizer settings
            bool escapeWhiteSpaces = true;
            bool treatWhitespaceAsSuffix = false;
            if (root.TryGetProperty("pre_tokenizer", out JsonElement preTokenizerElement) &&
                preTokenizerElement.ValueKind == JsonValueKind.Object)
            {
                ExtractMetaspaceSettings(preTokenizerElement, ref addDummyPrefix, ref escapeWhiteSpaces, ref treatWhitespaceAsSuffix);

                // A WhitespaceSplit pre-tokenizer splits on whitespace and drops the empties, which collapses runs of
                // whitespace and strips leading/trailing whitespace before Metaspace adds the dummy prefix. That is
                // exactly remove_extra_whitespaces, so honor it even when the normalizer carries no collapse step.
                // This is independent of chain-mode: the chain handles the normalizer's own steps, but whitespace
                // collapsing here comes from the pre-tokenizer and is applied by the model's Metaspace pass.
                if (PreTokenizerSplitsWhitespace(preTokenizerElement))
                {
                    removeExtraWhitespaces = true;
                }
            }

            // Merge the special tokens declared in added_tokens (authoritative source for their IDs) with any
            // caller-supplied special tokens; the caller's entries win on conflict.
            Dictionary<string, int> mergedSpecialTokens = ParseAddedTokens(root);
            if (specialTokens is not null)
            {
                foreach (var kvp in specialTokens)
                {
                    mergedSpecialTokens[kvp.Key] = kvp.Value;
                }
            }

            // Resolve the prefix/suffix special-token wrapping from the post_processor (if present), falling back
            // to the SentencePiece-conventional <s>/</s> names otherwise.
            ResolvePostProcessorAffixes(root, vocab, mergedSpecialTokens,
                out List<(int Id, string Token)> prefixTokens, out List<(int Id, string Token)> suffixTokens);

            // Ensure every wrapping token is registered as a special token so it is classified Control and round-trips on decode.
            foreach (var (id, token) in prefixTokens)
            {
                mergedSpecialTokens[token] = id;
            }
            foreach (var (id, token) in suffixTokens)
            {
                mergedSpecialTokens[token] = id;
            }

            int padId = mergedSpecialTokens.TryGetValue("<pad>", out int p) ? p : FindPieceId(vocab, "<pad>");

            SentencePieceUnigramModel model = new SentencePieceUnigramModel(
                vocab, unkId, addBeginningOfSentence, addEndOfSentence,
                precompiledCharsMap is not null ? precompiledCharsMap.AsSpan() : default,
                addDummyPrefix, escapeWhiteSpaces, treatWhitespaceAsSuffix, removeExtraWhitespaces, byteFallback,
                mergedSpecialTokens.Count > 0 ? mergedSpecialTokens : null,
                prefixTokens, suffixTokens, padId);

            if (chainNormalizer is not null)
            {
                model.Normalizer!.NormalizationChain = chainNormalizer;
            }

            return new SentencePieceTokenizer(model);
        }

        // Reads the special tokens (those marked "special": true) from the top-level added_tokens array.
        private static Dictionary<string, int> ParseAddedTokens(JsonElement root)
        {
            Dictionary<string, int> result = new Dictionary<string, int>();
            if (!root.TryGetProperty("added_tokens", out JsonElement addedTokens) || addedTokens.ValueKind != JsonValueKind.Array)
            {
                return result;
            }

            foreach (JsonElement entry in addedTokens.EnumerateArray())
            {
                if (entry.ValueKind != JsonValueKind.Object)
                {
                    continue;
                }

                if (!entry.TryGetProperty("special", out JsonElement specialElement) || specialElement.ValueKind != JsonValueKind.True)
                {
                    continue;
                }

                if (entry.TryGetProperty("content", out JsonElement contentElement) &&
                    entry.TryGetProperty("id", out JsonElement idElement) &&
                    contentElement.GetString() is string content)
                {
                    result[content] = idElement.GetInt32();
                }
            }

            return result;
        }

        // Resolves the ordered prefix/suffix special tokens that wrap an encoded sequence, from the post_processor.
        private static void ResolvePostProcessorAffixes(
            JsonElement root,
            IReadOnlyList<(string Piece, float Score)> vocab,
            IReadOnlyDictionary<string, int> specialTokens,
            out List<(int Id, string Token)> prefixTokens,
            out List<(int Id, string Token)> suffixTokens)
        {
            prefixTokens = new List<(int Id, string Token)>();
            suffixTokens = new List<(int Id, string Token)>();

            if (root.TryGetProperty("post_processor", out JsonElement postProcessor) &&
                postProcessor.ValueKind == JsonValueKind.Object)
            {
                ProcessPostProcessor(postProcessor, vocab, specialTokens, prefixTokens, suffixTokens);
                return;
            }

            // No post_processor: fall back to the SentencePiece-conventional names.
            AddAffixToken(prefixTokens, "<s>", vocab, specialTokens, required: false);
            AddAffixToken(suffixTokens, "</s>", vocab, specialTokens, required: false);
        }

        private static void ProcessPostProcessor(
            JsonElement postProcessor,
            IReadOnlyList<(string Piece, float Score)> vocab,
            IReadOnlyDictionary<string, int> specialTokens,
            List<(int Id, string Token)> prefixTokens,
            List<(int Id, string Token)> suffixTokens)
        {
            string? type = postProcessor.TryGetProperty("type", out JsonElement typeElement) ? typeElement.GetString() : null;

            switch (type)
            {
                case "TemplateProcessing":
                    ProcessTemplate(postProcessor, vocab, specialTokens, prefixTokens, suffixTokens);
                    break;

                case "RobertaProcessing":
                    AddProcessorAffix(postProcessor, "cls", prefixTokens, vocab, specialTokens);
                    AddProcessorAffix(postProcessor, "sep", suffixTokens, vocab, specialTokens);
                    break;

                case "BertProcessing":
                    AddProcessorAffix(postProcessor, "cls", prefixTokens, vocab, specialTokens);
                    AddProcessorAffix(postProcessor, "sep", suffixTokens, vocab, specialTokens);
                    break;

                case "Sequence":
                    if (postProcessor.TryGetProperty("processors", out JsonElement processors) && processors.ValueKind == JsonValueKind.Array)
                    {
                        foreach (JsonElement inner in processors.EnumerateArray())
                        {
                            if (inner.ValueKind == JsonValueKind.Object)
                            {
                                ProcessPostProcessor(inner, vocab, specialTokens, prefixTokens, suffixTokens);
                            }
                        }
                    }
                    break;

                default:
                    // ByteLevel and other processors do not contribute special-token wrapping; ignore them.
                    break;
            }
        }

        // Parses a TemplateProcessing "single" template into leading (prefix) and trailing (suffix) special tokens.
        private static void ProcessTemplate(
            JsonElement postProcessor,
            IReadOnlyList<(string Piece, float Score)> vocab,
            IReadOnlyDictionary<string, int> specialTokens,
            List<(int Id, string Token)> prefixTokens,
            List<(int Id, string Token)> suffixTokens)
        {
            if (!postProcessor.TryGetProperty("single", out JsonElement single) || single.ValueKind != JsonValueKind.Array)
            {
                return;
            }

            JsonElement? ppSpecialTokens = postProcessor.TryGetProperty("special_tokens", out JsonElement st) && st.ValueKind == JsonValueKind.Object
                ? st : (JsonElement?)null;

            bool seenSequence = false;
            foreach (JsonElement item in single.EnumerateArray())
            {
                if (item.ValueKind != JsonValueKind.Object)
                {
                    continue;
                }

                if (item.TryGetProperty("Sequence", out _))
                {
                    if (seenSequence)
                    {
                        throw new NotSupportedException("tokenizer.json post_processor templates with more than one sequence are not supported.");
                    }

                    seenSequence = true;
                }
                else if (item.TryGetProperty("SpecialToken", out JsonElement specialToken) &&
                         specialToken.TryGetProperty("id", out JsonElement idElement) &&
                         idElement.GetString() is string tokenName)
                {
                    int id = ResolveTemplateTokenId(tokenName, ppSpecialTokens, specialTokens, vocab);
                    (seenSequence ? suffixTokens : prefixTokens).Add((id, tokenName));
                }
            }

            if (!seenSequence)
            {
                throw new NotSupportedException("tokenizer.json post_processor template does not contain a sequence placeholder.");
            }
        }

        private static int ResolveTemplateTokenId(
            string tokenName,
            JsonElement? ppSpecialTokens,
            IReadOnlyDictionary<string, int> specialTokens,
            IReadOnlyList<(string Piece, float Score)> vocab)
        {
            if (ppSpecialTokens is JsonElement st &&
                st.TryGetProperty(tokenName, out JsonElement entry) &&
                entry.TryGetProperty("ids", out JsonElement ids) &&
                ids.ValueKind == JsonValueKind.Array &&
                ids.GetArrayLength() > 0)
            {
                return ids[0].GetInt32();
            }

            if (specialTokens.TryGetValue(tokenName, out int specialId))
            {
                return specialId;
            }

            int vocabId = FindPieceId(vocab, tokenName);
            if (vocabId < 0)
            {
                throw new InvalidDataException($"The tokenizer.json post_processor references special token '{tokenName}' that is not present in the vocabulary.");
            }

            return vocabId;
        }

        private static void AddProcessorAffix(
            JsonElement postProcessor,
            string property,
            List<(int Id, string Token)> target,
            IReadOnlyList<(string Piece, float Score)> vocab,
            IReadOnlyDictionary<string, int> specialTokens)
        {
            // Roberta/Bert processors store cls/sep as [token, id] arrays.
            if (postProcessor.TryGetProperty(property, out JsonElement el) && el.ValueKind == JsonValueKind.Array && el.GetArrayLength() >= 2 &&
                el[0].GetString() is string token)
            {
                target.Add((el[1].GetInt32(), token));
            }
        }

        private static void AddAffixToken(
            List<(int Id, string Token)> target,
            string tokenName,
            IReadOnlyList<(string Piece, float Score)> vocab,
            IReadOnlyDictionary<string, int> specialTokens,
            bool required)
        {
            int id = specialTokens.TryGetValue(tokenName, out int specialId) ? specialId : FindPieceId(vocab, tokenName);
            if (id >= 0)
            {
                target.Add((id, tokenName));
            }
            else if (required)
            {
                throw new InvalidDataException($"The tokenizer.json does not contain the required special token '{tokenName}'.");
            }
        }

        private static int FindPieceId(IReadOnlyList<(string Piece, float Score)> vocab, string token)
        {
            for (int i = 0; i < vocab.Count; i++)
            {
                if (vocab[i].Piece == token)
                {
                    return i;
                }
            }

            return -1;
        }

        private static byte[]? ExtractPrecompiledCharsMap(JsonElement normalizer)
        {
            if (!normalizer.TryGetProperty("type", out JsonElement typeElement))
            {
                return null;
            }

            string? type = typeElement.GetString();
            if (string.Equals(type, "Precompiled", StringComparison.OrdinalIgnoreCase))
            {
                if (normalizer.TryGetProperty("precompiled_charsmap", out JsonElement mapElement))
                {
                    string? base64 = mapElement.GetString();
                    if (base64 is not null)
                    {
                        return Convert.FromBase64String(base64);
                    }
                }
                return null;
            }
            else if (string.Equals(type, "Sequence", StringComparison.OrdinalIgnoreCase) &&
                     normalizer.TryGetProperty("normalizers", out JsonElement normalizersElement) &&
                     normalizersElement.ValueKind == JsonValueKind.Array)
            {
                // A Sequence may legitimately interleave the precompiled map with other steps (Nmt, Replace, ...).
                // Extract the precompiled map and ignore the steps we don't model rather than failing the load.
                byte[]? result = null;
                foreach (JsonElement inner in normalizersElement.EnumerateArray())
                {
                    if (inner.ValueKind != JsonValueKind.Object)
                    {
                        continue;
                    }

                    byte[]? innerResult = ExtractPrecompiledCharsMap(inner);
                    if (innerResult is not null)
                    {
                        result = innerResult;
                    }
                }
                return result;
            }

            // Other normalizer types (Nmt, Replace, Lowercase, ...) carry no precompiled map; treat as absent.
            return null;
        }

        // Detects whether the normalizer collapses extra whitespace, i.e. SentencePiece's remove_extra_whitespaces.
        // HF's SpmConverter emits this as a right-Strip plus a Replace of a runs-of-spaces Regex (" {2,}") -> "▁".
        private static bool NormalizerCollapsesWhitespace(JsonElement normalizer)
        {
            if (normalizer.ValueKind != JsonValueKind.Object || !normalizer.TryGetProperty("type", out JsonElement typeElement))
            {
                return false;
            }

            string? type = typeElement.GetString();

            if (string.Equals(type, "Strip", StringComparison.OrdinalIgnoreCase))
            {
                // A right-Strip removes trailing whitespace; treat its presence as the strip half of the behavior.
                return !normalizer.TryGetProperty("strip_right", out JsonElement stripRight) || stripRight.ValueKind != JsonValueKind.False;
            }

            if (string.Equals(type, "Replace", StringComparison.OrdinalIgnoreCase))
            {
                return ReplaceCollapsesSpaces(normalizer);
            }

            if (string.Equals(type, "Sequence", StringComparison.OrdinalIgnoreCase) &&
                normalizer.TryGetProperty("normalizers", out JsonElement normalizersElement) &&
                normalizersElement.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement inner in normalizersElement.EnumerateArray())
                {
                    if (NormalizerCollapsesWhitespace(inner))
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        // True only for a Replace whose Regex matches runs of two-or-more spaces, not a single-space Metaspace Replace.
        private static bool ReplaceCollapsesSpaces(JsonElement replace)
        {
            if (!replace.TryGetProperty("pattern", out JsonElement patternElement) ||
                patternElement.ValueKind != JsonValueKind.Object ||
                !patternElement.TryGetProperty("Regex", out JsonElement regexElement))
            {
                return false;
            }

            string? pattern = regexElement.GetString();
            if (pattern is null)
            {
                return false;
            }

            // Do not trim: HF's canonical patterns " {2,}" and " +" carry a significant leading space.
            switch (pattern)
            {
                case " {2,}":
                case " +":
                case "[ ]+":
                case "[ ]{2,}":
                case "\\s+":
                case "\\s{2,}":
                    return true;
                default:
                    return false;
            }
        }

        // Returns true if the pre-tokenizer splits on whitespace (WhitespaceSplit/Whitespace), recursing into a
        // Sequence. Such a split discards whitespace runs, matching SentencePiece's remove_extra_whitespaces.
        private static bool PreTokenizerSplitsWhitespace(JsonElement preTokenizer)
        {
            if (preTokenizer.ValueKind != JsonValueKind.Object || !preTokenizer.TryGetProperty("type", out JsonElement typeElement))
            {
                return false;
            }

            string? type = typeElement.GetString();
            if (string.Equals(type, "WhitespaceSplit", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(type, "Whitespace", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            if (string.Equals(type, "Sequence", StringComparison.OrdinalIgnoreCase) &&
                preTokenizer.TryGetProperty("pretokenizers", out JsonElement preTokenizersElement) &&
                preTokenizersElement.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement inner in preTokenizersElement.EnumerateArray())
                {
                    if (PreTokenizerSplitsWhitespace(inner))
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        private static void ExtractMetaspaceSettings(JsonElement preTokenizer, ref bool addDummyPrefix, ref bool escapeWhiteSpaces, ref bool treatWhitespaceAsSuffix)
        {
            if (!preTokenizer.TryGetProperty("type", out JsonElement typeElement))
            {
                return;
            }

            string? type = typeElement.GetString();
            if (string.Equals(type, "Metaspace", StringComparison.OrdinalIgnoreCase))
            {
                if (preTokenizer.TryGetProperty("add_prefix_space", out JsonElement addPrefixElement))
                {
                    addDummyPrefix = addPrefixElement.GetBoolean();
                }

                if (preTokenizer.TryGetProperty("replacement", out JsonElement replacementElement))
                {
                    // HF Metaspace's 'replacement' is the actual whitespace marker character. The SentencePiece model
                    // only supports U+2581 ('▁'); reject any other marker rather than silently not escaping spaces.
                    string? replacement = replacementElement.GetString();
                    if (replacement is not null && replacement != "\u2581") // U+2581 LOWER ONE EIGHTH BLOCK (▁)
                    {
                        throw new NotSupportedException(
                            $"The Metaspace 'replacement' '{replacement}' is not supported; only U+2581 ('\u2581') is supported.");
                    }

                    escapeWhiteSpaces = true;
                }

                if (preTokenizer.TryGetProperty("prepend_scheme", out JsonElement prependSchemeElement))
                {
                    string? scheme = prependSchemeElement.GetString();
                    // "never" suppresses the dummy prefix; "always"/"first" keep the default (true)
                    if (string.Equals(scheme, "never", StringComparison.OrdinalIgnoreCase))
                    {
                        addDummyPrefix = false;
                    }
                }
            }
            else if (string.Equals(type, "Sequence", StringComparison.OrdinalIgnoreCase) &&
                     preTokenizer.TryGetProperty("pretokenizers", out JsonElement preTokenizersElement) &&
                     preTokenizersElement.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement inner in preTokenizersElement.EnumerateArray())
                {
                    ExtractMetaspaceSettings(inner, ref addDummyPrefix, ref escapeWhiteSpaces, ref treatWhitespaceAsSuffix);
                }
            }
        }
    }
}
