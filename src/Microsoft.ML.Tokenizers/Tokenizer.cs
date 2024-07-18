// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// serves as an abstraction for concrete tokenizers, enabling the encoding of text into tokens and IDs, as well as the decoding of IDs back into text.
    /// </summary>
    public abstract class Tokenizer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Tokenizer"/> class.
        /// </summary>
        protected Tokenizer() { }

        /// <summary>
        /// Gets the PreTokenizer used by the Tokenizer.
        /// </summary>
        public virtual PreTokenizer? PreTokenizer => null;

        /// <summary>
        /// Gets the Normalizer in use by the Tokenizer.
        /// </summary>
        public virtual Normalizer? Normalizer => null;

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        /// <returns>The encoded results containing the list of encoded Ids.</returns>
        protected abstract EncodeResults<int> EncodeToIds(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings);

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(string text, bool considerPreTokenization = true, bool considerNormalization = true)
             => EncodeToIds(text, ReadOnlySpan<char>.Empty, new EncodeSettings { ConsiderPreTokenization = considerPreTokenization, ConsiderNormalization = considerNormalization }).Tokens;

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, bool considerPreTokenization = true, bool considerNormalization = true)
             => EncodeToIds(null, text, new EncodeSettings { ConsiderPreTokenization = considerPreTokenization, ConsiderNormalization = considerNormalization }).Tokens;

        /// <summary>
        /// Encodes input text to token Ids up to maximum number of tokens.
        /// <param name="text">The text to encode.</param>
        /// </summary>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is <see langword="false"/>, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to <see langword="null"/>.</param>
        /// <param name="charsConsumed">The characters count of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(string text, int maxTokenCount, out string? normalizedText, out int charsConsumed, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            EncodeResults<int> result = EncodeToIds(text, ReadOnlySpan<char>.Empty,
                                                    new EncodeSettings
                                                    {
                                                        ConsiderPreTokenization = considerPreTokenization,
                                                        ConsiderNormalization = considerNormalization,
                                                        MaxTokenCount = maxTokenCount
                                                    });

            normalizedText = result.NormalizedText;
            charsConsumed = result.CharsConsumed;

            return result.Tokens;
        }

        /// <summary>
        /// Encodes input text to token Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is <see langword="false"/>, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to <see langword="null"/>.</param>
        /// <param name="charsConsumed">The characters count of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedText, out int charsConsumed, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            EncodeResults<int> result = EncodeToIds(null, text,
                                                    new EncodeSettings
                                                    {
                                                        ConsiderPreTokenization = considerPreTokenization,
                                                        ConsiderNormalization = considerNormalization,
                                                        MaxTokenCount = maxTokenCount
                                                    });

            normalizedText = result.NormalizedText;
            charsConsumed = result.CharsConsumed;

            return result.Tokens;
        }

        /// <summary>
        /// Encodes input text to a list of <see cref="EncodedToken" />s.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        protected abstract EncodeResults<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings);

        /// <summary>
        /// Encodes input text to a list of <see cref="EncodedToken" />s.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is <see langword="false"/>, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to <see langword="null"/>.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded <see cref="EncodedToken" />s.</returns>
        public IReadOnlyList<EncodedToken> EncodeToTokens(string text, out string? normalizedString, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            EncodeResults<EncodedToken> result = EncodeToTokens(text, ReadOnlySpan<char>.Empty, new EncodeSettings { ConsiderPreTokenization = considerPreTokenization, ConsiderNormalization = considerNormalization });

            normalizedString = result.NormalizedText;
            return result.Tokens;
        }

        /// <summary>
        /// Encodes input text to a list of <see cref="EncodedToken" />s.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is <see langword="false"/>, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to <see langword="null"/>.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded <see cref="EncodedToken" />s.</returns>
        public IReadOnlyList<EncodedToken> EncodeToTokens(ReadOnlySpan<char> text, out string? normalizedString, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            EncodeResults<EncodedToken> result = EncodeToTokens(null, text, new EncodeSettings { ConsiderPreTokenization = considerPreTokenization, ConsiderNormalization = considerNormalization });

            normalizedString = result.NormalizedText;
            return result.Tokens;
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textSpan">The span of the text to encode which will be used if the <paramref name="text"/> is <see langword="null"/>.</param>
        /// <param name="settings">The settings used to encode the text.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        protected abstract int CountTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        public int CountTokens(string text, bool considerPreTokenization = true, bool considerNormalization = true)
            => CountTokens(text, ReadOnlySpan<char>.Empty, new EncodeSettings { ConsiderPreTokenization = considerPreTokenization, ConsiderNormalization = considerNormalization });

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        public int CountTokens(ReadOnlySpan<char> text, bool considerPreTokenization = true, bool considerNormalization = true)
            => CountTokens(null, text, new EncodeSettings { ConsiderPreTokenization = considerPreTokenization, ConsiderNormalization = considerNormalization });

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
        protected abstract int GetIndexByTokenCount(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings, bool fromEnd, out string? normalizedString, out int tokenCount);

        /// <summary>
        /// Find the index of the maximum encoding capacity without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is <see langword="false"/>, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to <see langword="null"/>.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the input text or the <paramref name="normalizedString"/> if the normalization is enabled.
        /// </returns>
        public int GetIndexByTokenCount(string text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
            => GetIndexByTokenCount(
                text,
                ReadOnlySpan<char>.Empty,
                new EncodeSettings { ConsiderPreTokenization = considerPreTokenization, ConsiderNormalization = considerNormalization, MaxTokenCount = maxTokenCount },
                fromEnd: false,
                out normalizedString,
                out tokenCount);

        /// <summary>
        /// Find the index of the maximum encoding capacity without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerPreTokenization" /> is <see langword="false"/>, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to <see langword="null"/>.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the input text or the <paramref name="normalizedString"/> if the normalization is enabled.
        /// </returns>
        public int GetIndexByTokenCount(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
            => GetIndexByTokenCount(
                null,
                text,
                new EncodeSettings { ConsiderPreTokenization = considerPreTokenization, ConsiderNormalization = considerNormalization, MaxTokenCount = maxTokenCount },
                fromEnd: false,
                out normalizedString,
                out tokenCount);

        /// <summary>
        /// Find the index of the maximum encoding capacity without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerPreTokenization" /> is <see langword="false"/>, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to <see langword="null"/>.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index of the first character to be included. In cases where no tokens fit, the result will be the text length; conversely,
        /// if all tokens fit, the result will be zero.
        /// </returns>
        public int GetIndexByTokenCountFromEnd(string text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
            => GetIndexByTokenCount(
                text,
                ReadOnlySpan<char>.Empty,
                new EncodeSettings { ConsiderPreTokenization = considerPreTokenization, ConsiderNormalization = considerNormalization, MaxTokenCount = maxTokenCount },
                fromEnd: true,
                out normalizedString,
                out tokenCount);

        /// <summary>
        /// Find the index of the maximum encoding capacity without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerPreTokenization" /> is <see langword="false"/>, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to <see langword="null"/>.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index of the first character to be included. In cases where no tokens fit, the result will be the text length; conversely,
        /// if all tokens fit, the result will be zero.
        /// </returns>
        public int GetIndexByTokenCountFromEnd(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
            => GetIndexByTokenCount(
                null,
                text,
                new EncodeSettings { ConsiderPreTokenization = considerPreTokenization, ConsiderNormalization = considerNormalization, MaxTokenCount = maxTokenCount },
                fromEnd: true,
                out normalizedString,
                out tokenCount);

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <returns>The decoded string.</returns>
        public abstract string? Decode(IEnumerable<int> ids);

        /// <summary>
        /// Decode the given ids back to text and store the result in the <paramref name="destination"/> span.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="destination">The span to store the decoded text.</param>
        /// <param name="idsConsumed">The number of ids consumed during the decoding.</param>
        /// <param name="charsWritten">The number of characters written to the destination span.</param>
        /// <returns>The operation status indicates whether all IDs were successfully decoded or if the <paramref name="destination"/> is too small to contain the entire decoded result.</returns>
        public abstract OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, out int idsConsumed, out int charsWritten);

        internal static IEnumerable<(int Offset, int Length)>? InitializeForEncoding(
                                                string? text,
                                                ReadOnlySpan<char> textSpan,
                                                bool considerPreTokenization,
                                                bool considerNormalization,
                                                Normalizer? normalizer,
                                                PreTokenizer? preTokenizer,
                                                out string? normalizedString,
                                                out ReadOnlySpan<char> textSpanToEncode,
                                                out int fullTextLength)
        {
            normalizedString = null;
            IEnumerable<(int Offset, int Length)>? splits = null;

            if (text is null)
            {
                if (considerNormalization && (normalizer is not null))
                {
                    normalizedString = normalizer.Normalize(textSpan.ToString());
                    textSpanToEncode = normalizedString.AsSpan();
                    fullTextLength = normalizedString.Length;
                    if (considerPreTokenization && preTokenizer is not null)
                    {
                        splits = preTokenizer.PreTokenize(normalizedString);
                    }
                }
                else
                {
                    textSpanToEncode = textSpan;
                    fullTextLength = textSpan.Length;
                    if (considerPreTokenization && preTokenizer is not null)
                    {
                        splits = preTokenizer.PreTokenize(textSpan);
                    }
                }
            }
            else
            {
                if (considerNormalization && (normalizer is not null))
                {
                    normalizedString = normalizer.Normalize(text);
                    textSpanToEncode = normalizedString.AsSpan();
                    fullTextLength = normalizedString.Length;
                    if (considerPreTokenization && preTokenizer is not null)
                    {
                        splits = preTokenizer.PreTokenize(normalizedString);
                    }
                }
                else
                {
                    textSpanToEncode = text.AsSpan();
                    fullTextLength = text.Length;
                    if (considerPreTokenization && preTokenizer is not null)
                    {
                        splits = preTokenizer.PreTokenize(text);
                    }
                }
            }

            return splits;
        }
    }
}
