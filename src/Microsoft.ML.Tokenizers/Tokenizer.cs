// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// A Tokenizer works as a pipeline. It processes some raw text as input and outputs a EncodingResult object.
    /// </summary>
    public partial class Tokenizer
    {
        /// <summary>
        /// Create a new Tokenizer object.
        /// </summary>
        /// <param name="model">The Model in use by the Tokenizer.</param>
        /// <param name="preTokenizer">The optional PreTokenizer in use by the Tokenizer. WhiteSpace PreTokenizer will be used if this parameter is null.</param>
        /// <param name="normalizer">The optional Normalizer in use by the Tokenizer.</param>
        /// <param name="decoder">The optional Decoder in use by the Tokenizer during the decoding operation to merge the given list of tokens in a string.</param>
        public Tokenizer(Model model, PreTokenizer? preTokenizer = null, Normalizer? normalizer = null, TokenizerDecoder? decoder = null)
        {
            Model = model;
            PreTokenizer = preTokenizer ?? WhiteSpace.Instance;
            Normalizer = normalizer;
            Decoder = decoder;
        }

        /// <summary>
        /// Gets the Model in use by the Tokenizer.
        /// </summary>
        public Model Model { get; }

        /// <summary>
        /// Gets or sets the PreTokenizer used by the Tokenizer.
        /// </summary>
        public PreTokenizer PreTokenizer { get; }

        /// <summary>
        /// Gets or sets the Normalizer in use by the Tokenizer.
        /// </summary>
        public Normalizer? Normalizer { get; }

        /// <summary>
        /// Gets or sets the Decoder in use by the Tokenizer.
        /// </summary>
        public TokenizerDecoder? Decoder { get; }

        /// <summary>
        /// Encodes input text to object has the tokens list, tokens Ids, tokens offset mapping.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the encoding.</param>
        /// <returns>The tokenization result includes the tokens list, tokens Ids, tokens offset mapping.</returns>
        public EncodingResult Encode(string text, bool considerSpecialTokens = true)
        {
            if (text is null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            string normalized = Normalizer is null ? text : Normalizer.Normalize(text);
            bool offsetsMappedToOriginal = true;

            EncodingResult encoding = new(text, normalized, PreTokenizer.PreTokenize(normalized, considerSpecialTokens), offsetsMappedToOriginal);

            foreach (Split split in encoding.Splits)
            {
                IReadOnlyList<Token> tokens = Model.Encode(split.TokenString, split.IsSpecialToken);
                foreach (Token token in tokens)
                {
                    token.Offset = (token.Offset.Index + split.Offset.Index, token.Offset.Length);
                }

                encoding.AddTokens(tokens);
            }

            return encoding;
        }

        /// <summary>
        /// Encodes input text to tokens Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the encoding.</param>
        /// <returns>The tokenization result includes the tokens list, tokens Ids, tokens offset mapping.</returns>
        public IReadOnlyList<int> EncodeToIds(string text, bool considerSpecialTokens = true)
        {
            if (text is null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            string normalized = Normalizer is not null ? Normalizer.Normalize(text) : text;
            List<int> idsList = new();

            foreach (Split split in PreTokenizer.PreTokenize(normalized, considerSpecialTokens))
            {
                Model.EncodeToIds(split.TokenSpan, split.IsSpecialToken, idsList);
            }

            return idsList;
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the encoding.</param>
        /// <returns>The number of tokens Ids that the input text will be encoded to.</returns>
        /// <exception cref="ArgumentNullException">The input text is null.</exception>
        /// <exception cref="ArgumentException">Unable to encode the text.</exception>
        public int CountTokens(string text, bool considerSpecialTokens = true)
        {
            if (text is null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            string normalized = Normalizer is not null ? Normalizer.Normalize(text) : text;

            int idsCount = 0;
            foreach (Split split in PreTokenizer.PreTokenize(normalized, considerSpecialTokens))
            {
                idsCount += Model.CountTokens(split.TokenSpan, split.IsSpecialToken);
            }

            return idsCount;
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the start within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="processedText">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will remain unchanged as the input text.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the encoding.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely, if all tokens fit, the result will be length of the <paramref name="processedText"/>.
        /// </returns>
        /// <exception cref="ArgumentNullException">The input text is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">The maximum token count must be greater than 0.</exception>
        public int IndexOfTokenCount(string text, int maxTokenCount, out string processedText, out int tokenCount, bool considerSpecialTokens = true)
            => IndexOf(text, maxTokenCount, fromStart: true, considerSpecialTokens, out processedText, out tokenCount);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="processedText">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will remain unchanged as the input text.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the encoding.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the <paramref name="processedText"/>; conversely, if all tokens fit, the result will be 0.
        /// </returns>
        /// <exception cref="ArgumentNullException">The input text is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">The maximum token count must be greater than 0.</exception>
        /// <remarks>
        /// If the whole text can be encoded within the token limit, the returned index will be 0.
        /// </remarks>
        public int LastIndexOfTokenCount(string text, int maxTokenCount, out string processedText, out int tokenCount, bool considerSpecialTokens = true)
            => IndexOf(text, maxTokenCount, fromStart: false, considerSpecialTokens, out processedText, out tokenCount);

        private int IndexOf(string text, int maxTokenCount, bool fromStart, bool considerSpecialTokens, out string processedText, out int tokenCount)
        {
            if (text is null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The max token count must be greater than 0.");
            }

            processedText = Normalizer is not null ? Normalizer.Normalize(text) : text;
            tokenCount = 0;

            IEnumerable<Split> splits = PreTokenizer.PreTokenize(processedText, considerSpecialTokens);
            foreach (Split split in (fromStart ? splits : splits.Reverse()))
            {
                int count = Model.CountTokens(split.TokenSpan, split.IsSpecialToken);
                if (tokenCount > maxTokenCount - count)
                {
                    return fromStart ? split.Offset.Index : split.Offset.Index + split.Offset.Length;
                }

                tokenCount += count;
            }

            return fromStart ? processedText.Length : 0;
        }

        /// <summary>
        /// Decodes the Id to the mapped token.
        /// </summary>
        /// <param name="id">The id to map to the token.</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the decoding.</param>
        /// <returns>The decoded string or null if there is no token mapped to the input id.</returns>
        public string? Decode(int id, bool considerSpecialTokens = true) => Model.MapIdToToken(id, considerSpecialTokens);

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="considerSpecialTokens">Whether the special tokens should be kept in the decoded string.</param>
        /// <returns>The decoded string.</returns>
        public string? Decode(IEnumerable<int> ids, bool considerSpecialTokens = true) => Model.Decode(ids, Decoder, considerSpecialTokens);
    }
}
