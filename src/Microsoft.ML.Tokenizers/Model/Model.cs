// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Represents a model used during Tokenization (like BPE or Word Piece or Unigram).
    /// </summary>
    public abstract class Model
    {
        /// <summary>
        /// Encode a split text string to a list of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="isSpecialToken">Indicate if the token is a special token.</param>
        /// <returns>The list of tokens generated from the text tokenization.</returns>
        public abstract IReadOnlyList<Token> Encode(string text, bool isSpecialToken = false);

        /// <summary>
        /// Encode a split text string to a list of Ids and add them to the accumulatedIds list.
        /// </summary>
        /// <param name="text">The text to split.</param>
        /// <param name="isSpecialToken">Indicate if the token is a special token.</param>
        /// <param name="accumulatedIds">The list of accumulated encoded Ids.</param>
        /// <remarks>
        /// This method does the default implementation that uses the Encode method to get the token's Ids.
        /// Tokenizer's models which care about performance may choose to override this method to provide a more efficient implementation.
        /// </remarks>
        public virtual void EncodeToIds(string text, bool isSpecialToken, IList<int> accumulatedIds)
        {
            if (accumulatedIds is null)
            {
                throw new ArgumentNullException(nameof(accumulatedIds));
            }

            var tokens = Encode(text);
            foreach (var token in tokens)
            {
                accumulatedIds.Add(token.Id);
            }
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="isSpecialToken">Indicate if the token is special token.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        /// <remarks>
        /// This method does the default implementation that uses the EncodeToIds method to get the number of token's Ids.
        /// Tokenizer's models which care about performance may choose to override this method to provide a more efficient implementation.
        /// </remarks>
        public virtual int CountTokens(string text, bool isSpecialToken)
        {
            var ids = new List<int>();
            EncodeToIds(text, isSpecialToken, ids);
            return ids.Count;
        }

        /// <summary>
        /// Map the token to encoded id with the option to skip the special tokens.
        /// </summary>
        /// <param name="token">The token to map to Id</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the encoding.</param>
        /// <returns>The mapped Id of the token.</returns>
        public abstract int? MapTokenToId(string token, bool considerSpecialTokens = true);

        /// <summary>
        /// Map the encoded Id to the token.
        /// </summary>
        /// <param name="id">The Id to map to the token.</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the decoding.</param>
        /// <param name="filterUnsupportedChars">Indicate if want to filter the unsupported characters during the decoding.</param>
        /// <returns>The mapped token of the Id.</returns>
        public abstract string? MapIdToToken(int id, bool considerSpecialTokens = true, bool filterUnsupportedChars = true);

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="considerSpecialTokens">Whether the special tokens should be kept in the decoded string.</param>
        /// <param name="filterUnsupportedChars">Indicate if want to filter the unsupported characters during the decoding.</param>
        /// <param name="decoder">The optional Decoder to merge the given list of tokens in a string.</param>
        /// <returns>The decoded string.</returns>
        public virtual string? Decode(IEnumerable<int> ids, TokenizerDecoder? decoder = null, bool considerSpecialTokens = true, bool filterUnsupportedChars = true)
        {
            List<string> tokens = new List<string>();

            foreach (int id in ids)
            {
                tokens.Add(MapIdToToken(id, considerSpecialTokens, filterUnsupportedChars) ?? "");
            }

            return decoder?.Decode(tokens) ?? string.Join("", tokens);
        }

        /// <summary>
        /// Gets the dictionary mapping tokens to Ids.
        /// </summary>
        public abstract IReadOnlyDictionary<string, int> GetVocab();

        /// <summary>
        /// Gets the dictionary size that map tokens to Ids.
        /// </summary>
        public abstract int GetVocabSize();
    }
}
