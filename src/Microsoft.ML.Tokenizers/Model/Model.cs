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
        /// Tokenize a split sequence string to a list of tokens.
        /// </summary>
        /// <param name="sequence">The text to tokenize.</param>
        /// <param name="isSpecialToken">Indicate if the token is a special token.</param>
        /// <returns>The list of tokens generated from the sequence tokenization.</returns>
        public abstract IReadOnlyList<Token> Tokenize(string sequence, bool isSpecialToken = false);

        /// <summary>
        /// Tokenize a split sequence string to a list of Ids and add them to the accumulatedIds list.
        /// </summary>
        /// <param name="sequence">The sequence to split.</param>
        /// <param name="isSpecialToken">Indicate if the token is a special token.</param>
        /// <param name="accumulatedIds">The list of accumulated tokenized Ids.</param>
        /// <remarks>
        /// This method does the default implementation that uses the Tokenize method to get the token's Ids.
        /// Tokenizer's models which care about performance may choose to override this method to provide a more efficient implementation.
        /// </remarks>
        public virtual void TokenizeToIds(string sequence, bool isSpecialToken, IList<int> accumulatedIds)
        {
            if (accumulatedIds is null)
            {
                throw new ArgumentNullException(nameof(accumulatedIds));
            }

            var tokens = Tokenize(sequence);
            foreach (var token in tokens)
            {
                accumulatedIds.Add(token.Id);
            }
        }

        /// <summary>
        /// Get the number of tokens that the input sequence will be encoded to.
        /// </summary>
        /// <param name="sequence">The text to tokenize.</param>
        /// <param name="isSpecialToken">Indicate if the token is special token.</param>
        /// <returns>The number of tokens that the input sequence will be encoded to.</returns>
        /// <remarks>
        /// This method does the default implementation that uses the TokenizeToIds method to get the number of token's Ids.
        /// Tokenizer's models which care about performance may choose to override this method to provide a more efficient implementation.
        /// </remarks>
        public virtual int CountTokens(string sequence, bool isSpecialToken)
        {
            var ids = new List<int>();
            TokenizeToIds(sequence, isSpecialToken, ids);
            return ids.Count;
        }

        /// <summary>
        /// Map the token to tokenized id with the option to skip the special tokens.
        /// </summary>
        /// <param name="token">The token to map to Id</param>
        /// <param name="skipSpecialTokens">Indicate if want to skip the special tokens during the encoding.</param>
        /// <returns>The mapped Id of the token.</returns>
        public abstract int? TokenToId(string token, bool skipSpecialTokens = false);

        /// <summary>
        /// Map the tokenized Id to the token.
        /// </summary>
        /// <param name="id">The Id to map to the token.</param>
        /// <param name="skipSpecialTokens">Indicate if want to skip the special tokens during the decoding.</param>
        /// <param name="filterUnsupportedChars">Indicate if want to filter the unsupported characters during the decoding.</param>
        /// <returns>The mapped token of the Id.</returns>
        public abstract string? IdToToken(int id, bool skipSpecialTokens = false, bool filterUnsupportedChars = true);

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="skipSpecialTokens">Whether the special tokens should be removed from the decoded string.</param>
        /// <param name="filterUnsupportedChars">Indicate if want to filter the unsupported characters during the decoding.</param>
        /// <param name="decoder">The optional Decoder to merge the given list of tokens in a string.</param>
        /// <returns>The decoded string.</returns>
        public virtual string? Decode(IEnumerable<int> ids, TokenizerDecoder? decoder = null, bool skipSpecialTokens = false, bool filterUnsupportedChars = true)
        {
            List<string> tokens = new List<string>();

            foreach (int id in ids)
            {
                tokens.Add(IdToToken(id, skipSpecialTokens, filterUnsupportedChars) ?? "");
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
