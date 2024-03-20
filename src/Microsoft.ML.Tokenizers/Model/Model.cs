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
        /// Encode a text to a list of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>The list of tokens generated from the text tokenization.</returns>
        public abstract IReadOnlyList<Token> Encode(string text);

        /// <summary>
        /// Encode a text to a list of Ids and add them to the accumulatedIds list.
        /// </summary>
        /// <param name="text">The text to encode. </param>
        /// <param name="accumulatedIds">The list of accumulated encoded Ids.</param>
        /// <remarks>
        /// This method does the default implementation that uses the Encode method to get the token's Ids.
        /// Tokenizer's models which care about performance may choose to override this method to provide a more efficient implementation.
        /// </remarks>
        public virtual void EncodeToIds(ReadOnlySpan<char> text, IList<int> accumulatedIds)
        {
            if (accumulatedIds is null)
            {
                throw new ArgumentNullException(nameof(accumulatedIds));
            }

            // Default implementation is not optimized for memory allocation. It is recommended to override this method for the sake of the performance.
            var tokens = Encode(text.ToString());
            foreach (var token in tokens)
            {
                accumulatedIds.Add(token.Id);
            }
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        /// <remarks>
        /// This method does the default implementation that uses the EncodeToIds method to get the number of token's Ids.
        /// Tokenizer's models which care about performance may choose to override this method to provide a more efficient implementation.
        /// </remarks>
        public virtual int CountTokens(ReadOnlySpan<char> text)
        {
            var ids = new List<int>();
            EncodeToIds(text, ids);
            return ids.Count;
        }

        /// <summary>
        /// Map the token to encoded id with the option to skip the special tokens.
        /// </summary>
        /// <param name="token">The token to map to Id</param>
        /// <returns>The mapped Id of the token.</returns>
        public abstract int? MapTokenToId(ReadOnlySpan<char> token);

        /// <summary>
        /// Map the encoded Id to the token.
        /// </summary>
        /// <param name="id">The Id to map to the token.</param>
        /// <returns>The mapped token of the Id.</returns>
        public abstract string? MapIdToToken(int id);

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="decoder">The optional Decoder to merge the given list of tokens in a string.</param>
        /// <returns>The decoded string.</returns>
        public virtual string? Decode(IEnumerable<int> ids, TokenizerDecoder? decoder = null)
        {
            List<string> tokens = new List<string>();

            foreach (int id in ids)
            {
                if (MapIdToToken(id) is string s)
                {
                    tokens.Add(s);
                }
            }

            return decoder?.Decode(tokens) ?? string.Concat(tokens);
        }
    }
}
