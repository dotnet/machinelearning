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
        public abstract IReadOnlyList<Token> Encode(ReadOnlySpan<char> text);

        /// <summary>
        /// Encode a text to a list of Ids and add them to the accumulatedIds list.
        /// </summary>
        /// <param name="text">The text to encode. </param>
        /// <param name="accumulatedIds">The list of accumulated encoded Ids.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        /// <remarks>
        /// This method does the default implementation that uses the Encode method to get the token's Ids.
        /// Tokenizer's models which care about performance may choose to override this method to provide a more efficient implementation.
        /// </remarks>
        public virtual int EncodeToIds(ReadOnlySpan<char> text, IList<int> accumulatedIds, out int textLength, int maxTokens = int.MaxValue)
        {
            if (accumulatedIds is null)
            {
                throw new ArgumentNullException(nameof(accumulatedIds));
            }

            // Default implementation is not optimized for memory allocation. It is recommended to override this method for the sake of the performance.
            textLength = 0;
            var tokens = Encode(text);

            int count = Math.Min(tokens.Count, maxTokens);

            for (int i = 0; i < count; i++)
            {
                textLength += tokens[i].Offset.Length;
                accumulatedIds.Add(tokens[i].Id);
            }

            return count;
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        /// <remarks>
        /// This method does the default implementation that uses the EncodeToIds method to get the number of token's Ids.
        /// Tokenizer's models which care about performance may choose to override this method to provide a more efficient implementation.
        /// </remarks>
        public virtual int CountTokens(ReadOnlySpan<char> text, out int textLength, int maxTokens = int.MaxValue)
        {
            if (maxTokens <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokens), "The maximum number of tokens must be greater than 0.");
            }

            var ids = new List<int>();

            if (maxTokens == int.MaxValue)
            {
                EncodeToIds(text, ids, out _);
                textLength = text.Length;
                return ids.Count;
            }

            IReadOnlyList<Token> tokens = Encode(text);
            textLength = 0;
            int count = Math.Min(tokens.Count, maxTokens);
            for (int i = 0; i < count; i++)
            {
                textLength += tokens[i].Offset.Length;
            }

            return count;
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textIndex">Starting from this index to the end of the text will encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        /// <remarks>
        /// This method does the default implementation that uses the EncodeToIds method to get the number of token's Ids.
        /// Tokenizer's models which care about performance may choose to override this method to provide a more efficient implementation.
        /// </remarks>
        public virtual int CountTokensFromEnd(ReadOnlySpan<char> text, out int textIndex, int maxTokens = int.MaxValue)
        {
            if (maxTokens <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokens), "The maximum number of tokens must be greater than 0.");
            }

            var ids = new List<int>();

            if (maxTokens == int.MaxValue)
            {
                EncodeToIds(text, ids, out _);
                textIndex = 0;
                return ids.Count;
            }

            IReadOnlyList<Token> tokens = Encode(text);
            textIndex = text.Length;
            int count = Math.Min(tokens.Count, maxTokens);

            int tokensCount = tokens.Count;
            int end = tokensCount - count;
            for (int i = tokensCount - 1; i >= end; i--)
            {
                textIndex -= tokens[i].Offset.Length;
            }

            return count;
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
