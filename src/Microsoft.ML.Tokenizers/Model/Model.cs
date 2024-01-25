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
        /// Tokenize a sequence string to a list of tokens.
        /// </summary>
        /// <param name="sequence">The sequence to tokenize.</param>
        /// <returns>The list of tokens generated from the sequence tokenization.</returns>
        public abstract IReadOnlyList<Token> Tokenize(string sequence);

        /// <summary>
        /// Tokenize a split sequence string to a list of tokens.
        /// </summary>
        /// <param name="sequence">The text to tokenize.</param>
        /// <param name="isSpecialToken">Indicate if the token is a special token.</param>
        /// <returns>The list of tokens generated from the sequence tokenization.</returns>
        public virtual IReadOnlyList<Token> Tokenize(string sequence, bool isSpecialToken) => Tokenize(sequence);

        /// <summary>
        /// Tokenize a split sequence string to a list of Ids and add them to the accumulatedIds list.
        /// </summary>
        /// <param name="sequence">The sequence to split.</param>
        /// <param name="isSpecialToken">Indicate if the token is a special token.</param>
        /// <param name="accumulatedIds">The list of accumulated tokenized Ids.</param>
        /// <returns>True if the operation succeeded, false otherwise.</returns>
        public virtual bool TokenizeToIds(string sequence, bool isSpecialToken, List<int> accumulatedIds)
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
            return true;
        }

        /// <summary>
        /// Map the token to tokenized Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <returns>The mapped Id of the token.</returns>
        public abstract int? TokenToId(string token);

        /// <summary>
        /// Map the token to tokenized id with the option to skip the special tokens.
        /// </summary>
        /// <param name="token">The token to map to Id</param>
        /// <param name="skipSpecialTokens">Indicate if want to skip the special tokens during the encoding.</param>
        /// <returns>The mapped Id of the token.</returns>
        public virtual int? TokenToId(string token, bool skipSpecialTokens) => TokenToId(token);

        /// <summary>
        /// Map the tokenized Id to the token.
        /// </summary>
        /// <param name="id">The Id to map to the token.</param>
        /// <param name="skipSpecialTokens">Indicate if want to skip the special tokens during the decoding.</param>
        /// <returns>The mapped token of the Id.</returns>
        public abstract string? IdToToken(int id, bool skipSpecialTokens = false);

        public abstract string? IdToString(int id, bool skipSpecialTokens = false);

        /// <summary>
        /// Gets the dictionary mapping tokens to Ids.
        /// </summary>
        public abstract IReadOnlyDictionary<string, int> GetVocab();

        /// <summary>
        /// Gets the dictionary size that map tokens to Ids.
        /// </summary>
        public abstract int GetVocabSize();

        /// <summary>
        /// Save the model data into the vocabulary and merges files.
        /// </summary>
        /// <param name="path">The file system path to store the generated files at.</param>
        /// <param name="prefix">Optional prefix for the generated file names.</param>
        /// <returns>The list of all saved files.</returns>
        public abstract string[] Save(string path, string? prefix = null);

        /// <summary>
        /// Gets a trainer object to use in training the model.
        /// </summary>
        public abstract Trainer? GetTrainer();

        /// <summary>
        /// Return true if the char is valid in the tokenizer; otherwise return false.
        /// </summary>
        /// <param name="ch"></param>
        /// <returns></returns>
        public abstract bool IsValidChar(char ch);
    }
}
