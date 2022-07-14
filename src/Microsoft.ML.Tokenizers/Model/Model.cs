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
        /// Map the token to tokenized Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <returns>The mapped Id of the token.</returns>
        public abstract int? TokenToId(string token);

        /// <summary>
        /// Map the tokenized Id to the token.
        /// </summary>
        /// <param name="id">The Id to map to the token.</param>
        /// <param name="skipSpecialTokens">Indicate if want to skip the special tokens during the decoding.</param>
        /// <returns>The mapped token of the Id.</returns>
        public abstract string? IdToToken(int id, bool skipSpecialTokens = false);

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
    }

}
