// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Options for the WordPiece tokenizer.
    /// </summary>
    public class WordPieceOptions
    {
#pragma warning disable MSML_NoInstanceInitializers
        internal const int DefaultMaxInputCharsPerWord = 100;
        internal const string DefaultContinuingSubwordPrefix = "##";

        /// <summary>
        /// Gets or sets the <see cref="PreTokenizer"/> to override the default normalizer, if desired.
        /// </summary>
        public PreTokenizer? PreTokenizer { get; set; }

        /// <summary>
        /// Gets or sets the <see cref="Normalizer"/> to override the default normalizer, if desired.
        /// </summary>
        public Normalizer? Normalizer { get; set; }

        /// <summary>
        /// Gets or set the special tokens to use.
        /// </summary>
        public IReadOnlyDictionary<string, int>? SpecialTokens { get; set; }

        /// <summary>
        /// Gets or set the unknown token to use.
        /// </summary>
        public string UnknownToken { get; set; } = "[UNK]";

        /// <summary>
        /// Gets or set the prefix to use for sub-words that are not the first part of a word.
        /// </summary>
        public string ContinuingSubwordPrefix { get; set; } = DefaultContinuingSubwordPrefix;

        /// <summary>
        /// Gets or set the maximum number of characters to consider for a single word.
        /// </summary>
        public int MaxInputCharsPerWord { get; set; } = DefaultMaxInputCharsPerWord;
#pragma warning restore MSML_NoInstanceInitializers
    }
}