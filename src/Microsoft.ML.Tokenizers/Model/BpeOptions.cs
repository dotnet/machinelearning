// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Options for the BPE tokenizer.
    /// </summary>
    public sealed class BpeOptions
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="BpeOptions"/> class.
        /// </summary>
        public BpeOptions(IEnumerable<(string Token, int Id)> vocabulary)
        {
            if (vocabulary == null)
            {
                throw new ArgumentNullException(nameof(vocabulary));
            }

            Vocabulary = vocabulary;
        }

        /// <summary>
        /// Gets or sets the vocabulary to use.
        /// </summary>
        public IEnumerable<(string Token, int Id)> Vocabulary { get; }

        /// <summary>
        /// Gets or sets the list of the merge strings used to merge tokens during encoding.
        /// </summary>
        public IEnumerable<string>? Merges { get; set; }

        /// <summary>
        /// Gets or sets the optional special tokens to use.
        /// </summary>
        public Dictionary<string, int>? SpecialTokens { get; set; }

        /// <summary>
        /// Gets or sets the optional normalizer to normalize the input text before encoding it.
        /// </summary>
        public Normalizer? Normalizer { get; set; }

        /// <summary>
        /// Gets or sets the optional pre-tokenizer to split the input text into tokens before encoding it.
        /// </summary>
        public PreTokenizer? PreTokenizer { get; set; }

        /// <summary>
        /// Gets or sets the Unknown token.
        /// </summary>
        public string? UnknownToken { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to merge the sequence of the unknown tokens together.
        /// </summary>
        public bool FuseUnknownTokens { get; set; }

        /// <summary>
        /// Gets or sets the optional prefix to be used for every subword that is not a beginning-of-word token
        /// </summary>
        public string? ContinuingSubwordPrefix { get; set; }

        /// <summary>
        /// Gets or sets the optional suffix to characterize the end-of-word and sub-word
        /// </summary>
        public string? EndOfWordSuffix { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to handle the input text in byte level.
        /// if true, the input text will be converted to UTF-8 bytes before encoding it.
        /// Additionally, some ASCII characters will be transformed to different characters (e.g Space character will be transformed to 'Ġ' character).
        /// </summary>
        public bool ByteLevel { get; set; }

        /// <summary>
        /// Gets or sets the optional beginning of sentence token to be used when encoding the input text.
        /// </summary>
        /// <remarks>
        /// When specified, this token will be added to the beginning of the input text before encoding it.
        /// This is useful for models that require a specific token to indicate the start of a sentence.
        /// This token should be present in the vocabulary.
        /// </remarks>
        public string? BeginningOfSentenceToken { get; set; }

        /// <summary>
        /// Gets or sets the optional end of sentence token to be used when encoding the input text.
        /// </summary>
        /// <remarks>
        /// When specified, this token will be added to the end of the input text before encoding it.
        /// This is useful for models that require a specific token to indicate the end of a sentence.
        /// This token should be present in the vocabulary.
        /// </remarks>
        public string? EndOfSentenceToken { get; set; }
    }
}
