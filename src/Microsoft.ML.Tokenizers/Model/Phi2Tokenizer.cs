// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Represent the Byte Pair Encoding model.
    /// Implement the Phi2 tokenizer described in https://huggingface.co/microsoft/phi-2
    /// </summary>
    public sealed class Phi2Tokenizer : CodeGenTokenizer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Phi2Tokenizer"/> class.
        /// </summary>
        /// <summary>
        /// Construct tokenizer's model object to use with the English Robert model.
        /// </summary>
        /// <param name="vocabularyPath">The JSON file path containing the dictionary of string keys and their ids.</param>
        /// <param name="mergePath">The file path containing the tokens's pairs list.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="addedTokens">The additional tokens to add to the vocabulary.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="unknownToken">The unknown token.</param>
        /// <param name="beginningOfSentenceToken">The beginning of sentence token.</param>
        /// <param name="endOfSentenceToken">The end of sentence token.</param>
        internal Phi2Tokenizer(
                string vocabularyPath,
                string mergePath,
                PreTokenizer? preTokenizer = null,
                Normalizer? normalizer = null,
                IReadOnlyDictionary<string, int>? addedTokens = null,
                bool addPrefixSpace = false,
                bool addBeginningOfSentence = false,
                bool addEndOfSentence = false,
                string? unknownToken = DefaultSpecialToken,
                string? beginningOfSentenceToken = DefaultSpecialToken,
                string? endOfSentenceToken = DefaultSpecialToken) :
                    base(vocabularyPath, mergePath, preTokenizer, normalizer, addedTokens, addPrefixSpace, addBeginningOfSentence,
                    addEndOfSentence, unknownToken, beginningOfSentenceToken, endOfSentenceToken)
        {
        }

        /// <summary>
        /// Construct tokenizer's model object to use with the English Robert model.
        /// </summary>
        /// <param name="vocabularyStream">The stream of a JSON file containing the dictionary of string keys and their ids.</param>
        /// <param name="mergeStream">The stream of a file containing the tokens's pairs list.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="addedTokens">The additional tokens to add to the vocabulary.</param>
        /// <param name="addPrefixSpace">Indicate whether to include a leading space before encoding the text.</param>
        /// <param name="addBeginningOfSentence">Indicate whether to include the beginning of sentence token in the encoding.</param>
        /// <param name="addEndOfSentence">Indicate whether to include the end of sentence token in the encoding.</param>
        /// <param name="unknownToken">The unknown token.</param>
        /// <param name="beginningOfSentenceToken">The beginning of sentence token.</param>
        /// <param name="endOfSentenceToken">The end of sentence token.</param>
        internal Phi2Tokenizer(
                Stream vocabularyStream,
                Stream mergeStream,
                PreTokenizer? preTokenizer = null,
                Normalizer? normalizer = null,
                IReadOnlyDictionary<string, int>? addedTokens = null,
                bool addPrefixSpace = false,
                bool addBeginningOfSentence = false,
                bool addEndOfSentence = false,
                string? unknownToken = DefaultSpecialToken,
                string? beginningOfSentenceToken = DefaultSpecialToken,
                string? endOfSentenceToken = DefaultSpecialToken) :
                base(vocabularyStream, mergeStream, preTokenizer, normalizer, addedTokens, addPrefixSpace, addBeginningOfSentence,
                    addEndOfSentence, unknownToken, beginningOfSentenceToken, endOfSentenceToken)
        {
        }

        /// <summary>
        /// Create a CodeGen Phi2 tokenizer from the given vocab and merges streams.
        /// </summary>
        /// <param name="vocabStream">The stream containing the vocab file.</param>
        /// <param name="mergesStream">The stream containing the merges file.</param>
        /// <param name="addPrefixSpace">Indicate whether to add a space before the token.</param>
        /// <param name="addBeginOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <returns>The CodeGen tokenizer object.</returns>
        /// <remarks>
        /// The tokenizer will be created according to the configuration specified in https://huggingface.co/microsoft/phi-2/raw/main/tokenizer.json.
        /// It is important to provide the similar vocab and merges files to the ones used in the training of the model.
        /// The vocab and merges files can be downloaded from the following links:
        ///     https://huggingface.co/microsoft/phi-2/resolve/main/vocab.json?download=true
        ///     https://huggingface.co/microsoft/phi-2/resolve/main/merges.txt?download=true
        /// </remarks>
        public static new Phi2Tokenizer Create(
            Stream vocabStream,
            Stream mergesStream,
            bool addPrefixSpace = false,
            bool addBeginOfSentence = false,
            bool addEndOfSentence = false)
        {
            if (vocabStream is null)
            {
                throw new ArgumentNullException(nameof(vocabStream));
            }

            if (mergesStream is null)
            {
                throw new ArgumentNullException(nameof(mergesStream));
            }

            return new Phi2Tokenizer(
                        vocabStream, mergesStream, new RegexPreTokenizer(TiktokenTokenizer.P50kBaseRegex(), CodeGenTokenizer.CodeGenAddedTokens), normalizer: null,
                        CodeGenTokenizer.CodeGenAddedTokens, addPrefixSpace: addPrefixSpace, addBeginningOfSentence: addBeginOfSentence, addEndOfSentence: addEndOfSentence);
        }
    }
}
