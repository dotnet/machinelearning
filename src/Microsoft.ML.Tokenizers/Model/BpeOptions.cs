// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

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
        /// <param name="vocabulary">The vocabulary to use.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="vocabulary"/> is null.</exception>
        public BpeOptions(IEnumerable<KeyValuePair<string, int>> vocabulary)
        {
            if (vocabulary == null)
            {
                throw new ArgumentNullException(nameof(vocabulary));
            }

            Vocabulary = vocabulary;
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="BpeOptions"/> class.
        /// </summary>
        /// <param name="vocabFile">The path to the vocabulary file.</param>
        /// <param name="mergesFile">The path to the merges file.</param>
        public BpeOptions(string vocabFile, string? mergesFile = null)
        {
            if (vocabFile is null)
            {
                throw new ArgumentNullException(nameof(vocabFile));
            }

            if (!File.Exists(vocabFile))
            {
                throw new ArgumentException($"Could not find the vocabulary file '{vocabFile}'.");
            }

            using Stream vocabStream = File.OpenRead(vocabFile);
            Dictionary<string, int>? dictionary = JsonSerializer.Deserialize<Dictionary<string, int>>(vocabStream);

            if (dictionary is null)
            {
                throw new InvalidOperationException($"The content of the vocabulary file '{vocabFile}' is not valid.");
            }

            Vocabulary = dictionary;

            if (mergesFile is not null)
            {
                if (!File.Exists(mergesFile))
                {
                    throw new ArgumentException($"Could not find the merges file '{mergesFile}'.");
                }

                using Stream mergesStream = File.OpenRead(mergesFile);
                using StreamReader reader = new(mergesStream);

                List<string> merges = new();

                int lineNumber = 0;
                string? line;

                while ((line = reader.ReadLine()) is not null)
                {
                    lineNumber++;
                    if (line.StartsWith("#version", StringComparison.Ordinal) || line.Length == 0)
                    {
                        continue;
                    }

                    // validate the merges format
                    int index = line.IndexOf(' ');
                    if (index < 0 || index == line.Length - 1 || line.IndexOf(' ', index + 1) >= 0)
                    {
                        throw new InvalidOperationException($"Invalid merge file format at line: {lineNumber}");
                    }

                    merges.Add(line);
                }

                Merges = merges;
            }
        }

        /// <summary>
        /// Gets or sets the vocabulary to use.
        /// </summary>
        public IEnumerable<KeyValuePair<string, int>> Vocabulary { get; }

        /// <summary>
        /// Gets or sets the list of the merge strings used to merge tokens during encoding.
        /// </summary>
        public IEnumerable<string>? Merges { get; set; }

        /// <summary>
        /// Gets or sets the optional special tokens to use.
        /// </summary>
        public IReadOnlyDictionary<string, int>? SpecialTokens { get; set; }

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
