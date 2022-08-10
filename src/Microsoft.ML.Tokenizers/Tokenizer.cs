// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// A Tokenizer works as a pipeline. It processes some raw text as input and outputs a TokenizerResult object.
    /// </summary>
    public class Tokenizer
    {
        /// <summary>
        /// Create a new Tokenizer object.
        /// </summary>
        /// <param name="model">The Model in use by the Tokenizer.</param>
        /// <param name="preTokenizer">The optional PreTokenizer in use by the Tokenizer. WhiteSpace PreTokenizer will be used if this parameter is null.</param>
        /// <param name="normalizer">The optional Normalizer in use by the Tokenizer.</param>
        public Tokenizer(Model model, PreTokenizer? preTokenizer = null, Normalizer? normalizer = null)
        {
            Model = model;
            PreTokenizer = preTokenizer ?? WhiteSpace.Instance;
            Normalizer = normalizer;
        }

        /// <summary>
        /// Gets the Model in use by the Tokenizer.
        /// </summary>
        public Model Model { get; }

        /// <summary>
        /// Gets or sets the PreTokenizer used by the Tokenizer.
        /// </summary>
        public PreTokenizer PreTokenizer { get; set; }

        /// <summary>
        /// Gets or sets the Normalizer in use by the Tokenizer.
        /// </summary>
        public Normalizer? Normalizer { get; set; }

        /// <summary>
        /// Gets or sets the Decoder in use by the Tokenizer.
        /// </summary>
        public TokenizerDecoder? Decoder { get; set; }

        /// <summary>
        /// Encodes input text to object has the tokens list, tokens Ids, tokens offset mapping.
        /// </summary>
        /// <param name="sequence">The text to tokenize.</param>
        /// <returns>The tokenization result includes the tokens list, tokens Ids, tokens offset mapping.</returns>
        public TokenizerResult Encode(string sequence)
        {
            if (sequence is null)
            {
                throw new ArgumentNullException(nameof(sequence));
            }

            string normalized;
            NormalizedString normalizedString = default;

            bool offsetsMappedToOriginal = true;
            if (Normalizer is not null)
            {
                normalizedString = Normalizer.Normalize(sequence);
                normalized = normalizedString.Normalized;

                offsetsMappedToOriginal = normalizedString.CanMapToOriginal;
            }
            else
            {
                normalized = sequence;
            }

            TokenizerResult encoding = new(sequence, normalized, PreTokenizer.PreTokenize(normalized), offsetsMappedToOriginal);

            if (Normalizer is null || !normalizedString.CanMapToOriginal || normalizedString.IsOneToOneMapping)
            {
                // Optimize the case we don't have to map the offsets.
                foreach (Split split in encoding.Splits)
                {
                    IReadOnlyList<Token> tokens = Model.Tokenize(split.TokenString);
                    foreach (Token token in tokens)
                    {
                        token.Offset = (token.Offset.Index + split.Offset.Index, token.Offset.End + split.Offset.Index);
                    }

                    encoding.AddTokens(tokens);
                }
            }
            else
            {
                Debug.Assert(normalizedString.NormalizedToOriginalMapping is not null);

                foreach (Split split in encoding.Splits)
                {
                    IReadOnlyList<Token> tokens = Model.Tokenize(split.TokenString);
                    foreach (Token token in tokens)
                    {
                        int index = normalizedString.NormalizedToOriginalMapping![token.Offset.Index + split.Offset.Index];
                        int end = normalizedString.NormalizedToOriginalMapping![token.Offset.End + split.Offset.Index - 1] + 1;

                        Debug.Assert(index < end && end >= 0 && index >= 0);

                        token.Offset = (index, end);
                    }

                    encoding.AddTokens(tokens);
                }
            }

            return encoding;
        }

        // skipSpecialTokens is used in post processing we don't support yet. We are keeping it to allow using it when we support post processing.
        /// <summary>
        /// Decodes the Id to the mapped token.
        /// </summary>
        /// <param name="id">The id to map to the token.</param>
        /// <param name="skipSpecialTokens">Indicate if want to skip the special tokens during the decoding.</param>
        /// <returns>The decoded string or null if there is no token mapped to the input id.</returns>
        public string? Decode(int id, bool skipSpecialTokens = false) => Model.IdToToken(id, skipSpecialTokens);

        // skipSpecialTokens is used in post processing we don't support yet. We are keeping it to allow using it when we support post processing.
        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="skipSpecialTokens">Whether the special tokens should be removed from the decoded string.</param>
        /// <returns>The decoded string.</returns>
        public string? Decode(IEnumerable<int> ids, bool skipSpecialTokens = false)
        {
            List<string> tokens = new List<string>();

            foreach (int id in ids)
            {
                tokens.Add(Model.IdToToken(id) ?? "");
            }

            return Decoder?.Decode(tokens) ?? string.Join("", tokens);
        }

        /// <summary>
        /// Train the tokenizer model using input files.
        /// </summary>
        /// <param name="trainer">An optional trainer that should be used to train our Model.</param>
        /// <param name="progress">Optional progress callback to report the training progress.</param>
        /// <param name="files">A list of the files that we should use for training.</param>
        public void TrainFromFiles(
                        Trainer? trainer,
                        ReportProgress? progress,
                        params string[] files)
        {
            Trainer? t = trainer ?? Model.GetTrainer();
            if (t == null)
            {
                throw new ArgumentNullException(nameof(trainer));
            }

            foreach (var file in files)
            {
                string[] lines = File.ReadAllLines(file);
                progress?.Invoke(new Progress(ProgressState.Start, $"{file}", lines.Length));

                t.Feed(lines, (s) =>
                {
                    string current = Normalizer is null ? s : Normalizer.Normalize(s).Normalized;
                    IReadOnlyList<Split> splits = PreTokenizer.PreTokenize(current);
                    List<string> list = new(splits.Count);
                    foreach (Split split in splits)
                    {
                        list.Add(split.TokenString);
                    }

                    progress?.Invoke(new Progress(ProgressState.Increment, null, 1));

                    return list;
                });
                progress?.Invoke(new Progress(ProgressState.End, null, lines.Length));
            }

            IReadOnlyList<AddedToken>? addedTokens = t.Train(Model);

            // To Do: support added vocabulary in the tokenizer which will include this returned special_tokens.
            // self.add_special_tokens(&special_tokens);
        }
    }
}
