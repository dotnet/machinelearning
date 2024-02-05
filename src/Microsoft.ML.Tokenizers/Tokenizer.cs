// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

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
        public TokenizerResult Encode(string sequence) => Encode(sequence, skipSpecialTokens: false);

        /// <summary>
        /// Encodes input text to object has the tokens list, tokens Ids, tokens offset mapping.
        /// </summary>
        /// <param name="sequence">The text to tokenize.</param>
        /// <param name="skipSpecialTokens">Indicate if want to skip the special tokens during the encoding.</param>
        /// <returns>The tokenization result includes the tokens list, tokens Ids, tokens offset mapping.</returns>
        public TokenizerResult Encode(string sequence, bool skipSpecialTokens = false)
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

            TokenizerResult encoding = new(sequence, normalized, PreTokenizer.PreTokenize(normalized, skipSpecialTokens), offsetsMappedToOriginal);

            if (Normalizer is null || !normalizedString.CanMapToOriginal || normalizedString.IsOneToOneMapping)
            {
                // Optimize the case we don't have to map the offsets.
                foreach (Split split in encoding.Splits)
                {
                    IReadOnlyList<Token> tokens = Model.Tokenize(split.TokenString, split.IsSpecialToken);
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
                    IReadOnlyList<Token> tokens = Model.Tokenize(split.TokenString, split.IsSpecialToken);
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

        /// <summary>
        /// Encodes input text to tokens Ids.
        /// </summary>
        /// <param name="sequence">The text to tokenize.</param>
        /// <param name="skipSpecialTokens">Indicate if want to skip the special tokens during the encoding.</param>
        /// <returns>The tokenization result includes the tokens list, tokens Ids, tokens offset mapping.</returns>
        public IReadOnlyList<int> EncodeToIds(string sequence, bool skipSpecialTokens = false)
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

            List<int> idsList = new();

            foreach (Split split in PreTokenizer.PreTokenize(normalized, skipSpecialTokens))
            {
                if (!Model.TokenizeToIds(split.TokenString, split.IsSpecialToken, idsList))
                {
                    throw new ArgumentException($"Unable to tokenize the sequence: {split.TokenString}");
                }
            }

            return idsList;
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
            if (Model is Tiktoken tiktoken)
            {
                // Tiktoken does not ensure a one-to-one mapping between IDs and tokens. Consequently, decoding individual IDs into tokens is not supported;
                // instead, decoding all IDs must be done collectively.
                // Here is example of case that map one character to multiple Ids:
                // '⭐' U-2B50 is mapped to Ids [2928, 99834] in the Tiktoken model.
                // In other words, the character '⭐' has UTF-8 code point 0xE2, 0xAD, 0x90, Tiktoken will map 0xE2 to [2928] and 0xAD, 0x90 to [99834].
                return tiktoken.IdsToString(ids, skipSpecialTokens);
            }

            List<string> tokens = new List<string>();

            foreach (int id in ids)
            {
                if (Model.GetType() == typeof(EnglishRoberta))
                    tokens.Add(Model.IdToString(id) ?? "");
                else
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
                    IEnumerable<Split> splits = PreTokenizer.PreTokenize(current);

                    List<string> list = new();
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

        public bool IsValidChar(char ch)
        {
            return Model.IsValidChar(ch);
        }

        private const string EndOfText = "<|endoftext|>";
        private const string FimPrefix = "<|fim_prefix|>";
        private const string FimMiddle = "<|fim_middle|>";
        private const string FimSuffix = "<|fim_suffix|>";
        private const string EndOfPrompt = "<|endofprompt|>";

        private static readonly HttpClient _httpClient = new HttpClient();

        private enum ModelEncoding
        {
            None,
            Cl100kBase,
            P50kBase,
            P50kEdit,
            R50kBase,
            GPT2
        }

        private static readonly IReadOnlyDictionary<string, ModelEncoding> _modelPrefixToEncoding =
                                                            new Dictionary<string, ModelEncoding>()
                                                            {
                                                                // chat
                                                                { "gpt-4-", ModelEncoding.Cl100kBase },  // e.g., gpt-4-0314, etc., plus gpt-4-32k
                                                                { "gpt-3.5-turbo-", ModelEncoding.Cl100kBase } // e.g, gpt-3.5-turbo-0301, -0401, etc.
                                                            };

        private static readonly IReadOnlyDictionary<string, ModelEncoding> _modelToEncoding =
                                                            new Dictionary<string, ModelEncoding>(StringComparer.OrdinalIgnoreCase)
                                                            {
                                                                // chat
                                                                { "gpt-4", ModelEncoding.Cl100kBase },
                                                                { "gpt-3.5-turbo", ModelEncoding.Cl100kBase },

                                                                // text
                                                                { "text-davinci-003", ModelEncoding.P50kBase },
                                                                { "text-davinci-002", ModelEncoding.P50kBase },
                                                                { "text-davinci-001", ModelEncoding.R50kBase },
                                                                { "text-curie-001", ModelEncoding.R50kBase },
                                                                { "text-babbage-001", ModelEncoding.R50kBase },
                                                                { "text-ada-001", ModelEncoding.R50kBase },
                                                                { "davinci", ModelEncoding.R50kBase },
                                                                { "curie", ModelEncoding.R50kBase },
                                                                { "babbage", ModelEncoding.R50kBase },
                                                                { "ada", ModelEncoding.R50kBase },

                                                                // code
                                                                { "code-davinci-002", ModelEncoding.P50kBase },
                                                                { "code-davinci-001", ModelEncoding.P50kBase },
                                                                { "code-cushman-002", ModelEncoding.P50kBase },
                                                                { "code-cushman-001", ModelEncoding.P50kBase },
                                                                { "davinci-codex", ModelEncoding.P50kBase },
                                                                { "cushman-codex", ModelEncoding.P50kBase },

                                                                // edit
                                                                { "text-davinci-edit-001", ModelEncoding.P50kEdit },
                                                                { "code-davinci-edit-001", ModelEncoding.P50kEdit },

                                                                // embeddings
                                                                { "text-embedding-ada-002", ModelEncoding.Cl100kBase },

                                                                // old embeddings
                                                                { "text-similarity-davinci-001", ModelEncoding.R50kBase },
                                                                { "text-similarity-curie-001", ModelEncoding.R50kBase },
                                                                { "text-similarity-babbage-001", ModelEncoding.R50kBase },
                                                                { "text-similarity-ada-001", ModelEncoding.R50kBase },
                                                                { "text-search-davinci-doc-001", ModelEncoding.R50kBase },
                                                                { "text-search-curie-doc-001", ModelEncoding.R50kBase },
                                                                { "text-search-babbage-doc-001", ModelEncoding.R50kBase },
                                                                { "text-search-ada-doc-001", ModelEncoding.R50kBase },
                                                                { "code-search-babbage-code-001", ModelEncoding.R50kBase },
                                                                { "code-search-ada-code-001", ModelEncoding.R50kBase },

                                                                //open source
                                                                { "gpt2", ModelEncoding.GPT2 }
                                                            };


        /// <summary>
        /// Create tokenizer based on model name
        /// </summary>
        /// <param name="modelName">Model name</param>
        /// <param name="extraSpecialTokens">Extra special tokens other than the built-in ones for the model</param>
        /// <param name="normalizer">To normalize the text before tokenization</param>
        /// <returns>The tokenizer</returns>
        public static async Task<Tokenizer> CreateByModelNameAsync(
                                                string modelName,
                                                IReadOnlyDictionary<string, int>? extraSpecialTokens = null,
                                                Normalizer? normalizer = null)
        {
            var encoder = ModelEncoding.None;

            if (!_modelToEncoding.TryGetValue(modelName, out encoder))
            {
                foreach (KeyValuePair<string, ModelEncoding> kvp in _modelPrefixToEncoding)
                {
                    if (modelName.StartsWith(kvp.Key, StringComparison.OrdinalIgnoreCase))
                    {
                        encoder = kvp.Value;
                        break;
                    }
                }
            }

            if (encoder == ModelEncoding.None)
            {
                throw new NotImplementedException($"Doesn't support this model [{modelName}]");
            }

            return await CreateByEncoderNameAsync(encoder, extraSpecialTokens, normalizer);
        }

        private const string Cl100kBaseRegexPattern = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        private const string P50kBaseRegexPattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

        const string Cl100kBaseVocabUrl = @"https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken";
        const string P50RegexUrl = @"https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken";
        const string R50RegexUrl = @"https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken";
        const string GPT2Url = @"https://pythia.blob.core.windows.net/public/encoding/gpt2.tiktoken";

        /// <summary>
        /// Create tokenizer based on encoder name and extra special tokens
        /// </summary>
        /// <param name="modelEncoding">Encoder label</param>
        /// <param name="extraSpecialTokens">Extra special tokens other than the built-in ones for the encoder</param>
        /// <param name="normalizer">To normalize the text before tokenization</param>
        /// <returns>The tokenizer</returns>
        /// <exception cref="NotImplementedException">Throws if the encoder is not supported</exception>
        private static async Task<Tokenizer> CreateByEncoderNameAsync(
                                                ModelEncoding modelEncoding,
                                                IReadOnlyDictionary<string, int>? extraSpecialTokens,
                                                Normalizer? normalizer)
        {
            switch (modelEncoding)
            {
                case ModelEncoding.Cl100kBase:
                    var specialTokens = new Dictionary<string, int>
                        { { EndOfText, 100257}, { FimPrefix, 100258}, { FimMiddle, 100259}, { FimSuffix, 100260}, { EndOfPrompt, 100276} };
                    return await CreateTikTokenTokenizerAsync(Cl100kBaseRegexPattern, Cl100kBaseVocabUrl, specialTokens, extraSpecialTokens, normalizer);

                case ModelEncoding.P50kBase:
                    specialTokens = new Dictionary<string, int> { { EndOfText, 50256 } };
                    return await CreateTikTokenTokenizerAsync(P50kBaseRegexPattern, P50RegexUrl, specialTokens, extraSpecialTokens, normalizer);

                case ModelEncoding.P50kEdit:
                    specialTokens = new Dictionary<string, int>
                        { { EndOfText, 50256 }, { FimPrefix, 50281 }, { FimMiddle, 50282 }, { FimSuffix, 50283 } };
                    return await CreateTikTokenTokenizerAsync(P50kBaseRegexPattern, P50RegexUrl, specialTokens, extraSpecialTokens, normalizer);

                case ModelEncoding.R50kBase:
                    specialTokens = new Dictionary<string, int> { { EndOfText, 50256 } };
                    return await CreateTikTokenTokenizerAsync(P50kBaseRegexPattern, R50RegexUrl, specialTokens, extraSpecialTokens, normalizer);

                case ModelEncoding.GPT2:
                    specialTokens = new Dictionary<string, int> { { EndOfText, 50256 }, };
                    return await CreateTikTokenTokenizerAsync(P50kBaseRegexPattern, GPT2Url, specialTokens, extraSpecialTokens, normalizer);

                default:
                    Debug.Assert(false, $"Unexpected encoder [{modelEncoding}]");
                    throw new NotImplementedException($"Doesn't support this encoder [{modelEncoding}]");
            }
        }

        private static readonly Dictionary<string, (Dictionary<byte[], int>, Dictionary<string, int>, IReadOnlyDictionary<int, byte[]>)> _tiktokenCache = new(StringComparer.OrdinalIgnoreCase);

        /// <summary>
        /// Create tokenizer based on regex pattern, BPE rank file and special tokens
        /// </summary>
        /// <param name="regexPatternStr">Regex pattern to break a long string</param>
        /// <param name="mergeableRanksFileUrl">BPE rank file</param>
        /// <param name="specialTokens">Special tokens mapping</param>
        /// <param name="extraSpecialTokens">Extra special tokens other than the built-in ones for the encoder</param>
        /// <param name="normalizer">To normalize the text before tokenization</param>
        /// <returns>The tokenizer</returns>
        private static async Task<Tokenizer> CreateTikTokenTokenizerAsync(
                                                string regexPatternStr,
                                                string mergeableRanksFileUrl,
                                                Dictionary<string, int> specialTokens,
                                                IReadOnlyDictionary<string, int>? extraSpecialTokens,
                                                Normalizer? normalizer)
        {
            if (extraSpecialTokens is not null)
            {
                specialTokens = specialTokens.Concat(extraSpecialTokens).ToDictionary(pair => pair.Key, pair => pair.Value);
            }

            if (!_tiktokenCache.TryGetValue(mergeableRanksFileUrl, out (Dictionary<byte[], int> encoder, Dictionary<string, int> vocab, IReadOnlyDictionary<int, byte[]> decoder) cache))
            {
                using (Stream stream = await _httpClient.GetStreamAsync(mergeableRanksFileUrl))
                {
                    cache = Tiktoken.LoadTikTokenBpe(stream);
                }
                _tiktokenCache.Add(mergeableRanksFileUrl, cache);
            }

            return new Tokenizer(new Tiktoken(cache.encoder, cache.decoder, cache.vocab, specialTokens), new TikTokenPreTokenizer(new Regex(regexPatternStr, RegexOptions.Compiled), specialTokens), normalizer);
        }
    }
}
