// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// A Tokenizer works as a pipeline. It processes some raw text as input and outputs a EncodingResult object.
    /// </summary>
    public partial class Tokenizer
    {
        /// <summary>
        /// Create a new Tokenizer object.
        /// </summary>
        /// <param name="model">The Model in use by the Tokenizer.</param>
        /// <param name="preTokenizer">The optional PreTokenizer in use by the Tokenizer. WhiteSpace PreTokenizer will be used if this parameter is null.</param>
        /// <param name="normalizer">The optional Normalizer in use by the Tokenizer.</param>
        /// <param name="decoder">The optional Decoder in use by the Tokenizer during the decoding operation to merge the given list of tokens in a string.</param>
        public Tokenizer(Model model, PreTokenizer? preTokenizer = null, Normalizer? normalizer = null, TokenizerDecoder? decoder = null)
        {
            Model = model;
            PreTokenizer = preTokenizer ?? WhiteSpace.Instance;
            Normalizer = normalizer;
            Decoder = decoder;
        }

        /// <summary>
        /// Gets the Model in use by the Tokenizer.
        /// </summary>
        public Model Model { get; }

        /// <summary>
        /// Gets or sets the PreTokenizer used by the Tokenizer.
        /// </summary>
        public PreTokenizer PreTokenizer { get; }

        /// <summary>
        /// Gets or sets the Normalizer in use by the Tokenizer.
        /// </summary>
        public Normalizer? Normalizer { get; }

        /// <summary>
        /// Gets or sets the Decoder in use by the Tokenizer.
        /// </summary>
        public TokenizerDecoder? Decoder { get; }

        /// <summary>
        /// Encodes input text to object has the tokens list, tokens Ids, tokens offset mapping.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the encoding.</param>
        /// <returns>The tokenization result includes the tokens list, tokens Ids, tokens offset mapping.</returns>
        public EncodingResult Encode(string text, bool considerSpecialTokens = true)
        {
            if (text is null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            string normalized = Normalizer is null ? text : Normalizer.Normalize(text);
            bool offsetsMappedToOriginal = true;

            EncodingResult encoding = new(text, normalized, PreTokenizer.PreTokenize(normalized, considerSpecialTokens), offsetsMappedToOriginal);

            foreach (Split split in encoding.Splits)
            {
                IReadOnlyList<Token> tokens = Model.Encode(split.TokenString, split.IsSpecialToken);
                foreach (Token token in tokens)
                {
                    token.Offset = (token.Offset.Index + split.Offset.Index, token.Offset.Length);
                }

                encoding.AddTokens(tokens);
            }

            return encoding;
        }

        /// <summary>
        /// Encodes input text to tokens Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the encoding.</param>
        /// <returns>The tokenization result includes the tokens list, tokens Ids, tokens offset mapping.</returns>
        public IReadOnlyList<int> EncodeToIds(string text, bool considerSpecialTokens = true)
        {
            if (text is null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            string normalized = Normalizer is not null ? Normalizer.Normalize(text) : text;
            List<int> idsList = new();

            foreach (Split split in PreTokenizer.PreTokenize(normalized, considerSpecialTokens))
            {
                Model.EncodeToIds(split.TokenSpan, split.IsSpecialToken, idsList);
            }

            return idsList;
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the encoding.</param>
        /// <returns>The number of tokens Ids that the input text will be encoded to.</returns>
        /// <exception cref="ArgumentNullException">The input text is null.</exception>
        /// <exception cref="ArgumentException">Unable to encode the text.</exception>
        public int CountTokens(string text, bool considerSpecialTokens = true)
        {
            if (text is null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            string normalized = Normalizer is not null ? Normalizer.Normalize(text) : text;

            int idsCount = 0;
            foreach (Split split in PreTokenizer.PreTokenize(normalized, considerSpecialTokens))
            {
                idsCount += Model.CountTokens(split.TokenSpan, split.IsSpecialToken);
            }

            return idsCount;
        }

        /// <summary>
        /// Find the maximum encoding capacity from beginning within the input text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the encoding.</param>
        /// <returns>
        /// - The entire normalized text.
        /// - The length of text from the beginning of the normalized which is limited by the maximum token count.
        /// - The token count can be generated using the provided length which should be smaller than the maximum token count.
        /// </returns>
        /// <exception cref="ArgumentNullException">The input text is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">The maximum token count must be greater than 0.</exception>
        /// <remarks>
        /// If the tokenizer has a normalizer, the returned text will be the normalized text. Otherwise the returned text will be the input text.
        /// If the provided <paramref name="maxTokenCount"/> is greater than the token count of the input text, the returned length will be the length of the input text.
        /// If the provided <paramref name="maxTokenCount"/> is smaller enough to hold smallest number of grouped Ids, the returned length will be 0 and returned TokenCount will be 0.
        /// </remarks>
        public (string Text, int Length, int TokenCount) TrimSuffixWithinTokenLimit(string text, int maxTokenCount, bool considerSpecialTokens = true)
        {
            (string Text, int Offset, int Length, int TokenCount) result = TrimWithinTokenLimit(text, maxTokenCount, trimSuffix: true, considerSpecialTokens);
            return (result.Text, result.Length, result.TokenCount);
        }

        /// <summary>
        /// Find the maximum encoding capacity from the end within the input text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the encoding.</param>
        /// <returns>
        /// - The entire normalized text.
        /// - The starting offset within the returned normalized text for token counting.
        /// - The token count can be generated which should be smaller than the maximum token count.
        /// </returns>
        /// <exception cref="ArgumentNullException">The input text is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">The maximum token count must be greater than 0.</exception>
        /// <remarks>
        /// If the tokenizer has a normalizer, the returned text will be the normalized text. Otherwise the returned text will be the input text.
        /// If the provided <paramref name="maxTokenCount"/> is greater than the token count of the input text, the returned Offset will be 0.
        /// If the provided <paramref name="maxTokenCount"/> is smaller enough to hold smallest number of grouped Ids, the returned Offset will be equal to normalized text length and the returned TokenCount will be 0.
        /// </remarks>
        public (string Text, int Offset, int TokenCount) TrimPrefixWithinTokenLimit(string text, int maxTokenCount, bool considerSpecialTokens = true)
        {
            (string Text, int Offset, int Length, int TokenCount) result = TrimWithinTokenLimit(text, maxTokenCount, trimSuffix: false, considerSpecialTokens);
            return (result.Text, result.Offset, result.TokenCount);
        }

        private (string Text, int Offset, int Length, int TokenCount) TrimWithinTokenLimit(string text, int maxTokenCount, bool trimSuffix = true, bool considerSpecialTokens = true)
        {
            if (text is null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The max token count must be greater than 0.");
            }

            string normalized = Normalizer is not null ? Normalizer.Normalize(text) : text;
            int idsCount = 0;

            IEnumerable<Split> splits = PreTokenizer.PreTokenize(normalized, considerSpecialTokens);
            foreach (Split split in (trimSuffix ? splits : splits.Reverse()))
            {
                int tokenCount = Model.CountTokens(split.TokenSpan, split.IsSpecialToken);
                if (tokenCount > maxTokenCount - idsCount)
                {
                    return trimSuffix ?
                        (normalized, 0, split.Offset.Index, idsCount) :
                        (normalized, split.Offset.Index + split.Offset.Length, normalized.Length - split.Offset.Index - split.Offset.Length, idsCount);
                }

                idsCount += tokenCount;
            }

            return (normalized, 0, normalized.Length, idsCount);
        }

        /// <summary>
        /// Decodes the Id to the mapped token.
        /// </summary>
        /// <param name="id">The id to map to the token.</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the decoding.</param>
        /// <returns>The decoded string or null if there is no token mapped to the input id.</returns>
        public string? Decode(int id, bool considerSpecialTokens = true) => Model.MapIdToToken(id, considerSpecialTokens);

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="considerSpecialTokens">Whether the special tokens should be kept in the decoded string.</param>
        /// <returns>The decoded string.</returns>
        public string? Decode(IEnumerable<int> ids, bool considerSpecialTokens = true) => Model.Decode(ids, Decoder, considerSpecialTokens);

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

        private static readonly (string Prefix, ModelEncoding Encoding)[] _modelPrefixToEncoding =
                                                            [
                                                                // chat
                                                                ("gpt-4-", ModelEncoding.Cl100kBase),  // e.g., gpt-4-0314, etc., plus gpt-4-32k
                                                                ("gpt-3.5-turbo-", ModelEncoding.Cl100kBase) // e.g, gpt-3.5-turbo-0301, -0401, etc.
                                                            ];

        private static readonly Dictionary<string, ModelEncoding> _modelToEncoding =
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
                                                                // https://platform.openai.com/docs/guides/embeddings/what-are-embeddings
                                                                { "text-embedding-ada-002", ModelEncoding.Cl100kBase },
                                                                { "text-embedding-3-small", ModelEncoding.Cl100kBase },
                                                                { "text-embedding-3-large", ModelEncoding.Cl100kBase },

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

                                                                // open source
                                                                { "gpt2", ModelEncoding.GPT2 }
                                                            };

        private static ModelEncoding GetModelEncoding(string modelName)
        {
            if (!_modelToEncoding.TryGetValue(modelName, out ModelEncoding encoder))
            {
                foreach ((string Prefix, ModelEncoding Encoding) in _modelPrefixToEncoding)
                {
                    if (modelName.StartsWith(Prefix, StringComparison.OrdinalIgnoreCase))
                    {
                        encoder = Encoding;
                        break;
                    }
                }
            }

            if (encoder == ModelEncoding.None)
            {
                throw new NotSupportedException($"The model '{modelName}' is not supported.");
            }

            return encoder;
        }

        internal static (Dictionary<string, int> SpecialTokens, Regex Regex) GetTiktokenConfigurations(string modelName)
        {
            ModelEncoding modelEncoding = GetModelEncoding(modelName);

            switch (modelEncoding)
            {
                case ModelEncoding.Cl100kBase:
                    return (new Dictionary<string, int>
                        { { EndOfText, 100257}, { FimPrefix, 100258}, { FimMiddle, 100259}, { FimSuffix, 100260}, { EndOfPrompt, 100276} }, Cl100kBaseRegex());

                case ModelEncoding.P50kBase:
                    return (new Dictionary<string, int> { { EndOfText, 50256 } }, P50kBaseRegex());

                case ModelEncoding.P50kEdit:
                    return (new Dictionary<string, int>
                        { { EndOfText, 50256 }, { FimPrefix, 50281 }, { FimMiddle, 50282 }, { FimSuffix, 50283 } }, P50kBaseRegex());

                case ModelEncoding.R50kBase:
                    return (new Dictionary<string, int> { { EndOfText, 50256 } }, P50kBaseRegex());

                case ModelEncoding.GPT2:
                    return (new Dictionary<string, int> { { EndOfText, 50256 }, }, P50kBaseRegex());

                default:
                    Debug.Assert(false, $"Unexpected encoder [{modelEncoding}]");
                    throw new NotSupportedException($"The model '{modelName}' is not supported.");
            }
        }

        /// <summary>
        /// Create tokenizer based on model name
        /// </summary>
        /// <param name="modelName">Model name</param>
        /// <param name="extraSpecialTokens">Extra special tokens other than the built-in ones for the model</param>
        /// <param name="normalizer">To normalize the text before tokenization</param>
        /// <param name="cancellationToken"><see cref="CancellationToken"/> used to request cancellation of the operation.</param>
        /// <returns>The tokenizer</returns>
        public static Task<Tokenizer> CreateByModelNameAsync(
                                                string modelName,
                                                IReadOnlyDictionary<string, int>? extraSpecialTokens = null,
                                                Normalizer? normalizer = null,
                                                CancellationToken cancellationToken = default)
        {
            try
            {
                return CreateByEncoderNameAsync(modelName, GetModelEncoding(modelName), extraSpecialTokens, normalizer, cancellationToken);
            }
            catch (Exception ex)
            {
                return Task.FromException<Tokenizer>(ex);
            }
        }

        // Regex patterns based on https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py

        private const string Cl100kBaseRegexPattern = /*lang=regex*/ @"'(?i:[sdmt]|re|ve|ll)|(?>[^\r\n\p{L}\p{N}]?)\p{L}+|\p{N}{1,3}| ?(?>[^\s\p{L}\p{N}]+)[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
        private const string P50kBaseRegexPattern = /*lang=regex*/ @"'(?:[sdmt]|re|ve|ll)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

        private const string Cl100kBaseVocabUrl = @"https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken";
        private const string P50RanksUrl = @"https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken";
        private const string R50RanksUrl = @"https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken";
        private const string GPT2Url = @"https://pythia.blob.core.windows.net/public/encoding/gpt2.tiktoken";

#if NET7_0_OR_GREATER
        [GeneratedRegex(Cl100kBaseRegexPattern)]
        private static partial Regex Cl100kBaseRegex();

        [GeneratedRegex(P50kBaseRegexPattern)]
        internal static partial Regex P50kBaseRegex();
#else
        private static Regex? _cl100kBaseRegex;
        private static Regex Cl100kBaseRegex() => _cl100kBaseRegex ??= new Regex(Cl100kBaseRegexPattern, RegexOptions.Compiled);

        private static Regex? _p50kBaseRegex;
        internal static Regex P50kBaseRegex() => _p50kBaseRegex ??= new Regex(P50kBaseRegexPattern, RegexOptions.Compiled);
#endif

        /// <summary>
        /// Create tokenizer based on encoder name and extra special tokens
        /// </summary>
        /// <param name="modelName">Model name</param>
        /// <param name="modelEncoding">Encoder label</param>
        /// <param name="extraSpecialTokens">Extra special tokens other than the built-in ones for the encoder</param>
        /// <param name="normalizer">To normalize the text before tokenization</param>
        /// <param name="cancellationToken"><see cref="CancellationToken"/> used to request cancellation of the operation.</param>
        /// <returns>The tokenizer</returns>
        /// <exception cref="NotSupportedException">Throws if the model name is not supported</exception>
        private static Task<Tokenizer> CreateByEncoderNameAsync(
                                                string modelName,
                                                ModelEncoding modelEncoding,
                                                IReadOnlyDictionary<string, int>? extraSpecialTokens,
                                                Normalizer? normalizer,
                                                CancellationToken cancellationToken)
        {
            switch (modelEncoding)
            {
                case ModelEncoding.Cl100kBase:
                    var specialTokens = new Dictionary<string, int>
                        { { EndOfText, 100257}, { FimPrefix, 100258}, { FimMiddle, 100259}, { FimSuffix, 100260}, { EndOfPrompt, 100276} };
                    return CreateTikTokenTokenizerAsync(Cl100kBaseRegex(), Cl100kBaseVocabUrl, specialTokens, extraSpecialTokens, normalizer, cancellationToken);

                case ModelEncoding.P50kBase:
                    specialTokens = new Dictionary<string, int> { { EndOfText, 50256 } };
                    return CreateTikTokenTokenizerAsync(P50kBaseRegex(), P50RanksUrl, specialTokens, extraSpecialTokens, normalizer, cancellationToken);

                case ModelEncoding.P50kEdit:
                    specialTokens = new Dictionary<string, int>
                        { { EndOfText, 50256 }, { FimPrefix, 50281 }, { FimMiddle, 50282 }, { FimSuffix, 50283 } };
                    return CreateTikTokenTokenizerAsync(P50kBaseRegex(), P50RanksUrl, specialTokens, extraSpecialTokens, normalizer, cancellationToken);

                case ModelEncoding.R50kBase:
                    specialTokens = new Dictionary<string, int> { { EndOfText, 50256 } };
                    return CreateTikTokenTokenizerAsync(P50kBaseRegex(), R50RanksUrl, specialTokens, extraSpecialTokens, normalizer, cancellationToken);

                case ModelEncoding.GPT2:
                    specialTokens = new Dictionary<string, int> { { EndOfText, 50256 }, };
                    return CreateTikTokenTokenizerAsync(P50kBaseRegex(), GPT2Url, specialTokens, extraSpecialTokens, normalizer, cancellationToken);

                default:
                    Debug.Assert(false, $"Unexpected encoder [{modelEncoding}]");
                    throw new NotSupportedException($"The model '{modelName}' is not supported.");
            }
        }

        private static readonly ConcurrentDictionary<string, (Dictionary<ReadOnlyMemory<byte>, int> encoder, Dictionary<StringSpanOrdinalKey, int> vocab, Dictionary<int, ReadOnlyMemory<byte>> decoder)> _tiktokenCache = new(StringComparer.OrdinalIgnoreCase);

        /// <summary>
        /// Create tokenizer based on regex pattern, BPE rank file and special tokens
        /// </summary>
        /// <param name="regex">Regex to break a long string</param>
        /// <param name="mergeableRanksFileUrl">BPE rank file</param>
        /// <param name="specialTokens">Special tokens mapping. This may be mutated by the method.</param>
        /// <param name="extraSpecialTokens">Extra special tokens other than the built-in ones for the encoder</param>
        /// <param name="normalizer">To normalize the text before tokenization</param>
        /// <param name="cancellationToken"><see cref="CancellationToken"/> used to request cancellation of the operation.</param>
        /// <returns>The tokenizer</returns>
        private static async Task<Tokenizer> CreateTikTokenTokenizerAsync(
            Regex regex,
            string mergeableRanksFileUrl,
            Dictionary<string, int> specialTokens,
            IReadOnlyDictionary<string, int>? extraSpecialTokens,
            Normalizer? normalizer,
            CancellationToken cancellationToken)
        {
            if (extraSpecialTokens is not null)
            {
                foreach (var extraSpecialToken in extraSpecialTokens)
                {
                    specialTokens.Add(extraSpecialToken.Key, extraSpecialToken.Value);
                }
            }

            if (!_tiktokenCache.TryGetValue(mergeableRanksFileUrl, out (Dictionary<ReadOnlyMemory<byte>, int> encoder, Dictionary<StringSpanOrdinalKey, int> vocab, Dictionary<int, ReadOnlyMemory<byte>> decoder) cache))
            {
                using (Stream stream = await Helpers.GetStreamAsync(_httpClient, mergeableRanksFileUrl, cancellationToken).ConfigureAwait(false))
                {
                    cache = await Tiktoken.LoadTikTokenBpeAsync(stream, useAsync: true, cancellationToken).ConfigureAwait(false);
                }

                _tiktokenCache.TryAdd(mergeableRanksFileUrl, cache);
            }

            return new Tokenizer(new Tiktoken(cache.encoder, cache.decoder, cache.vocab, specialTokens), new TikTokenPreTokenizer(regex, specialTokens), normalizer);
        }
    }
}
