// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Sentencepiece;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
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
        /// <returns>The tokenization result includes the tokens list, tokens Ids, tokens offset mapping.</returns>
        public EncodingResult Encode(string text)
        {
            if (text is null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            string normalized = Normalizer is null ? text : Normalizer.Normalize(text);
            bool offsetsMappedToOriginal = true;

            EncodingResult encoding = new(text, normalized, PreTokenizer.PreTokenize(normalized), offsetsMappedToOriginal);

            foreach (Split split in encoding.Splits)
            {
                IReadOnlyList<Token> tokens = Model.Encode(split.TokenString.AsSpan());
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
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(string text)
        {
            if (text is null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            string normalized = Normalizer is not null ? Normalizer.Normalize(text) : text;
            List<int> idsList = new();

            foreach (Split split in PreTokenizer.PreTokenize(normalized))
            {
                Model.EncodeToIds(split.TokenSpan, idsList, out _);
            }

            return idsList;
        }

        /// <summary>
        /// Encodes input text to tokens Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="processedText">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will remain unchanged as the input text.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <returns>The list of encoded Ids.</returns>
        public IReadOnlyList<int> EncodeToIds(string text, int maxTokenCount, out string processedText, out int textLength)
        {
            processedText = text;
            textLength = 0;

            if (text is null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than 0.");
            }

            if (Normalizer is not null)
            {
                processedText = Normalizer.Normalize(text);
            }

            List<int> idsList = new();

            foreach (Split split in PreTokenizer.PreTokenize(processedText))
            {
                Model.EncodeToIds(split.TokenSpan, idsList, out int length, maxTokenCount - idsList.Count);
                if (length < split.Offset.Length || idsList.Count >= maxTokenCount)
                {
                    break;
                }
            }

            return idsList;
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>The number of tokens Ids that the input text will be encoded to.</returns>
        /// <exception cref="ArgumentNullException">The input text is null.</exception>
        /// <exception cref="ArgumentException">Unable to encode the text.</exception>
        public int CountTokens(string text)
        {
            if (text is null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            string normalized = Normalizer is not null ? Normalizer.Normalize(text) : text;

            int idsCount = 0;
            foreach (Split split in PreTokenizer.PreTokenize(normalized))
            {
                idsCount += Model.CountTokens(split.TokenSpan, out _);
            }

            return idsCount;
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the start within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="processedText">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will remain unchanged as the input text.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely, if all tokens fit, the result will be length of the <paramref name="processedText"/>.
        /// </returns>
        /// <exception cref="ArgumentNullException">The input text is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">The maximum token count must be greater than 0.</exception>
        public int IndexOfTokenCount(string text, int maxTokenCount, out string processedText, out int tokenCount)
            => IndexOf(text, maxTokenCount, out processedText, out tokenCount);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="processedText">If the tokenizer's normalization is enabled, the input text will be represented in its normalization form; otherwise, it will remain unchanged as the input text.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the <paramref name="processedText"/>; conversely, if all tokens fit, the result will be 0.
        /// </returns>
        /// <exception cref="ArgumentNullException">The input text is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">The maximum token count must be greater than 0.</exception>
        /// <remarks>
        /// If the whole text can be encoded within the token limit, the returned index will be 0.
        /// </remarks>
        public int LastIndexOfTokenCount(string text, int maxTokenCount, out string processedText, out int tokenCount)
            => LastIndexOf(text, maxTokenCount, out processedText, out tokenCount);

        private int IndexOf(string text, int maxTokenCount, out string processedText, out int tokenCount)
        {
            if (text is null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The max token count must be greater than 0.");
            }

            processedText = Normalizer is not null ? Normalizer.Normalize(text) : text;
            tokenCount = 0;

            IEnumerable<Split> splits = PreTokenizer.PreTokenize(processedText);
            foreach (Split split in splits)
            {
                tokenCount += Model.CountTokens(split.TokenSpan, out int textLength, maxTokenCount - tokenCount);
                if (textLength < split.Offset.Length || tokenCount >= maxTokenCount)
                {
                    return split.Offset.Index + textLength;
                }
            }

            return processedText.Length;
        }

        private int LastIndexOf(string text, int maxTokenCount, out string processedText, out int tokenCount)
        {
            if (text is null)
            {
                throw new ArgumentNullException(nameof(text));
            }

            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The max token count must be greater than 0.");
            }

            processedText = Normalizer is not null ? Normalizer.Normalize(text) : text;
            tokenCount = 0;

            IEnumerable<Split> splits = PreTokenizer.PreTokenize(processedText);
            foreach (Split split in splits.Reverse())
            {
                tokenCount += Model.CountTokensFromEnd(split.TokenSpan, out int textIndex, maxTokenCount - tokenCount);
                if (textIndex > 0 || tokenCount >= maxTokenCount)
                {
                    return split.Offset.Index + textIndex;
                }
            }

            return 0;
        }

        /// <summary>
        /// Decodes the Id to the mapped token.
        /// </summary>
        /// <param name="id">The id to map to the token.</param>
        /// <returns>The decoded string or null if there is no token mapped to the input id.</returns>
        public string? Decode(int id) => Model.MapIdToToken(id);

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <returns>The decoded string.</returns>
        public string? Decode(IEnumerable<int> ids) => Model.Decode(ids, Decoder);

        /// <summary>
        /// Create a Tiktoken tokenizer based on model name and vocab file.
        /// </summary>
        /// <param name="modelName">Model name</param>
        /// <param name="vocabStream">The stream to the BPE vocab file.</param>
        /// <param name="extraSpecialTokens">Extra special tokens other than the built-in ones for the model</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <param name="normalizer">To normalize the text before tokenization</param>
        /// <returns>The tokenizer</returns>
        public static Tokenizer CreateTiktokenForModel(
                                    string modelName,
                                    Stream vocabStream,
                                    IReadOnlyDictionary<string, int>? extraSpecialTokens = null,
                                    int cacheSize = LruCache<int[]>.DefaultCacheSize,
                                    Normalizer? normalizer = null)
        {
            if (string.IsNullOrEmpty(modelName))
            {
                throw new ArgumentNullException(nameof(modelName));
            }

            (Dictionary<string, int> SpecialTokens, Regex Regex, string _) tiktokenConfiguration = Tiktoken.GetTiktokenConfigurations(modelName);

            if (extraSpecialTokens is not null)
            {
                foreach (var extraSpecialToken in extraSpecialTokens)
                {
                    tiktokenConfiguration.SpecialTokens.Add(extraSpecialToken.Key, extraSpecialToken.Value);
                }
            }

            return new Tokenizer(
                            new Tiktoken(vocabStream, tiktokenConfiguration.SpecialTokens, cacheSize),
                            new TiktokenPreTokenizer(tiktokenConfiguration.Regex, tiktokenConfiguration.SpecialTokens),
                            normalizer);
        }

        /// <summary>
        /// Create a Tiktoken tokenizer based on model name and vocab file.
        /// </summary>
        /// <param name="modelName">Model name</param>
        /// <param name="vocabStream">The stream to the BPE vocab file.</param>
        /// <param name="extraSpecialTokens">Extra special tokens other than the built-in ones for the model</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <param name="normalizer">To normalize the text before tokenization</param>
        /// <param name="cancellationToken"><see cref="CancellationToken"/> used to request cancellation of the operation.</param>
        /// <returns>The tokenizer</returns>
        public static async Task<Tokenizer> CreateTiktokenForModelAsync(
                                    string modelName,
                                    Stream vocabStream,
                                    IReadOnlyDictionary<string, int>? extraSpecialTokens = null,
                                    int cacheSize = LruCache<int[]>.DefaultCacheSize,
                                    Normalizer? normalizer = null,
                                    CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(modelName))
            {
                throw new ArgumentNullException(nameof(modelName));
            }

            (Dictionary<string, int> SpecialTokens, Regex Regex, string _) tiktokenConfiguration = Tiktoken.GetTiktokenConfigurations(modelName);

            if (extraSpecialTokens is not null)
            {
                foreach (var extraSpecialToken in extraSpecialTokens)
                {
                    tiktokenConfiguration.SpecialTokens.Add(extraSpecialToken.Key, extraSpecialToken.Value);
                }
            }

            return new Tokenizer(
                            await Tiktoken.CreateAsync(vocabStream, tiktokenConfiguration.SpecialTokens, cacheSize, cancellationToken).ConfigureAwait(false),
                            new TiktokenPreTokenizer(tiktokenConfiguration.Regex, tiktokenConfiguration.SpecialTokens),
                            normalizer);
        }

        /// <summary>
        /// Create tokenizer based on model name
        /// </summary>
        /// <param name="modelName">Model name</param>
        /// <param name="extraSpecialTokens">Extra special tokens other than the built-in ones for the model</param>
        /// <param name="normalizer">To normalize the text before tokenization</param>
        /// <returns>The tokenizer</returns>
        public static Tokenizer CreateTiktokenForModel(string modelName, IReadOnlyDictionary<string, int>? extraSpecialTokens = null, Normalizer? normalizer = null)
                        => Tiktoken.CreateTokenizerForModel(modelName, extraSpecialTokens, normalizer);

        /// <summary>
        /// Create tokenizer based on encoding name
        /// </summary>
        /// <param name="encodingName">Encoding name</param>
        /// <param name="extraSpecialTokens">Extra special tokens other than the built-in ones for the encoding</param>
        /// <param name="normalizer">To normalize the text before tokenization</param>
        /// <returns>The tokenizer</returns>
        public static Tokenizer CreateTiktokenForEncoding(string encodingName, IReadOnlyDictionary<string, int>? extraSpecialTokens = null, Normalizer? normalizer = null)
        {
            if (string.IsNullOrEmpty(encodingName))
            {
                throw new ArgumentNullException(nameof(encodingName));
            }

            Tiktoken.ModelEncoding modelEncoding;
            if (encodingName.Equals(Tiktoken.Cl100kBaseEncodingName, StringComparison.OrdinalIgnoreCase))
            {
                modelEncoding = Tiktoken.ModelEncoding.Cl100kBase;
            }
            else if (encodingName.Equals(Tiktoken.P50kBaseEncodingName, StringComparison.OrdinalIgnoreCase))
            {
                modelEncoding = Tiktoken.ModelEncoding.P50kBase;
            }
            else if (encodingName.Equals(Tiktoken.P50kEditEncodingName, StringComparison.OrdinalIgnoreCase))
            {
                modelEncoding = Tiktoken.ModelEncoding.P50kEdit;
            }
            else if (encodingName.Equals(Tiktoken.R50kBaseEncodingName, StringComparison.OrdinalIgnoreCase))
            {
                modelEncoding = Tiktoken.ModelEncoding.R50kBase;
            }
            else
            {
                throw new ArgumentException($"The encoding name '{encodingName}' is not supported. The only supported encoding names are: {Tiktoken.Cl100kBaseEncodingName}, {Tiktoken.P50kBaseEncodingName}, {Tiktoken.P50kEditEncodingName}, and {Tiktoken.R50kBaseEncodingName}.", nameof(encodingName));
            }

            return Tiktoken.CreateTokenizerForModel(modelEncoding, modelName: null, extraSpecialTokens, normalizer);
        }

        /// <summary>
        /// Create a SentencePieceBpe tokenizer from the given model stream. The model stream should contain the SentencePiece Bpe model according to
        /// https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto specification.
        /// </summary>
        /// <param name="modelStream">The stream containing the SentencePiece Bpe model.</param>
        /// <param name="addBeginOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        public static Tokenizer CreateLlama(
            Stream modelStream,
            bool addBeginOfSentence = true,
            bool addEndOfSentence = false)
        {
            ModelProto modelProto = ModelProto.Parser.ParseFrom(modelStream);

            if (modelProto is null)
            {
                throw new ArgumentNullException(nameof(modelProto));
            }

            if (modelProto.TrainerSpec.ModelType != TrainerSpec.Types.ModelType.Bpe)
            {
                throw new ArgumentException("The model type is not Bpe.", nameof(modelProto));
            }

            if (modelProto.NormalizerSpec.Name != "identity" && !string.IsNullOrEmpty(modelProto.NormalizerSpec.Name))
            {
                throw new ArgumentException($"Normalization '{modelProto.NormalizerSpec.Name}' is not supported.", nameof(modelProto));
            }

            LlamaNormalizer normalizer = new(
                                    modelProto.NormalizerSpec.RemoveExtraWhitespaces,
                                    modelProto.NormalizerSpec.AddDummyPrefix,
                                    modelProto.NormalizerSpec.EscapeWhitespaces,
                                    modelProto.TrainerSpec.TreatWhitespaceAsSuffix);

            return new Tokenizer(
                        new SentencePieceBpe(modelProto, addBeginOfSentence, addEndOfSentence),
                        SentencePiecePreTokenizer.Instance,
                        normalizer);
        }
    }
}
