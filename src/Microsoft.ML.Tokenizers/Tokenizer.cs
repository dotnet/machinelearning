// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Sentencepiece;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// serves as an abstraction for concrete tokenizers, enabling the encoding of text into tokens and IDs, as well as the decoding of IDs back into text.
    /// </summary>
    public abstract class Tokenizer
    {
        /// <summary>
        /// Gets the PreTokenizer used by the Tokenizer.
        /// </summary>
        public virtual PreTokenizer? PreTokenizer => null;

        /// <summary>
        /// Gets the Normalizer in use by the Tokenizer.
        /// </summary>
        public virtual Normalizer? Normalizer => null;

        /// <summary>
        /// Encodes input text a list of <see cref="Token" />s with string value of the token, id, and offset.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The tokenization result includes a list of <see cref="Token" />s with string value of the token, id, and offset.</returns>
        public virtual IReadOnlyList<Token> Encode(string text, out string? normalizedString, bool considerPreTokenization = true, bool considerNormalization = true) => Encode(text.AsSpan(), out normalizedString, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Encodes input text a list of <see cref="Token" />s with string value of the token, id, and offset.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The tokenization result includes a list of <see cref="Token" />s with string value of the token, id, and offset.</returns>
        public abstract IReadOnlyList<Token> Encode(ReadOnlySpan<char> text, out string? normalizedString, bool considerPreTokenization = true, bool considerNormalization = true);

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public virtual IReadOnlyList<int> EncodeToIds(string text, bool considerPreTokenization = true, bool considerNormalization = true) => EncodeToIds(text.AsSpan(), considerPreTokenization, considerNormalization);

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public abstract IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, bool considerPreTokenization = true, bool considerNormalization = true);

        /// <summary>
        /// Encodes input text to token Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public virtual IReadOnlyList<int> EncodeToIds(string text, int maxTokenCount, out string? normalizedText, out int textLength, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(text.AsSpan(), maxTokenCount, out normalizedText, out textLength, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Encodes input text to token Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public abstract IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedText, out int textLength, bool considerPreTokenization = true, bool considerNormalization = true);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        public virtual int CountTokens(string text, bool considerPreTokenization = true, bool considerNormalization = true)
            => CountTokens(text.AsSpan(), considerPreTokenization, considerNormalization);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        public abstract int CountTokens(ReadOnlySpan<char> text, bool considerPreTokenization = true, bool considerNormalization = true);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the start within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the text or the <paramref name="normalizedString"/> if the normalization is enabled.
        /// </returns>
        public virtual int IndexOfTokenCount(string text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
            => IndexOfTokenCount(text.AsSpan(), maxTokenCount, out normalizedString, out tokenCount, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the start within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index immediately following the last character to be included. In cases where no tokens fit, the result will be 0; conversely,
        /// if all tokens fit, the result will be length of the text or the <paramref name="normalizedString"/> if the normalization is enabled.
        /// </returns>
        public abstract int IndexOfTokenCount(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="processedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the <paramref name="processedText"/>; conversely, if all tokens fit, the result will be 0.
        /// </returns>
        public virtual int LastIndexOfTokenCount(string text, int maxTokenCount, out string? processedText, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
            => LastIndexOfTokenCount(text.AsSpan(), maxTokenCount, out processedText, out tokenCount, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="processedText">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the <paramref name="processedText"/>; conversely, if all tokens fit, the result will be 0.
        /// </returns>
        public abstract int LastIndexOfTokenCount(ReadOnlySpan<char> text, int maxTokenCount, out string? processedText, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true);

        /// <summary>
        /// Map the token to encoded Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <returns>The mapped Id of the token.</returns>
        public virtual int? MapTokenToId(string token)
        {
            if (token is null)
            {
                throw new ArgumentNullException(nameof(token));
            }

            return MapTokenToId(token.AsSpan());
        }

        /// <summary>
        /// Map the token to encoded Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <returns>The mapped Id of the token.</returns>
        public abstract int? MapTokenToId(ReadOnlySpan<char> token);

        /// <summary>
        /// Decodes the Id to the mapped token.
        /// </summary>
        /// <param name="id">The id to map to the token.</param>
        /// <returns>The decoded string or null if there is no token mapped to the input id.</returns>
        public abstract string? MapIdToToken(int id);

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <returns>The decoded string.</returns>
        public virtual string? Decode(IEnumerable<int> ids)
        {
            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
            }

            ValueStringBuilder sb = new ValueStringBuilder();

            foreach (int id in ids)
            {
                if (MapIdToToken(id) is string s)
                {
                    sb.Append(s);
                }
            }

            return sb.ToString();
        }

        //
        // Factory Methods
        //

        /// <summary>
        /// Create a new Tiktoken tokenizer's object asynchronously.
        /// </summary>
        /// <param name="vocabStream">The stream to the BPE vocab file.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="specialTokens">The dictionary mapping special tokens to Ids.</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <param name="cancellationToken"><see cref="CancellationToken"/> used to request cancellation of the operation.</param>
        /// <returns>The tokenizer's object.</returns>
        public static async Task<Tokenizer> CreateTiktokenAsync(
                            Stream vocabStream,
                            PreTokenizer? preTokenizer,
                            Normalizer? normalizer,
                            IReadOnlyDictionary<string, int>? specialTokens = null,
                            int cacheSize = LruCache<int[]>.DefaultCacheSize,
                            CancellationToken cancellationToken = default)
        {
            if (vocabStream is null)
            {
                throw new ArgumentNullException(nameof(vocabStream));
            }

            (Dictionary<ReadOnlyMemory<byte>, int> encoder, Dictionary<StringSpanOrdinalKey, (int Id, string Token)> vocab, Dictionary<int, ReadOnlyMemory<byte>> decoder) =
                        await Tiktoken.LoadTiktokenBpeAsync(vocabStream, useAsync: true, cancellationToken).ConfigureAwait(false);

            return new Tiktoken(encoder, decoder, vocab, preTokenizer, specialTokens, normalizer, cacheSize);
        }

        /// <summary>
        /// Create a new Tiktoken tokenizer's object asynchronously.
        /// </summary>
        /// <param name="vocabFilePath">The BPE vocab file.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="specialTokensEncoder">The dictionary mapping special tokens to Ids.</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <param name="cancellationToken"><see cref="CancellationToken"/> used to request cancellation of the operation.</param>
        /// <returns>The tokenizer's object.</returns>
        public static async Task<Tokenizer> CreateTiktokenAsync(
                                string vocabFilePath,
                                PreTokenizer? preTokenizer,
                                Normalizer? normalizer,
                                IReadOnlyDictionary<string, int>? specialTokensEncoder = null,
                                int cacheSize = LruCache<int[]>.DefaultCacheSize,
                                CancellationToken cancellationToken = default)
        {
            if (vocabFilePath is null)
            {
                throw new ArgumentNullException(nameof(vocabFilePath));
            }

            using Stream vocabStream = File.OpenRead(vocabFilePath);
            return await CreateTiktokenAsync(vocabStream, preTokenizer, normalizer, specialTokensEncoder, cacheSize, cancellationToken).ConfigureAwait(false);
        }

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

            return new Tiktoken(vocabStream,
                                new TiktokenPreTokenizer(tiktokenConfiguration.Regex, tiktokenConfiguration.SpecialTokens),
                                tiktokenConfiguration.SpecialTokens,
                                normalizer,
                                cacheSize);
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

            return await CreateTiktokenAsync(vocabStream,
                                new TiktokenPreTokenizer(tiktokenConfiguration.Regex, tiktokenConfiguration.SpecialTokens),
                                normalizer,
                                tiktokenConfiguration.SpecialTokens,
                                cacheSize, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Create tokenizer based on model name
        /// </summary>
        /// <param name="modelName">Model name</param>
        /// <param name="extraSpecialTokens">Extra special tokens other than the built-in ones for the model</param>
        /// <param name="normalizer">To normalize the text before tokenization</param>
        /// <returns>The tokenizer</returns>
        public static Tokenizer CreateTiktokenForModel(string modelName, IReadOnlyDictionary<string, int>? extraSpecialTokens = null, Normalizer? normalizer = null)
                        => Tiktoken.CreateForModel(Tiktoken.GetModelEncoding(modelName), modelName, extraSpecialTokens, normalizer);

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
            else if (encodingName.Equals(Tiktoken.O200kBaseEncodingName, StringComparison.OrdinalIgnoreCase))
            {
                modelEncoding = Tiktoken.ModelEncoding.O200kBase;
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

            return Tiktoken.CreateForModel(modelEncoding, modelName: null, extraSpecialTokens, normalizer);
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

            SentencePieceNormalizer normalizer = new(
                                    modelProto.NormalizerSpec.RemoveExtraWhitespaces,
                                    modelProto.NormalizerSpec.AddDummyPrefix,
                                    modelProto.NormalizerSpec.EscapeWhitespaces,
                                    modelProto.TrainerSpec.TreatWhitespaceAsSuffix);

            return new SentencePieceBpe(modelProto, addBeginOfSentence, addEndOfSentence);
        }

        /// <summary>
        /// Create a CodeGen tokenizer from the given vocab and merges streams.
        /// </summary>
        /// <param name="vocabStream">The stream containing the vocab file.</param>
        /// <param name="mergesStream">The stream containing the merges file.</param>
        /// <param name="addPrefixSpace">Indicate whether to add a space before the token.</param>
        /// <param name="addBeginOfSentence">Indicate emitting the beginning of sentence token during the encoding.</param>
        /// <param name="addEndOfSentence">Indicate emitting the end of sentence token during the encoding.</param>
        /// <returns>The CodeGen tokenizer object.</returns>
        /// <remarks>
        /// The tokenizer will be created according to the configuration specified in https://huggingface.co/Salesforce/codegen-350M-mono/raw/main/tokenizer.json.
        /// It is important to provide the similar vocab and merges files to the ones used in the training of the model.
        /// The vocab and merges files can be downloaded from the following links:
        ///     https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/vocab.json?download=true
        ///     https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/merges.txt?download=true
        /// </remarks>
        public static Tokenizer CreateCodeGen(
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

            return new CodeGen(
                        vocabStream,
                        mergesStream,
                        new TiktokenPreTokenizer(Tiktoken.P50kBaseRegex(), CodeGen.CodeGenAddedTokens),
                        normalizer: null,
                        CodeGen.CodeGenAddedTokens,
                        addPrefixSpace: addPrefixSpace,
                        addBeginningOfSentence: addBeginOfSentence,
                        addEndOfSentence: addEndOfSentence);
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
        public static Tokenizer CreatePhi2(
            Stream vocabStream,
            Stream mergesStream,
            bool addPrefixSpace = false,
            bool addBeginOfSentence = false,
            bool addEndOfSentence = false)
            => CreateCodeGen(vocabStream, mergesStream, addPrefixSpace, addBeginOfSentence, addEndOfSentence);

        internal static IEnumerable<(int Offset, int Length)>? InitializeForEncoding(
                                                string? text,
                                                ReadOnlySpan<char> textSpan,
                                                bool considerPreTokenization,
                                                bool considerNormalization,
                                                Normalizer? normalizer,
                                                PreTokenizer? preTokenizer,
                                                out string? normalizedString,
                                                out ReadOnlySpan<char> textSpanToEncode)
        {
            normalizedString = null;
            IEnumerable<(int Offset, int Length)>? splits = null;

            if (text is null)
            {
                if (considerNormalization && (normalizer is not null))
                {
                    normalizedString = normalizer.Normalize(textSpan.ToString());
                    textSpanToEncode = normalizedString.AsSpan();
                    if (considerPreTokenization && preTokenizer is not null)
                    {
                        splits = preTokenizer.PreTokenize(normalizedString);
                    }
                }
                else
                {
                    textSpanToEncode = textSpan;
                    if (considerPreTokenization && preTokenizer is not null)
                    {
                        splits = preTokenizer.PreTokenize(textSpan);
                    }
                }
            }
            else
            {
                if (considerNormalization && (normalizer is not null))
                {
                    normalizedString = normalizer.Normalize(text);
                    textSpanToEncode = normalizedString.AsSpan();
                    if (considerPreTokenization && preTokenizer is not null)
                    {
                        splits = preTokenizer.PreTokenize(normalizedString);
                    }
                }
                else
                {
                    textSpanToEncode = text.AsSpan();
                    if (considerPreTokenization && preTokenizer is not null)
                    {
                        splits = preTokenizer.PreTokenize(text);
                    }
                }
            }

            return splits;
        }
    }
}
