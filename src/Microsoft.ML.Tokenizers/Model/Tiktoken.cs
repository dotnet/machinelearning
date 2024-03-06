// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Represent the rapid Byte Pair Encoding model commonly referred to as Tiktoken.
    /// </summary>
    public sealed partial class Tiktoken : Model
    {
        private readonly Dictionary<ReadOnlyMemory<byte>, int> _encoder;
        private readonly Dictionary<int, ReadOnlyMemory<byte>> _decoder;
        private readonly LruCache<int[]> _cache;
        private readonly Dictionary<StringSpanOrdinalKey, int>? _specialTokensEncoder;
        private Dictionary<string, int>? _specialTokensEncoderOriginal;
        private readonly Dictionary<int, string>? _specialTokensDecoder;
        private readonly Dictionary<StringSpanOrdinalKey, int> _vocab;
        private IReadOnlyDictionary<string, int>? _vocabOriginal;

        /// <summary>
        /// Create a new Tiktoken tokenizer's model object.
        /// </summary>
        /// <param name="vocabFilePath">The path to the BPE vocab file.</param>
        /// <param name="specialTokens">The dictionary mapping special tokens to Ids.</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="vocabFilePath"/> is null or empty.</exception>
        /// <exception cref="InvalidOperationException">Thrown when failed to load the BPE vocab file.</exception>
        public Tiktoken(string vocabFilePath, IReadOnlyDictionary<string, int>? specialTokens = null, int cacheSize = LruCache<int[]>.DefaultCacheSize) :
            this(string.IsNullOrEmpty(vocabFilePath) ? throw new ArgumentNullException(nameof(vocabFilePath)) : File.OpenRead(vocabFilePath), specialTokens, cacheSize, disposeStream: true)
        {
        }

        /// <summary>
        /// Create a new Tiktoken tokenizer's model object.
        /// </summary>
        /// <param name="vocabStream">The stream to the BPE vocab file.</param>
        /// <param name="specialTokens">The dictionary mapping special tokens to Ids.</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="vocabStream"/> is null or empty.</exception>
        /// <exception cref="InvalidOperationException">Thrown when failed to load the BPE vocab file.</exception>
        public Tiktoken(Stream vocabStream, IReadOnlyDictionary<string, int>? specialTokens = null, int cacheSize = LruCache<int[]>.DefaultCacheSize) :
            this(vocabStream ?? throw new ArgumentNullException(nameof(vocabStream)), specialTokens, cacheSize, disposeStream: false)
        {
        }

        /// <summary>
        /// Create a new Tiktoken tokenizer's model object.
        /// </summary>
        /// <param name="encoder">The dictionary mapping token utf-8 bytes to Ids.</param>
        /// <param name="decoder">The dictionary mapping Ids to token utf-8 bytes.</param>
        /// <param name="vocab">The dictionary mapping string tokens to Ids.</param>
        /// <param name="specialTokens">The dictionary mapping special tokens to Ids.</param>
        /// <param name="cacheSize">The max size of the cache to use.</param>
        internal Tiktoken(
            Dictionary<ReadOnlyMemory<byte>, int> encoder,
            Dictionary<int, ReadOnlyMemory<byte>> decoder,
            Dictionary<StringSpanOrdinalKey, int> vocab,
            IReadOnlyDictionary<string, int>? specialTokens,
            int cacheSize = LruCache<int[]>.DefaultCacheSize)
        {
            _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
            _decoder = decoder ?? throw new ArgumentNullException(nameof(decoder));
            _vocab = vocab ?? throw new ArgumentNullException(nameof(vocab));

            Debug.Assert(encoder.Count == decoder.Count);

            _encoder = encoder!;
            _decoder = decoder!;
            _vocab = vocab!;
            _cache = new LruCache<int[]>(cacheSize);

            (_specialTokensEncoder, _specialTokensDecoder) = CreateEncoderDecoder(specialTokens);
        }

        private Tiktoken(Stream vocabStream, IReadOnlyDictionary<string, int>? specialTokens, int cacheSize, bool disposeStream)
        {
            try
            {
                _cache = new LruCache<int[]>(cacheSize);
                (_encoder, _vocab, _decoder) = LoadTikTokenBpeAsync(vocabStream, useAsync: false).GetAwaiter().GetResult();
                (_specialTokensEncoder, _specialTokensDecoder) = CreateEncoderDecoder(specialTokens);
            }
            finally
            {
                if (disposeStream)
                {
                    vocabStream.Dispose();
                }
            }
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
        public static Tokenizer CreateByModelName(
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

            (Dictionary<string, int> SpecialTokens, Regex Regex) tiktokenConfiguration = GetTiktokenConfigurations(modelName);

            if (extraSpecialTokens is not null)
            {
                foreach (var extraSpecialToken in extraSpecialTokens)
                {
                    tiktokenConfiguration.SpecialTokens.Add(extraSpecialToken.Key, extraSpecialToken.Value);
                }
            }

            return new Tokenizer(
                            new Tiktoken(vocabStream, tiktokenConfiguration.SpecialTokens, cacheSize),
                            new TikTokenPreTokenizer(tiktokenConfiguration.Regex, tiktokenConfiguration.SpecialTokens),
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
        public static async Task<Tokenizer> CreateByModelNameAsync(
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

            (Dictionary<string, int> SpecialTokens, Regex Regex) tiktokenConfiguration = GetTiktokenConfigurations(modelName);

            if (extraSpecialTokens is not null)
            {
                foreach (var extraSpecialToken in extraSpecialTokens)
                {
                    tiktokenConfiguration.SpecialTokens.Add(extraSpecialToken.Key, extraSpecialToken.Value);
                }
            }

            return new Tokenizer(
                            await CreateAsync(vocabStream, tiktokenConfiguration.SpecialTokens, cacheSize, cancellationToken).ConfigureAwait(false),
                            new TikTokenPreTokenizer(tiktokenConfiguration.Regex, tiktokenConfiguration.SpecialTokens),
                            normalizer);
        }


        private static (Dictionary<StringSpanOrdinalKey, int>?, Dictionary<int, string>?) CreateEncoderDecoder(IReadOnlyDictionary<string, int>? specialTokens)
        {
            if (specialTokens is not null)
            {
                var encoder = specialTokens.ToDictionary(e => new StringSpanOrdinalKey(e.Key), e => e.Value);
                return (encoder, encoder.ToDictionary(kvp => kvp.Value, kvp => kvp.Key.Data!));
            }

            return (null, null);
        }

        /// <summary>
        /// Create a new Tiktoken tokenizer's model object asynchronously.
        /// </summary>
        /// <param name="vocabStream">The stream to the BPE vocab file.</param>
        /// <param name="specialTokens">The dictionary mapping special tokens to Ids.</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <param name="cancellationToken"><see cref="CancellationToken"/> used to request cancellation of the operation.</param>
        /// <returns>Tiktoken tokenizer's object.</returns>
        public static async Task<Tiktoken> CreateAsync(
                            Stream vocabStream,
                            IReadOnlyDictionary<string, int>? specialTokens = null,
                            int cacheSize = LruCache<int[]>.DefaultCacheSize,
                            CancellationToken cancellationToken = default)
        {
            if (vocabStream is null)
            {
                throw new ArgumentNullException(nameof(vocabStream));
            }

            (Dictionary<ReadOnlyMemory<byte>, int> encoder, Dictionary<StringSpanOrdinalKey, int> vocab, Dictionary<int, ReadOnlyMemory<byte>> decoder) =
                        await LoadTikTokenBpeAsync(vocabStream, useAsync: true, cancellationToken).ConfigureAwait(false);

            return new Tiktoken(encoder, decoder, vocab, specialTokens, cacheSize);
        }

        /// <summary>
        /// Create a new Tiktoken tokenizer's object asynchronously.
        /// </summary>
        /// <param name="vocabFilePath">The BPE vocab file.</param>
        /// <param name="specialTokensEncoder">The dictionary mapping special tokens to Ids.</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <param name="cancellationToken"><see cref="CancellationToken"/> used to request cancellation of the operation.</param>
        /// <returns>Tiktoken tokenizer's model object.</returns>
        public static async Task<Tiktoken> CreateAsync(
                                string vocabFilePath,
                                IReadOnlyDictionary<string, int>? specialTokensEncoder = null,
                                int cacheSize = LruCache<int[]>.DefaultCacheSize,
                                CancellationToken cancellationToken = default)
        {
            if (vocabFilePath is null)
            {
                throw new ArgumentNullException(nameof(vocabFilePath));
            }

            using Stream vocabStream = File.OpenRead(vocabFilePath);
            return await CreateAsync(vocabStream, specialTokensEncoder, cacheSize, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Load BPE vocab dictionary from a stream.
        /// </summary>
        /// <param name="vocabStream">Stream to the BPE vocab file</param>
        /// <param name="useAsync">Whether to perform I/O synchronously or asynchronously.</param>
        /// <param name="cancellationToken"><see cref="CancellationToken"/> used to request cancellation of the operation.</param>
        /// <returns>Map of byte[] to integer token id</returns>
        /// <exception cref="InvalidOperationException"></exception>
        internal static async ValueTask<(Dictionary<ReadOnlyMemory<byte>, int>, Dictionary<StringSpanOrdinalKey, int>, Dictionary<int, ReadOnlyMemory<byte>>)> LoadTikTokenBpeAsync(
            Stream vocabStream, bool useAsync, CancellationToken cancellationToken = default)
        {
            var encoder = new Dictionary<ReadOnlyMemory<byte>, int>(ReadOnlyMemoryByteComparer.Instance);
            var vocab = new Dictionary<StringSpanOrdinalKey, int>();
            var decoder = new Dictionary<int, ReadOnlyMemory<byte>>();

            try
            {
                using (StreamReader reader = new StreamReader(vocabStream))
                {
                    while (true)
                    {
                        string? line = useAsync ?
                            await Helpers.ReadLineAsync(reader, cancellationToken).ConfigureAwait(false) :
                            reader.ReadLine();
                        if (string.IsNullOrWhiteSpace(line))
                        {
                            if (line is null)
                            {
                                break;
                            }
                            continue;
                        }

                        int spaceIndex = line.IndexOf(' ');
                        if (spaceIndex <= 0 || spaceIndex >= line.Length - 1 || line.IndexOf(' ', spaceIndex + 1) >= 0)
                        {
                            throw new FormatException($"Invalid format in the BPE vocab file stream");
                        }

                        if (Helpers.TryParseInt32(line, spaceIndex + 1, out int rank))
                        {
                            byte[] tokenBytes = Helpers.FromBase64String(line, 0, spaceIndex);

                            encoder[tokenBytes] = rank;
                            decoder[rank] = tokenBytes;

                            string decodedToken = Encoding.UTF8.GetString(tokenBytes);

                            if (decodedToken.IndexOf('\uFFFD') < 0)
                            {
                                vocab[new StringSpanOrdinalKey(decodedToken)] = rank;
                            }
                        }
                        else
                        {
                            throw new FormatException($"Can't parse {line.Substring(spaceIndex)} to integer");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to load from BPE vocab file stream: {ex.Message}", ex);
            }

            return (encoder, vocab, decoder);
        }

        /// <summary>
        /// Encode a split text string to a list of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="isSpecialToken">Indicate if the text is a special token.</param>
        /// <returns>The list of tokens generated from the text tokenization.</returns>
        public override IReadOnlyList<Token> Encode(string text, bool isSpecialToken = false)
        {
            Token[] tokens;

            if (string.IsNullOrEmpty(text))
            {
                return Array.Empty<Token>();
            }

            if (isSpecialToken)
            {
                if (_specialTokensEncoder?.TryGetValue(text, out int id) is true)
                {
                    return new List<Token> { new(id, text, (0, text.Length)) };
                }

                throw new InvalidOperationException($"The special token {text} doesn't exist in the tokenizer");
            }

            if (_cache.TryGetValue(text, out int[]? ids))
            {
                tokens = new Token[ids.Length];
                tokens[0] = new Token(ids[0], text, (0, text.Length));
                for (int i = 1; i < ids.Length; i++)
                {
                    // One word split mapped to multiple Ids. Make the offset of the remaining token point at the end with zero width.
                    tokens[i] = new Token(ids[i], "", (text.Length, 0));
                }

                return tokens;
            }

            // cache miss
            if (_vocab.TryGetValue(text, out int mappedId))
            {
                return new Token[1] { new(mappedId, text, (0, text.Length)) };
            }

            byte[] arrayPoolArray = ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(text.Length));
            int encodedLength = GetUtf8Bytes(text.AsSpan(), arrayPoolArray);

            int[] encodedIds = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder);
            Debug.Assert(encodedIds.Length > 0);
            _cache.Add(text, encodedIds);

            tokens = new Token[encodedIds.Length];
            tokens[0] = new Token(encodedIds[0], text, (0, text.Length));
            for (int i = 1; i < encodedIds.Length; i++)
            {
                // One word split mapped to multiple Ids. Make the offset of the remaining token point at the end with zero width.
                tokens[i] = new Token(encodedIds[i], "", (text.Length, 0));
            }

            ArrayPool<byte>.Shared.Return(arrayPoolArray);
            return tokens;
        }

        /// <summary>
        /// Encode text to a list of Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="isSpecialToken">Indicate if the text is a special token.</param>
        /// <param name="accumulatedIds">The list of accumulated Ids.</param>
        public override void EncodeToIds(ReadOnlySpan<char> text, bool isSpecialToken, IList<int> accumulatedIds)
        {
            if (text.IsEmpty)
            {
                return;
            }

            if (isSpecialToken)
            {
                if (_specialTokensEncoder?.TryGetValue(text, out int id) is true)
                {
                    accumulatedIds.Add(id);
                }

                return;
            }

            if (_cache.TryGetValue(text, out int[]? tokenIds))
            {
                accumulatedIds.AddRange(tokenIds);
                return;
            }

            if (_vocab.TryGetValue(text, out int mappedId))
            {
                accumulatedIds.Add(mappedId);
                return;
            }

            byte[] arrayPoolArray = ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(text.Length));
            int encodedLength = GetUtf8Bytes(text, arrayPoolArray);

            int[] encodedIds = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder);
            _cache.Add(text.ToString(), encodedIds);

            accumulatedIds.AddRange(encodedIds);

            ArrayPool<byte>.Shared.Return(arrayPoolArray);
            return;
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to tokenize.</param>
        /// <param name="isSpecialToken">Indicate if the token is special token.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        public override int CountTokens(ReadOnlySpan<char> text, bool isSpecialToken)
        {
            if (text.IsEmpty)
            {
                return 0;
            }

            if (isSpecialToken && _specialTokensEncoder is not null)
            {
                return _specialTokensEncoder.TryGetValue(text, out _) ? 1 : 0;
            }

            if (_cache.TryGetValue(text, out int[] ids))
            {
                return ids.Length;
            }

            if (_vocab.TryGetValue(text, out _))
            {
                return 1;
            }

            byte[] arrayPoolArray = ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(text.Length));
            int encodedLength = GetUtf8Bytes(text, arrayPoolArray);

            int[] encodedIds = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder);
            _cache.Add(text.ToString(), encodedIds);

            ArrayPool<byte>.Shared.Return(arrayPoolArray);
            return encodedIds.Length;
        }

        /// <summary>
        /// Map the token to encoded Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the encoding.</param>
        /// <returns>The mapped Id of the token.</returns>
        public override int? MapTokenToId(ReadOnlySpan<char> token, bool considerSpecialTokens = true)
        {
            if (token.IsEmpty)
            {
                return 0;
            }

            if (considerSpecialTokens && _specialTokensEncoder is not null)
            {
                if (_specialTokensEncoder.TryGetValue(token, out int specialTokenId))
                {
                    return specialTokenId;
                }
            }

            if (_cache.TryGetValue(token, out int[] ids))
            {
                if (ids.Length == 1)
                {
                    return ids[0];
                }

                return null;
            }

            if (_vocab.TryGetValue(token, out int id))
            {
                return id;
            }

            byte[] arrayPoolArray = ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(token.Length));
            try
            {
                int encodedLength = GetUtf8Bytes(token, arrayPoolArray);

                int[] idsToCache = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder);
                _cache.Add(token.ToString(), idsToCache);

                if (idsToCache.Length == 1)
                {
                    return idsToCache[0];
                }

                return null;
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(arrayPoolArray);
            }
        }

        /// <summary>
        /// Map the encoded Id to the token.
        /// </summary>
        /// <param name="id">The Id to map to the token.</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the decoding.</param>
        /// <returns>The mapped token of the Id.</returns>
        public override string? MapIdToToken(int id, bool considerSpecialTokens = true)
        {
            if (considerSpecialTokens && _specialTokensDecoder is not null && _specialTokensDecoder.TryGetValue(id, out string? token))
            {
                return token;
            }

            if (_decoder.TryGetValue(id, out ReadOnlyMemory<byte> tokenBytes))
            {
                return GetString(tokenBytes.Span);
            }

            return null;
        }

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="considerSpecialTokens">Whether the special tokens should be kept in the decoded string.</param>
        /// <param name="decoder">The optional Decoder to merge the given list of tokens in a string.</param>
        /// <returns>The decoded string.</returns>
        public override string? Decode(IEnumerable<int> ids, TokenizerDecoder? decoder = null, bool considerSpecialTokens = true)
        {

            // Tiktoken doesn't guarantee a one-to-one correspondence between IDs and UTF-16 words.
            // Consequently, decoding individual IDs into UTF-16 string is not supported; instead, decoding all IDs must be performed collectively.
            // Here's an example case that maps one character to multiple IDs:
            // '⭐' U-2B50 is mapped to IDs [2928, 99834] in the Tiktoken model.
            // In other words, the character '⭐' with UTF-8 code point 0xE2, 0xAD, 0x90 will be mapped by Tiktoken as follows: 0xE2 to [2928]
            // and 0xAD, 0x90 to [99834]. Decoding 2928 and 99834 individually won't reconstruct the original UTF-16 string '⭐' U-2B50;
            // decoding all IDs together is required to get the expected result.
            if (ids is null)
            {
                return null;
            }

            byte[]? arrayPoolArray = null;
            try
            {
                Span<byte> utf8Bytes = stackalloc byte[256];
                int utf8ByteCount = 0;

                bool useSpecialTokens = considerSpecialTokens && _specialTokensDecoder is not null;

                foreach (int id in ids)
                {
                    if (_decoder.TryGetValue(id, out ReadOnlyMemory<byte> tokenBytes))
                    {
                        if ((uint)utf8ByteCount + (uint)tokenBytes.Length > (uint)utf8Bytes.Length)
                        {
                            ArrayPoolGrow(ref utf8Bytes, ref arrayPoolArray, utf8ByteCount + tokenBytes.Length);
                        }

                        tokenBytes.Span.CopyTo(utf8Bytes.Slice(utf8ByteCount));
                        utf8ByteCount += tokenBytes.Length;
                    }
                    else if (useSpecialTokens && _specialTokensDecoder!.TryGetValue(id, out string? token))
                    {
                        while (true)
                        {
                            if (TryGetUtf8Bytes(token.AsSpan(), utf8Bytes.Slice(utf8ByteCount), out int bytesWritten))
                            {
                                utf8ByteCount += bytesWritten;
                                break;
                            }

                            ArrayPoolGrow(ref utf8Bytes, ref arrayPoolArray, utf8ByteCount + Encoding.UTF8.GetByteCount(token));
                        }
                    }
                    else
                    {
                        return null;
                    }
                }

                return GetString(utf8Bytes.Slice(0, utf8ByteCount));
            }
            finally
            {
                if (arrayPoolArray is not null)
                {
                    ArrayPool<byte>.Shared.Return(arrayPoolArray);
                }
            }

            static void ArrayPoolGrow(ref Span<byte> utf8Bytes, ref byte[]? arrayPoolArray, int requiredCapacity)
            {
                byte[] tmp = ArrayPool<byte>.Shared.Rent(Math.Max(utf8Bytes.Length * 2, requiredCapacity));
                utf8Bytes.CopyTo(tmp.AsSpan());
                byte[]? toReturn = arrayPoolArray;
                utf8Bytes = arrayPoolArray = tmp;
                if (toReturn is not null)
                {
                    ArrayPool<byte>.Shared.Return(toReturn);
                }
            }
        }

        /// <summary>
        /// Gets the dictionary mapping tokens to Ids.
        /// </summary>
        /// <remarks>This may not contain the full set of vocabulary tokens, use Encoder to get the full set of vocabulary.</remarks>
        public IReadOnlyDictionary<string, int> Vocab => _vocabOriginal ??= _vocab.ToDictionary(kvp => kvp.Key.Data!, kvp => kvp.Value);

        /// <summary>
        /// Gets the dictionary mapping special tokens to Ids.
        /// </summary>
        public IReadOnlyDictionary<string, int>? SpecialTokensEncoder => _specialTokensEncoderOriginal ??= _specialTokensEncoder?.ToDictionary(kvp => kvp.Key.Data!, kvp => kvp.Value);

        /// <summary>
        /// Gets the dictionary mapping token bytes to Ids.
        /// </summary>
        public IReadOnlyDictionary<ReadOnlyMemory<byte>, int> Encoder => _encoder;

        /// <summary>
        /// Gets the dictionary mapping Ids to token utf-8 bytes.
        /// </summary>
        public IReadOnlyDictionary<int, ReadOnlyMemory<byte>> Decoder => _decoder;

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

        private static unsafe int GetUtf8Bytes(ReadOnlySpan<char> source, Span<byte> destination)
        {
#if NETCOREAPP
            return Encoding.UTF8.GetBytes(source, destination);
#else
            fixed (char* sourcePtr = source)
            fixed (byte* destPtr = destination)
            {
                return Encoding.UTF8.GetBytes(sourcePtr, source.Length, destPtr, destination.Length);
            }
#endif
        }

        private static unsafe bool TryGetUtf8Bytes(ReadOnlySpan<char> source, Span<byte> destination, out int bytesWritten)
        {
#if NET8_0_OR_GREATER
            return Encoding.UTF8.TryGetBytes(source, destination, out bytesWritten);
#else
            fixed (char* sourcePtr = source)
            fixed (byte* destPtr = destination)
            {
                if (Encoding.UTF8.GetByteCount(sourcePtr, source.Length) <= destination.Length)
                {
                    bytesWritten = Encoding.UTF8.GetBytes(sourcePtr, source.Length, destPtr, destination.Length);
                    return true;
                }

                bytesWritten = 0;
                return false;
            }
#endif
        }

        private static unsafe string GetString(ReadOnlySpan<byte> utf8Bytes)
        {
#if NETCOREAPP
            return Encoding.UTF8.GetString(utf8Bytes);
#else
            fixed (byte* sourcePtr = utf8Bytes)
            {
                return Encoding.UTF8.GetString(sourcePtr, utf8Bytes.Length);
            }
#endif
        }
    }
}
