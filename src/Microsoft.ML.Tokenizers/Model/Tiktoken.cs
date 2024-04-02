// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
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
        private readonly LruCache<(int[] Bytes, string Token)> _cache;
        private readonly Dictionary<StringSpanOrdinalKey, (int Id, string Token)> _vocab;
        private IReadOnlyDictionary<string, int>? _vocabOriginal;
        private const int MaxWordLengthToCache = 15;

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
            Dictionary<StringSpanOrdinalKey, (int Id, string Token)> vocab,
            IReadOnlyDictionary<string, int>? specialTokens,
            int cacheSize = LruCache<int[]>.DefaultCacheSize)
        {
            _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
            _decoder = decoder ?? throw new ArgumentNullException(nameof(decoder));
            _vocab = vocab ?? throw new ArgumentNullException(nameof(vocab));

            _encoder = encoder!;
            _decoder = decoder!;
            _vocab = vocab!;
            _cache = new LruCache<(int[] Bytes, string Token)>(cacheSize);

            SpecialTokens = specialTokens;
            CacheSpecialTokensEncoding(specialTokens);
        }

        private Tiktoken(Stream vocabStream, IReadOnlyDictionary<string, int>? specialTokens, int cacheSize, bool disposeStream)
        {
            try
            {
                _cache = new LruCache<(int[] Bytes, string Token)>(cacheSize);
                (_encoder, _vocab, _decoder) = LoadTiktokenBpeAsync(vocabStream, useAsync: false).GetAwaiter().GetResult();

                SpecialTokens = specialTokens;
                CacheSpecialTokensEncoding(specialTokens);
            }
            finally
            {
                if (disposeStream)
                {
                    vocabStream.Dispose();
                }
            }
        }

        private void CacheSpecialTokensEncoding(IReadOnlyDictionary<string, int>? specialTokens)
        {
            Debug.Assert(_cache is not null);
            Debug.Assert(_decoder is not null);

            if (specialTokens is not null)
            {
                foreach (KeyValuePair<string, int> specialToken in specialTokens)
                {
                    _decoder![specialToken.Value] = Encoding.UTF8.GetBytes(specialToken.Key);
                    _cache!.Add(specialToken.Key, (new[] { specialToken.Value }, specialToken.Key));
                }
            }
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

            (Dictionary<ReadOnlyMemory<byte>, int> encoder, Dictionary<StringSpanOrdinalKey, (int Id, string Token)> vocab, Dictionary<int, ReadOnlyMemory<byte>> decoder) =
                        await LoadTiktokenBpeAsync(vocabStream, useAsync: true, cancellationToken).ConfigureAwait(false);

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
        internal static async ValueTask<(Dictionary<ReadOnlyMemory<byte>, int>, Dictionary<StringSpanOrdinalKey, (int Id, string Token)>, Dictionary<int, ReadOnlyMemory<byte>>)> LoadTiktokenBpeAsync(
            Stream vocabStream, bool useAsync, CancellationToken cancellationToken = default)
        {
            var encoder = new Dictionary<ReadOnlyMemory<byte>, int>(ReadOnlyMemoryByteComparer.Instance);
            var vocab = new Dictionary<StringSpanOrdinalKey, (int Id, string Token)>();
            var decoder = new Dictionary<int, ReadOnlyMemory<byte>>();

            try
            {
                using (StreamReader reader = new StreamReader(vocabStream))
                {
                    string? line;
                    do
                    {
                        line = useAsync ?
                            await Helpers.ReadLineAsync(reader, cancellationToken).ConfigureAwait(false) :
                            reader.ReadLine();
                    } while (line is not null && line.Length == 0);

                    if (line is not null && line.IndexOf(' ') < 0)
                    {
                        // We generate the ranking using the line number
                        int lineNumber = 0;
                        do
                        {
                            if (line.Length > 0)
                            {
                                AddData(Convert.FromBase64String(line), lineNumber);
                            }
                            lineNumber++;
                        } while ((line = useAsync ? await Helpers.ReadLineAsync(reader, cancellationToken).ConfigureAwait(false) : reader.ReadLine()) is not null);
                    }

                    while (line is not null)
                    {
                        if (line.Length > 0)
                        {
                            int spaceIndex = line.IndexOf(' ');
                            if (spaceIndex <= 0 || spaceIndex >= line.Length - 1 || line.IndexOf(' ', spaceIndex + 1) >= 0)
                            {
                                throw new FormatException($"Invalid format in the BPE vocab file stream");
                            }

                            if (Helpers.TryParseInt32(line, spaceIndex + 1, out int rank))
                            {
                                AddData(Helpers.FromBase64String(line, 0, spaceIndex), rank);
                            }
                            else
                            {
                                throw new FormatException($"Can't parse {line.Substring(spaceIndex)} to integer");
                            }

                            line = useAsync ?
                                await Helpers.ReadLineAsync(reader, cancellationToken).ConfigureAwait(false) :
                                reader.ReadLine();
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to load from BPE vocab file stream: {ex.Message}", ex);
            }

            return (encoder, vocab, decoder);

            void AddData(byte[] tokenBytes, int rank)
            {
                encoder[tokenBytes] = rank;
                decoder[rank] = tokenBytes;

                string decodedToken = Encoding.UTF8.GetString(tokenBytes);

                if (decodedToken.IndexOf('\uFFFD') < 0)
                {
                    vocab[new StringSpanOrdinalKey(decodedToken)] = (rank, decodedToken);
                }
            }
        }

        /// <summary>
        /// Encode text to a list of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>The list of tokens generated from the text tokenization.</returns>
        public override IReadOnlyList<Token> Encode(ReadOnlySpan<char> text)
        {
            Token[] tokens;

            if (text.IsEmpty)
            {
                return Array.Empty<Token>();
            }

            if (_cache.TryGetValue(text, out (int[] Ids, string Token) value))
            {
                tokens = new Token[value.Ids.Length];
                tokens[0] = new Token(value.Ids[0], value.Token, (0, value.Token.Length));
                for (int i = 1; i < value.Ids.Length; i++)
                {
                    // One word split mapped to multiple Ids. Make the offset of the remaining token point at the end with zero width.
                    tokens[i] = new Token(value.Ids[i], "", (text.Length, 0));
                }

                return tokens;
            }

            // cache miss
            if (_vocab.TryGetValue(text, out (int Id, string Token) mappedId))
            {
                return new Token[1] { new(mappedId.Id, mappedId.Token, (0, mappedId.Token.Length)) };
            }

            byte[] arrayPoolArray = ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(text.Length));
            int encodedLength = Helpers.GetUtf8Bytes(text, arrayPoolArray);

            int[] encodedIds = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder);
            ArrayPool<byte>.Shared.Return(arrayPoolArray);

            Debug.Assert(encodedIds.Length > 0);
            string textAsString = text.ToString();

            if (text.Length <= MaxWordLengthToCache)
            {
                _cache.Add(textAsString, (encodedIds, textAsString));
            }

            tokens = new Token[encodedIds.Length];
            tokens[0] = new Token(encodedIds[0], textAsString, (0, text.Length));
            for (int i = 1; i < encodedIds.Length; i++)
            {
                // One word split mapped to multiple Ids. Make the offset of the remaining token point at the end with zero width.
                tokens[i] = new Token(encodedIds[i], "", (text.Length, 0));
            }

            return tokens;
        }

        /// <summary>
        /// Encode text to a list of Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="accumulatedIds">The list of accumulated Ids.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        public override int EncodeToIds(ReadOnlySpan<char> text, IList<int> accumulatedIds, out int textLength, int maxTokens = int.MaxValue)
        {
            Debug.Assert(maxTokens > 0);

            if (text.IsEmpty)
            {
                textLength = 0;
                return 0;
            }

            if (_cache.TryGetValue(text, out (int[] Ids, string Token) value))
            {
                if (value.Ids.Length <= maxTokens)
                {
                    accumulatedIds.AddRange(value.Ids);
                    textLength = text.Length;
                    return value.Ids.Length;
                }

                textLength = 0;
                return 0;
            }

            if (_vocab.TryGetValue(text, out (int Id, string Token) mappedId))
            {
                textLength = text.Length;
                accumulatedIds.Add(mappedId.Id);
                return 1;
            }

            byte[] arrayPoolArray = ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(text.Length));
            int encodedLength = Helpers.GetUtf8Bytes(text, arrayPoolArray);

            int[] encodedIds = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder);

            if (text.Length <= MaxWordLengthToCache)
            {
                string textAsString = text.ToString();
                _cache.Add(textAsString, (encodedIds, textAsString));
            }

            int result;
            if (encodedIds.Length <= maxTokens)
            {
                accumulatedIds.AddRange(encodedIds);
                textLength = text.Length;
                result = encodedIds.Length;
            }
            else
            {
                textLength = 0;
                result = 0;
            }

            ArrayPool<byte>.Shared.Return(arrayPoolArray);
            return result;
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        public override int CountTokens(ReadOnlySpan<char> text, out int textLength, int maxTokens = int.MaxValue)
        {
            Debug.Assert(maxTokens > 0);

            if (text.IsEmpty)
            {
                textLength = 0;
                return 0;
            }

            if (_cache.TryGetValue(text, out (int[] Ids, string Token) value))
            {
                if (value.Ids.Length <= maxTokens)
                {
                    textLength = text.Length;
                    return value.Ids.Length;
                }

                textLength = 0;
                return 0;
            }

            if (_vocab.TryGetValue(text, out _))
            {
                textLength = text.Length;
                return 1;
            }

            byte[] arrayPoolArray = ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(text.Length));
            int encodedLength = Helpers.GetUtf8Bytes(text, arrayPoolArray);

            int[] encodedIds = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder);
            if (text.Length <= MaxWordLengthToCache)
            {
                string textAsString = text.ToString();
                _cache.Add(textAsString, (encodedIds, textAsString));
            }

            int result;
            if (encodedIds.Length <= maxTokens)
            {
                textLength = text.Length;
                result = encodedIds.Length;
            }
            else
            {
                textLength = 0;
                result = 0;
            }

            ArrayPool<byte>.Shared.Return(arrayPoolArray);
            return result;
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textIndex">Starting from this index to the end of the text will encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        public override int CountTokensFromEnd(ReadOnlySpan<char> text, out int textIndex, int maxTokens = int.MaxValue)
        {
            Debug.Assert(maxTokens > 0);

            if (text.IsEmpty)
            {
                textIndex = 0;
                return 0;
            }

            if (_cache.TryGetValue(text, out (int[] Ids, string Token) value))
            {
                if (value.Ids.Length <= maxTokens)
                {
                    textIndex = 0;
                    return value.Ids.Length;
                }

                textIndex = text.Length;
                return 0;
            }

            if (_vocab.TryGetValue(text, out _))
            {
                textIndex = 0;
                return 1;
            }

            byte[] arrayPoolArray = ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(text.Length));
            int encodedLength = Helpers.GetUtf8Bytes(text, arrayPoolArray);

            int[] encodedIds = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder);

            if (text.Length <= MaxWordLengthToCache)
            {
                string textAsString = text.ToString();
                _cache.Add(textAsString, (encodedIds, textAsString));
            }

            int result;
            if (encodedIds.Length <= maxTokens)
            {
                textIndex = 0;
                result = encodedIds.Length;
            }
            else
            {
                textIndex = text.Length;
                result = 0;
            }

            ArrayPool<byte>.Shared.Return(arrayPoolArray);
            return result;
        }

        /// <summary>
        /// Map the token to encoded Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <returns>The mapped Id of the token.</returns>
        public override int? MapTokenToId(ReadOnlySpan<char> token)
        {
            if (token.IsEmpty)
            {
                return 0;
            }

            if (_cache.TryGetValue(token, out (int[] Ids, string Token) value))
            {
                if (value.Ids.Length == 1)
                {
                    return value.Ids[0];
                }

                return null;
            }

            if (_vocab.TryGetValue(token, out (int Id, string Token) id))
            {
                return id.Id;
            }

            byte[] arrayPoolArray = ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(token.Length));
            try
            {
                int encodedLength = Helpers.GetUtf8Bytes(token, arrayPoolArray);

                int[] idsToCache = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder);

                if (token.Length <= MaxWordLengthToCache)
                {
                    string tokenAsString = token.ToString();
                    _cache.Add(tokenAsString, (idsToCache, tokenAsString));
                }

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
        /// <returns>The mapped token of the Id.</returns>
        public override string? MapIdToToken(int id)
        {
            if (_decoder.TryGetValue(id, out ReadOnlyMemory<byte> tokenBytes))
            {
                return Helpers.GetString(tokenBytes.Span);
            }

            return null;
        }

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="decoder">The optional Decoder to merge the given list of tokens in a string.</param>
        /// <returns>The decoded string.</returns>
        public override string? Decode(IEnumerable<int> ids, TokenizerDecoder? decoder = null)
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
                    else
                    {
                        return null;
                    }
                }

                return Helpers.GetString(utf8Bytes.Slice(0, utf8ByteCount));
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
        public IReadOnlyDictionary<string, int> Vocab => _vocabOriginal ??= _vocab.ToDictionary(kvp => kvp.Key.Data!, kvp => kvp.Value.Id);

        /// <summary>
        /// Gets the dictionary mapping special tokens to Ids.
        /// </summary>
        public IReadOnlyDictionary<string, int>? SpecialTokens { get; }

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

        internal enum ModelEncoding
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
                                                                ( "gpt-4-", ModelEncoding.Cl100kBase),          // e.g., gpt-4-0314, etc., plus gpt-4-32k
                                                                ( "gpt-3.5-turbo-", ModelEncoding.Cl100kBase),  // e.g, gpt-3.5-turbo-0301, -0401, etc.
                                                                ( "gpt-35-turbo-", ModelEncoding.Cl100kBase )   // Azure deployment name
                                                            ];

        private static readonly Dictionary<string, ModelEncoding> _modelToEncoding =
                                                            new Dictionary<string, ModelEncoding>(StringComparer.OrdinalIgnoreCase)
                                                            {
                                                                // chat
                                                                { "gpt-4", ModelEncoding.Cl100kBase },
                                                                { "gpt-3.5-turbo", ModelEncoding.Cl100kBase },
                                                                { "gpt-3.5-turbo-16k", ModelEncoding.Cl100kBase },
                                                                { "gpt-35-turbo", ModelEncoding.Cl100kBase },     // Azure deployment name
                                                                { "gpt-35-turbo-16k", ModelEncoding.Cl100kBase }, // Azure deployment name

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

        internal static ModelEncoding GetModelEncoding(string modelName)
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

        internal static (Dictionary<string, int> SpecialTokens, Regex Regex, string VocabFile) GetTiktokenConfigurations(string modelName) => GetTiktokenConfigurations(GetModelEncoding(modelName), modelName);

        internal static (Dictionary<string, int> SpecialTokens, Regex Regex, string VocabFile) GetTiktokenConfigurations(ModelEncoding modelEncoding, string? modelName = null)
        {
            switch (modelEncoding)
            {
                case ModelEncoding.Cl100kBase:
                    return (new Dictionary<string, int>
                        { { EndOfText, 100257}, { FimPrefix, 100258}, { FimMiddle, 100259}, { FimSuffix, 100260}, { EndOfPrompt, 100276} }, Cl100kBaseRegex(), Cl100kBaseVocabFile);

                case ModelEncoding.P50kBase:
                    return (new Dictionary<string, int> { { EndOfText, 50256 } }, P50kBaseRegex(), P50RanksFile);

                case ModelEncoding.P50kEdit:
                    return (new Dictionary<string, int>
                        { { EndOfText, 50256 }, { FimPrefix, 50281 }, { FimMiddle, 50282 }, { FimSuffix, 50283 } }, P50kBaseRegex(), P50RanksFile);

                case ModelEncoding.R50kBase:
                    return (new Dictionary<string, int> { { EndOfText, 50256 } }, P50kBaseRegex(), R50RanksFile);

                case ModelEncoding.GPT2:
                    return (new Dictionary<string, int> { { EndOfText, 50256 }, }, P50kBaseRegex(), GPT2File);

                default:
                    throw new NotSupportedException($"The model '{modelName ?? modelEncoding.ToString()}' is not supported.");
            }
        }

        // Regex patterns based on https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py

        private const string Cl100kBaseRegexPattern = /*lang=regex*/ @"'(?i:[sdmt]|re|ve|ll)|(?>[^\r\n\p{L}\p{N}]?)\p{L}+|\p{N}{1,3}| ?(?>[^\s\p{L}\p{N}]+)[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
        private const string P50kBaseRegexPattern = /*lang=regex*/ @"'(?:[sdmt]|re|ve|ll)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

        private const string Cl100kBaseVocabFile = "cl100k_base.tiktoken.deflate";  // "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        private const string P50RanksFile = "p50k_base.tiktoken.deflate";           // "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken"
        private const string R50RanksFile = "r50k_base.tiktoken.deflate";           // "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"
        private const string GPT2File = "gpt2.tiktoken.deflate";                    // "https://pythia.blob.core.windows.net/public/encoding/gpt2.tiktoken"

        internal const string Cl100kBaseEncodingName = "cl100k_base";
        internal const string P50kBaseEncodingName = "p50k_base";
        internal const string P50kEditEncodingName = "p50k_edit";
        internal const string R50kBaseEncodingName = "r50k_base";

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

        private static readonly ConcurrentDictionary<string, (Dictionary<ReadOnlyMemory<byte>, int> encoder, Dictionary<StringSpanOrdinalKey, (int Id, string Token)> vocab, Dictionary<int, ReadOnlyMemory<byte>> decoder)> _tiktokenCache = new(StringComparer.OrdinalIgnoreCase);

        internal static Tokenizer CreateTokenizerForModel(
                                                string modelName,
                                                IReadOnlyDictionary<string, int>? extraSpecialTokens = null,
                                                Normalizer? normalizer = null)
        {
            if (string.IsNullOrEmpty(modelName))
            {
                throw new ArgumentNullException(nameof(modelName));
            }

            return CreateTokenizerForModel(GetModelEncoding(modelName), modelName, extraSpecialTokens, normalizer);
        }

        internal static Tokenizer CreateTokenizerForModel(
                                                ModelEncoding modelEncoding,
                                                string? modelName = null,
                                                IReadOnlyDictionary<string, int>? extraSpecialTokens = null,
                                                Normalizer? normalizer = null)
        {
            (Dictionary<string, int> SpecialTokens, Regex Regex, string VocabFile) tiktokenConfiguration = Tiktoken.GetTiktokenConfigurations(modelEncoding, modelName);

            if (extraSpecialTokens is not null)
            {
                foreach (var extraSpecialToken in extraSpecialTokens)
                {
                    tiktokenConfiguration.SpecialTokens.Add(extraSpecialToken.Key, extraSpecialToken.Value);
                }
            }

            if (!_tiktokenCache.TryGetValue(
                    tiktokenConfiguration.VocabFile,
                    out (Dictionary<ReadOnlyMemory<byte>, int> encoder, Dictionary<StringSpanOrdinalKey, (int Id, string Token)> vocab, Dictionary<int, ReadOnlyMemory<byte>> decoder) cache))
            {
                using Stream compressedStream = typeof(Tokenizer).Assembly.GetManifestResourceStream(tiktokenConfiguration.VocabFile)!;
                using Stream deflateStream = new DeflateStream(compressedStream, CompressionMode.Decompress);

                cache = LoadTiktokenBpeAsync(deflateStream, useAsync: false).GetAwaiter().GetResult();

                _tiktokenCache.TryAdd(tiktokenConfiguration.VocabFile, cache);
            }

            return new Tokenizer(
                            new Tiktoken(cache.encoder, cache.decoder, cache.vocab, tiktokenConfiguration.SpecialTokens, LruCache<int[]>.DefaultCacheSize),
                            new TiktokenPreTokenizer(tiktokenConfiguration.Regex, tiktokenConfiguration.SpecialTokens),
                            normalizer);
        }
    }
}
