// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Represent the rapid Byte Pair Encoding model commonly referred to as Tiktoken.
    /// </summary>
    public sealed class Tiktoken : Model
    {
        private readonly IReadOnlyDictionary<ReadOnlyMemory<byte>, int> _encoder = null!;
        private readonly IReadOnlyDictionary<int, byte[]> _decoder = null!;
        private readonly LruCache<string, int[]>? _cache;
        private readonly IReadOnlyDictionary<string, int>? _specialTokensEncoder;
        private readonly Dictionary<int, string>? _specialTokensDecoder;
        private readonly IReadOnlyDictionary<string, int> _vocab = null!;

        /// <summary>
        /// Create a new Tiktoken tokenizer's model object.
        /// </summary>
        /// <param name="tikTokenBpeFile">The path to the BPE rank file.</param>
        /// <param name="specialTokensEncoder">The dictionary mapping special tokens to Ids.</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="tikTokenBpeFile"/> is null or empty.</exception>
        /// <exception cref="InvalidOperationException">Thrown when failed to load the BPE rank file.</exception>
        public Tiktoken(string tikTokenBpeFile, IReadOnlyDictionary<string, int>? specialTokensEncoder = null, int cacheSize = LruCache<string, int[]>.DefaultCacheSize) :
            this(string.IsNullOrEmpty(tikTokenBpeFile) ? throw new ArgumentNullException(nameof(tikTokenBpeFile)) : File.OpenRead(tikTokenBpeFile), specialTokensEncoder, cacheSize, disposeStream: true)
        {
        }

        /// <summary>
        /// Create a new Tiktoken tokenizer's model object.
        /// </summary>
        /// <param name="tikTokenBpeFileStream">The stream to the BPE rank file.</param>
        /// <param name="specialTokensEncoder">The dictionary mapping special tokens to Ids.</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="tikTokenBpeFileStream"/> is null or empty.</exception>
        /// <exception cref="InvalidOperationException">Thrown when failed to load the BPE rank file.</exception>
        public Tiktoken(Stream tikTokenBpeFileStream, IReadOnlyDictionary<string, int>? specialTokensEncoder = null, int cacheSize = LruCache<string, int[]>.DefaultCacheSize) :
            this(tikTokenBpeFileStream ?? throw new ArgumentNullException(nameof(tikTokenBpeFileStream)), specialTokensEncoder, cacheSize, disposeStream: false)
        {
        }

        /// <summary>
        /// Create a new Tiktoken tokenizer's model object.
        /// </summary>
        /// <param name="encoder">The dictionary mapping token utf-8 bytes to Ids.</param>
        /// <param name="decoder">The dictionary mapping Ids to token utf-8 bytes.</param>
        /// <param name="vocab">The dictionary mapping string tokens to Ids.</param>
        /// <param name="specialTokensEncoder">The dictionary mapping special tokens to Ids.</param>
        /// <param name="cacheSize">The max size of the cache to use.</param>
        public Tiktoken(
            IReadOnlyDictionary<ReadOnlyMemory<byte>, int> encoder,
            IReadOnlyDictionary<int, byte[]>? decoder,
            IReadOnlyDictionary<string, int>? vocab,
            IReadOnlyDictionary<string, int>? specialTokensEncoder = null,
            int cacheSize = LruCache<string, int[]>.DefaultCacheSize) : this(cacheSize)
        {
            if (encoder is null)
            {
                throw new ArgumentNullException(nameof(encoder));
            }

            _encoder = encoder;

            if (decoder is null)
            {
                decoder = encoder.ToDictionary(kvp => kvp.Value, kvp => kvp.Key.ToArray());
            }

            if (encoder.Count != decoder.Count)
            {
                throw new ArgumentException("The encoder and decoder dictionaries must have the same number of elements");
            }

            _decoder = decoder!;

            if (vocab is null)
            {
                var vocab1 = new Dictionary<string, int>();
                foreach (var kvp in encoder)
                {
                    string s = Encoding.UTF8.GetString(kvp.Key.ToArray());

                    // Don't add mal-formed utf8 converted bytes to the vocab.
                    if (!StringContainInvalidChars(s))
                    {
                        vocab1[s] = kvp.Value;
                    }
                }
                vocab = vocab1;
            }

            _vocab = vocab;

            _specialTokensEncoder = specialTokensEncoder;
            if (_specialTokensEncoder is not null)
            {
                _specialTokensDecoder = _specialTokensEncoder.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
            }
        }

        private Tiktoken(Stream tikTokenBpeFileStream, IReadOnlyDictionary<string, int>? specialTokensEncoder, int cacheSize, bool disposeStream) : this(cacheSize)
        {
            try
            {
                (_encoder, _vocab, _decoder) = LoadTikTokenBpeAsync(tikTokenBpeFileStream, useAsync: false).GetAwaiter().GetResult();

                _specialTokensEncoder = specialTokensEncoder;
                if (_specialTokensEncoder is not null)
                {
                    _specialTokensDecoder = _specialTokensEncoder.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
                }
            }
            finally
            {
                if (disposeStream)
                {
                    tikTokenBpeFileStream.Dispose();
                }
            }
        }

        private Tiktoken(int cacheSize)
        {
            if (cacheSize < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(cacheSize));
            }

            if (cacheSize > 0)
            {
                _cache = new LruCache<string, int[]>(cacheSize);
            }
        }

        /// <summary>
        /// Create a new Tiktoken tokenizer's model object asynchronously.
        /// </summary>
        /// <param name="tikTokenBpeFileStream">The stream to the BPE rank file.</param>
        /// <param name="specialTokensEncoder">The dictionary mapping special tokens to Ids.</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <param name="cancellationToken"><see cref="CancellationToken"/> used to request cancellation of the operation.</param>
        /// <returns>Tiktoken tokenizer's object.</returns>
        public static async Task<Tiktoken> CreateAsync(
                            Stream tikTokenBpeFileStream,
                            IReadOnlyDictionary<string, int>? specialTokensEncoder = null,
                            int cacheSize = LruCache<string, int[]>.DefaultCacheSize,
                            CancellationToken cancellationToken = default)
        {
            if (tikTokenBpeFileStream is null)
            {
                throw new ArgumentNullException(nameof(tikTokenBpeFileStream));
            }

            (IReadOnlyDictionary<ReadOnlyMemory<byte>, int> encoder, Dictionary<string, int> vocab, IReadOnlyDictionary<int, byte[]> decoder) =
                        await LoadTikTokenBpeAsync(tikTokenBpeFileStream, useAsync: true, cancellationToken).ConfigureAwait(false);

            return new Tiktoken(encoder, decoder, vocab, specialTokensEncoder, cacheSize);
        }

        /// <summary>
        /// Create a new Tiktoken tokenizer's object asynchronously.
        /// </summary>
        /// <param name="tikTokenBpeFile">The BPE rank file.</param>
        /// <param name="specialTokensEncoder">The dictionary mapping special tokens to Ids.</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <param name="cancellationToken"><see cref="CancellationToken"/> used to request cancellation of the operation.</param>
        /// <returns>Tiktoken tokenizer's model object.</returns>
        public static async Task<Tiktoken> CreateAsync(
                                string tikTokenBpeFile,
                                IReadOnlyDictionary<string, int>? specialTokensEncoder = null,
                                int cacheSize = LruCache<string, int[]>.DefaultCacheSize,
                                CancellationToken cancellationToken = default)
        {
            if (tikTokenBpeFile is null)
            {
                throw new ArgumentNullException(nameof(tikTokenBpeFile));
            }

            using Stream tikTokenBpeFileStream = File.OpenRead(tikTokenBpeFile);
            return await CreateAsync(tikTokenBpeFileStream, specialTokensEncoder, cacheSize, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Load BPE rank dictionary from a stream.
        /// </summary>
        /// <param name="tikTokenBpeFileStream">Stream to the BPE rank file</param>
        /// <param name="useAsync">Whether to perform I/O synchronously or asynchronously.</param>
        /// <param name="cancellationToken"><see cref="CancellationToken"/> used to request cancellation of the operation.</param>
        /// <returns>Map of byte[] to integer token id</returns>
        /// <exception cref="InvalidOperationException"></exception>
        internal static async ValueTask<(Dictionary<ReadOnlyMemory<byte>, int>, Dictionary<string, int>, IReadOnlyDictionary<int, byte[]>)> LoadTikTokenBpeAsync(
            Stream tikTokenBpeFileStream, bool useAsync, CancellationToken cancellationToken = default)
        {
            var encoder = new Dictionary<ReadOnlyMemory<byte>, int>(ReadOnlyMemoryByteComparer.Instance);
            var vocab = new Dictionary<string, int>();
            var decoder = new Dictionary<int, byte[]>();

            try
            {
                using (StreamReader reader = new StreamReader(tikTokenBpeFileStream))
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
                            throw new FormatException($"Invalid format in the BPE encoder file stream");
                        }

                        if (Helpers.TryParseInt32(line, spaceIndex + 1, out int rank))
                        {
                            byte[] tokenBytes = Helpers.FromBase64String(line, 0, spaceIndex);

                            encoder[tokenBytes] = rank;
                            decoder[rank] = tokenBytes;

                            string decodedToken = Encoding.UTF8.GetString(tokenBytes);

                            if (!StringContainInvalidChars(decodedToken))
                            {
                                vocab[decodedToken] = rank;
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
                throw new InvalidOperationException($"Failed to load from BPE encoder file stream: {ex.Message}", ex);
            }

            return (encoder, vocab, decoder);
        }

        /// <summary>
        /// Gets the dictionary mapping special tokens to Ids.
        /// </summary>
        /// <returns>The dictionary mapping special tokens to Ids.</returns>
        public IReadOnlyDictionary<string, int>? SpecialTokensEncoder => _specialTokensEncoder;

        /// <summary>
        /// Encode a split text string to a list of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="isSpecialToken">Indicate if the token is a special token.</param>
        /// <returns>The list of tokens generated from the text tokenization.</returns>
        public override IReadOnlyList<Token> Encode(string text, bool isSpecialToken)
        {
            Token[] tokens;

            if (string.IsNullOrEmpty(text))
            {
                return Array.Empty<Token>();
            }

            if (isSpecialToken)
            {
                if (_specialTokensEncoder is null)
                {
                    throw new InvalidOperationException($"The tokenizer doesn't have special tokens");
                }

                if (_specialTokensEncoder.TryGetValue(text, out int id))
                {
                    return new List<Token> { new(id, text, (0, text.Length)) };
                }

                throw new InvalidOperationException($"The special token {text} doesn't exist in the tokenizer");
            }

            if (_cache?.Lookup(text, out int[] ids) is true)
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
            _cache?.Add(text, encodedIds);

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
        /// Encode a split text string to a list of Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="isSpecialToken">Indicate if the token is a special token.</param>
        /// <param name="accumulatedIds">The list of accumulated Ids.</param>
        public override void EncodeToIds(string text, bool isSpecialToken, IList<int> accumulatedIds)
        {
            if (string.IsNullOrEmpty(text))
            {
                return;
            }

            if (isSpecialToken)
            {
                if (_specialTokensEncoder is not null && _specialTokensEncoder.TryGetValue(text, out int id))
                {
                    accumulatedIds.Add(id);
                }

                return;
            }

            if (_cache?.Lookup(text, out int[] tokenIds) is true)
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
            int encodedLength = GetUtf8Bytes(text.AsSpan(), arrayPoolArray);

            int[] encodedIds = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder);
            _cache?.Add(text, encodedIds);

            accumulatedIds.AddRange(encodedIds);

            ArrayPool<byte>.Shared.Return(arrayPoolArray);
            return;
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="isSpecialToken">Indicate if the token is special token.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        public override int CountTokens(string text, bool isSpecialToken)
        {
            if (string.IsNullOrEmpty(text))
            {
                return 0;
            }

            if (isSpecialToken && _specialTokensEncoder is not null)
            {
                return _specialTokensEncoder.TryGetValue(text, out _) ? 1 : 0;
            }

            if (_cache?.Lookup(text, out int[] ids) is true)
            {
                return ids.Length;
            }

            if (_vocab.TryGetValue(text, out _))
            {
                return 1;
            }

            byte[] arrayPoolArray = ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(text.Length));
            int encodedLength = GetUtf8Bytes(text.AsSpan(), arrayPoolArray);

            int[] encodedIds = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder);
            _cache?.Add(text, encodedIds);

            ArrayPool<byte>.Shared.Return(arrayPoolArray);
            return encodedIds.Length;
        }

        /// <summary>
        /// Map the token to encoded Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <param name="considerSpecialTokens">Indicate if want to consider the special tokens during the encoding.</param>
        /// <returns>The mapped Id of the token.</returns>
        public override int? TokenToId(string token, bool considerSpecialTokens = true)
        {
            if (string.IsNullOrEmpty(token))
            {
                return 0;
            }

            if (considerSpecialTokens && _specialTokensEncoder is not null && _specialTokensEncoder.TryGetValue(token, out int specialTokenId))
            {
                return specialTokenId;
            }

            if (_cache?.Lookup(token, out int[] ids) is true)
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
                int encodedLength = GetUtf8Bytes(token.AsSpan(), arrayPoolArray);

                int[] idsToCache = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder);
                _cache?.Add(token, idsToCache);

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
        /// <param name="filterUnsupportedChars">Indicate if want to filter the unsupported characters during the decoding.</param>
        /// <returns>The mapped token of the Id.</returns>
        public override string? IdToToken(int id, bool considerSpecialTokens = true, bool filterUnsupportedChars = true)
        {
            if (considerSpecialTokens && _specialTokensDecoder is not null && _specialTokensDecoder.TryGetValue(id, out string? token))
            {
                return token;
            }

            if (_decoder.TryGetValue(id, out byte[]? tokenBytes))
            {
                return Encoding.UTF8.GetString(tokenBytes);
            }

            return null;
        }

        /// <summary>
        /// Decode the given ids, back to a String.
        /// </summary>
        /// <param name="ids">The list of ids that we want to decode.</param>
        /// <param name="considerSpecialTokens">Whether the special tokens should be kept in the decoded string.</param>
        /// <param name="filterUnsupportedChars">Indicate if want to filter the unsupported characters during the decoding.</param>
        /// <param name="decoder">The optional Decoder to merge the given list of tokens in a string.</param>
        /// <returns>The decoded string.</returns>
        public override string? Decode(IEnumerable<int> ids, TokenizerDecoder? decoder = null, bool considerSpecialTokens = true, bool filterUnsupportedChars = true)
        {
            // Tiktoken does not ensure a one-to-one mapping between IDs and tokens. Consequently, decoding individual IDs into tokens is not supported;
            // instead, decoding all IDs must be done collectively.
            // Here is example of case that map one character to multiple Ids:
            // '⭐' U-2B50 is mapped to Ids [2928, 99834] in the Tiktoken model.
            // In other words, the character '⭐' has UTF-8 code point 0xE2, 0xAD, 0x90, Tiktoken will map 0xE2 to [2928] and 0xAD, 0x90 to [99834].

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
                    if (_decoder.TryGetValue(id, out byte[]? tokenBytes))
                    {
                        if ((uint)utf8ByteCount + (uint)tokenBytes.Length > (uint)utf8Bytes.Length)
                        {
                            ArrayPoolGrow(ref utf8Bytes, ref arrayPoolArray, utf8ByteCount + tokenBytes.Length);
                        }

                        tokenBytes.AsSpan().CopyTo(utf8Bytes.Slice(utf8ByteCount));
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
        public override IReadOnlyDictionary<string, int> GetVocab() => _vocab;

        /// <summary>
        /// Gets the dictionary mapping token utf-8 bytes to Ids.
        /// </summary>
        public IReadOnlyDictionary<ReadOnlyMemory<byte>, int> Encoder => _encoder;

        /// <summary>
        /// Gets the dictionary mapping Ids to token utf-8 bytes.
        /// </summary>
        public IReadOnlyDictionary<int, byte[]>? Decoder => _decoder;

        /// <summary>
        /// Gets the dictionary size that map tokens to Ids.
        /// </summary>
        public override int GetVocabSize() => _vocab.Count;

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

        private static bool StringContainInvalidChars(string text)
        {
            for (int i = 0; i < text.Length; i++)
            {
                if (text[i] == 0xFFFD)
                {
                    return true;
                }
            }

            return false;
        }
    }
}
