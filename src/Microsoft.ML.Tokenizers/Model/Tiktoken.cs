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
        private readonly Dictionary<ReadOnlyMemory<byte>, int> _encoder;
        private readonly IReadOnlyDictionary<int, byte[]> _decoder;
        private readonly LruCache<int[]> _cache;
        private readonly Dictionary<StringSpanOrdinalKey, int>? _specialTokensEncoder;
        private readonly Dictionary<int, string>? _specialTokensDecoder;
        private readonly IReadOnlyDictionary<string, int> _vocabOriginal;
        private readonly Dictionary<StringSpanOrdinalKey, int> _vocab;

        /// <summary>
        /// Create a new Tiktoken tokenizer object.
        /// </summary>
        /// <param name="tikTokenBpeFile">The path to the BPE rank file.</param>
        /// <param name="specialTokensEncoder">The dictionary mapping special tokens to Ids.</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="tikTokenBpeFile"/> is null or empty.</exception>
        /// <exception cref="InvalidOperationException">Thrown when failed to load the BPE rank file.</exception>
        public Tiktoken(string tikTokenBpeFile, IReadOnlyDictionary<string, int>? specialTokensEncoder = null, int cacheSize = LruCache<int[]>.DefaultCacheSize) :
            this(string.IsNullOrEmpty(tikTokenBpeFile) ? throw new ArgumentNullException(nameof(tikTokenBpeFile)) : File.OpenRead(tikTokenBpeFile), specialTokensEncoder, cacheSize, disposeStream: true)
        {
        }

        /// <summary>
        /// Create a new Tiktoken tokenizer object.
        /// </summary>
        /// <param name="tikTokenBpeFileStream">The stream to the BPE rank file.</param>
        /// <param name="specialTokensEncoder">The dictionary mapping special tokens to Ids.</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="tikTokenBpeFileStream"/> is null or empty.</exception>
        /// <exception cref="InvalidOperationException">Thrown when failed to load the BPE rank file.</exception>
        public Tiktoken(Stream tikTokenBpeFileStream, IReadOnlyDictionary<string, int>? specialTokensEncoder = null, int cacheSize = LruCache<int[]>.DefaultCacheSize) :
            this(tikTokenBpeFileStream ?? throw new ArgumentNullException(nameof(tikTokenBpeFileStream)), specialTokensEncoder, cacheSize, disposeStream: false)
        {
        }

        internal Tiktoken(
            Dictionary<ReadOnlyMemory<byte>, int> encoder,
            Dictionary<int, byte[]> decoder,
            Dictionary<StringSpanOrdinalKey, int> vocab,
            IReadOnlyDictionary<string, int>? specialTokensEncoder = null,
            int cacheSize = LruCache<int[]>.DefaultCacheSize)
        {
            Debug.Assert(encoder is not null);
            Debug.Assert(decoder is not null);
            Debug.Assert(vocab is not null);

            _encoder = encoder!;
            _decoder = decoder!;
            _vocab = vocab!;
            _cache = new LruCache<int[]>(cacheSize);

            _vocabOriginal = _vocab.ToDictionary(kvp => kvp.Key.Data!, kvp => kvp.Value);
            (_specialTokensEncoder, _specialTokensDecoder) = CreateEncoderDecoder(specialTokensEncoder);
        }

        private Tiktoken(Stream tikTokenBpeFileStream, IReadOnlyDictionary<string, int>? specialTokensEncoder, int cacheSize, bool disposeStream)
        {
            try
            {
                _cache = new LruCache<int[]>(cacheSize);

                (_encoder, _vocab, _decoder) = LoadTikTokenBpeAsync(tikTokenBpeFileStream, useAsync: false).GetAwaiter().GetResult();
                _vocabOriginal = _vocab.ToDictionary(kvp => kvp.Key.Data!, kvp => kvp.Value);
                (_specialTokensEncoder, _specialTokensDecoder) = CreateEncoderDecoder(specialTokensEncoder);
            }
            finally
            {
                if (disposeStream)
                {
                    tikTokenBpeFileStream.Dispose();
                }
            }
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
        /// Load BPE rank dictionary from a stream.
        /// </summary>
        /// <param name="tikTokenBpeFileStream">Stream to the BPE rank file</param>
        /// <param name="useAsync">Whether to perform I/O synchronously or asynchronously.</param>
        /// <param name="cancellationToken"><see cref="CancellationToken"/> used to request cancellation of the operation.</param>
        /// <returns>Map of byte[] to integer token id</returns>
        /// <exception cref="InvalidOperationException"></exception>
        internal static async ValueTask<(Dictionary<ReadOnlyMemory<byte>, int>, Dictionary<StringSpanOrdinalKey, int>, Dictionary<int, byte[]>)> LoadTikTokenBpeAsync(
            Stream tikTokenBpeFileStream, bool useAsync, CancellationToken cancellationToken = default)
        {
            var encoder = new Dictionary<ReadOnlyMemory<byte>, int>(ReadOnlyMemoryByteComparer.Instance);
            var vocab = new Dictionary<StringSpanOrdinalKey, int>();
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

                            vocab[new StringSpanOrdinalKey(decodedToken)] = rank;
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
        /// Tokenize a split sequence string to a list of tokens.
        /// </summary>
        /// <param name="sequence">The text to tokenize.</param>
        /// <param name="isSpecialToken">Indicate if the token is a special token.</param>
        /// <returns>The list of tokens generated from the sequence tokenization.</returns>
        public override IReadOnlyList<Token> Tokenize(string sequence, bool isSpecialToken)
        {
            Token[] tokens;

            if (string.IsNullOrEmpty(sequence))
            {
                return Array.Empty<Token>();
            }

            if (isSpecialToken)
            {
                if (_specialTokensEncoder is null)
                {
                    throw new InvalidOperationException($"The tokenizer doesn't have special tokens");
                }

                if (_specialTokensEncoder.TryGetValue(new StringSpanOrdinalKey(sequence), out int id))
                {
                    return new List<Token> { new(id, sequence, (0, sequence.Length)) };
                }

                throw new InvalidOperationException($"The special token {sequence} doesn't exist in the tokenizer");
            }

            if (_cache.TryGetValue(sequence, out int[]? ids))
            {
                tokens = new Token[ids.Length];
                tokens[0] = new Token(ids[0], sequence, (0, sequence.Length));
                for (int i = 1; i < ids.Length; i++)
                {
                    // One word split mapped to multiple Ids. Make the offset of the remaining token point at the end with zero width.
                    tokens[i] = new Token(ids[i], "", (sequence.Length, sequence.Length));
                }

                return tokens;
            }

            // cache miss
            if (_vocab.TryGetValue(new StringSpanOrdinalKey(sequence), out int mappedId))
            {
                return new Token[1] { new(mappedId, sequence, (0, sequence.Length)) };
            }

            byte[] arrayPoolArray = ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(sequence.Length));
            int encodedLength = GetUtf8Bytes(sequence.AsSpan(), arrayPoolArray);

            int[] encodedIds = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder);
            Debug.Assert(encodedIds.Length > 0);
            _cache.Add(sequence, encodedIds);

            tokens = new Token[encodedIds.Length];
            tokens[0] = new Token(encodedIds[0], sequence, (0, sequence.Length));
            for (int i = 1; i < encodedIds.Length; i++)
            {
                // One word split mapped to multiple Ids. Make the offset of the remaining token point at the end with zero width.
                tokens[i] = new Token(encodedIds[i], "", (sequence.Length, sequence.Length));
            }

            ArrayPool<byte>.Shared.Return(arrayPoolArray);
            return tokens;
        }

        /// <summary>
        /// Tokenize a split sequence string to a list of Ids.
        /// </summary>
        /// <param name="sequence">The sequence to tokenize.</param>
        /// <param name="isSpecialToken">Indicate if the token is a special token.</param>
        /// <param name="accumulatedIds">The list of accumulated Ids.</param>
        public override void TokenizeToIds(string sequence, bool isSpecialToken, IList<int> accumulatedIds) =>
            TokenizeToIds(sequence.AsSpan(), sequence, isSpecialToken, accumulatedIds);

        /// <summary>
        /// Tokenize a split sequence string to a list of Ids.
        /// </summary>
        /// <param name="sequence">The sequence to tokenize.</param>
        /// <param name="isSpecialToken">Indicate if the token is a special token.</param>
        /// <param name="accumulatedIds">The list of accumulated Ids.</param>
        public override void TokenizeToIds(ReadOnlySpan<char> sequence, bool isSpecialToken, IList<int> accumulatedIds) =>
            TokenizeToIds(sequence, null, isSpecialToken, accumulatedIds);

        private unsafe void TokenizeToIds(ReadOnlySpan<char> sequence, string? sequenceString, bool isSpecialToken, IList<int> accumulatedIds)
        {
            if (sequence.IsEmpty)
            {
                return;
            }

            if (isSpecialToken)
            {
                fixed (char* ptr = sequence)
                {
                    if (_specialTokensEncoder?.TryGetValue(new StringSpanOrdinalKey(ptr, sequence.Length), out int id) is true)
                    {
                        accumulatedIds.Add(id);
                    }
                }

                return;
            }

            if (_cache.TryGetValue(sequence, out int[]? tokenIds))
            {
                accumulatedIds.AddRange(tokenIds);
                return;
            }

            fixed (char* ptr = sequence)
            {
                if (_vocab.TryGetValue(new StringSpanOrdinalKey(ptr, sequence.Length), out int mappedId))
                {
                    accumulatedIds.Add(mappedId);
                    return;
                }
            }

            byte[] arrayPoolArray = ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(sequence.Length));
            int encodedLength = GetUtf8Bytes(sequence, arrayPoolArray);

            int[] encodedIds = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder);
            _cache.Add(sequenceString ?? sequence.ToString(), encodedIds);

            accumulatedIds.AddRange(encodedIds);

            ArrayPool<byte>.Shared.Return(arrayPoolArray);
            return;
        }

        /// <summary>
        /// Get the number of tokens that the input sequence will be encoded to.
        /// </summary>
        /// <param name="sequence">The text to tokenize.</param>
        /// <param name="isSpecialToken">Indicate if the token is special token.</param>
        /// <returns>The number of tokens that the input sequence will be encoded to.</returns>
        public override int CountTokens(string sequence, bool isSpecialToken) =>
            CountTokens(sequence.AsSpan(), sequence, isSpecialToken);

        /// <summary>
        /// Get the number of tokens that the input sequence will be encoded to.
        /// </summary>
        /// <param name="sequence">The text to tokenize.</param>
        /// <param name="isSpecialToken">Indicate if the token is special token.</param>
        /// <returns>The number of tokens that the input sequence will be encoded to.</returns>
        public override int CountTokens(ReadOnlySpan<char> sequence, bool isSpecialToken) =>
            CountTokens(sequence, null, isSpecialToken);

        private unsafe int CountTokens(ReadOnlySpan<char> sequence, string? sequenceString, bool isSpecialToken)
        {
            if (sequence.IsEmpty)
            {
                return 0;
            }

            if (isSpecialToken && _specialTokensEncoder is not null)
            {
                fixed (char* ptr = sequence)
                {
                    return _specialTokensEncoder.TryGetValue(new StringSpanOrdinalKey(ptr, sequence.Length), out _) ? 1 : 0;
                }
            }

            if (_cache.TryGetValue(sequence, out int[] ids))
            {
                return ids.Length;
            }

            fixed (char* ptr = sequence)
            {
                if (_vocab.TryGetValue(new StringSpanOrdinalKey(ptr, sequence.Length), out _))
                {
                    return 1;
                }
            }

            byte[] arrayPoolArray = ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(sequence.Length));
            int encodedLength = GetUtf8Bytes(sequence, arrayPoolArray);

            int[] encodedIds = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder);
            _cache.Add(sequenceString ?? sequence.ToString(), encodedIds);

            ArrayPool<byte>.Shared.Return(arrayPoolArray);
            return encodedIds.Length;
        }

        public override int? TokenToId(ReadOnlySpan<char> token) => TokenToId(token, skipSpecialTokens: false);
        public override int? TokenToId(string token) => TokenToId(token, skipSpecialTokens: false);

        /// <summary>
        /// Map the token to tokenized Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <param name="skipSpecialTokens">Indicate if want to skip the special tokens during the encoding.</param>
        /// <returns>The mapped Id of the token.</returns>
        public override int? TokenToId(string token, bool skipSpecialTokens) =>
            TokenToId(token.AsSpan(), token, skipSpecialTokens);

        /// <summary>
        /// Map the token to tokenized Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <param name="skipSpecialTokens">Indicate if want to skip the special tokens during the encoding.</param>
        /// <returns>The mapped Id of the token.</returns>
        public override unsafe int? TokenToId(ReadOnlySpan<char> token, bool skipSpecialTokens) =>
            TokenToId(token, null, skipSpecialTokens);

        private unsafe int? TokenToId(ReadOnlySpan<char> token, string? tokenString, bool skipSpecialTokens)
        {
            if (token.IsEmpty)
            {
                return 0;
            }

            if (!skipSpecialTokens && _specialTokensEncoder is not null)
            {
                fixed (char* ptr = token)
                {
                    if (_specialTokensEncoder.TryGetValue(new StringSpanOrdinalKey(ptr, token.Length), out int specialTokenId))
                    {
                        return specialTokenId;
                    }
                }
            }

            if (_cache.TryGetValue(token, out int[]? ids))
            {
                if (ids.Length == 1)
                {
                    return ids[0];
                }

                return null;
            }

            fixed (char* ptr = token)
            {
                if (_vocab.TryGetValue(new StringSpanOrdinalKey(ptr, token.Length), out int id))
                {
                    return id;
                }
            }

            byte[] arrayPoolArray = ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(token.Length));
            try
            {
                int encodedLength = GetUtf8Bytes(token, arrayPoolArray);

                int[] idsToCache = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder);
                _cache.Add(tokenString ?? token.ToString(), idsToCache);

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
        /// Map the tokenized Id to the token.
        /// </summary>
        /// <param name="id">The Id to map to the token.</param>
        /// <param name="skipSpecialTokens">Indicate if want to skip the special tokens during the decoding.</param>
        /// <returns>The mapped token of the Id.</returns>
        public override string? IdToToken(int id, bool skipSpecialTokens = false)
        {
            if (!skipSpecialTokens && _specialTokensDecoder is not null && _specialTokensDecoder.TryGetValue(id, out string? token))
            {
                return token;
            }

            if (_decoder.TryGetValue(id, out byte[]? tokenBytes))
            {
                return Encoding.UTF8.GetString(tokenBytes);
            }

            return null;
        }

        internal string? IdsToString(IEnumerable<int> ids, bool skipSpecialTokens = false)
        {
            if (ids is null)
            {
                return null;
            }

            byte[]? arrayPoolArray = null;
            try
            {
                Span<byte> utf8Bytes = stackalloc byte[256];
                int utf8ByteCount = 0;

                bool useSpecialTokens = !skipSpecialTokens && _specialTokensDecoder is not null;

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
        public override IReadOnlyDictionary<string, int> GetVocab() => _vocabOriginal;

        /// <summary>
        /// Gets the dictionary size that map tokens to Ids.
        /// </summary>
        public override int GetVocabSize() => _vocabOriginal.Count;

        /// <summary>
        /// Save the model data into the vocabulary and merges files.
        /// </summary>
        /// <param name="path">The file system path to store the generated files at.</param>
        /// <param name="prefix">Optional prefix for the generated file names.</param>
        /// <returns>The list of all saved files.</returns>
        public override string[] Save(string path, string? prefix = null) => throw new NotImplementedException();

        /// <summary>
        /// Gets a trainer object to use in training the model.
        /// </summary>
        public override Trainer? GetTrainer() => throw new NotImplementedException();

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
