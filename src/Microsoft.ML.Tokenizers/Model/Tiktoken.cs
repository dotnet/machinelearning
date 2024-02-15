// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Represent the rapid Byte Pair Encoding model commonly referred to as Tiktoken.
    /// </summary>
    public sealed class Tiktoken : Model
    {
        private Dictionary<byte[], int> _encoder = null!;
        private IReadOnlyDictionary<int, byte[]> _decoder = null!;
        private readonly LruCache<string, int[]> _cache;
        private IReadOnlyDictionary<string, int>? _specialTokensEncoder;
        private Dictionary<int, string>? _specialTokensDecoder;

        private Dictionary<string, int> _vocab = null!;
        private static readonly List<Token> _emptyTokenList = new();

        /// <summary>
        /// Create a new Tiktoken tokenizer object.
        /// </summary>
        /// <param name="tikTokenBpeFile">The path to the BPE rank file.</param>
        /// <param name="specialTokensEncoder">The dictionary mapping special tokens to Ids.</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="tikTokenBpeFile"/> is null or empty.</exception>
        /// <exception cref="InvalidOperationException">Thrown when failed to load the BPE rank file.</exception>
        public Tiktoken(string tikTokenBpeFile, IReadOnlyDictionary<string, int>? specialTokensEncoder = null, int cacheSize = LruCache<string, int[]>.DefaultCacheSize) : this(cacheSize)
        {
            if (string.IsNullOrEmpty(tikTokenBpeFile))
            {
                throw new ArgumentNullException(nameof(tikTokenBpeFile));
            }

            using (Stream stream = File.OpenRead(tikTokenBpeFile))
            {
                Initialize(stream, specialTokensEncoder);
            }
        }

        /// <summary>
        /// Create a new Tiktoken tokenizer object.
        /// </summary>
        /// <param name="tikTokenBpeFileStream">The stream to the BPE rank file.</param>
        /// <param name="specialTokensEncoder">The dictionary mapping special tokens to Ids.</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="tikTokenBpeFileStream"/> is null or empty.</exception>
        /// <exception cref="InvalidOperationException">Thrown when failed to load the BPE rank file.</exception>
        public Tiktoken(Stream tikTokenBpeFileStream, IReadOnlyDictionary<string, int>? specialTokensEncoder = null, int cacheSize = LruCache<string, int[]>.DefaultCacheSize) : this(cacheSize)
        {
            Initialize(tikTokenBpeFileStream, specialTokensEncoder);
        }

        internal Tiktoken(
                    Dictionary<byte[], int> encoder,
                    IReadOnlyDictionary<int, byte[]> decoder,
                    Dictionary<string, int> vocab,
                    IReadOnlyDictionary<string, int>? specialTokensEncoder = null,
                    int cacheSize = LruCache<string, int[]>.DefaultCacheSize) : this(cacheSize)
        {
            Debug.Assert(encoder is not null);
            Debug.Assert(decoder is not null);
            Debug.Assert(vocab is not null);

            _encoder = encoder!;
            _vocab = vocab!;
            _decoder = decoder!;

            _specialTokensEncoder = specialTokensEncoder;
            if (_specialTokensEncoder is not null)
            {
                _specialTokensDecoder = _specialTokensEncoder.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
            }
        }

        private Tiktoken(int cacheSize)
        {
            _cache = new LruCache<string, int[]>(cacheSize);
        }

        private void Initialize(Stream tikTokenBpeFileStream, IReadOnlyDictionary<string, int>? specialTokensEncoder = null)
        {
            if (tikTokenBpeFileStream is null)
            {
                throw new ArgumentNullException(nameof(tikTokenBpeFileStream));
            }

            (_encoder, _vocab, _decoder) = LoadTikTokenBpe(tikTokenBpeFileStream);

            _specialTokensEncoder = specialTokensEncoder;
            if (_specialTokensEncoder is not null)
            {
                _specialTokensDecoder = _specialTokensEncoder.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
            }
        }

        /// <summary>
        /// Load BPE rank dictionary from a stream.
        /// </summary>
        /// <param name="tikTokenBpeFileStream">Stream to the BPE rank file</param>
        /// <returns>Map of byte[] to integer token id</returns>
        /// <exception cref="InvalidOperationException"></exception>
        internal static (Dictionary<byte[], int>, Dictionary<string, int>, IReadOnlyDictionary<int, byte[]>) LoadTikTokenBpe(Stream tikTokenBpeFileStream)
        {
            var encoder = new Dictionary<byte[], int>(new ByteArrayComparer());
            var vocab = new Dictionary<string, int>();
            var decoder = new Dictionary<int, byte[]>();

            try
            {
                using (StreamReader reader = new StreamReader(tikTokenBpeFileStream))
                {
                    while (!reader.EndOfStream)
                    {
                        string? line = reader.ReadLine();
                        if (string.IsNullOrWhiteSpace(line))
                        {
                            continue;
                        }

                        int spaceIndex = line.IndexOf(' ');
                        if (spaceIndex <= 0 || spaceIndex >= line.Length - 1 || line.IndexOf(' ', spaceIndex + 1) >= 0)
                        {
                            throw new FormatException($"Invalid format in the BPE encoder file stream");
                        }

                        byte[] tokenBytes = Helpers.FromBase64String(line, 0, spaceIndex);

                        if (Helpers.TryParseInt32(line, spaceIndex + 1, out int rank))
                        {
                            encoder[tokenBytes] = rank;
                            decoder[rank] = tokenBytes;

                            string decodedToken = Encoding.UTF8.GetString(tokenBytes);

                            vocab[decodedToken] = rank;
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
        /// Tokenize a split sequence string to a list of tokens.
        /// </summary>
        /// <param name="sequence">The text to tokenize.</param>
        /// <param name="isSpecialToken">Indicate if the token is a special token.</param>
        /// <returns>The list of tokens generated from the sequence tokenization.</returns>
        public override IReadOnlyList<Token> Tokenize(string sequence, bool isSpecialToken)
        {
            List<Token> tokens;

            if (string.IsNullOrEmpty(sequence))
            {
                return _emptyTokenList;
            }

            if (isSpecialToken)
            {
                if (_specialTokensEncoder is null)
                {
                    throw new InvalidOperationException($"The tokenizer doesn't have special tokens");
                }

                if (_specialTokensEncoder.TryGetValue(sequence, out int id))
                {
                    return new List<Token> { new(id, sequence, (0, sequence.Length)) };
                }

                throw new InvalidOperationException($"The special token {sequence} doesn't exist in the tokenizer");
            }

            if (_cache.Lookup(sequence, out int[] ids))
            {
                tokens = new(ids.Length);
                tokens.Add(new Token(ids[0], sequence, (0, sequence.Length)));
                for (int i = 1; i < ids.Length; i++)
                {
                    // One word split mapped to multiple Ids. Make the offset of the remaining token point at the end with zero width.
                    tokens.Add(new Token(ids[i], "", (sequence.Length, sequence.Length)));
                }

                return tokens;
            }

            // cache miss
            if (_vocab.TryGetValue(sequence, out int mappedId))
            {
                return new List<Token> { new(mappedId, sequence, (0, sequence.Length)) };
            }

            int[] encodedIds = BytePairEncoder.BytePairEncode(Encoding.UTF8.GetBytes(sequence), _encoder);
            _cache.Add(sequence, encodedIds);

            tokens = new List<Token>(encodedIds.Length);
            tokens.Add(new Token(encodedIds[0], sequence, (0, sequence.Length)));
            for (int i = 1; i < encodedIds.Length; i++)
            {
                // One word split mapped to multiple Ids. Make the offset of the remaining token point at the end with zero width.
                tokens.Add(new Token(encodedIds[i], "", (sequence.Length, sequence.Length)));
            }

            return tokens;
        }

        /// <summary>
        /// Tokenize a split sequence string to a list of Ids.
        /// </summary>
        /// <param name="sequence">The sequence to tokenize.</param>
        /// <param name="isSpecialToken">Indicate if the token is a special token.</param>
        /// <param name="accumulatedIds">The list of accumulated Ids.</param>
        public override void TokenizeToIds(string sequence, bool isSpecialToken, IList<int> accumulatedIds)
        {
            if (string.IsNullOrEmpty(sequence))
            {
                return;
            }

            if (isSpecialToken)
            {
                if (_specialTokensEncoder is not null && _specialTokensEncoder.TryGetValue(sequence, out int id))
                {
                    accumulatedIds.Add(id);
                }

                return;
            }

            if (_cache.Lookup(sequence, out int[] tokenIds))
            {
                accumulatedIds.AddRange(tokenIds);
                return;
            }

            if (_vocab.TryGetValue(sequence, out int mappedId))
            {
                accumulatedIds.Add(mappedId);
                return;
            }

            int[] encodedIds = BytePairEncoder.BytePairEncode(Encoding.UTF8.GetBytes(sequence), _encoder);
            _cache.Add(sequence, encodedIds);

            accumulatedIds.AddRange(encodedIds);
            return;
        }

        /// <summary>
        /// Get the number of token's Ids that the input sequence will be encoded to.
        /// </summary>
        /// <param name="sequence">The text to tokenize.</param>
        /// <param name="isSpecialToken">Indicate if the token is special token.</param>
        /// <returns>The number of token's Ids that the input sequence will be encoded to.</returns>
        public override int GetTokenizedIdsCount(string sequence, bool isSpecialToken)
        {
            if (string.IsNullOrEmpty(sequence))
            {
                return 0;
            }

            if (isSpecialToken && _specialTokensEncoder is not null)
            {
                return _specialTokensEncoder.TryGetValue(sequence, out int id) ? 1 : 0;
            }

            if (_cache.Lookup(sequence, out int[] ids))
            {
                return ids.Length;
            }

            if (_vocab.TryGetValue(sequence, out int mappedId))
            {
                return 1;
            }

            int[] encodedIds = BytePairEncoder.BytePairEncode(Encoding.UTF8.GetBytes(sequence), _encoder);
            _cache.Add(sequence, encodedIds);

            return encodedIds.Length;
        }

        /// <summary>
        /// Map the token to tokenized Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <returns>The mapped Id of the token.</returns>
        public override int? TokenToId(string token) => TokenToId(token, skipSpecialTokens: false);

        /// <summary>
        /// Map the token to tokenized Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <param name="skipSpecialTokens">Indicate if want to skip the special tokens during the encoding.</param>
        /// <returns>The mapped Id of the token.</returns>
        public override int? TokenToId(string token, bool skipSpecialTokens)
        {
            if (string.IsNullOrEmpty(token))
            {
                return 0;
            }

            if (!skipSpecialTokens && _specialTokensEncoder is not null && _specialTokensEncoder.TryGetValue(token, out int specialTokenId))
            {
                return specialTokenId;
            }

            if (_cache.Lookup(token, out int[] ids))
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

            int[] idsToCache = BytePairEncoder.BytePairEncode(Encoding.UTF8.GetBytes(token), _encoder);
            _cache.Add(token, idsToCache);

            if (idsToCache.Length == 1)
            {
                return idsToCache[0];
            }

            return null;
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

            List<byte> utf8Bytes = new();
            bool useSpecialTokens = !skipSpecialTokens && _specialTokensDecoder is not null;

            foreach (int id in ids)
            {
                if (_decoder.TryGetValue(id, out byte[]? tokenBytes))
                {
                    utf8Bytes.AddRange(tokenBytes);
                }
                else if (useSpecialTokens && _specialTokensDecoder!.TryGetValue(id, out string? token))
                {
                    utf8Bytes.AddRange(Encoding.UTF8.GetBytes(token));
                }
                else
                {
                    return null;
                }
            }

            return utf8Bytes.Count > 0 ? Encoding.UTF8.GetString(utf8Bytes.ToArray()) : string.Empty;
        }

        /// <summary>
        /// Gets the dictionary mapping tokens to Ids.
        /// </summary>
        public override IReadOnlyDictionary<string, int> GetVocab() => _vocab;

        /// <summary>
        /// Gets the dictionary size that map tokens to Ids.
        /// </summary>
        public override int GetVocabSize() => _vocab.Count;

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
    }
}