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
    /// Represent the rapid Byte Pair Encoding tokenizer.
    /// </summary>
    public sealed partial class Tiktoken : Tokenizer
    {
        private readonly Dictionary<ReadOnlyMemory<byte>, int> _encoder;
        private readonly Dictionary<int, ReadOnlyMemory<byte>> _decoder;
        private readonly LruCache<(int Id, int TokenIndex, int TokenLength)[]> _cache;
        private readonly Dictionary<StringSpanOrdinalKey, (int Id, string Token)> _vocab;
        private IReadOnlyDictionary<string, int>? _vocabOriginal;
        private const int MaxWordLengthToCache = 15;
        private readonly PreTokenizer? _preTokenizer;
        private readonly Normalizer? _normalizer;

        /// <summary>
        /// Create a new Tiktoken tokenizer's object.
        /// </summary>
        /// <param name="vocabFilePath">The path to the BPE vocab file.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="specialTokens">The dictionary mapping special tokens to Ids.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="vocabFilePath"/> is null or empty.</exception>
        /// <exception cref="InvalidOperationException">Thrown when failed to load the BPE vocab file.</exception>
        public Tiktoken(string vocabFilePath, PreTokenizer? preTokenizer, IReadOnlyDictionary<string, int>? specialTokens = null, Normalizer? normalizer = null, int cacheSize = LruCache<int[]>.DefaultCacheSize) :
            this(string.IsNullOrEmpty(vocabFilePath) ? throw new ArgumentNullException(nameof(vocabFilePath)) : File.OpenRead(vocabFilePath), preTokenizer, specialTokens, normalizer, cacheSize, disposeStream: true)
        {
        }

        /// <summary>
        /// Create a new Tiktoken tokenizer's object.
        /// </summary>
        /// <param name="vocabStream">The stream to the BPE vocab file.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="specialTokens">The dictionary mapping special tokens to Ids.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="cacheSize">The size of the cache to use.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="vocabStream"/> is null or empty.</exception>
        /// <exception cref="InvalidOperationException">Thrown when failed to load the BPE vocab file.</exception>
        public Tiktoken(Stream vocabStream, PreTokenizer? preTokenizer, IReadOnlyDictionary<string, int>? specialTokens = null, Normalizer? normalizer = null, int cacheSize = LruCache<int[]>.DefaultCacheSize) :
            this(vocabStream ?? throw new ArgumentNullException(nameof(vocabStream)), preTokenizer, specialTokens, normalizer, cacheSize, disposeStream: false)
        {
        }

        /// <summary>
        /// Create a new Tiktoken tokenizer's object.
        /// </summary>
        /// <param name="encoder">The dictionary mapping token utf-8 bytes to Ids.</param>
        /// <param name="decoder">The dictionary mapping Ids to token utf-8 bytes.</param>
        /// <param name="vocab">The dictionary mapping string tokens to Ids.</param>
        /// <param name="preTokenizer">The pre-tokenizer to use.</param>
        /// <param name="specialTokens">The dictionary mapping special tokens to Ids.</param>
        /// <param name="normalizer">The normalizer to use.</param>
        /// <param name="cacheSize">The max size of the cache to use.</param>
        internal Tiktoken(
            Dictionary<ReadOnlyMemory<byte>, int> encoder,
            Dictionary<int, ReadOnlyMemory<byte>> decoder,
            Dictionary<StringSpanOrdinalKey, (int Id, string Token)> vocab,
            PreTokenizer? preTokenizer,
            IReadOnlyDictionary<string, int>? specialTokens,
            Normalizer? normalizer = null,
            int cacheSize = LruCache<int[]>.DefaultCacheSize)
        {
            _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
            _decoder = decoder ?? throw new ArgumentNullException(nameof(decoder));
            _vocab = vocab ?? throw new ArgumentNullException(nameof(vocab));

            _encoder = encoder!;
            _decoder = decoder!;
            _vocab = vocab!;

            _preTokenizer = preTokenizer;
            _normalizer = normalizer;

            _cache = new LruCache<(int Id, int TokenIndex, int TokenLength)[]>(cacheSize);

            SpecialTokens = specialTokens;
            CacheSpecialTokensEncoding(specialTokens);
        }

        private Tiktoken(Stream vocabStream, PreTokenizer? preTokenizer, IReadOnlyDictionary<string, int>? specialTokens, Normalizer? normalizer, int cacheSize, bool disposeStream)
        {
            try
            {
                _cache = new LruCache<(int Id, int TokenIndex, int TokenLength)[]>(cacheSize);
                (_encoder, _vocab, _decoder) = LoadTiktokenBpeAsync(vocabStream, useAsync: false).GetAwaiter().GetResult();

                _preTokenizer = preTokenizer;
                _normalizer = normalizer;

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

        /// <summary>
        /// Gets the PreTokenizer used by the Tokenizer.
        /// </summary>
        public override PreTokenizer? PreTokenizer => _preTokenizer;

        /// <summary>
        /// Gets the Normalizer in use by the Tokenizer.
        /// </summary>
        public override Normalizer? Normalizer => _normalizer;

        private void CacheSpecialTokensEncoding(IReadOnlyDictionary<string, int>? specialTokens)
        {
            Debug.Assert(_cache is not null);
            Debug.Assert(_decoder is not null);

            if (specialTokens is not null)
            {
                foreach (KeyValuePair<string, int> specialToken in specialTokens)
                {
                    _decoder![specialToken.Value] = Encoding.UTF8.GetBytes(specialToken.Key);
                    _cache!.Add(specialToken.Key, new[] { (Id: specialToken.Value, TokenIndex0: 0, TokenLength: specialToken.Key.Length) });
                }
            }
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
        /// Encodes input text a list of <see cref="Token" />s with string value of the token, id, and offset.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The tokenization result includes a list of <see cref="Token" />s with string value of the token, id, and offset.</returns>
        public override IReadOnlyList<Token> Encode(string text, out string? normalizedString, bool considerPreTokenization = true, bool considerNormalization = true) => Encode(text, Span<char>.Empty, out normalizedString, considerPreTokenization, considerNormalization);

        /// <summary>
        /// Encodes input text a list of <see cref="Token" />s with string value of the token, id, and offset.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The tokenization result includes a list of <see cref="Token" />s with string value of the token, id, and offset.</returns>
        public override IReadOnlyList<Token> Encode(ReadOnlySpan<char> text, out string? normalizedString, bool considerPreTokenization = true, bool considerNormalization = true) => Encode(null, text, out normalizedString, considerPreTokenization, considerNormalization);

        private IReadOnlyList<Token> Encode(string? text, ReadOnlySpan<char> textSpan, out string? normalizedString, bool considerPreTokenization, bool considerNormalization)
        {
            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedString = null;
                return [];
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(text, textSpan, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedString, out ReadOnlySpan<char> textSpanToEncode);

            List<Token> tokens = new();

            if (splits is not null)
            {
                foreach ((int Offset, int Length) split in splits)
                {
                    Encode(textSpanToEncode.Slice(split.Offset, split.Length), tokens, split.Offset);
                }
            }
            else
            {
                Encode(textSpanToEncode, tokens, 0);
            }

            return tokens;
        }

        /// <summary>
        /// Encode text to a list of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="tokens">The list of tokens to populate.</param>
        /// <param name="offset">The offset to start encoding from.</param>
        private void Encode(ReadOnlySpan<char> text, List<Token> tokens, int offset)
        {
            Debug.Assert(!text.IsEmpty);

            if (_cache.TryGetValue(text, out (int Id, int TokenIndex, int TokenLength)[] value))
            {
                for (int i = 0; i < value.Length; i++)
                {
                    tokens.Add(new Token(
                                        value[i].Id,
                                        value[i].TokenLength == 0 ? string.Empty : text.Slice(value[i].TokenIndex, value[i].TokenLength).ToString(),
                                        (value[i].TokenIndex + offset, value[i].TokenLength)));
                }

                return;
            }

            // cache miss
            if (_vocab.TryGetValue(text, out (int Id, string Token) mappedId))
            {
                tokens.Add(new Token(mappedId.Id, mappedId.Token, (offset, mappedId.Token.Length)));
                return;
            }

            int utf8Length = Encoding.UTF8.GetMaxByteCount(text.Length);
            byte[] arrayPoolArray = arrayPoolArray = ArrayPool<byte>.Shared.Rent(utf8Length);
            int[]? indexMappingArray = null;
            Span<int> indexMappingSpan = utf8Length + 1 <= 128 ? stackalloc int[128] : (indexMappingArray = ArrayPool<int>.Shared.Rent(utf8Length + 1));
            int encodedLength = Helpers.EncodeToUtf8(text, arrayPoolArray, indexMappingSpan);
            Debug.Assert(encodedLength < indexMappingSpan.Length);
            indexMappingSpan[encodedLength] = text.Length;

            (int Id, int TokenIndex, int TokenLength)[] encodedTokens = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder, indexMappingSpan.Slice(0, encodedLength + 1));
            ArrayPool<byte>.Shared.Return(arrayPoolArray);
            if (indexMappingArray is not null)
            {
                ArrayPool<int>.Shared.Return(indexMappingArray);
            }

            Debug.Assert(encodedTokens.Length > 0);
            string textAsString = text.ToString();

            if (text.Length <= MaxWordLengthToCache)
            {
                _cache.Add(textAsString, encodedTokens);
            }

            for (int i = 0; i < encodedTokens.Length; i++)
            {
                tokens.Add(new Token(
                                encodedTokens[i].Id,
                                encodedTokens[i].TokenLength == 0 ? string.Empty : text.Slice(encodedTokens[i].TokenIndex, encodedTokens[i].TokenLength).ToString(),
                                (encodedTokens[i].TokenIndex + offset, encodedTokens[i].TokenLength)));
            }
        }

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public override IReadOnlyList<int> EncodeToIds(string text, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(text, Span<char>.Empty, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Encodes input text to token Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public override IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(null, text, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Encodes input text to token Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public override IReadOnlyList<int> EncodeToIds(string text, int maxTokenCount, out string? normalizedString, out int textLength, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(text, Span<char>.Empty, considerPreTokenization, considerNormalization, out normalizedString, out textLength, maxTokenCount);

        /// <summary>
        /// Encodes input text to token Ids up to maximum number of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The list of encoded Ids.</returns>
        public override IReadOnlyList<int> EncodeToIds(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedString, out int textLength, bool considerPreTokenization = true, bool considerNormalization = true)
            => EncodeToIds(null, text, considerPreTokenization, considerNormalization, out normalizedString, out textLength, maxTokenCount);

        private IReadOnlyList<int> EncodeToIds(string? text, ReadOnlySpan<char> textSpan, bool considerPreTokenization, bool considerNormalization, out string? normalizedString, out int textLength, int maxTokenCount = int.MaxValue)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                textLength = 0;
                normalizedString = null;
                return [];
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(text, textSpan, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedString, out ReadOnlySpan<char> textSpanToEncode);

            List<int> ids = new();

            if (splits is not null)
            {
                textLength = 0;
                foreach ((int Offset, int Length) split in splits)
                {
                    EncodeToIds(textSpanToEncode.Slice(split.Offset, split.Length), ids, out int length, maxTokenCount - ids.Count);
                    textLength = split.Offset + length;

                    if (length < split.Length || ids.Count >= maxTokenCount)
                    {
                        break;
                    }
                }
            }
            else
            {
                EncodeToIds(textSpanToEncode, ids, out textLength);
            }

            return ids;
        }

        /// <summary>
        /// Encode text to a list of Ids.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="accumulatedIds">The list of accumulated Ids.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokenCount">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        private int EncodeToIds(ReadOnlySpan<char> text, IList<int> accumulatedIds, out int textLength, int maxTokenCount = int.MaxValue)
        {
            Debug.Assert(maxTokenCount > 0);

            if (text.IsEmpty)
            {
                textLength = 0;
                return 0;
            }

            if (_cache.TryGetValue(text, out (int Id, int TokenIndex, int TokenLength)[] value))
            {
                return EncodeToIdsResult(value, accumulatedIds, maxTokenCount, text.Length, out textLength);
            }

            if (_vocab.TryGetValue(text, out (int Id, string Token) mappedId))
            {
                textLength = text.Length;
                accumulatedIds.Add(mappedId.Id);
                return 1;
            }

            int utf8Length = Encoding.UTF8.GetMaxByteCount(text.Length);
            byte[] arrayPoolArray = arrayPoolArray = ArrayPool<byte>.Shared.Rent(utf8Length);
            int[]? indexMappingArray = null;
            Span<int> indexMappingSpan = utf8Length + 1 <= 128 ? stackalloc int[128] : (indexMappingArray = ArrayPool<int>.Shared.Rent(utf8Length + 1));
            int encodedLength = Helpers.EncodeToUtf8(text, arrayPoolArray, indexMappingSpan);
            Debug.Assert(encodedLength < indexMappingSpan.Length);
            indexMappingSpan[encodedLength] = text.Length;

            (int Id, int TokenIndex, int TokenLength)[] encodedTokens = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder, indexMappingSpan.Slice(0, encodedLength + 1));
            ArrayPool<byte>.Shared.Return(arrayPoolArray);
            if (indexMappingArray is not null)
            {
                ArrayPool<int>.Shared.Return(indexMappingArray);
            }

            if (text.Length <= MaxWordLengthToCache)
            {
                string textAsString = text.ToString();
                _cache.Add(textAsString, encodedTokens);
            }

            return EncodeToIdsResult(encodedTokens, accumulatedIds, maxTokenCount, text.Length, out textLength);
        }

        private int EncodeToIdsResult((int Id, int TokenIndex, int TokenLength)[] tokens, IList<int>? accumulatedIds, int maxTokens, int fullTextLength, out int textLength)
        {
            textLength = 0;

            if (tokens.Length <= maxTokens)
            {
                if (accumulatedIds is not null)
                {
                    foreach (var t in tokens)
                    {
                        accumulatedIds.Add(t.Id);
                    }
                }

                textLength = fullTextLength;
                return tokens.Length;
            }

            int tokenCount;
            for (tokenCount = 0; tokenCount < maxTokens; tokenCount++)
            {
                int overlapIndex = tokens[tokenCount].TokenIndex + tokens[tokenCount].TokenLength;
                // maxTokens is less than tokens.Count, so it is safe to index maxTokens.
                if (tokens[tokenCount + 1].TokenIndex < overlapIndex)
                {
                    // Ensure we'll not break the text in the middle of a code-point
                    int j = tokenCount + 2;
                    while (j < tokens.Length && tokens[j].TokenIndex < overlapIndex)
                    {
                        j++;
                    }

                    if (j <= maxTokens)
                    {
                        // append encountered tokens to the accumulatedIds
                        for (int k = tokenCount; k < j; k++)
                        {
                            accumulatedIds?.Add(tokens[k].Id);
                        }
                        tokenCount = j - 1;
                        textLength = tokens[tokenCount].TokenIndex + tokens[tokenCount].TokenLength;
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    accumulatedIds?.Add(tokens[tokenCount].Id);
                    textLength = tokens[tokenCount].TokenIndex + tokens[tokenCount].TokenLength;
                }
            }

            return tokenCount;
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        public override int CountTokens(string text, bool considerPreTokenization = true, bool considerNormalization = true)
            => CountTokens(text, Span<char>.Empty, considerPreTokenization, considerNormalization, out _, out _);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>The number of token Ids that the input text will be encoded to.</returns>
        public override int CountTokens(ReadOnlySpan<char> text, bool considerPreTokenization = true, bool considerNormalization = true)
            => CountTokens(null, text, considerPreTokenization, considerNormalization, out _, out _);

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
        public override int IndexOfTokenCount(string text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            tokenCount = CountTokens(text, Span<char>.Empty, considerPreTokenization, considerNormalization, out normalizedString, out int textLength, maxTokenCount);
            return textLength;
        }

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
        public override int IndexOfTokenCount(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
        {
            tokenCount = CountTokens(null, text, considerPreTokenization, considerNormalization, out normalizedString, out int textLength, maxTokenCount);
            return textLength;
        }

        private int CountTokens(string? text, ReadOnlySpan<char> textSpan, bool considerPreTokenization, bool considerNormalization, out string? normalizedString, out int textLength, int maxTokenCount = int.MaxValue)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The maximum number of tokens must be greater than zero.");
            }

            textLength = 0;
            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedString = null;
                return 0;
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(text, textSpan, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedString, out ReadOnlySpan<char> textSpanToEncode);

            int count = 0;
            if (splits is not null)
            {
                foreach ((int Offset, int Length) split in splits)
                {
                    count += CountTokens(textSpanToEncode.Slice(split.Offset, split.Length), out int length, maxTokenCount - count);
                    textLength = split.Offset + length;

                    if (length < split.Length || count >= maxTokenCount)
                    {
                        break;
                    }
                }
            }
            else
            {
                count = CountTokens(textSpanToEncode, out textLength, maxTokenCount);
            }

            return count;
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textLength">The length of the text that encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        private int CountTokens(ReadOnlySpan<char> text, out int textLength, int maxTokens = int.MaxValue)
        {
            Debug.Assert(maxTokens > 0);

            if (text.IsEmpty)
            {
                textLength = 0;
                return 0;
            }

            if (_cache.TryGetValue(text, out (int Id, int TokenIndex, int TokenLength)[] value))
            {
                return EncodeToIdsResult(value, accumulatedIds: null, maxTokens, text.Length, out textLength);
            }

            if (_vocab.TryGetValue(text, out _))
            {
                textLength = text.Length;
                return 1;
            }

            int utf8Length = Encoding.UTF8.GetMaxByteCount(text.Length);
            byte[] arrayPoolArray = arrayPoolArray = ArrayPool<byte>.Shared.Rent(utf8Length);
            int[]? indexMappingArray = null;
            Span<int> indexMappingSpan = utf8Length + 1 <= 128 ? stackalloc int[128] : (indexMappingArray = ArrayPool<int>.Shared.Rent(utf8Length + 1));
            int encodedLength = Helpers.EncodeToUtf8(text, arrayPoolArray, indexMappingSpan);
            Debug.Assert(encodedLength < indexMappingSpan.Length);
            indexMappingSpan[encodedLength] = text.Length;

            (int Id, int TokenIndex, int TokenLength)[] encodedTokens = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder, indexMappingSpan.Slice(0, encodedLength + 1));
            ArrayPool<byte>.Shared.Return(arrayPoolArray);
            if (indexMappingArray is not null)
            {
                ArrayPool<int>.Shared.Return(indexMappingArray);
            }

            if (text.Length <= MaxWordLengthToCache)
            {
                string textAsString = text.ToString();
                _cache.Add(textAsString, encodedTokens);
            }

            return EncodeToIdsResult(encodedTokens, accumulatedIds: null, maxTokens, text.Length, out textLength);
        }

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the text or the <paramref name="normalizedString"/> if normalization is enabled;
        /// conversely, if all tokens fit, the result will be 0.
        /// </returns>
        public override int LastIndexOfTokenCount(string text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
            => LastIndexOf(text, Span<char>.Empty, maxTokenCount, considerPreTokenization, considerNormalization, out normalizedString, out tokenCount);

        /// <summary>
        /// Find the index of the maximum encoding capacity from the end within the text without surpassing the token limit.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="maxTokenCount">The maximum token count to limit the encoding capacity.</param>
        /// <param name="normalizedString">If the tokenizer's normalization is enabled or <paramRef name="considerNormalization" /> is false, this will be set to <paramRef name="text" /> in its normalized form; otherwise, this value will be set to null.</param>
        /// <param name="tokenCount">The token count can be generated which should be smaller than the maximum token count.</param>
        /// <param name="considerPreTokenization">Indicate whether to consider pre-tokenization before tokenization.</param>
        /// <param name="considerNormalization">Indicate whether to consider normalization before tokenization.</param>
        /// <returns>
        /// The start index of the maximum encoding capacity within the processed text without surpassing the token limit.
        /// It represents the index at the first character to be included. In cases where no tokens fit, the result will be length of the <paramref name="normalizedString"/>; conversely, if all tokens fit, the result will be 0.
        /// </returns>
        public override int LastIndexOfTokenCount(ReadOnlySpan<char> text, int maxTokenCount, out string? normalizedString, out int tokenCount, bool considerPreTokenization = true, bool considerNormalization = true)
            => LastIndexOf(null, text, maxTokenCount, considerPreTokenization, considerNormalization, out normalizedString, out tokenCount);

        private int LastIndexOf(string? text, ReadOnlySpan<char> textSpan, int maxTokenCount, bool considerPreTokenization, bool considerNormalization, out string? normalizedString, out int tokenCount)
        {
            if (maxTokenCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxTokenCount), "The max token count must be greater than 0.");
            }

            if (string.IsNullOrEmpty(text) && textSpan.IsEmpty)
            {
                normalizedString = null;
                tokenCount = 0;
                return 0;
            }

            IEnumerable<(int Offset, int Length)>? splits = InitializeForEncoding(text, textSpan, considerPreTokenization, considerNormalization, _normalizer, _preTokenizer, out normalizedString, out ReadOnlySpan<char> textSpanToEncode);

            if (splits is not null)
            {
                tokenCount = 0;
                foreach ((int Offset, int Length) split in splits.Reverse())
                {
                    tokenCount += CountTokensFromEnd(textSpanToEncode.Slice(split.Offset, split.Length), out int textIndex, maxTokenCount - tokenCount);
                    if (textIndex > 0 || tokenCount >= maxTokenCount)
                    {
                        return split.Offset + textIndex;
                    }
                }

                return 0;
            }
            else
            {
                tokenCount = CountTokensFromEnd(textSpanToEncode, out int textLength, maxTokenCount);
                return textLength;
            }
        }

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="textIndex">Starting from this index to the end of the text will encompasses the maximum encoded tokens.</param>
        /// <param name="maxTokens">The maximum number of tokens to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to.</returns>
        private int CountTokensFromEnd(ReadOnlySpan<char> text, out int textIndex, int maxTokens = int.MaxValue)
        {
            Debug.Assert(maxTokens > 0);

            if (text.IsEmpty)
            {
                textIndex = 0;
                return 0;
            }

            if (_cache.TryGetValue(text, out (int Id, int TokenIndex, int TokenLength)[] value))
            {
                return EncodeToIdsFromEndResult(value, accumulatedIds: null, maxTokens, text.Length, out textIndex);
            }

            if (_vocab.TryGetValue(text, out _))
            {
                textIndex = 0;
                return 1;
            }

            int utf8Length = Encoding.UTF8.GetMaxByteCount(text.Length);
            byte[] arrayPoolArray = arrayPoolArray = ArrayPool<byte>.Shared.Rent(utf8Length);
            int[]? indexMappingArray = null;
            Span<int> indexMappingSpan = utf8Length + 1 <= 128 ? stackalloc int[128] : (indexMappingArray = ArrayPool<int>.Shared.Rent(utf8Length + 1));
            int encodedLength = Helpers.EncodeToUtf8(text, arrayPoolArray, indexMappingSpan);
            Debug.Assert(encodedLength < indexMappingSpan.Length);
            indexMappingSpan[encodedLength] = text.Length;

            (int Id, int TokenIndex, int TokenLength)[] encodedTokens = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder, indexMappingSpan.Slice(0, encodedLength + 1));
            ArrayPool<byte>.Shared.Return(arrayPoolArray);
            if (indexMappingArray is not null)
            {
                ArrayPool<int>.Shared.Return(indexMappingArray);
            }

            if (text.Length <= MaxWordLengthToCache)
            {
                string textAsString = text.ToString();
                _cache.Add(textAsString, encodedTokens);
            }

            return EncodeToIdsFromEndResult(encodedTokens, accumulatedIds: null, maxTokens, text.Length, out textIndex);
        }

        private int EncodeToIdsFromEndResult((int Id, int TokenIndex, int TokenLength)[] tokens, IList<int>? accumulatedIds, int maxTokens, int fullTextLength, out int textIndex)
        {
            textIndex = fullTextLength;

            if (tokens.Length <= maxTokens)
            {
                if (accumulatedIds is not null)
                {
                    foreach (var t in tokens)
                    {
                        accumulatedIds.Add(t.Id);
                    }
                }

                textIndex = 0;
                return tokens.Length;
            }

            int index = tokens.Length - maxTokens;

            // avoid breaking the text in the middle of a code-point
            while (index < tokens.Length && tokens[index].TokenIndex < tokens[index - 1].TokenIndex + tokens[index - 1].TokenLength)
            {
                index++;
            }

            for (int i = index; i < tokens.Length; i++)
            {
                accumulatedIds?.Add(tokens[i].Id);
            }

            textIndex = index >= tokens.Length ? fullTextLength : tokens[index].TokenIndex;
            return tokens.Length - index;
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
                return null;
            }

            if (_cache.TryGetValue(token, out (int Id, int TokenIndex, int TokenLength)[] value))
            {
                if (value.Length == 1)
                {
                    return value[0].Id;
                }

                return null;
            }

            if (_vocab.TryGetValue(token, out (int Id, string Token) id))
            {
                return id.Id;
            }

            int utf8Length = Encoding.UTF8.GetMaxByteCount(token.Length);
            byte[] arrayPoolArray = arrayPoolArray = ArrayPool<byte>.Shared.Rent(utf8Length);
            int[]? indexMappingArray = null;
            Span<int> indexMappingSpan = utf8Length + 1 <= 128 ? stackalloc int[128] : (indexMappingArray = ArrayPool<int>.Shared.Rent(utf8Length + 1));
            int encodedLength = Helpers.EncodeToUtf8(token, arrayPoolArray, indexMappingSpan);
            Debug.Assert(encodedLength < indexMappingSpan.Length);
            indexMappingSpan[encodedLength] = token.Length;

            (int Id, int TokenIndex, int TokenLength)[] encodedTokens = BytePairEncoder.BytePairEncode(arrayPoolArray.AsMemory(0, encodedLength), _encoder, indexMappingSpan.Slice(0, encodedLength + 1));

            ArrayPool<byte>.Shared.Return(arrayPoolArray);
            if (indexMappingArray is not null)
            {
                ArrayPool<int>.Shared.Return(indexMappingArray);
            }

            if (token.Length <= MaxWordLengthToCache)
            {
                string tokenAsString = token.ToString();
                _cache.Add(tokenAsString, encodedTokens);
            }

            if (encodedTokens.Length == 1)
            {
                return encodedTokens[0].Id;
            }

            return null;
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
        /// <returns>The decoded string.</returns>
        public override string? Decode(IEnumerable<int> ids)
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
            GPT2,
            O200kBase
        }

        private static readonly (string Prefix, ModelEncoding Encoding)[] _modelPrefixToEncoding =
                                                            [
                                                                // chat
                                                                ( "gpt-4o-", ModelEncoding.O200kBase),    // e.g., gpt-4o-2024-05-13
                                                                ( "gpt-4-", ModelEncoding.Cl100kBase),    // e.g., gpt-4-0314, etc., plus gpt-4-32k
                                                                ( "gpt-3.5-", ModelEncoding.Cl100kBase),  // e.g, gpt-3.5-turbo-0301, -0401, etc.
                                                                ( "gpt-35-", ModelEncoding.Cl100kBase )   // Azure deployment name
                                                            ];

        private static readonly Dictionary<string, ModelEncoding> _modelToEncoding =
                                                            new Dictionary<string, ModelEncoding>(StringComparer.OrdinalIgnoreCase)
                                                            {
                                                                // chat
                                                                { "gpt-4o", ModelEncoding.O200kBase },
                                                                { "gpt-4", ModelEncoding.Cl100kBase },
                                                                { "gpt-3.5-turbo", ModelEncoding.Cl100kBase },
                                                                { "gpt-3.5-turbo-16k", ModelEncoding.Cl100kBase },
                                                                { "gpt-35", ModelEncoding.Cl100kBase },           // Azure deployment name
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

                case ModelEncoding.O200kBase:
                    return (new Dictionary<string, int> { { EndOfText, 199999 }, { EndOfPrompt, 200018 } }, O200kBaseRegex(), O200kBaseFile);

                default:
                    throw new NotSupportedException($"The model '{modelName ?? modelEncoding.ToString()}' is not supported.");
            }
        }

        // Regex patterns based on https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py

        private const string Cl100kBaseRegexPattern = /*lang=regex*/ @"'(?i:[sdmt]|re|ve|ll)|(?>[^\r\n\p{L}\p{N}]?)\p{L}+|\p{N}{1,3}| ?(?>[^\s\p{L}\p{N}]+)[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
        private const string P50kBaseRegexPattern = /*lang=regex*/ @"'(?:[sdmt]|re|ve|ll)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
        private const string O200kBaseRegexPattern = /*lang=regex*/ @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

        private const string Cl100kBaseVocabFile = "cl100k_base.tiktoken.deflate";  // "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        private const string P50RanksFile = "p50k_base.tiktoken.deflate";           // "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken"
        private const string R50RanksFile = "r50k_base.tiktoken.deflate";           // "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"
        private const string GPT2File = "gpt2.tiktoken.deflate";                    // "https://fossies.org/linux/misc/whisper-20231117.tar.gz/whisper-20231117/whisper/assets/gpt2.tiktoken?m=b"
        private const string O200kBaseFile = "o200k_base.tiktoken.deflate";         // "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"

        internal const string Cl100kBaseEncodingName = "cl100k_base";
        internal const string P50kBaseEncodingName = "p50k_base";
        internal const string P50kEditEncodingName = "p50k_edit";
        internal const string R50kBaseEncodingName = "r50k_base";
        internal const string O200kBaseEncodingName = "o200k_base";

#if NET7_0_OR_GREATER
        [GeneratedRegex(Cl100kBaseRegexPattern)]
        private static partial Regex Cl100kBaseRegex();

        [GeneratedRegex(P50kBaseRegexPattern)]
        internal static partial Regex P50kBaseRegex();

        [GeneratedRegex(O200kBaseRegexPattern)]
        internal static partial Regex O200kBaseRegex();
#else
        private static Regex? _cl100kBaseRegex;
        private static Regex Cl100kBaseRegex() => _cl100kBaseRegex ??= new Regex(Cl100kBaseRegexPattern, RegexOptions.Compiled);

        private static Regex? _p50kBaseRegex;
        internal static Regex P50kBaseRegex() => _p50kBaseRegex ??= new Regex(P50kBaseRegexPattern, RegexOptions.Compiled);

        private static Regex? _o200kBaseRegex;
        internal static Regex O200kBaseRegex() => _o200kBaseRegex ??= new Regex(O200kBaseRegexPattern, RegexOptions.Compiled);
#endif

        private static readonly ConcurrentDictionary<string, (Dictionary<ReadOnlyMemory<byte>, int> encoder, Dictionary<StringSpanOrdinalKey, (int Id, string Token)> vocab, Dictionary<int, ReadOnlyMemory<byte>> decoder)> _tiktokenCache = new(StringComparer.OrdinalIgnoreCase);

        internal static Tokenizer CreateForModel(
                                    ModelEncoding modelEncoding,
                                    string? modelName = null,
                                    IReadOnlyDictionary<string, int>? extraSpecialTokens = null,
                                    Normalizer? normalizer = null)
        {
            (Dictionary<string, int> SpecialTokens, Regex Regex, string VocabFile) tiktokenConfiguration = GetTiktokenConfigurations(modelEncoding, modelName);

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

            return new Tiktoken(
                        cache.encoder,
                        cache.decoder,
                        cache.vocab,
                        new TiktokenPreTokenizer(tiktokenConfiguration.Regex, tiktokenConfiguration.SpecialTokens),
                        tiktokenConfiguration.SpecialTokens,
                        normalizer,
                        LruCache<int[]>.DefaultCacheSize);
        }
    }
}
