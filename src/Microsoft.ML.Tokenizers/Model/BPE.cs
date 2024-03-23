// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Represent the Byte Pair Encoding model.
    /// </summary>
    public sealed class Bpe : Model
    {
        /// A [Byte Pair Encoding](https://www.aclweb.org/anthology/P16-1162/) model.

        private string? _unknownToken;

        /// <summary>
        /// Gets or Sets unknown token. The unknown token to be used when we encounter an unknown char
        /// </summary>
        public string? UnknownToken
        {
            get
            {
                return _unknownToken;
            }

            private set
            {
                _unknownToken = value;

                if (VocabReverse.TryGetValue(0, out string? v))
                {
                    if (v == value)
                    {
                        return;
                    }

                    VocabReverse.Remove(0);
                    _vocab.Remove(new StringSpanOrdinalKey(v));
                }


                if (value is not null)
                {
                    _vocab[new StringSpanOrdinalKey(value)] = 0;
                    VocabReverse[0] = value;
                }
            }
        }

        /// <summary>
        /// A prefix to be used for every subword that is not a beginning-of-word
        /// </summary>
        public string? ContinuingSubwordPrefix { get; }

        /// <summary>
        /// An optional suffix to characterize and end-of-word sub-word
        /// </summary>
        public string? EndOfWordSuffix { get; }

        /// <summary>
        /// Gets or sets whether allowing multiple unknown tokens get fused
        /// </summary>
        public bool FuseUnknownTokens { get; }

        /// <summary>
        /// Construct a new Bpe model object to use for text encoding.
        /// </summary>
        /// <param name="vocabFile">The JSON file path containing the dictionary of string keys and their ids.</param>
        /// <param name="mergesFile">The file path containing the tokens's pairs list.</param>
        /// <param name="unknownToken"> The unknown token to be used by the model.</param>
        /// <param name="continuingSubwordPrefix">The prefix to attach to sub-word units that don’t represent a beginning of word.</param>
        /// <param name="endOfWordSuffix">The suffix to attach to sub-word units that represent an end of word.</param>
        /// <param name="fuseUnknownTokens">Indicate whether allowing multiple unknown tokens get fused.</param>
        public Bpe(string vocabFile, string? mergesFile, string? unknownToken = null, string? continuingSubwordPrefix = null, string? endOfWordSuffix = null, bool fuseUnknownTokens = false) :
            this(vocabFile is null ? throw new ArgumentNullException(nameof(vocabFile)) : File.Open(vocabFile, FileMode.Open, FileAccess.Read),
                mergesFile is null ? null : File.Open(mergesFile, FileMode.Open, FileAccess.Read), unknownToken, continuingSubwordPrefix, endOfWordSuffix, fuseUnknownTokens, disposeStreams: true)
        {
        }

        /// <summary>
        /// Construct a new Bpe model object to use for text encoding.
        /// </summary>
        /// <param name="vocabStream">The JSON stream containing the dictionary of string keys and their ids.</param>
        /// <param name="mergesStream">The stream containing the tokens's pairs list.</param>
        /// <param name="unknownToken"> The unknown token to be used by the model.</param>
        /// <param name="continuingSubwordPrefix">The prefix to attach to sub-word units that don’t represent a beginning of word.</param>
        /// <param name="endOfWordSuffix">The suffix to attach to sub-word units that represent an end of word.</param>
        /// <param name="fuseUnknownTokens">Indicate whether allowing multiple unknown tokens get fused.</param>
        public Bpe(Stream vocabStream, Stream? mergesStream, string? unknownToken = null, string? continuingSubwordPrefix = null, string? endOfWordSuffix = null, bool fuseUnknownTokens = false) :
                this(vocabStream, mergesStream, unknownToken, continuingSubwordPrefix, endOfWordSuffix, fuseUnknownTokens, disposeStreams: false)
        {
        }

        private Bpe(Stream vocabStream, Stream? mergesStream, string? unknownToken, string? continuingSubwordPrefix, string? endOfWordSuffix, bool fuseUnknownTokens, bool disposeStreams)
        {
            try
            {
                if (vocabStream is null)
                {
                    throw new ArgumentNullException(nameof(vocabStream));
                }

                FuseUnknownTokens = fuseUnknownTokens;
                ContinuingSubwordPrefix = continuingSubwordPrefix;
                EndOfWordSuffix = endOfWordSuffix;

                (Dictionary<StringSpanOrdinalKey, int>? vocab1, Vec<(string, string)> merges) = ReadModelData(vocabStream, mergesStream);
                _vocab = vocab1 ?? new Dictionary<StringSpanOrdinalKey, int>();
                Cache = new StringSpanOrdinalKeyCache<Word>();

                VocabReverse = new();

                foreach (KeyValuePair<StringSpanOrdinalKey, int> kvp in _vocab)
                {
                    VocabReverse.Add(kvp.Value, kvp.Key.Data!);
                }


                UnknownToken = unknownToken ?? (VocabReverse.TryGetValue(0, out string? unkToken) ? unkToken : null);

                int prefixLen = ContinuingSubwordPrefix is null ? 0 : ContinuingSubwordPrefix.Length;

                Merges = new();
                for (int i = 0; i < merges.Count; i++)
                {
                    (string a, string b) mergeValues = merges[i];

                    if (!_vocab.TryGetValue(mergeValues.a, out int aId))
                    {
                        throw new InvalidOperationException($"Trying to merge a token '{mergeValues.a}' which not exist in the vocabulary.");
                    }

                    if (!_vocab.TryGetValue(mergeValues.b, out int bId))
                    {
                        throw new InvalidOperationException($"Trying to merge a token '{mergeValues.b}' which not exist in the vocabulary.");
                    }

                    if (mergeValues.b.Length <= prefixLen)
                    {
                        throw new InvalidOperationException($"The merge value '{mergeValues.b}' is too short to be merged with a prefix of length {prefixLen}. This implies that the merge file is either damaged or missing the prefix in its entries.");
                    }

                    string newToken = $"{mergeValues.a}{mergeValues.b.Substring(prefixLen)}";
                    if (!_vocab.TryGetValue(newToken, out int newId))
                    {
                        throw new InvalidOperationException($"Trying to merge a token '{newToken}' which not exist in the vocabulary.");
                    }

                    Merges.Add(new Pair<int>(aId, bId), (i, newId));
                }
            }
            finally
            {
                if (disposeStreams)
                {
                    vocabStream.Dispose();
                    mergesStream?.Dispose();
                }
            }
        }

        /// <summary>
        /// Gets the Bpe decoder object.
        /// </summary>
        public static TokenizerDecoder Decoder { get; } = new BpeDecoder();

        /// <summary>
        /// Encode a text string to a list of tokens.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>The list of tokens generated from the text tokenization.</returns>
        public override IReadOnlyList<Token> Encode(ReadOnlySpan<char> text)
        {
            if (text.Length == 0)
            {
                return EmptyTokensList;
            }

            return EncodeWithCache(text);
        }

        /// <summary>
        /// Encode a split text string to a list of Ids and add them to the accumulatedIds list.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="accumulatedIds">The list of accumulated encoded Ids.</param>
        public override void EncodeToIds(ReadOnlySpan<char> text, IList<int> accumulatedIds) => EncodeToIdsWithCache(text, accumulatedIds);

        /// <summary>
        /// Get the number of tokens that the input text will be encoded to.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>The number of tokens that the input text will be encoded to. This parameter is ignored in this model.</returns>
        public override int CountTokens(ReadOnlySpan<char> text) => EncodeToIdsWithCache(text, null);

        /// <summary>
        /// Map the token to encoded Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <returns>The mapped Id of the token.</returns>
        public override int? MapTokenToId(ReadOnlySpan<char> token) => _vocab.TryGetValue(token, out int value) ? value : null;

        /// <summary>
        /// Map the encoded Id to the token.
        /// </summary>
        /// <param name="id">The Id to map to the token.</param>
        /// <returns>The mapped token of the Id.</returns>
        public override string? MapIdToToken(int id)
        {
            if (VocabReverse.TryGetValue(id, out string? value))
            {
                return value;
            }

            return null;
        }

        /// <summary>
        /// Gets the dictionary mapping tokens to Ids.
        /// </summary>
        public IReadOnlyDictionary<string, int> Vocab => _vocabOriginal ??= _vocab.ToDictionary(kvp => kvp.Key.Data!, kvp => kvp.Value);

        /// Read the given files to extract the vocab and merges
        internal static (Dictionary<StringSpanOrdinalKey, int>?, Vec<(string, string)>) ReadModelData(Stream vocab, Stream? merges)
        {
            JsonSerializerOptions options = new() { Converters = { StringSpanOrdinalKeyConverter.Instance } };
            Dictionary<StringSpanOrdinalKey, int>? dic = JsonSerializer.Deserialize<Dictionary<StringSpanOrdinalKey, int>>(vocab, options) as Dictionary<StringSpanOrdinalKey, int>;

            return (dic, ConvertMergesToHashmap(merges));
        }

        /// The vocabulary assigns a number to each token.
        private readonly Dictionary<StringSpanOrdinalKey, int> _vocab;

        private Dictionary<string, int>? _vocabOriginal;

        /// Contains the mapping between Pairs and their (rank, newId).
        internal Dictionary<Pair<int>, (int, int)> Merges { get; }

        /// Contains the cache for optimizing the encoding step.
        internal StringSpanOrdinalKeyCache<Word>? Cache { get; }

        internal static readonly int DefaultCacheCapacity = 10_000;

        /// Reversed vocabulary, to rebuild the text.
        internal SortedDictionary<int, string> VocabReverse { get; }

        /// Dropout probability for merges. 0 = no dropout is the default. At 1.0, tokenization will
        /// perform no merges, so the result will just be characters.
        internal float? Dropout { get; }

        /// Converts the merges strings (for example from `merges.txt` file) with the format
        /// "{pair_a} {pair_b}" into the format expected by the BPE struct
        internal static Vec<(string, string)> ConvertMergesToHashmap(Stream? mergesStream)
        {
            if (mergesStream is null)
            {
                return new Vec<(string, string)>();
            }

            using StreamReader reader = new StreamReader(mergesStream);

            Vec<(string, string)> merges = new(1000);

            int lineNumber = 0;
            while (true)
            {
                string? line = reader.ReadLine();
                if (line is null)
                {
                    break;
                }

                lineNumber++;
                if (line.StartsWith("#version", StringComparison.Ordinal) || line.Length == 0)
                {
                    continue;
                }

                int index = line.IndexOf(' ');
                if (index < 0 || index == line.Length - 1 || line.IndexOf(' ', index + 1) >= 0)
                {
                    throw new InvalidOperationException($"Invalid merger file format at line: {lineNumber}");
                }
                merges.Push((line.Substring(0, index), line.Substring(index + 1)));
            }

            return merges;
        }

        private readonly Dictionary<char, string> _charToString = new Dictionary<char, string>();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal string CharToString(char c)
        {
            if (_charToString.TryGetValue(c, out string? v))
            {
                return v;
            }

            string s = c.ToString();
            _charToString[c] = s;
            return s;
        }

        internal Word MergeWord(ReadOnlySpan<char> w)
        {
            Word word = Word.WithCapacity(w.Length);
            (int Id, int Len)? unk = null;
            int i = 0;

            Span<char> buffer = stackalloc char[256];
            scoped ReadOnlySpan<char> s;

            while (i < w.Length)
            {
                int length;

                if (Char.IsHighSurrogate(w[i]) && i < w.Length - 1 && Char.IsLowSurrogate(w[i + 1]))
                {
                    length = 2;
                    s = w.Slice(i, 2);
                }
                else
                {
                    length = 1;
                    s = w.Slice(i, 1);
                }

                // Add the `continuing_subword_prefix` if relevant
                if (i > 0 && ContinuingSubwordPrefix is not null)
                {
                    if (ContinuingSubwordPrefix.Length + s.Length <= buffer.Length)
                    {
                        ContinuingSubwordPrefix.AsSpan().CopyTo(buffer);
                        s.CopyTo(buffer.Slice(ContinuingSubwordPrefix.Length));
                        s = buffer.Slice(0, ContinuingSubwordPrefix.Length + s.Length);
                    }
                    else
                    {
#if NETCOREAPP
                        s = $"{ContinuingSubwordPrefix}{s}".AsSpan();
#else
                        string s1 = s.Length == 1 ? CharToString(s[0]) : s.ToString();
                        s = $"{ContinuingSubwordPrefix}{s1}".AsSpan();
#endif
                    }
                }

                // Add the `end_of_word_suffix` if relevant
                if (i + length >= w.Length && EndOfWordSuffix is not null)
                {
                    if (s.Length + EndOfWordSuffix.Length <= buffer.Length)
                    {
                        s.CopyTo(buffer);
                        EndOfWordSuffix.AsSpan().CopyTo(buffer.Slice(s.Length));
                        s = buffer.Slice(0, s.Length + EndOfWordSuffix.Length);
                    }
                    else
                    {
#if NETCOREAPP
                        s = $"{s}{EndOfWordSuffix}".AsSpan();
#else
                        string s1 = s.Length == 1 ? CharToString(s[0]) : s.ToString();
                        s = $"{s1}{EndOfWordSuffix}".AsSpan();
#endif
                    }
                }

                if (_vocab.TryGetValue(s, out int id))
                {
                    if (unk.HasValue)
                    {
                        word.Add(unk.Value.Id, unk.Value.Len);
                        unk = null;
                    }
                    word.Add(id, length);
                }
                else if (UnknownToken is not null)
                {
                    if (unk.HasValue)
                    {
                        if (FuseUnknownTokens)
                        {
                            // Fuse unk
                            unk = (unk.Value.Id, unk.Value.Len + length);
                        }
                        else
                        {
                            // Do not fuse unk, add the previous one
                            word.Add(unk.Value.Id, unk.Value.Len);
                            if (!_vocab.TryGetValue(UnknownToken, out int value))
                            {
                                throw new InvalidOperationException($"Unknown Token Out Of Vocabulary.");
                            }
                            unk = (value, length);
                        }
                    }
                    else
                    {
                        if (!_vocab.TryGetValue(UnknownToken, out int value))
                        {
                            throw new InvalidOperationException($"Unknown Token Out Of Vocabulary.");
                        }
                        unk = (value, length);
                    }
                }

                i += length;
            }

            if (unk.HasValue)
            {
                word.Add(unk.Value.Id, unk.Value.Len);
            }

            word.MergeAll(Merges, Dropout);
            return word;
        }

        internal List<Token> WordToTokens(ref Word word) => word.ToTokens(VocabReverse);

        internal List<Token> EncodeWithCache(ReadOnlySpan<char> text)
        {
            Word word;
            if (Cache is not null)
            {
                if (Cache.TryGetValue(text, out word))
                {
                    return WordToTokens(ref word);
                }

                word = MergeWord(text);
                Cache.Set(text.ToString(), word);
            }
            else
            {
                word = MergeWord(text);
            }

            return WordToTokens(ref word);
        }

        internal int WordToIds(ref Word word, IList<int>? accumulatedIds)
        {
            if (accumulatedIds is not null)
            {
                word.PopulateIds(accumulatedIds);
            }

            return word.SymbolsCount;
        }

        internal int EncodeToIdsWithCache(ReadOnlySpan<char> text, IList<int>? accumulatedIds)
        {
            Word word;

            if (Cache is not null)
            {
                if (Cache.TryGetValue(text, out Word hit))
                {
                    return WordToIds(ref hit, accumulatedIds);
                }

                word = MergeWord(text);
                Cache.Set(text.ToString(), word);
            }
            else
            {
                word = MergeWord(text);
            }

            return WordToIds(ref word, accumulatedIds);
        }

        internal static readonly List<Token> EmptyTokensList = new();
    }
}
