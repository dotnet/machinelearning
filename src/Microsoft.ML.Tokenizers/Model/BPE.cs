// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
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

            set
            {
                _unknownToken = value;

                if (value is null)
                {
                    if (VocabReverse.TryGetValue(0, out string v))
                    {
                        VocabReverse.Remove(0);
                        if (Vocab.TryGetValue(v, out int id))
                        {
                            Vocab.Remove(v);
                        }
                    }
                }
                else
                {
                    Vocab[value] = 0;
                    VocabReverse[0] = value;
                }
            }
        }

        /// <summary>
        /// An optional prefix to use on any sub-word that exist only behind another one
        /// </summary>
        public string? ContinuingSubwordPrefix { get; set; }

        /// <summary>
        /// An optional suffix to characterize and end-of-word sub-word
        /// </summary>
        public string? EndOfWordSuffix { get; set; }

        /// <summary>
        /// Gets or sets whether allowing multiple unknown tokens get fused
        /// </summary>
        public bool FuseUnknownTokens { get; set; }

        /// <summary>
        /// Construct a new Bpe model object with no tokenization vocabulary. This constructor is useful only in the training scenario.
        /// </summary>
        public Bpe()
        {
            Vocab = new();
            VocabReverse = new();
            Merges = new();

            UnknownToken = "[Unk]";
        }

        /// <summary>
        /// Construct a new Bpe model object to use for sentence tokenization and tokenizer training.
        /// </summary>
        /// <param name="vocabFile">The JSON file path containing the dictionary of string keys and their ids.</param>
        /// <param name="mergesFile">The file path containing the tokens's pairs list.</param>
        /// <param name="unknownToken"> The unknown token to be used by the model.</param>
        /// <param name="continuingSubwordPrefix">The prefix to attach to sub-word units that don’t represent a beginning of word.</param>
        /// <param name="endOfWordSuffix">The suffix to attach to sub-word units that represent an end of word.</param>
        public Bpe(string vocabFile, string? mergesFile, string? unknownToken = null, string? continuingSubwordPrefix = null, string? endOfWordSuffix = null)
        {
            ContinuingSubwordPrefix = continuingSubwordPrefix;
            EndOfWordSuffix = endOfWordSuffix;

            (Dictionary<string, int>? vocab1, Vec<(string, string)> merges) = ReadFile(vocabFile, mergesFile);
            Vocab = vocab1 ?? new Dictionary<string, int>();

            VocabReverse = new();

            foreach (KeyValuePair<string, int> kvp in Vocab)
            {
                VocabReverse.Add(kvp.Value, kvp.Key);
            }

            if (unknownToken is null && VocabReverse.TryGetValue(0, out string unkToken))
            {
                unknownToken = unkToken;
            }

            UnknownToken = unknownToken;

            int prefixLen = ContinuingSubwordPrefix is null ? 0 : ContinuingSubwordPrefix.Length;

            Merges = new();
            for (int i = 0; i < merges.Count; i++)
            {
                (string a, string b) mergeValues = merges[i];

                if (!Vocab.TryGetValue(mergeValues.a, out int aId))
                {
                    throw new InvalidOperationException($"Trying to merge a token {mergeValues.a} which not exist in the vocabulary.");
                }

                if (!Vocab.TryGetValue(mergeValues.b, out int bId))
                {
                    throw new InvalidOperationException($"Trying to merge a token {mergeValues.b} which not exist in the vocabulary.");
                }

                string newToken = $"{mergeValues.a}{mergeValues.b.Substring(prefixLen)}";
                if (!Vocab.TryGetValue(newToken, out int newId))
                {
                    throw new InvalidOperationException($"Trying to merge a token {newToken} which not exist in the vocabulary.");
                }

                Merges.Add(new Pair<int>(aId, bId), (i, newId));
            }
        }

        /// <summary>
        /// Gets the Bpe decoder object.
        /// </summary>
        public static TokenizerDecoder Decoder { get; } = new BpeDecoder();

        /// <summary>
        /// Tokenize a sequence string to a list of tokens.
        /// </summary>
        /// <param name="sequence">The sequence to tokenize.</param>
        /// <returns>The list of tokens generated from the sequence tokenization.</returns>
        public override IReadOnlyList<Token> Tokenize(string sequence)
        {
            if (sequence.Length == 0)
            {
                return EmptyTokensList;
            }

            if (!Dropout.HasValue)
            {
                return TokenizeWithCache(sequence);
            }

            Word word = MergeWord(sequence);

            return WordToTokens(ref word);
        }

        /// <summary>
        /// Map the token to tokenized Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <returns>The mapped Id of the token.</returns>
        public override int? TokenToId(string token)
        {
            if (Vocab.TryGetValue(token, out int value))
            {
                return value;
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
            if (VocabReverse.TryGetValue(id, out string value))
            {
                return value;
            }

            return null;
        }

        /// <summary>
        /// Gets the dictionary mapping tokens to Ids.
        /// </summary>
        public override IReadOnlyDictionary<string, int> GetVocab() => Vocab;

        /// <summary>
        /// Gets the dictionary size that map tokens to Ids.
        /// </summary>
        public override int GetVocabSize() => Vocab.Count;

        /// <summary>
        /// Gets a trainer object to use in training the model and generate the vocabulary and merges data.
        /// </summary>
        public override Trainer? GetTrainer() => new BpeTrainer();

        /// <summary>
        /// Save the model data into the vocabulary and merges files.
        /// </summary>
        /// <param name="path">The file system path to store the generated files at.</param>
        /// <param name="prefix">Optional prefix for the generated file names.</param>
        /// <returns>The list of all saved files.</returns>
        public override string[] Save(string path, string? prefix = null)
        {
            // Write vocab.json
            string vocabFileNname = prefix is null ? "vocab.json" : $"{prefix}-vocab.json";
            string vocabPath = Path.Combine(path, vocabFileNname);
            string serialized = JsonSerializer.Serialize(VocabReverse, new JsonSerializerOptions { Converters = { new DictReversingConverter() } });
            File.WriteAllText(vocabPath, serialized, System.Text.Encoding.UTF8);

            // Write merges.txt
            string mergeFileName = prefix is null ? "merges.txt" : $"{prefix}-merges.txt";
            string mergePath = Path.Combine(path, mergeFileName);
            (Pair<int> pair, int rank)[] pairsArray = new (Pair<int>, int)[Merges.Count];
            int i = 0;
            foreach (var p in Merges)
            {
                pairsArray[i++] = (p.Key, p.Value.Item1 /* rank */);
            }
            Array.Sort(pairsArray, (x, y) => x.rank.CompareTo(y.rank));
            using StreamWriter file = new(mergePath, append: false, System.Text.Encoding.UTF8);
            file.WriteLine("#version: 0.2 - Trained by `huggingface/tokenizers`");
            foreach (var p in pairsArray)
            {
                file.WriteLine($"{VocabReverse[p.pair.First]} {VocabReverse[p.pair.Second]}");
            }

            return new string[] { vocabPath, mergePath };
        }

        /// Read the given files to extract the vocab and merges
        internal static (Dictionary<string, int>?, Vec<(string, string)>) ReadFile(string? vocab, string? merges)
        {
            Dictionary<string, int>? dic;
            using (Stream stream = File.OpenRead(vocab))
            {
                dic = JsonSerializer.Deserialize<Dictionary<string, int>>(stream) as Dictionary<string, int>;
            }

            return (dic, ConvertMergesToHashmap(merges));
        }

        /// The vocabulary assigns a number to each token.
        internal Dictionary<string, int> Vocab { get; set; }

        /// Contains the mapping between Pairs and their (rank, newId).
        internal Dictionary<Pair<int>, (int, int)> Merges { get; set; }

        /// Contains the cache for optimizing the encoding step.
        internal Cache<string, Word>? Cache { get; set; }

        internal static readonly int DefaultCacheCapacity = 10_000;

        /// Reversed vocabulary, to rebuild sentences.
        internal SortedDictionary<int, string> VocabReverse { get; set; }

        /// Dropout probability for merges. 0 = no dropout is the default. At 1.0, tokenization will
        /// perform no merges, so the result will just be characters.
        internal float? Dropout { get; set; }

        /// Converts the merges strings (for example from `merges.txt` file) with the format
        /// "{pair_a} {pair_b}" into the format expected by the BPE struct
        internal static Vec<(string, string)> ConvertMergesToHashmap(string? mergesFile)
        {
            if (mergesFile is null)
            {
                return new Vec<(string, string)>();
            }

            Vec<(string, string)> merges = new(1000);

            int lineNumber = 0;
            foreach (string line in System.IO.File.ReadLines(mergesFile))
            {
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

        /// Reset the cache.
        internal void ClearCache() => Cache?.Clear();

        private readonly Dictionary<char, string> _charToString = new Dictionary<char, string>();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal string CharToString(char c)
        {
            if (_charToString.TryGetValue(c, out string v))
            {
                return v;
            }

            string s = c.ToString();
            _charToString[c] = s;
            return s;
        }

        internal Word MergeWord(string w)
        {
            Word word = Word.WithCapacity((int)w.Length);
            (int Id, int Len)? unk = null;
            int i = 0;

            while (i < w.Length)
            {
                int length;
                string s;

                if (Char.IsHighSurrogate(w[i]) && i < w.Length - 1 && Char.IsLowSurrogate(w[i + 1]))
                {
                    length = 2;
                    s = w.Substring(i, (int)length);
                }
                else
                {
                    length = 1;
                    s = CharToString(w[i]);
                }

                // Add the `continuing_subword_prefix` if relevant
                if (i > 0 && ContinuingSubwordPrefix is not null)
                {
                    s = $"{ContinuingSubwordPrefix}{s}";
                }

                // Add the `end_of_word_suffix` if relevant
                if (i + length >= w.Length && EndOfWordSuffix is not null)
                {
                    s = $"{s}{EndOfWordSuffix}";
                }

                if (Vocab.TryGetValue(s, out int id))
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
                            if (!Vocab.TryGetValue(UnknownToken, out int value))
                            {
                                throw new InvalidOperationException($"Unknown Token Out Of Vocabulary.");
                            }
                            unk = (value, length);
                        }
                    }
                    else
                    {
                        if (!Vocab.TryGetValue(UnknownToken, out int value))
                        {
                            throw new InvalidOperationException($"Unknown Token Out Of Vocabulary.");
                        }
                        unk = (value, length);
                    }
                }

                i += (int)length;
            }

            if (unk.HasValue)
            {
                word.Add(unk.Value.Id, unk.Value.Len);
            }

            word.MergeAll(Merges, Dropout);
            return word;
        }

        // internal Word.Enumerator WordToTokens(Word word) => word.GetIterator(VocabReverse);
        internal List<Token> WordToTokens(ref Word word)
        {
            List<Token> tokens = new(word.SymbolsCount);

            foreach (Token token in word.GetIterator(VocabReverse))
            {
                tokens.Add(token);
            }

            return tokens;
        }

        internal List<Token> TokenizeWithCache(string sequence)
        {
            if (Cache is not null)
            {
                Word? hit = Cache.Get(sequence);
                if (hit.HasValue)
                {
                    Word w = hit.Value;
                    return WordToTokens(ref w);
                }
            }

            Word word = MergeWord(sequence);
            List<Token> tokens = WordToTokens(ref word);

            if (Cache is not null)
            {
                Cache.Set(sequence, word);
            }

            return tokens;
        }

        internal static readonly List<Token> EmptyTokensList = new();
    }
}
