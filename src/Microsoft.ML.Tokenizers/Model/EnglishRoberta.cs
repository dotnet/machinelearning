// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Represent the Byte Pair Encoding model.
    /// </summary>
    public sealed class EnglishRoberta : Model
    {
        private readonly HighestOccurrenceMapping _vocabIdToHighestOccurrence;
        private readonly IReadOnlyDictionary<string, int> _vocab;
        private readonly SortedDictionary<int, string> _vocabReverse;
        private readonly Dictionary<(string, string), int> _mergeRanks;
        private readonly IReadOnlyDictionary<char, char> _byteToUnicode;
        private readonly IReadOnlyDictionary<char, char> _unicodeToByte;
        private readonly string[] _charToString;
        private readonly Cache<string, IReadOnlyList<Token>> _cache;

        /// <summary>
        /// Construct tokenizer object to use with the English Robert model.
        /// </summary>
        /// <param name="vocabularyPath">The JSON file path containing the dictionary of string keys and their ids.</param>
        /// <param name="mergePath">The file path containing the tokens's pairs list.</param>
        /// <param name="highestOccurrenceMappingPath">Remap the original GPT-2 model Ids to high occurrence ranks and values.</param>
        public EnglishRoberta(string vocabularyPath, string mergePath, string highestOccurrenceMappingPath)
        {
            if (vocabularyPath is null)
            {
                throw new ArgumentNullException(nameof(vocabularyPath));
            }

            if (mergePath is null)
            {
                throw new ArgumentNullException(nameof(mergePath));
            }

            if (highestOccurrenceMappingPath is null)
            {
                throw new ArgumentNullException(nameof(highestOccurrenceMappingPath));
            }

            using Stream vocabularyStream = File.OpenRead(vocabularyPath);
            using Stream mergeStream = File.OpenRead(mergePath);
            using Stream highestOccurrenceMappingStream = File.OpenRead(highestOccurrenceMappingPath);

            // vocabularyPath like encoder.json
            // merge file like vocab.bpe
            // highestOccurrenceMappingPath like dict.txt

            _vocabIdToHighestOccurrence = GetHighestOccurrenceMapping(highestOccurrenceMappingStream);
            _vocab = GetVocabulary(vocabularyStream);
            _vocabReverse = _vocab.ReverseSorted();
            _mergeRanks = GetMergeRanks(mergeStream);
            int maxCharValue = GetByteToUnicode(out _byteToUnicode);
            _charToString = new string[maxCharValue];
            for (char c = (char)0; c < (char)maxCharValue; c++)
            {
                _charToString[c] = c.ToString();
            }

            _unicodeToByte = _byteToUnicode.Reverse();
            _cache = new Cache<string, IReadOnlyList<Token>>();
        }

        /// <summary>
        /// Construct tokenizer object to use with the English Robert model.
        /// </summary>
        /// <param name="vocabularyStream">The stream of a JSON file containing the dictionary of string keys and their ids.</param>
        /// <param name="mergeStream">The stream of a file containing the tokens's pairs list.</param>
        /// <param name="highestOccurrenceMappingStream">Remap the original GPT-2 model Ids to high occurrence ranks and values.</param>
        public EnglishRoberta(Stream vocabularyStream, Stream mergeStream, Stream highestOccurrenceMappingStream)
        {
            if (vocabularyStream is null)
            {
                throw new ArgumentNullException(nameof(vocabularyStream));
            }

            if (mergeStream is null)
            {
                throw new ArgumentNullException(nameof(mergeStream));
            }

            if (highestOccurrenceMappingStream is null)
            {
                throw new ArgumentNullException(nameof(highestOccurrenceMappingStream));
            }

            _vocabIdToHighestOccurrence = GetHighestOccurrenceMapping(highestOccurrenceMappingStream);
            _vocab = GetVocabulary(vocabularyStream);
            _vocabReverse = _vocab.ReverseSorted();
            _mergeRanks = GetMergeRanks(mergeStream);
            int maxCharValue = GetByteToUnicode(out _byteToUnicode);
            _charToString = new string[maxCharValue];
            for (char c = (char)0; c < (char)maxCharValue; c++)
            {
                _charToString[c] = c.ToString();
            }

            _unicodeToByte = _byteToUnicode.Reverse();
            _cache = new Cache<string, IReadOnlyList<Token>>();
        }

        //
        // Public Model interfaces implementation
        //

        /// <summary>
        /// Gets the dictionary mapping tokens to Ids.
        /// </summary>
        public override IReadOnlyDictionary<string, int> GetVocab() => _vocab;

        /// <summary>
        /// Gets the dictionary size that map tokens to Ids.
        /// </summary>
        public override int GetVocabSize() => _vocab.Count;

        /// <summary>
        /// Map the tokenized Id to the token.
        /// </summary>
        /// <param name="id">The Id to map to the token.</param>
        /// <param name="skipSpecialTokens">Indicate if want to skip the special tokens during the decoding.</param>
        /// <returns>The mapped token of the Id.</returns>
        public override string? IdToToken(int id, bool skipSpecialTokens = false) =>
            skipSpecialTokens && id < 0 ? null : _vocabReverse.TryGetValue(id, out var value) ? value : null;

        /// <summary>
        /// Save the model data into the vocabulary, merges, and occurrence mapping files.
        /// </summary>
        /// <param name="path">The file system path to store the generated files at.</param>
        /// <param name="prefix">Optional prefix for the generated file names.</param>
        /// <returns>The list of all saved files.</returns>
        public override string[] Save(string path, string? prefix = null)
        {
            // Write vocab.json
            string vocabFileNname = prefix is null ? "vocab.json" : $"{prefix}-vocab.json";
            string vocabPath = Path.Combine(path, vocabFileNname);
            string serialized = JsonSerializer.Serialize(_vocabReverse, new JsonSerializerOptions { Converters = { new DictReversingConverter() } });
            File.WriteAllText(vocabPath, serialized, System.Text.Encoding.UTF8);

            // Write merges.txt
            string mergeFileName = prefix is null ? "merges.txt" : $"{prefix}-merges.txt";
            string mergePath = Path.Combine(path, mergeFileName);

            KeyValuePair<(string, string), int>[] mergeArray = _mergeRanks.ToArray();
            Array.Sort(mergeArray, (x, y) => x.Value.CompareTo(y.Value));

            using StreamWriter file = new(mergePath, append: false, System.Text.Encoding.UTF8);
            file.WriteLine("#version: 0.2");
            foreach (var p in mergeArray)
            {
                if (p.Value == int.MaxValue)
                {
                    // Skip the entries which we added during the runs.
                    continue;
                }
                file.WriteLine($"{p.Key.Item1} {p.Key.Item2}");
            }

            // Write high occurrence mapping file
            string highOccurrenceFileName = prefix is null ? "dict.txt" : $"{prefix}-dict.txt";
            string highOccurrencePath = Path.Combine(path, highOccurrenceFileName);
            using StreamWriter file1 = new(highOccurrencePath, append: false, System.Text.Encoding.UTF8);
            _vocabIdToHighestOccurrence.Save(file1);

            return new string[] { vocabPath, mergePath, highOccurrencePath };
        }

        /// <summary>
        /// Tokenize a sequence string to a list of tokens.
        /// </summary>
        /// <param name="sequence">The sequence to tokenize.</param>
        /// <returns>The list of tokens generated from the sequence tokenization.</returns>
        public override IReadOnlyList<Token> Tokenize(string sequence)
        {
            var bpeTokens = new List<string>();

            Span<char> token = stackalloc char[100];
            Span<int> indexMapping = stackalloc int[100];

            if (sequence.Length > 100)
            {
                token = new char[sequence.Length].AsSpan();
                indexMapping = new int[sequence.Length].AsSpan();
            }

            int newTokenIndex = 0;
            for (int i = 0; i < sequence.Length; i++)
            {
                if (_byteToUnicode.TryGetValue(sequence[i], out var value))
                {
                    token[newTokenIndex] = value;
                    indexMapping[newTokenIndex] = i;
                    newTokenIndex++;
                }
            }

            if (newTokenIndex == 0)
            {
                return Bpe.EmptyTokensList;
            }

            IReadOnlyList<Token>? hit = _cache.Get(sequence);
            if (hit is not null)
            {
                return ModifyTokenListOffsets(hit, indexMapping);
            }

            IReadOnlyList<Token> result = BpeToken(token.Slice(0, newTokenIndex), indexMapping);
            _cache.Set(sequence, result);
            return result;
        }

        /// <summary>
        /// Map the token to tokenized Id.
        /// </summary>
        /// <param name="token">The token to map to the Id.</param>
        /// <returns>The mapped Id of the token.</returns>
        public override int? TokenToId(string token) => _vocab.TryGetValue(token, out var value) ? value : null;

        /// <summary>
        /// Gets a trainer object to use in training the model and generate the vocabulary and merges data.
        /// </summary>
        /// <remarks>
        /// This tokenizer doesn't support training so this method will return null. Consider using Bpe.GetTrainer() for training.
        /// </remarks>
        public override Trainer? GetTrainer() => null;

        /// <summary>
        /// Convert a list of tokens Ids to highest occurrence rankings.
        /// </summary>
        /// <param name="ids">The Ids list to map to the high occurrence rank.</param>
        /// <returns>The list of ranks mapped from the list of Ids.</returns>
        public IReadOnlyList<int> IdsToOccurrenceRanks(IReadOnlyList<int> ids)
        {
            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
            }

            List<int> list = new List<int>(ids.Count);

            foreach (int id in ids)
            {
                list.Add(id <= 0 ? -id : _vocabIdToHighestOccurrence.IdToOccurrenceRank(id));
            }

            return list;
        }

        /// <summary>
        /// Convert a list of tokens Ids to highest occurrence values.
        /// </summary>
        /// <param name="ids">The Ids list to map to the high occurrence values.</param>
        /// <returns>The list of occurrence values mapped from the list of Ids.</returns>
        public IReadOnlyList<int> IdsToOccurrenceValues(IReadOnlyList<int> ids)
        {
            if (ids is null)
            {
                throw new ArgumentNullException(nameof(ids));
            }

            List<int> list = new List<int>(ids.Count);

            foreach (int id in ids)
            {
                list.Add(id <= 0 ? 0 : _vocabIdToHighestOccurrence.IdToOccurrenceValue(id));
            }

            return list;
        }

        /// <summary>
        /// Convert a list of highest occurrence rankings to tokens Ids list .
        /// </summary>
        /// <param name="ranks">The high occurrence ranks list to map to the Ids list.</param>
        /// <returns>The list of Ids mapped from the list of ranks.</returns>
        public IReadOnlyList<int> OccurrenceRanksIds(IReadOnlyList<int> ranks)
        {
            if (ranks is null)
            {
                throw new ArgumentNullException(nameof(ranks));
            }

            List<int> list = new List<int>(ranks.Count);

            foreach (int rank in ranks)
            {
                list.Add(_vocabIdToHighestOccurrence.OccurrenceRankToId(rank));
            }

            return list;
        }

        /// <summary>
        /// Gets the index of the pad symbol inside the symbols list.
        /// </summary>
        public int PadIndex => _vocabIdToHighestOccurrence.PadIndex;

        /// <summary>
        /// Gets the symbols list length.
        /// </summary>
        public int SymbolsCount => _vocabIdToHighestOccurrence.Count;

        /// <summary>
        /// Add the mask symbol to the symbols list.
        /// </summary>
        /// <param name="mask">The mask symbol.</param>
        /// <returns>The index of the mask symbol in the symbols list.</returns>
        public int AddMaskSymbol(string mask = "<mask>") => _vocabIdToHighestOccurrence.AddMaskSymbol(mask);

        //
        // Private & Internal methods
        //

        private IReadOnlyList<Token> ModifyTokenListOffsets(IReadOnlyList<Token> tokens, Span<int> indexMapping)
        {
            int index = 0;

            for (int i = 0; i < tokens.Count; i++)
            {
                Debug.Assert(index + tokens[i].Value.Length <= indexMapping.Length);

                if (tokens[i].Offset != (indexMapping[index], indexMapping[index + tokens[i].Value.Length - 1] + 1))
                {
                    List<Token> list = new List<Token>(tokens.Count);
                    for (int j = 0; j < i; j++)
                    {
                        list.Add(tokens[j]);
                    }

                    for (int j = i; j < tokens.Count; j++)
                    {
                        list.Add(new Token(tokens[j].Id, tokens[j].Value, (indexMapping[index], indexMapping[index + tokens[j].Value.Length - 1] + 1)));
                        index += tokens[j].Value.Length;
                    }

                    return list;
                }

                index += tokens[i].Value.Length;
            }

            return tokens;
        }

        private static HighestOccurrenceMapping GetHighestOccurrenceMapping(Stream highestOccurrenceMappingStream) =>
            HighestOccurrenceMapping.Load(highestOccurrenceMappingStream);

        private Dictionary<string, int> GetVocabulary(Stream vocabularyStream)
        {
            Dictionary<string, int>? vocab;
            try
            {
                vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(vocabularyStream) as Dictionary<string, int>;
            }
            catch (Exception e)
            {
                throw new ArgumentException($"Problems met when parsing JSON vocabulary object.{Environment.NewLine}Error message: {e.Message}");
            }

            if (vocab is null)
            {
                throw new ArgumentException($"Failed to read the vocabulary file.");
            }

            if (_vocabIdToHighestOccurrence.BosWord is not null)
            {
                vocab[_vocabIdToHighestOccurrence.BosWord] = -_vocabIdToHighestOccurrence.BosIndex;
            }

            if (_vocabIdToHighestOccurrence.EosWord is not null)
            {
                vocab[_vocabIdToHighestOccurrence.EosWord] = -_vocabIdToHighestOccurrence.EosIndex;
            }

            if (_vocabIdToHighestOccurrence.UnkWord is not null)
            {
                vocab[_vocabIdToHighestOccurrence.UnkWord] = -_vocabIdToHighestOccurrence.UnkIndex;
            }

            if (_vocabIdToHighestOccurrence.PadWord is not null)
            {
                vocab[_vocabIdToHighestOccurrence.PadWord] = -_vocabIdToHighestOccurrence.PadIndex;
            }

            return vocab;
        }

        private Dictionary<(string, string), int> GetMergeRanks(Stream mergeStream)
        {
            List<string> splitContents = new();

            try
            {
                using StreamReader reader = new StreamReader(mergeStream);
                while (reader.Peek() >= 0)
                {
                    splitContents.Add(reader.ReadLine());
                }
            }
            catch (Exception e)
            {
                throw new IOException($"Cannot read the file Merge file.{Environment.NewLine}Error message: {e.Message}", e);
            }

            var mergeRanks = new Dictionary<(string, string), int>();

            // We ignore the first and last line in the file
            for (int i = 1; i < splitContents.Count - 1; i++)
            {
                var split = splitContents[i].Split(' ');
                if (split.Length != 2 || string.IsNullOrEmpty(split[0]) || string.IsNullOrEmpty(split[1]))
                {
                    throw new Exception($"Invalid format of merge file: \"{splitContents[i]}\"");
                }

                mergeRanks.Add((split[0], split[1]), i);
            }

            return mergeRanks;
        }

        /// <summary>
        /// Returns list of utf-8 bytes and a corresponding list of unicode chars.
        /// This mapping is to make unseen characters (such as control characters) displayable.
        /// </summary>
        private static int GetByteToUnicode(out IReadOnlyDictionary<char, char> byteToUnicode)
        {
            var byteToUnicodeMapping = Enumerable.Range('!', '~' - '!' + 1)
                .Concat(Enumerable.Range('¡', '¬' - '¡' + 1))
                .Concat(Enumerable.Range('®', 'ÿ' - '®' + 1))
                .ToDictionary(b => (char)b, b => (char)b);

            const int numChars = 256;
            var n = 0;
            foreach (var b in Enumerable.Range(0, numChars))
            {
                if (!byteToUnicodeMapping.ContainsKey((char)b))
                {
                    byteToUnicodeMapping.Add((char)b, (char)(numChars + n));
                    ++n;
                }
            }

            byteToUnicode = byteToUnicodeMapping;
            return numChars + n;
        }

        /// <summary>
        /// Encode a token into BPE-ed sub-tokens. E.g., "playing" into ["play", "ing"].
        /// </summary>
        private List<Token> BpeToken(Span<char> token, Span<int> indexMapping)
        {
            List<string> word = new(token.Length);
            foreach (char c in token)
            {
                Debug.Assert(c < _charToString.Length);
                word.Add(_charToString[c]);
            }

            HashSet<(string, string)> pairs = WordToPairs(word);

            if (pairs.Count == 0)
            {
                string tokenValue = token.ToString();
                return new List<Token> { new Token(_vocab[tokenValue], tokenValue, (indexMapping[0], indexMapping[token.Length - 1] + 1)) };
            }

            while (true)
            {
                /* while conditions */
                // if only one element left, merge is finished (with the whole word merged)
                if (word.Count == 1)
                {
                    break;
                }

                // get the most frequent bi-gram pair
                var (first, second) = pairs.ArgMin(pair => _mergeRanks.GetOrAdd(pair, int.MaxValue));
                if (!_mergeRanks.ContainsKey((first, second)))
                {
                    break;
                }
                /* end while conditions */

                // search and merge all (first, second) pairs in {word}
                var newWord = new List<string>();
                var i = 0;
                while (i < word.Count)
                {
                    // find the next occurrence of {first} and add the elements before into {newWord}
                    var j = word.IndexOf(first, i);
                    if (j == -1)
                    {
                        newWord.AddRange(word.Skip(i));
                        break;
                    }
                    else
                    {
                        newWord.AddRange(word.Skip(i).Take(j - i));
                        i = j;
                    }

                    // check the next element is {second} or not
                    if (i < word.Count - 1 && word[i + 1] == second)
                    {
                        newWord.Add(first + second);
                        i += 2;
                    }
                    else
                    {
                        newWord.Add(word[i]);
                        i += 1;
                    }
                }

                word = newWord;

                // otherwise, continue merging
                pairs = WordToPairs(word);
            }

            var tokens = new List<Token>(word.Count);
            int index = 0;

            foreach (string w in word)
            {
                tokens.Add(new Token(_vocab[w], w, (indexMapping[index], indexMapping[index + w.Length - 1] + 1)));
                index += w.Length;
            }

            return tokens;
        }

        /// <summary>
        /// Extract element pairs in an aggregating word. E.g. [p, l, ay] into [(p,l), (l,ay)].
        /// If word contains 0 or 1 element, an empty HashSet will be returned.
        /// </summary>
        private static HashSet<(string, string)> WordToPairs(IReadOnlyList<string> word)
        {
            var pairs = new HashSet<(string, string)>();
            if (word.Count <= 1)
            {
                return pairs;
            }

            var prevElem = word[0];
            foreach (var elem in word.Skip(1))
            {
                pairs.Add((prevElem, elem));
                prevElem = elem;
            }

            return pairs;
        }
    }

    /// <summary>
    /// HighestOccurrenceMapping maps the GPT-2 vocabulary Id to highest occurrence value came from dict.txt file
    /// </summary>
    internal sealed class HighestOccurrenceMapping
    {
        public const int NumSpecialSymbols = 4;

        public string? PadWord { get; }
        public string? EosWord { get; }
        public string? UnkWord { get; }
        public string? BosWord { get; }

        public int PadIndex { get; }
        public int EosIndex { get; }
        public int UnkIndex { get; }
        public int BosIndex { get; }

        public string? MaskWord { get; private set; }
        public int MaskIndex { get; private set; }

        private readonly List<(int Id, int OccurrenceScore)> _symbols;
        private readonly Dictionary<int, int> _idToIndex;
        private readonly Dictionary<string, int> _stringSymbolToIndexMapping;

        /// <exception cref="ArgumentNullException">Any of `pad`, `eos`, `unk` and `bos` is `null`.</exception>
        public HighestOccurrenceMapping(string pad = "<pad>", string eos = "</s>", string unk = "<unk>", string bos = "<s>", string[]? extraSpecialSymbols = null)
        {
            _idToIndex = new Dictionary<int, int>();
            _symbols = new List<(int, int)>();
            _stringSymbolToIndexMapping = new Dictionary<string, int>();

            BosWord = bos;
            PadWord = pad;
            EosWord = eos;
            UnkWord = unk;
            BosIndex = ReserveStringSymboleSlot(bos);
            PadIndex = ReserveStringSymboleSlot(pad);
            EosIndex = ReserveStringSymboleSlot(eos);
            UnkIndex = ReserveStringSymboleSlot(unk);

            if (extraSpecialSymbols is not null)
            {
                foreach (var symbol in extraSpecialSymbols)
                {
                    ReserveStringSymboleSlot(symbol);
                }
            }
        }

        public int IdToOccurrenceRank(int id)
        {
            if ((uint)id <= NumSpecialSymbols)
                return id;

            return _idToIndex.TryGetValue(id, out int rank) ? rank : UnkIndex;
        }

        public int IdToOccurrenceValue(int id)
        {
            if ((uint)id <= NumSpecialSymbols)
                return 0;

            if (_idToIndex.TryGetValue(id, out int rank))
            {
                Debug.Assert(rank < _symbols.Count);
                return _symbols[rank].OccurrenceScore;
            }

            return 0;
        }

        public int OccurrenceRankToId(int rank)
        {
            if ((uint)rank >= _symbols.Count)
            {
                return UnkIndex;
            }

            return _symbols[rank].Id;
        }

        private int ReserveStringSymboleSlot(string symbol, int defaultOccurrence = -1)
        {
            if (symbol is null)
            {
                throw new ArgumentNullException(nameof(symbol), $"argument {nameof(symbol)} should not be null.");
            }

            if (!_stringSymbolToIndexMapping.TryGetValue(symbol, out int idx))
            {
                idx = _symbols.Count;
                _symbols.Add((-1, defaultOccurrence));
                _stringSymbolToIndexMapping[symbol] = idx;
            }

            return idx;
        }

        public int AddSymbol(int id, int highOccuranceScore)
        {
            if (!_idToIndex.TryGetValue(id, out int idx))
            {
                idx = _symbols.Count;
                _symbols.Add((id, highOccuranceScore));
                _idToIndex[id] = idx;
            }

            return idx;
        }

        public int AddMaskSymbol(string mask = "<mask>")
        {
            MaskWord = mask;
            MaskIndex = ReserveStringSymboleSlot(mask, 1);
            return MaskIndex;
        }

        /// <exception cref="ArgumentOutOfRangeException">`idx` is negative.</exception>
        public int this[int idx]
        {
            get
            {
                if (idx < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(idx), $"Index should be non-negative, got {idx}.");
                }

                return idx < _symbols.Count ? _symbols[idx].Id : UnkIndex;
            }
        }

        public int Count => _symbols.Count;

        public bool Equals(HighestOccurrenceMapping other) => _idToIndex.SequenceEqual(other._idToIndex);

        public bool Contains(string symbol) => symbol != null && _stringSymbolToIndexMapping.ContainsKey(symbol);

        public bool Contains(int id) => _idToIndex.ContainsKey(id);

        /// <exception cref="ArgumentNullException">`symbol` is `null`.</exception>
        public int IndexOf(int id) => _idToIndex.ContainsKey(id) ? _idToIndex[id] : UnkIndex;

        /// <summary>
        /// Loads the mapping from a text file with the format:
        ///     13 850314647
        ///     262 800385005
        ///     11 800251374
        ///     284 432911125
        ///     ...
        /// </summary>
        public static HighestOccurrenceMapping Load(Stream stream)
        {
            var mapping = new HighestOccurrenceMapping();
            mapping.AddFromStream(stream);
            return mapping;
        }

        /// <summary>
        /// Loads a pre-existing vocabulary from a text stream and adds its symbols to this instance.
        /// </summary>
        public void AddFromStream(Stream stream)
        {
            Debug.Assert(stream is not null);
            using StreamReader reader = new StreamReader(stream);

            while (reader.Peek() >= 0)
            {
                string line = reader.ReadLine();

                var splitLine = line.Trim().Split(' ');
                if (splitLine.Length != 2)
                {
                    throw new ArgumentException("Incorrect vocabulary format, expected \"<token> <cnt>\"");
                }

                if (!int.TryParse(splitLine[1], out int occurrenceScore))
                {
                    throw new ArgumentException($"Cannot parse the line: '{line}'.");
                }

                if (!int.TryParse(splitLine[0], out var id))
                {
                    ReserveStringSymboleSlot(splitLine[0], occurrenceScore);
                }
                else
                {
                    AddSymbol(id, occurrenceScore);
                }
            }
        }

        public void Save(StreamWriter file)
        {
            for (int i = NumSpecialSymbols; i < _symbols.Count; i++)
            {
                (int id, int occurrenceScore) symbol = _symbols[i];
                if (symbol.id >= 0 && symbol.occurrenceScore >= 0)
                {
                    file.WriteLine($"{symbol.id} {symbol.occurrenceScore}");
                }
            }

            foreach (KeyValuePair<string, int> kvp in _stringSymbolToIndexMapping)
            {
                if (_symbols[kvp.Value].OccurrenceScore >= 0)
                {
                    file.WriteLine($"{kvp.Key} {_symbols[kvp.Value].OccurrenceScore}");
                }
            }
        }
    }
}
