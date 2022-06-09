// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.Runtime;
using Microsoft.ML.TorchSharp.Extensions;
using Microsoft.ML.TorchSharp.Utils;
using Newtonsoft.Json;

namespace Microsoft.ML.TorchSharp.NasBert.Preprocessing
{
    internal class BpeTokenizer
    {
        // Singleton
        private static BpeTokenizer _instance;
        private static string _path;

        private const string EncoderJsonName = "encoder.json";
        private const string MergeName = "vocab.bpe";
        private const string DictName = "dict.txt";

        private static readonly Uri _encoderJsonUrl = new Uri("https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json");
        private static readonly Uri _mergeUrl = new Uri("https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe");
        private static readonly Uri _dictUrl = new Uri("https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt");

        internal readonly Vocabulary Vocabulary;
        private readonly IReadOnlyDictionary<string, int> _encoder;
        private readonly IReadOnlyDictionary<int, string> _decoder;
        private readonly (string, string)[] _merges;
        private readonly DefaultDictionary<(string, string), int> _mergeRanks;
        private readonly IReadOnlyDictionary<char, char> _byteToUnicode;
        private readonly IReadOnlyDictionary<char, char> _unicodeToByte;
        private const string Pattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

        internal const int InitToken = 0;
        internal const int SeperatorToken = 2;

        /// <summary>
        /// Get the singleton of BPE tokenizer with vocabulary files in <paramref name="path"/>.
        /// </summary>
        public static BpeTokenizer GetInstance(IChannel ch, string path = null)
        {
            if (_instance == null)
            {
                _path = path ?? "";
                _instance = new BpeTokenizer(ch);
            }

            return _instance;
        }

        private BpeTokenizer(IChannel ch)
        {
            Vocabulary = GetVocabulary(ch);
            _encoder = GetEncoder(ch);
            _decoder = _encoder.Reverse();
            _merges = GetMerges(ch);
            _mergeRanks = GetMergeRanks();
            _byteToUnicode = GetByteToUnicode();
            _unicodeToByte = _byteToUnicode.Reverse();
        }

        private static Vocabulary GetVocabulary(IChannel ch)
        {
            _ = FileUtils.LoadFromFileOrDownloadFromWeb(_path, DictName, _dictUrl, ch);
            return Vocabulary.Load(System.IO.Path.Combine(_path, DictName));
        }

        private Dictionary<string, int> GetEncoder(IChannel ch)
        {
            var contents = FileUtils.LoadFromFileOrDownloadFromWeb(_path, EncoderJsonName, _encoderJsonUrl, ch);

            // Parse JSON
            try
            {
                var jsonResult = (Dictionary<string, int>)JsonConvert.DeserializeObject<Dictionary<string, int>>(contents);

                jsonResult[Vocabulary.BosWord] = -Vocabulary.BosIndex;
                jsonResult[Vocabulary.EosWord] = -Vocabulary.EosIndex;
                jsonResult[Vocabulary.UnkWord] = -Vocabulary.UnkIndex;
                jsonResult[Vocabulary.PadWord] = -Vocabulary.PadIndex;

                return jsonResult;
            }
            catch (JsonException e)
            {
                throw new JsonException($"Problems met when parsing JSON object in {EncoderJsonName}.\n" +
                                        $"Error message: {e.Message}");
            }
        }

        private static (string, string)[] GetMerges(IChannel ch)
        {
            var contents = FileUtils.LoadFromFileOrDownloadFromWeb(_path, MergeName, _mergeUrl, ch);

            // Parse merge info
            var splitContents = contents.Split('\n');
            var merges = splitContents.Where((source, index) => index != splitContents.Length - 1 && index != 0).Select(line =>     // Was [1..^1] originally
            {
                var split = line.Split(' ');
                if (split.Length != 2 || string.IsNullOrEmpty(split[0]) || string.IsNullOrEmpty(split[1]))
                {
                    throw new Exception("Invalid format of merge file: \"{line}\"");
                }
                return (split[0], split[1]);
            }).ToArray();
            return merges;
        }

        private DefaultDictionary<(string, string), int> GetMergeRanks()
        {
            var mergeRanks = new DefaultDictionary<(string, string), int>(() => int.MaxValue);
            for (var i = 0; i < _merges.Length; ++i)
            {
                mergeRanks.Add(_merges[i], i);
            }

            return mergeRanks;
        }

        /// <summary>
        /// Returns list of utf-8 bytes and a corresponding list of unicode chars.
        /// This mapping is to make unseen characters (such as control characters) displayable.
        /// </summary>
        private static Dictionary<char, char> GetByteToUnicode()
        {
            var byteToUnicode = Enumerable.Range('!', '~' - '!' + 1)
                .Concat(Enumerable.Range('¡', '¬' - '¡' + 1))
                .Concat(Enumerable.Range('®', 'ÿ' - '®' + 1))
                .ToDictionary(b => (char)b, b => (char)b);

            const int numChars = 256;
            var n = 0;
            foreach (var b in Enumerable.Range(0, numChars))
            {
                if (byteToUnicode.ContainsKey((char)b)) continue;
                byteToUnicode.Add((char)b, (char)(numChars + n));
                ++n;
            }

            return byteToUnicode;
        }

        /// <summary>
        /// Encode a text string to the token IDs using BPE.
        /// </summary>
        public IList<int> Encode(string text)
        {
            var bpeTokens = Bpe(text);
            var bpeTokenIds = bpeTokens.Select(token => _encoder[token]).ToList();
            return bpeTokenIds;
        }

        /// <summary>
        /// Encode a text string to the converted token IDs using BPE.
        /// </summary>
        public IList<int> EncodeToConverted(string text)
        {
            var origin = Encode(text);
            var converted = origin.Select(token => token <= 0
                ? -token
                : Vocabulary.IndexOf($"{token}"));
            return converted.ToList();
        }

        /// <summary>
        /// Decode origin token IDs and return the corresponding string.
        /// Origin token ID is the token ID after BPE processing.
        /// </summary>
        public string DecodeOrigin(IEnumerable<int> tokenIds)
        {
            var tokens = tokenIds.Select(id => _decoder[id]);
            var textChars = string.Join("", tokens)
                .Where(c => _unicodeToByte.ContainsKey(c))
                .Select(c => _unicodeToByte[c]);
            var text = new string(textChars.ToArray());
            return text;
        }

        /// <summary>
        /// Decode converted token IDs and return the corresponding string.
        /// Origin token ID is the token ID after BPE processing, and <see cref="Vocabulary"/> defines a mapping
        ///     between origin token IDs and converted token IDs.
        /// </summary>
        public string DecodeConverted(IEnumerable<int> tokenIds)
        {
            // 1. not to decode padding tokens
            // 2. special tokens (BOS, EOS, PAD, UNK) in vocabulary are presented as strings rather than integers,
            //    so they will cause parsing failure. We treat their IDs as negative integers to avoid conflict with
            //    normal tokens. Other unrecognized tokens will be treated as token#13, which is ".".
            var tokenArray = tokenIds
                .Where(token => token != Vocabulary.PadIndex)
                .Select(token => int.TryParse(Vocabulary[token], out var result) ? result :
                    token < Vocabulary.NumSpecialSymbols ? -token :
                    13)
                .ToArray();
            var text = DecodeOrigin(tokenArray);
            return text;
        }


        /// <summary>
        /// Encode text with several tokens into BPE-ed sub-tokens.
        /// </summary>
        private List<string> Bpe(string text)
        {
            var bpeTokens = new List<string>();
            var tokenMatches = Regex.Matches(text, Pattern);
            for (var i = 0; i < tokenMatches.Count; ++i)
            {
                var token = tokenMatches[i].Value;
                var convertedToken = string.Join("", token
                    .Where(b => _byteToUnicode.ContainsKey(b))
                    .Select(b => _byteToUnicode[b]));
                if (convertedToken.Length == 0) continue;
                bpeTokens.AddRange(BpeToken(convertedToken));
            }

            return bpeTokens;
        }

        /// <summary>
        /// Encode a token into BPE-ed sub-tokens. E.g., "playing" into ["play", "ing"].
        /// </summary>
        private List<string> BpeToken(string token)
        {
            var word = token.Select(c => c.ToString()).ToList();
            var pairs = WordToPairs(word);

            if (pairs.Count == 0)
            {
                return new List<string> { token };
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
                var (first, second) = pairs.ArgMin(pair => _mergeRanks[pair]);
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

            return word;
        }

        /// <summary>
        /// Extract element pairs in an aggregating word. E.g. [p, l, ay] into [(p,l), (l,ay)].
        /// If word contains 0 or 1 element, an empty HashSet will be returned.
        /// </summary>
        private static HashSet<(string, string)> WordToPairs(IReadOnlyList<string> word)
        {
            var pairs = new HashSet<(string, string)>();
            if (word.Count <= 1) return pairs;

            var prevElem = word[0];
            foreach (var elem in word.Skip(1))
            {
                pairs.Add((prevElem, elem));
                prevElem = elem;
            }

            return pairs;
        }
    }
}
