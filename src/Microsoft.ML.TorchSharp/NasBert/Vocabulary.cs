// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.TorchSharp.Extensions;
using TorchSharp;

namespace Microsoft.ML.TorchSharp.NasBert
{
    internal sealed class Vocabulary
    {
        public const int NumSpecialSymbols = 4;

        public string PadWord { get; }
        public string EosWord { get; }
        public string UnkWord { get; }
        public string BosWord { get; }

        public int PadIndex { get; }
        public int EosIndex { get; }
        public int UnkIndex { get; }
        public int BosIndex { get; }

        public string MaskWord { get; private set; }
        public int MaskIndex { get; private set; }

        private readonly List<string> _symbols;
        private readonly Dictionary<string, int> _indices;
        private readonly List<int> _counter;

        public IReadOnlyDictionary<string, int> Indices => _indices;
        public IReadOnlyList<int> Counter => _counter;

        /// <exception cref="ArgumentNullException">Any of `pad`, `eos`, `unk` and `bos` is `null`.</exception>
        public Vocabulary(string pad = "<pad>", string eos = "</s>", string unk = "<unk>", string bos = "<s>",
            string[] extraSpecialSymbols = null)
        {
            _indices = new Dictionary<string, int>();
            _counter = new List<int>();
            _symbols = new List<string>();

            PadWord = pad;
            EosWord = eos;
            UnkWord = unk;
            BosWord = bos;
            BosIndex = AddSymbol(bos);
            PadIndex = AddSymbol(pad);
            EosIndex = AddSymbol(eos);
            UnkIndex = AddSymbol(unk);

            if (extraSpecialSymbols != null)
            {
                foreach (var symbol in extraSpecialSymbols)
                {
                    AddSymbol(symbol);
                }
            }
        }

        /// <summary>
        /// Add a word to the vocabulary.
        /// </summary>
        /// <exception cref="ArgumentNullException">`word` is `null`.</exception>
        public int AddSymbol(string word, int n = 1)
        {
            if (word == null)
            {
                throw new ArgumentNullException(nameof(word), $"argument {nameof(word)} should not be null.");
            }

            int idx;
            if (_indices.ContainsKey(word))
            {
                idx = _indices[word];
                _counter[idx] += n;
            }
            else
            {
                idx = _symbols.Count;
                _indices[word] = idx;
                _symbols.Add(word);
                _counter.Add(n);
            }

            return idx;
        }

        public int AddMaskSymbol(string mask = "<mask>")
        {
            MaskWord = mask;
            MaskIndex = AddSymbol(mask);
            return MaskIndex;
        }

        /// <exception cref="ArgumentOutOfRangeException">`idx` is negative.</exception>
        public string this[int idx]
        {
            get
            {
                if (idx < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(idx), $"Index should be non-negative, got {idx}.");
                }

                return idx < _symbols.Count ? _symbols[idx] : UnkWord;
            }
        }

        public int Count => _symbols.Count;

        public bool Equals(Vocabulary other) => _indices.SequenceEqual(other._indices);

        public bool Contains(string symbol) => symbol != null && _indices.ContainsKey(symbol);

        /// <exception cref="ArgumentNullException">`symbol` is `null`.</exception>
        public int IndexOf(string symbol) => _indices.ContainsKey(symbol) ? _indices[symbol] : UnkIndex;

        /// <summary>
        /// Convert a tensor of token indices to a string.
        /// Can optionally remove BPE symbols or escape "&lt;unk&gt;" words.
        /// </summary>
        public string Tensor2String(torch.Tensor tensor, string bpeSymbol = null, bool escapeUnk = false)
        {
            if (tensor.IsNull()) return string.Empty;
            bpeSymbol ??= "";

            List<string> subStrings;

            if (tensor.dim() == 2)
            {
                subStrings = Enumerable.Range(0, (int)tensor.shape[0])
                    .Select(i => Tensor2String(tensor[i], bpeSymbol, escapeUnk))
                    .ToList();
                return string.Join("\n", subStrings);
            }

            subStrings = Enumerable.Range(0, (int)tensor.shape[0])
                .Select(i => _symbols[i])
                .ToList();
            var sentence = string.Join(" ", subStrings);
            return ProcessBpeSymbol(sentence, bpeSymbol);
        }

        private static string ProcessBpeSymbol(string sentence, string bpeSymbol)
        {
            if (bpeSymbol == "sentencepiece")
                sentence = sentence.Replace(" ", "").Replace("\u2581", " ").Trim();
            if (bpeSymbol != null)
                sentence = (sentence + " ").Replace(bpeSymbol, "").TrimEnd();
            return sentence;
        }

        /// <summary>
        /// Updates vocabulary from new vocabulary.
        /// </summary>
        public void Update(Vocabulary newVocabulary)
        {
            foreach (var pair in newVocabulary.Indices)
            {
                AddSymbol(pair.Key, newVocabulary.Counter[pair.Value]);
            }
        }

        /// <summary>
        /// Loads the vocabulary from a text file with the format:
        ///     &lt;symbol0&gt; &lt;count0&gt;
        ///     &lt;symbol1&gt; &lt;count1&gt;
        ///     ...
        /// </summary>
        public static Vocabulary Load(string fileName)
        {
            var vocabulary = new Vocabulary();
            vocabulary.AddFromFile(fileName);
            return vocabulary;
        }

        /// <summary>
        /// Loads a pre-existing vocabulary from a text file and adds its symbols to this instance.
        /// </summary>
        public void AddFromFile(string fileName)
        {
            var lines = File.ReadAllLines(fileName, Encoding.UTF8);

            foreach (var line in lines)
            {
                var splitLine = line.Trim().Split(' ');
                if (splitLine.Length != 2)
                {
                    throw new ArgumentException("Incorrect vocabulary format, expected \"<token> <cnt>\"");
                }

                var word = splitLine[0];
                if (int.TryParse(splitLine[1], out var count))
                {
                    AddSymbol(word, count);
                }
                else
                {
                    throw new ArgumentException($"Cannot parse {splitLine[1]} as an integer. File line: \"{line}\".");
                }
            }
        }
    }
}
