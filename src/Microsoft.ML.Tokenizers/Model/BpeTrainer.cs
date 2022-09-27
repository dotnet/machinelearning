// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// The Bpe trainer responsible to train the Bpe model.
    /// </summary>
    public sealed class BpeTrainer : Trainer
    {
        /// <summary>
        /// Gets the size of the final vocabulary, including all tokens and alphabet.
        /// </summary>
        public int VocabSize { get; }

        /// <summary>
        /// Gets the minimum frequency a pair should have in order to be merged.
        /// </summary>
        public int MinFrequency { get; }

        /// <summary>
        /// Gets the list of special tokens the model should know of.
        /// </summary>
        public IReadOnlyList<AddedToken>? SpecialTokens { get; }

        /// <summary>
        /// Gets the maximum different characters to keep in the alphabet.
        /// </summary>
        public int? LimitAlphabet { get; }

        /// <summary>
        /// Gets the list of characters to include in the initial alphabet, even if not seen in the training dataset.
        /// If the strings contain more than one character, only the first one is kept.
        /// </summary>
        public HashSet<char>? InitialAlphabet { get; }

        /// <summary>
        /// Gets the prefix to be used for every sub-word that is not a beginning-of-word.
        /// </summary>
        public string? ContinuingSubwordPrefix { get; }

        /// <summary>
        /// Gets the suffix to be used for every sub-word that is a end-of-word.
        /// </summary>
        public string? EndOfWordSuffix { get; }

        private Dictionary<string, int> Words { get; set; }

        /// <summary>
        /// Construct a new BpeTrainer object using the default values.
        /// </summary>
        public BpeTrainer() : this(null)
        {
        }

        /// <summary>
        /// Construct a new BpeTrainer object.
        /// </summary>
        /// <param name="specialTokens">The list of special tokens the model should know of.</param>
        /// <param name="minFrequency">The minimum frequency a pair should have in order to be merged.</param>
        /// <param name="vocabSize">the size of the final vocabulary, including all tokens and alphabet.</param>
        /// <param name="progress">Callback for the training progress updates.</param>
        /// <param name="limitAlphabet">The list of characters to include in the initial alphabet.</param>
        /// <param name="initialAlphabet">The JSON file path containing the dictionary of string keys and their ids</param>
        /// <param name="continuingSubwordPrefix">the prefix to be used for every sub-word that is not a beginning-of-word.</param>
        /// <param name="endOfWordSuffix">the suffix to be used for every sub-word that is a end-of-word.</param>
        public BpeTrainer(
                IEnumerable<AddedToken>? specialTokens,
                int minFrequency = 0,
                int vocabSize = 30000,
                ReportProgress? progress = null,
                int? limitAlphabet = null,
                HashSet<char>? initialAlphabet = null,
                string? continuingSubwordPrefix = null,
                string? endOfWordSuffix = null)
        {
            MinFrequency = minFrequency;
            VocabSize = vocabSize;
            Progress = progress;
            SpecialTokens = new List<AddedToken>(specialTokens);
            LimitAlphabet = limitAlphabet;
            InitialAlphabet = initialAlphabet;
            ContinuingSubwordPrefix = continuingSubwordPrefix;
            EndOfWordSuffix = endOfWordSuffix;
            Words = new();
        }

        /// Add the provided special tokens to the initial vocabulary
        private void AddSpecialTokens(Dictionary<string, int> w2Id, ref Vec<string> id2W)
        {
            if (SpecialTokens is not null)
            {
                foreach (var token in SpecialTokens)
                {
                    if (!w2Id.ContainsKey(token.Content))
                    {
                        id2W.Push(token.Content);
                        w2Id.Add(token.Content, id2W.Count - 1);
                    }
                }
            }
        }

        private void ComputeAlphabet(Dictionary<string, int> wc, Dictionary<string, int> w2Id, ref Vec<string> id2W)
        {
            // Compute the alphabet from seen words
            Dictionary<char, int> alphabet = new();
            foreach (KeyValuePair<string, int> kvp in wc)
            {
                foreach (char c in kvp.Key)
                {
                    if (alphabet.ContainsKey(c))
                    {
                        alphabet[c] = alphabet[c] + kvp.Value;
                    }
                    else
                    {
                        alphabet[c] = kvp.Value;
                    }
                }
            }

            // Also include anything from the provided initial alphabet
            if (InitialAlphabet is not null)
            {
                foreach (char c in InitialAlphabet)
                {
                    alphabet[c] = int.MaxValue;
                }
            }

            List<KeyValuePair<char, int>> kept = new List<KeyValuePair<char, int>>(alphabet.Count);
            foreach (KeyValuePair<char, int> kvp in alphabet)
            {
                kept.Add(kvp);
            }

            // Compute the number of chars to remove from the alphabet
            // If `limit_alphabet < initial_alphabet.len()`, some of these initial characters
            // will be removed
            int toRemove = LimitAlphabet.HasValue && alphabet.Count > LimitAlphabet.Value ? (int)(alphabet.Count - LimitAlphabet.Value) : 0;

            // Remove the unwanted chars
            if (toRemove > 0)
            {
                kept.Sort((x, y) => (int)x.Value - (int)y.Value);
                kept.RemoveRange(0, toRemove);
            }

            // Keep the initial alphabet (sorted for determinism)
            kept.Sort((x, y) => (int)x.Key - (int)y.Key);

            foreach (KeyValuePair<char, int> kvp in kept)
            {
                string s = kvp.Key.ToString();
                if (!w2Id.ContainsKey(s))
                {
                    id2W.Push(s);
                    w2Id[s] = (int)id2W.Count - 1;
                }
            }
        }

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

        /// Tokenize words and add sub-words to the vocabulary when relevant
        private (Vec<Word>, Vec<int>) TokenizeWords(Dictionary<string, int> wc, Dictionary<string, int> w2Id, ref Vec<string> id2w)
        {
            Vec<Word> words = new Vec<Word>(wc.Count);
            Vec<int> counts = new Vec<int>(wc.Count);

            foreach (KeyValuePair<string, int> kvp in wc)
            {
                Word currentWord = new Word();
                counts.Push(kvp.Value);

                for (int i = 0; i < kvp.Key.Length; i++)
                {
                    char c = kvp.Key[i]; ;
                    string s = CharToString(c);
                    if (w2Id.ContainsKey(s))
                    {
                        // Found the initial char in the authorized alphabet
                        // Add the `continuing_subword_prefix` if relevant
                        if (i != 0 && ContinuingSubwordPrefix is not null)
                        {
                            s = $"{ContinuingSubwordPrefix}{s}";
                        }

                        // Add the `end_of_word_suffix` if relevant
                        if (i == kvp.Key.Length - 1 && EndOfWordSuffix is not null)
                        {
                            s = $"{s}{EndOfWordSuffix}";
                        }

                        // Insert the new formed string if necessary
                        if (!w2Id.ContainsKey(s))
                        {
                            id2w.Push(s);
                            w2Id[s] = (int)(id2w.Count - 1);
                        }

                        currentWord.Add(w2Id[s], 1);  // We do not care about the len here
                    }
                }

                words.Push(currentWord);
                Progress?.Invoke(new Progress(ProgressState.Increment, null, 1));
            }

            return (words, counts);
        }

        private (Dictionary<Pair<int>, int>, Dictionary<Pair<int>, HashSet<int>>) CountPairs(ref Vec<Word> words, ref Vec<int> counts)
        {
            if (words.Count <= 0)
            {
                return (new(), new());
            }

            Dictionary<Pair<int>, int> pairCounts = new Dictionary<Pair<int>, int>();
            Dictionary<Pair<int>, HashSet<int>> whereToUpdate = new();

            for (int i = 0; i < words.Count; i++)
            {
                ref Word word = ref words[i];

                int j = 0;
                Vec<int> chars = word.GetChars();
                while (j < chars.Count - 1)
                {
                    Pair<int> curPair = new Pair<int>(chars[j], chars[j + 1]);

                    // Initialize pair_counts and where_to_update for this pair if we just saw it
                    if (!pairCounts.ContainsKey(curPair))
                    {
                        pairCounts[curPair] = 0;
                    }

                    // Then update counts
                    int count = counts[i];

                    if (!whereToUpdate.TryGetValue(curPair, out HashSet<int> h))
                    {
                        h = new HashSet<int>();
                        whereToUpdate[curPair] = h;
                    }

                    h.Add(i);

                    pairCounts[curPair] = pairCounts[curPair] + (int)count;

                    j++;
                }

                Progress?.Invoke(new Progress(ProgressState.Increment, null, 1));
            }

            return (pairCounts, whereToUpdate);
        }

        private IReadOnlyList<AddedToken>? DoTrain(Dictionary<string, int> wordCounts, Bpe model)
        {
            Dictionary<string, int> wordToId = new(VocabSize);
            Vec<string> idToWord = new(VocabSize);

            //
            // 1. Add all special tokens to the vocabulary
            //
            AddSpecialTokens(wordToId, ref idToWord);

            //
            // 2. Compute the initial alphabet
            //
            ComputeAlphabet(wordCounts, wordToId, ref idToWord);

            //
            // 3. Tokenize words
            //
            Progress?.Invoke(new Progress(ProgressState.Start, "Tokenize words", wordCounts.Count));
            (Vec<Word> words, Vec<int> counts) = TokenizeWords(wordCounts, wordToId, ref idToWord);
            Progress?.Invoke(new Progress(ProgressState.End, null, wordCounts.Count));

            //
            // 4. Count pairs in words
            //
            Progress?.Invoke(new Progress(ProgressState.Start, "Count pairs", wordCounts.Count));
            (Dictionary<Pair<int>, int> pairCounts, Dictionary<Pair<int>, HashSet<int>> whereToUpdate) = CountPairs(ref words, ref counts);

            // Insert them in the queue
            PriorityQueue<BpeTrainerMerge> queue = new(pairCounts.Count);

            foreach (KeyValuePair<Pair<int>, HashSet<int>> kvp in whereToUpdate)
            {
                int count = pairCounts[kvp.Key];
                if (count > 0)
                {
                    queue.Enqueue(new BpeTrainerMerge(kvp.Key, count, kvp.Value));
                }
            }

            whereToUpdate.Clear();
            Progress?.Invoke(new Progress(ProgressState.End, null, words.Count));

            //
            // 5. Do merges
            //
            Progress?.Invoke(new Progress(ProgressState.End, "Compute merges", VocabSize));
            Vec<(Pair<int>, int)> merges = new();

            while (true)
            {
                // Stop as soon as we have a big enough vocabulary
                if (wordToId.Count >= VocabSize || queue.Count == 0)
                {
                    break;
                }

                BpeTrainerMerge top = queue.Dequeue();

                if (top.Count != pairCounts[top.Pair])
                {
                    top.Count = pairCounts[top.Pair];
                    queue.Enqueue(top);
                    continue;
                }

                if (top.Count < 1 || MinFrequency > top.Count)
                {
                    break;
                }

                string partA = idToWord[(int)top.Pair.First];
                string partB = idToWord[(int)top.Pair.Second];

                // Build new token
                if (ContinuingSubwordPrefix is not null)
                {
                    if (partB.StartsWith(ContinuingSubwordPrefix, StringComparison.Ordinal))
                    {
                        partB = partB.Substring(ContinuingSubwordPrefix.Length);
                    }
                }

                string newToken = $"{partA}{partB}";

                // Insert new token if it does not already exist
                if (!wordToId.TryGetValue(newToken, out int newTokenId))
                {
                    newTokenId = (int)idToWord.Count;
                    idToWord.Push(newToken);
                    wordToId[newToken] = newTokenId;
                }

                merges.Push((top.Pair, newTokenId));

                Vec<((Pair<int>, int), int)> changes = new();

                // Merge the new pair in every words
                foreach (int i in top.Pos)
                {
                    ref Word w = ref words[(int)i];

                    Vec<(Pair<int>, int)> m = w.Merge(top.Pair.First, top.Pair.Second, newTokenId);

                    for (int j = 0; j < m.Count; j++)
                    {
                        changes.Push((m[j], i));
                    }
                }

                // Introduce new formed pairs
                for (int j = 0; j < changes.Count; j++)
                {
                    ((Pair<int> p, int change), int iw) = changes[j];
                    int count = (int)(change * counts[(int)iw]);

                    pairCounts[p] = pairCounts.TryGetValue(p, out int c) ? c + count : count;

                    if (change > 0)
                    {
                        if (!whereToUpdate.TryGetValue(p, out HashSet<int> h))
                        {
                            h = new();
                            whereToUpdate[p] = h;
                        }
                        h.Add(iw);
                    }
                }

                foreach (KeyValuePair<Pair<int>, HashSet<int>> kvp in whereToUpdate)
                {
                    int count = pairCounts[kvp.Key];
                    if (count > 0)
                    {
                        queue.Enqueue(new BpeTrainerMerge(kvp.Key, count, kvp.Value));
                    }
                }
                whereToUpdate.Clear();

                Progress?.Invoke(new Progress(ProgressState.Increment, null, 1));
            }

            Progress?.Invoke(new Progress(ProgressState.End, null, merges.Count));

            // Transfer new vocab & options to model
            model.Vocab = wordToId;

            if (SpecialTokens is not null)
            {
                for (int i = 0; i < SpecialTokens.Count; i++)
                {
                    model.Vocab[SpecialTokens[(int)i].Content] = i;
                }

                if (SpecialTokens.Count > 0)
                {
                    model.UnknownToken = SpecialTokens[0].Content;
                }
            }

            model.VocabReverse = new();

            foreach (KeyValuePair<string, int> kvp in model.Vocab)
            {
                model.VocabReverse[kvp.Value] = kvp.Key;
            }

            model.Merges = new();

            for (int i = 0; i < merges.Count; i++)
            {
                (Pair<int> p, int v) = merges[i];
                model.Merges[p] = ((int)i, v);
            }

            model.ContinuingSubwordPrefix = ContinuingSubwordPrefix;
            model.EndOfWordSuffix = EndOfWordSuffix;

            return SpecialTokens;
        }

        /// <summary>
        /// Process the input sequences and feed the result to the model.
        /// </summary>
        /// <param name="sequences">The list of sequences to feed the trainer.</param>
        /// <param name="process">Optional process callback for reporting the training progress update.</param>
        public override void Feed(IEnumerable<string> sequences, Func<string, IEnumerable<string>> process)
        {
            foreach (string s in sequences)
            {
                foreach (string word in process(s))
                {
                    Words[word] = Words.TryGetValue(word, out int value) ? value + 1 : 1;
                }
            }
        }

        /// <summary>
        /// Perform the actual training and update the input model with the new vocabularies and merges data.
        /// </summary>
        /// <param name="model">The model to train. This has to be BpeModel to work with BpeTrainer.</param>
        /// <returns>The list of the added tokens.</returns>
        public override IReadOnlyList<AddedToken>? Train(Model model)
        {
            if (model is Bpe bpeModel)
            {
                return DoTrain(Words, bpeModel);
            }

            throw new Exception($"BpeTrainer work only with Bpe Model.");
        }
    }

    internal struct BpeTrainerMerge : IEquatable<BpeTrainerMerge>, IComparable<BpeTrainerMerge>
    {
        public BpeTrainerMerge(Pair<int> pair, int count, HashSet<int> pos)
        {
            Pair = pair;
            Count = count;
            Pos = pos;
        }

        public Pair<int> Pair { get; set; }
        public int Count { get; set; }
        public HashSet<int> Pos { get; set; }

        public int CompareTo(BpeTrainerMerge other)
        {
            if (Count != other.Count)
            {
                // return Count.CompareTo(other.Count);
                return other.Count.CompareTo(Count);
            }

            return Pair.CompareTo(other.Pair);
        }

        public override int GetHashCode()
        {
            int hashcode = 23;
            hashcode = (hashcode * 37) + Count.GetHashCode();
            hashcode = (hashcode * 37) + Pair.First.GetHashCode();
            hashcode = (hashcode * 37) + Pair.Second.GetHashCode();
            return hashcode;
        }

        public bool Equals(BpeTrainerMerge other) => Count == other.Count && Pair.Equals(other.Pair);
    }
}
