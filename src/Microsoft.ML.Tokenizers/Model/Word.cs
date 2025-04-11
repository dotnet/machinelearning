// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    internal struct Word
    {
        [ThreadStatic]
        private static Random? _random;
        private Vec<Symbol> _symbols;

        public Word() => _symbols = new Vec<Symbol>();

        public Word(int capacity)
        {
            if (capacity > int.MaxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(capacity));
            }
            _symbols = new Vec<Symbol>(capacity);
        }

        public static Word WithCapacity(int capacity) => new Word(capacity);

        public int SymbolsCount => _symbols.Count;

        public void Add(int c, int charLength)
        {
            int prev = -1;
            int next = -1;

            int len = _symbols.Count;

            if (len > 0)
            {
                // Update `next` on the previous one
                _symbols[len - 1].Next = len;
                prev = len - 1;
            }

            _symbols.Push(new Symbol(c, prev, next, charLength));
        }

        public Vec<(Pair<int>, int)> Merge(int c1, int c2, int replacement)
        {
            Vec<(Pair<int>, int)> changes = new();
            int i = 0;

            while (true)
            {
                if (i >= _symbols.Count)
                {
                    break;
                }

                // Found a pair
                if (_symbols[i].C == c1 && i + 1 < _symbols.Count && _symbols[i + 1].C == c2)
                {
                    Symbol first = _symbols[i];
                    Symbol second = _symbols[i + 1];

                    // If there are other characters before the pair
                    if (i > 0)
                    {
                        changes.Push((Pair<int>.Create(_symbols[i - 1].C, first.C), -1));
                        changes.Push((Pair<int>.Create(_symbols[i - 1].C, replacement), 1));
                    }

                    // Remove in place
                    // Insert replacement before first char of pair
                    // Remove first char of pair
                    // And then the second

                    _symbols[i].C = replacement;
                    _symbols[i].Prev = first.Prev;
                    _symbols[i].Next = second.Next;
                    _symbols[i].Len = first.Len + second.Len;

                    _symbols.Remove(i + 1);

                    // If there are other characters after the pair
                    if (i < _symbols.Count - 1)
                    {
                        changes.Push((Pair<int>.Create(second.C, _symbols[i + 1].C), -1));
                        changes.Push((Pair<int>.Create(replacement, _symbols[i + 1].C), 1));
                    }
                }

                i += 1;
            }

            return changes;
        }

        public void MergeAll(Dictionary<Pair<int>, (int, int)> merges, float? dropout, ref PriorityQueue<Merge>? priorityQueue)
        {
            priorityQueue ??= new PriorityQueue<Merge>(_symbols.Count);
            priorityQueue.Clear();

            Vec<Merge> skip = new Vec<Merge>(priorityQueue.Count);

            for (int i = 0; i < _symbols.Count - 1; i++)
            {
                if (merges.TryGetValue(Pair<int>.Create(_symbols[i].C, _symbols[i + 1].C), out (int m1, int m2) value))
                {
                    priorityQueue.Enqueue(new Merge(i, value.m1, value.m2));
                }
            }

            while (priorityQueue.Count > 0)
            {
                Merge top = priorityQueue.Dequeue();
                if (dropout.HasValue && (_random ??= new()).NextDouble() < dropout)
                {
                    skip.Push(top);
                }
                else
                {
                    // Re-insert the skipped elements
                    for (int i = 0; i < skip.Count; i++)
                    {
                        priorityQueue.Enqueue(skip[i]);
                    }
                    skip.Clear();

                    // Do nothing if we are the last symbol
                    if (_symbols.Count == 0 || _symbols[top.Pos].Len == 0 || _symbols[top.Pos].Next == -1)
                    {
                        continue;
                    }

                    int nextPos = _symbols[top.Pos].Next;
                    Symbol right = _symbols[nextPos];

                    // Make sure we are not processing an expired queue entry
                    Pair<int> targetNewPair = Pair<int>.Create(_symbols[top.Pos].C, right.C);
                    if (!merges.TryGetValue(targetNewPair, out (int m1, int m2) value) || value.m2 != top.NewId)
                    {
                        continue;
                    }

                    // Otherwise, let's merge
                    _symbols[top.Pos].MergeWith(ref right, top.NewId);

                    // Tag the right part as removed
                    _symbols[nextPos].Len = 0;

                    // Update `prev` on the new `next` to the current pos
                    if (right.Next > -1 && right.Next < _symbols.Count)
                    {
                        _symbols[right.Next].Prev = top.Pos;
                    }

                    // Insert the new pair formed with the previous symbol
                    Symbol current = _symbols[top.Pos];
                    if (current.Prev >= 0)
                    {
                        int prev = current.Prev;
                        Symbol prevSymbol = _symbols[prev];
                        Pair<int> newPair = Pair<int>.Create(prevSymbol.C, current.C);

                        if (merges.TryGetValue(newPair, out value))
                        {
                            priorityQueue.Enqueue(new Merge(current.Prev, value.m1, value.m2));
                        }
                    }

                    // Insert the new pair formed with the next symbol
                    int next = current.Next;
                    if ((uint)next < (uint)_symbols.Count)
                    {
                        Symbol nextSymbol = _symbols[next];
                        Pair<int> newPair = Pair<int>.Create(current.C, nextSymbol.C);
                        if (merges.TryGetValue(newPair, out value))
                        {
                            priorityQueue.Enqueue(new Merge(top.Pos, value.m1, value.m2));
                        }
                    }
                }
            }

            // Filter out the removed symbols
            for (int i = _symbols.Count - 1; i >= 0; i--)
            {
                if (_symbols[i].Len == 0)
                {
                    _symbols.Remove(i);
                }
            }
        }

        public void PopulateIds(IList<int> accumulatedIds)
        {
            for (int i = 0; i < SymbolsCount; i++)
            {
                accumulatedIds.Add(_symbols[i].C);
            }
        }

        public int PopulateIdsUpToMax(IList<int> accumulatedIds, int maxTokens, out int charsConsumed)
        {
            charsConsumed = 0;

            int count = Math.Min(SymbolsCount, maxTokens);

            for (int i = 0; i < count; i++)
            {
                accumulatedIds.Add(_symbols[i].C);
                charsConsumed += _symbols[i].Len;
            }

            return count;
        }

        public int PopulateIdsUpToMaxFromEnd(IList<int> accumulatedIds, int maxTokens, int fullTextLength, out int textIndex)
        {
            textIndex = fullTextLength;

            int count = Math.Min(SymbolsCount, maxTokens);

            for (int i = SymbolsCount - count; i < SymbolsCount; i++)
            {
                accumulatedIds.Add(_symbols[i].C);
                textIndex -= _symbols[i].Len;
            }

            return count;
        }

        public int CountIdsUpToMax(int maxTokens, out int charsConsumed)
        {
            charsConsumed = 0;

            int count = Math.Min(SymbolsCount, maxTokens);

            for (int i = 0; i < count; i++)
            {
                charsConsumed += _symbols[i].Len;
            }

            return count;
        }

        public int CountIdsUpToMaxFromEnd(int maxTokens, int fullTextLength, out int textIndex)
        {
            textIndex = fullTextLength;

            int count = Math.Min(SymbolsCount, maxTokens);

            for (int i = SymbolsCount - count; i < SymbolsCount; i++)
            {
                textIndex -= _symbols[i].Len;
            }

            return count;
        }

        public Vec<int> GetChars()
        {
            Vec<int> chars = new Vec<int>();
            for (int i = 0; i < _symbols.Count; i++)
            {
                chars.Push(_symbols[i].C);
            }

            return chars;
        }

        public override string ToString()
        {
            if (_symbols.Count == 0)
            {
                return "[]";
            }

            StringBuilder sb = new StringBuilder();
            sb.Append('[');
            sb.Append($"{_symbols[0].C}");
            for (int i = 1; i < _symbols.Count; i++)
            {
                sb.Append($", {_symbols[i].C}");
            }
            sb.Append(']');
            return sb.ToString();
        }

        public void ToTokens(SortedDictionary<int, string> vocabReverse, List<EncodedToken> tokens, int offset, ReadOnlySpan<int> mapping)
        {
            int index = 0;

            if (mapping.IsEmpty)
            {
                for (int i = 0; i < SymbolsCount; i++)
                {
                    int endIndex = index + _symbols[i].Len;
                    tokens.Add(new EncodedToken(_symbols[i].C, vocabReverse[_symbols[i].C], new Range(index + offset, index + offset + _symbols[i].Len)));
                    index += _symbols[i].Len;
                }
            }
            else
            {
                for (int i = 0; i < SymbolsCount; i++)
                {
                    int endIndex = index + _symbols[i].Len;

                    int mappedIndex = mapping[index];
                    int mappedEndIndex = mapping[endIndex - 1] + 1;

                    tokens.Add(new EncodedToken(_symbols[i].C, vocabReverse[_symbols[i].C], new Range(mappedIndex + offset, mappedEndIndex + offset)));
                    index += _symbols[i].Len;
                }
            }
        }
    }
}
