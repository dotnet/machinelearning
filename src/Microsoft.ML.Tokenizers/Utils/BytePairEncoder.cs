// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// This class implements the byte pair encoding algorithm.
    /// </summary>
    internal static class BytePairEncoder
    {
        public static (int Id, int TokenIndex, int TokenLength)[] BytePairEncode(ReadOnlyMemory<byte> mergingBytes, IReadOnlyDictionary<ReadOnlyMemory<byte>, int> ranks, ReadOnlySpan<int> indexMappingSpan)
        {
            if (mergingBytes.Length == 1)
            {
                return [(ranks[mergingBytes], 0, 1)];
            }

            // For large inputs, use heap-based algorithm to avoid O(n²) behavior.
            // Threshold of 128 chosen empirically: linear scan is cache-friendly for small inputs,
            // while heap overhead (O(log n) per operation) becomes worthwhile for larger inputs.
            // Based on upstream tiktoken using 100, adjusted upward for C#'s efficient span operations.
            if (mergingBytes.Length > 128)
            {
                return BytePairEncodeLarge(mergingBytes, ranks, indexMappingSpan);
            }

            (int Index, int Rank)[]? arrayPoolArray = null;
            int requiredLength = mergingBytes.Length + 1;
            Span<(int Index, int Rank)> byteIndicesAndRanks = requiredLength <= 64 ?
                stackalloc (int, int)[64] :
                (arrayPoolArray = ArrayPool<(int, int)>.Shared.Rent(requiredLength));
            byteIndicesAndRanks = byteIndicesAndRanks.Slice(0, requiredLength);

            for (int i = 0; i < byteIndicesAndRanks.Length; i++)
            {
                byteIndicesAndRanks[i] = (i, int.MaxValue);
            }

            int GetRank(Span<(int Index, int Rank)> byteIndicesAndRanks, int startIndex, int skip = 0)
            {
                if (startIndex + skip + 2 < byteIndicesAndRanks.Length)
                {
                    var slice = mergingBytes.SliceStartEnd(byteIndicesAndRanks[startIndex].Index, byteIndicesAndRanks[startIndex + skip + 2].Index);
                    if (ranks.TryGetValue(slice, out var rank))
                    {
                        return rank;
                    }
                }

                return int.MaxValue;
            }

            for (int i = 0; i < byteIndicesAndRanks.Length - 2; i++)
            {
                int rank = GetRank(byteIndicesAndRanks, i);
                if (rank != int.MaxValue)
                {
                    byteIndicesAndRanks[i].Rank = rank;
                }
            }

            while (byteIndicesAndRanks.Length > 1)
            {
                var minRank = (Index: 0, Rank: int.MaxValue);
                for (int i = 0; i < byteIndicesAndRanks.Length - 1; i++)
                {
                    if (byteIndicesAndRanks[i].Rank < minRank.Rank)
                    {
                        minRank = (i, byteIndicesAndRanks[i].Rank);
                    }
                }

                if (minRank.Rank != int.MaxValue)
                {
                    int j = minRank.Index;
                    byteIndicesAndRanks[j].Rank = GetRank(byteIndicesAndRanks, j, 1);
                    if (j > 0)
                    {
                        byteIndicesAndRanks[j - 1].Rank = GetRank(byteIndicesAndRanks, j - 1, 1);
                    }

                    byteIndicesAndRanks.Slice(j + 2).CopyTo(byteIndicesAndRanks.Slice(j + 1));
                    byteIndicesAndRanks = byteIndicesAndRanks.Slice(0, byteIndicesAndRanks.Length - 1);
                }
                else
                {
                    break;
                }
            }

            var result = new (int Id, int TokenIndex, int TokenLength)[byteIndicesAndRanks.Length - 1];
            for (int i = 0; i < result.Length; i++)
            {
                int startIndex = byteIndicesAndRanks[i].Index;
                int endIndex = byteIndicesAndRanks[i + 1].Index;

                int mappedStartIndex = indexMappingSpan[startIndex];
                int mappedEndIndex = indexMappingSpan[endIndex];

                int finalEndIndex = endIndex;

                if (finalEndIndex > 0 && indexMappingSpan[finalEndIndex - 1] == mappedEndIndex)
                {
                    // The partial character/element should be included in the current token.
                    finalEndIndex++;
                    while (finalEndIndex < indexMappingSpan.Length && indexMappingSpan[finalEndIndex] == mappedEndIndex)
                    {
                        finalEndIndex++;
                    }
                }

                result[i] = (ranks[mergingBytes.SliceStartEnd(startIndex, endIndex)], mappedStartIndex, indexMappingSpan[finalEndIndex] - mappedStartIndex);
            }

            if (arrayPoolArray is not null)
            {
                ArrayPool<(int, int)>.Shared.Return(arrayPoolArray);
            }

            return result;
        }

        private struct State
        {
            public int Prev;
            public int End;
            public int NextEnd;
            public int NextRank;
            public int CurRank;
        }

        private struct MergeEntry : IComparable<MergeEntry>
        {
            public int Rank;
            public int Start;

            public int CompareTo(MergeEntry other)
            {
                // Min-heap by rank (lower rank = higher priority)
                // If ranks are equal, prefer lower start index
                int rankComparison = other.Rank.CompareTo(Rank);
                if (rankComparison != 0)
                {
                    return rankComparison;
                }
                return other.Start.CompareTo(Start);
            }
        }

        private static (int Id, int TokenIndex, int TokenLength)[] BytePairEncodeLarge(ReadOnlyMemory<byte> mergingBytes, IReadOnlyDictionary<ReadOnlyMemory<byte>, int> ranks, ReadOnlySpan<int> indexMappingSpan)
        {
            State[]? statePoolArray = null;
            int stateLength = mergingBytes.Length;
            Span<State> state = stateLength <= 256 ?
                stackalloc State[256] :
                (statePoolArray = ArrayPool<State>.Shared.Rent(stateLength));
            state = state.Slice(0, stateLength);

            state[0] = new State
            {
                Prev = int.MaxValue,
                End = 1,
                NextEnd = 2,
                NextRank = int.MaxValue,
                CurRank = int.MaxValue
            };

            // Initial capacity: in the worst case, every adjacent pair is a valid merge candidate.
            // In practice, many pairs won't be in the vocabulary, so this over-allocates slightly,
            // but List resizing is cheap and this avoids multiple reallocations during initialization.
            var heap = new PriorityQueue<MergeEntry>(mergingBytes.Length - 1);

            for (int i = 0; i < mergingBytes.Length - 1; i++)
            {
                var slice = mergingBytes.Slice(i, 2);
                if (ranks.TryGetValue(slice, out int rank))
                {
                    heap.Enqueue(new MergeEntry { Start = i, Rank = rank });
                    state[i].NextRank = rank;
                }

                state[i + 1] = new State
                {
                    Prev = i,
                    End = i + 2,
                    NextEnd = i + 3,
                    NextRank = int.MaxValue,
                    CurRank = int.MaxValue
                };
            }

            // Local function to add a potential merge to the heap.
            // Captures: mergingBytes, ranks from outer scope.
            void PotentialMerge(Span<State> stateSpan, PriorityQueue<MergeEntry> heapQueue, int start, int nextEndItem)
            {
                stateSpan[start].NextEnd = nextEndItem;
                stateSpan[start].NextRank = int.MaxValue;

                if (nextEndItem <= mergingBytes.Length)
                {
                    var slice = mergingBytes.Slice(start, nextEndItem - start);
                    if (ranks.TryGetValue(slice, out int rank))
                    {
                        heapQueue.Enqueue(new MergeEntry { Start = start, Rank = rank });
                        stateSpan[start].NextRank = rank;
                    }
                }
            }

            while (heap.Count > 0)
            {
                MergeEntry left = heap.Dequeue();

                if (left.Rank == int.MaxValue)
                {
                    break;
                }

                if (left.Rank != state[left.Start].NextRank)
                {
                    continue;
                }

                int leftStart = left.Start;
                int rightStart = state[leftStart].End;
                int rightEnd = state[leftStart].NextEnd;
                int rightNextEnd = state[rightStart].NextEnd;

                state[leftStart].CurRank = state[leftStart].NextRank;
                state[leftStart].End = rightEnd;
                PotentialMerge(state, heap, leftStart, rightNextEnd);

                if (rightEnd < state.Length)
                {
                    state[rightEnd].Prev = leftStart;
                }

                if (leftStart > 0)
                {
                    int prevStart = state[leftStart].Prev;
                    PotentialMerge(state, heap, prevStart, rightEnd);
                }

                state[rightStart].NextRank = int.MaxValue;
            }

            var resultList = new List<(int Id, int TokenIndex, int TokenLength)>();
            int currentIndex = 0;

            while (currentIndex < state.Length)
            {
                int startIndex = currentIndex;
                int endIndex = state[currentIndex].End;

                int mappedStartIndex = indexMappingSpan[startIndex];
                int mappedEndIndex = indexMappingSpan[endIndex];

                int finalEndIndex = endIndex;

                // Handle partial characters/elements at token boundaries.
                // If the byte at endIndex-1 maps to the same character as endIndex,
                // extend the token to include the complete character.
                if (finalEndIndex > 0 && indexMappingSpan[finalEndIndex - 1] == mappedEndIndex)
                {
                    finalEndIndex++;
                    while (finalEndIndex < indexMappingSpan.Length && indexMappingSpan[finalEndIndex] == mappedEndIndex)
                    {
                        finalEndIndex++;
                    }
                }

                int tokenId;
                if (state[currentIndex].CurRank != int.MaxValue)
                {
                    tokenId = state[currentIndex].CurRank;
                }
                else
                {
                    tokenId = ranks[mergingBytes.SliceStartEnd(startIndex, endIndex)];
                }

                resultList.Add((tokenId, mappedStartIndex, indexMappingSpan[finalEndIndex] - mappedStartIndex));

                currentIndex = state[currentIndex].End;
            }

            if (statePoolArray is not null)
            {
                ArrayPool<State>.Shared.Return(statePoolArray);
            }

            return resultList.ToArray();
        }

        private static ReadOnlyMemory<byte> SliceStartEnd(this ReadOnlyMemory<byte> memory, int start, int end) => memory.Slice(start, end - start);
    }
}
