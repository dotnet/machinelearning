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

        private static ReadOnlyMemory<byte> SliceStartEnd(this ReadOnlyMemory<byte> memory, int start, int end) => memory.Slice(start, end - start);
    }
}
