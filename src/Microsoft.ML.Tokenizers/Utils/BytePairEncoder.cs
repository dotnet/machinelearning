// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// This class implements the byte pair encoding algorithm.
    /// </summary>
    internal static class BytePairEncoder
    {
        public static int[] BytePairEncode(byte[] mergingBytes, IReadOnlyDictionary<byte[], int> ranks)
        {
            if (mergingBytes.Length == 1)
            {
                return new int[] { ranks[mergingBytes] };
            }

            var byteIndicesAndRanks = new List<(int Index, int Rank)>();
            for (int i = 0; i < mergingBytes.Length + 1; i++)
            {
                byteIndicesAndRanks.Add((i, int.MaxValue));
            }
            int GetRank(int startIndex, int skip = 0)
            {
                if (startIndex + skip + 2 < byteIndicesAndRanks.Count)
                {
                    var slice = mergingBytes.Slice(byteIndicesAndRanks[startIndex].Index, byteIndicesAndRanks[startIndex + skip + 2].Index);
                    if (ranks.TryGetValue(slice, out var rank))
                    {
                        return rank;
                    }
                }
                return int.MaxValue;
            }
            for (int i = 0; i < byteIndicesAndRanks.Count - 2; i++)
            {
                var rank = GetRank(i);
                if (rank != int.MaxValue)
                {
                    byteIndicesAndRanks[i] = (byteIndicesAndRanks[i].Index, rank);
                }
            }
            while (byteIndicesAndRanks.Count > 1)
            {
                var minRank = (Index: 0, Rank: int.MaxValue);
                for (int i = 0; i < byteIndicesAndRanks.Count - 1; i++)
                {
                    if (byteIndicesAndRanks[i].Rank < minRank.Rank)
                    {
                        minRank = (i, byteIndicesAndRanks[i].Rank);
                    }
                }
                if (minRank.Rank != int.MaxValue)
                {
                    int j = minRank.Index;
                    byteIndicesAndRanks[j] = (byteIndicesAndRanks[j].Index, GetRank(j, 1));
                    if (j > 0)
                    {
                        byteIndicesAndRanks[j - 1] = (byteIndicesAndRanks[j - 1].Index, GetRank(j - 1, 1));
                    }
                    byteIndicesAndRanks.RemoveAt(j + 1);
                }
                else
                {
                    break;
                }
            }

            var outList = new int[byteIndicesAndRanks.Count - 1];
            for (int i = 0; i < byteIndicesAndRanks.Count - 1; i++)
            {
                outList[i] = ranks[mergingBytes.Slice(byteIndicesAndRanks[i].Index, byteIndicesAndRanks[i + 1].Index)];
            }
            return outList;
        }

        private static T[] Slice<T>(this T[] array, int start, int end)
        {
            var length = end - start;
            var result = new T[length];
            Array.Copy(array, start, result, 0, length);
            return result;
        }
    }
}