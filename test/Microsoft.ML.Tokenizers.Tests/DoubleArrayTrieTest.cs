// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class DoubleArrayTrieTests
    {
        private static readonly (string Key, int Value)[] _entries = new[]
        {
            // Higher ranges and surrogates
            ("\uD83D\uDE00ab", 26),     // \uF0B7 ordered before \uD83D\uDE00 in utf-8
            ("\uF0B7ab", 25),           // \uF0B7 ordered before \uD83D\uDE00 in utf-8

            // Different scripts
            ("\u0627\u0644", 24),           // Arabic
            ("\u0391\u0393", 23),           // Greek

            // Higher ranges and surrogates proceeded by Latin
            ("a\uD83D\uDE00b", 22),     // \uF0B7 ordered before \uD83D\uDE00 in utf-8
            ("a\uF0B7b", 21),           // \uF0B7 ordered before \uD83D\uDE00 in utf-8

            ("abcdefghijklmnopqrstu", 20),
            ("abcdefghijklmnopqrst", 19),
            ("abcdefghijklmnopqrs", 18),
            ("abcdefghijklmnopqr", 17),
            ("abcdefghijklmnopq", 16),
            ("abcdefghijklmnop", 15),
            ("abcdefghijklmno", 14),
            ("abcdefghijklmn", 13),
            ("abcdefghijklm", 12),
            ("abcdefghijkl", 11),
            ("abcdefghij", 10),
            ("abcdefghi", 9),
            ("abcdefgh", 8),
            ("abcdefg", 7),
            ("abcdef", 6),
            ("abcde", 5),
            ("abcd", 4),
            ("abc", 3),
            ("ab", 2),
            ("a", 1)
        };

        [Fact]
        public void DoubleArrayTrieTest()
        {
            SortedDictionary<string, int> dict = new SortedDictionary<string, int>(OrdinalUtf8StringComparer.Instance);
            foreach (var (key, value) in _entries)
            {
                dict.Add(key, value);
            }

            //
            // Ensure expected order by OrdinalUtf8StringComparer
            //

            int i = 1;
            foreach (var kvp in dict)
            {
                Assert.Equal(i, kvp.Value); // Validate the sort order
                i++;
            }

            //
            // test DoubleArrayTrie with prefix search
            //

            DoubleArrayTrie trie = new DoubleArrayTrie(dict);
            DoubleArrayResultPair[] doubleArrayResultPairs = new DoubleArrayResultPair[_entries.Length];

            foreach (var (key, value) in _entries)
            {
                byte[] utf8Bytes = Encoding.UTF8.GetBytes(key);
                int resultCount = trie.CommonPrefixSearch(utf8Bytes, doubleArrayResultPairs);
                for (i = 0; i < resultCount; i++)
                {
                    Assert.True(doubleArrayResultPairs[i].Value <= value);
                    Assert.StartsWith(Helpers.GetString(utf8Bytes.AsSpan(0, doubleArrayResultPairs[i].Length)), key, StringComparison.Ordinal);
                }
            }

            //
            // test DoubleArrayTrie with travers search
            //

            foreach (var (key, value) in _entries)
            {
                byte[] utf8Bytes = Encoding.UTF8.GetBytes(key);

                int nodePos = 0;
                int keyPos = 0;

                int result = trie.Traverse(utf8Bytes, ref nodePos, ref keyPos, utf8Bytes.Length);

                Assert.True(trie.ArrayUnits[nodePos].HasLeaf);
                Assert.Equal(utf8Bytes.Length, keyPos);
                Assert.Equal(value, result);
            }
        }
    }
}
