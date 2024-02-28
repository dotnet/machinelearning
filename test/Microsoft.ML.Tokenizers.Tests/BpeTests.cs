// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class BpeTests
    {
        private const string UnknownToken = "[unk]";

        public static IEnumerable<object?[]> BpeData
        {
            get
            {
                // vocab, merges, sentence, offsets, ids, expectedTokens, fuseUnknownToken
                yield return new object?[]
                {
                    new Dictionary<string, int>() { { "a", 1 }, { "b", 2 } },
                    null,
                    "c",
                    new (int, int)[] { (0, 1) },
                    new int[] { 0 },
                    new string[] { UnknownToken },
                    false,
                    "[unk]"
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>() { { "a", 1 }, { "b", 2 } },
                    null,
                    "a",
                    new (int, int)[] { (0, 1) },
                    new int[] { 1 },
                    new string[] { "a" },
                    false,
                    "a"
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>() { { "a", 1 }, { "b", 2 } },
                    null,
                    "b",
                    new (int, int)[] { (0, 1) },
                    new int[] { 2 },
                    new string[] { "b" },
                    false,
                    "b"
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>() { { "a", 1 }, { "b", 2 } },
                    null,
                    "abc",
                    new (int, int)[] { (0, 1), (1, 1), (2, 1) },
                    new int[] { 1, 2, 0 },
                    new string[] { "a", "b", UnknownToken },
                    false,
                    "ab[unk]"
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>() { { "a", 1 }, { "b", 2 } },
                    null,
                    "a b c",
                    new (int, int)[] { (0, 1), (2, 1), (4, 1) },
                    new int[] { 1, 2, 0 },
                    new string[] { "a", "b", UnknownToken },
                    false,
                    "ab[unk]"
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>() { { "a", 1 }, { "b", 2 }, { "ab", 3 } },
                    new (string, string)[] { ("a", "b") },
                    "ab c",
                    new (int, int)[] { (0, 2), (3, 1) },
                    new int[] { 3, 0 },
                    new string[] { "ab", UnknownToken },
                    false,
                    "ab[unk]"
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>() { { "a", 1 }, { "b", 2 }, { "c", 3 }, { "ab", 4 }, { "abc", 5 } },
                    new (string, string)[] { ("a", "b"), ("ab", "c") },
                    "abc",
                    new (int, int)[] { (0, 3) },
                    new int[] { 5 },
                    new string[] { "abc" },
                    false,
                    "abc"
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>(),
                    null,
                    "abc",
                    new (int, int)[] { (0, 1), (1, 1), (2, 1) },
                    new int[] { 0, 0, 0 },
                    new string[] { UnknownToken, UnknownToken, UnknownToken },
                    false,
                    "[unk][unk][unk]"
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>(),
                    null,
                    "abc",
                    new (int, int)[] { (0, 3) },
                    new int[] { 0 },
                    new string[] { UnknownToken },
                    true,
                    "[unk]"
                };
            }
        }

        [Theory]
        [MemberData(nameof(BpeData))]
        public void SimpleTestWithUnknownToken(Dictionary<string, int> vocab, (string, string)[]? merges, string sentence, (int, int)[] offsets, int[] ids, string[] expectedTokens, bool fuseUnknownToken, string decodedTokens)
        {
            string vocabFile = WriteToVocabFile(vocab);
            string? mergesFile = merges is null ? null : WriteToMergeFile(merges);

            try
            {
                Bpe bpe = new Bpe(vocabFile, mergesFile, UnknownToken, null, null, fuseUnknownToken);

                Assert.Equal(vocab.Count + 1u, bpe.Vocab.Count);
                Tokenizer tokenizer = new Tokenizer(bpe);

                EncodingResult encoding = tokenizer.Encode(sentence);
                IReadOnlyList<int> idsList = tokenizer.EncodeToIds(sentence);

                Assert.Equal(expectedTokens.Length, encoding.Tokens.Count);
                Assert.Equal(offsets.Length, encoding.Offsets.Count);
                Assert.Equal(ids.Length, encoding.Ids.Count);
                Assert.Equal(ids.Length, idsList.Count);
                Assert.Equal(ids.Length, tokenizer.CountTokens(sentence));
                Assert.Equal(decodedTokens, tokenizer.Decode(encoding.Ids));

                for (int i = 0; i < encoding.Tokens.Count; i++)
                {
                    Assert.Equal(expectedTokens[i], encoding.Tokens[i]);
                    Assert.Equal(offsets[i], encoding.Offsets[i]);
                    Assert.Equal(ids[i], encoding.Ids[i]);
                    Assert.Equal(ids[i], idsList[i]);
                    Assert.Equal(encoding.Tokens[i], tokenizer.Model.MapIdToToken(encoding.Ids[i]));
                    Assert.Equal(encoding.Ids[i], tokenizer.Model.MapTokenToId(encoding.Tokens[i]));
                    Assert.Equal(encoding.Tokens[i], tokenizer.Decode(encoding.Ids[i]));
                }
            }
            finally
            {
                Utils.DeleteFile(vocabFile);
                if (mergesFile is not null)
                {
                    Utils.DeleteFile(mergesFile);
                }
            }
        }

        public static IEnumerable<object?[]> BpeTestData
        {
            get
            {
                // string to tokenize, produced tokens, the token offsets
                yield return new object?[]
                {
                    "the brown fox jumped over the lazy dog!",
                    new string[] {"the", "brown", "fox", "jumped", "over", "the", "lazy", "dog", "!"},
                    new (int, int)[] {(0, 3), (4, 9), (10, 13), (14, 20), (21, 25), (26, 29), (30, 34), (35, 38), (38, 39)}
                };
                yield return new object?[]
                {
                    "he traveled to Egypt during the summer, the weather was hot and ammunition." ,
                    new string[] {"he", "traveled", "to", "Egypt", "during", "the", "summer", ",", "the", "weather", "was", "hot", "and", "ammunition", "."},
                    new (int, int)[] {(0, 2), (3, 11), (12, 14), (15, 20), (21, 27), (28, 31), (32, 38), (38, 39), (40, 43), (44, 51), (52, 55), (56, 59), (60, 63), (64, 74), (74, 75)}
                };
                yield return new object?[]
                {
                    "She played many games and she felt exhausted afterward",
                    new string[] {"She", "played", "many", "games", "and", "she", "felt", "exhausted", "afterward"},
                    new (int, int)[] {(0, 3), (4, 10), (11, 15), (16, 21), (22, 25), (26, 29), (30, 34), (35, 44), (45, 54)}
                };
                yield return new object?[]
                {
                    "Hello, y'all! How are you 😁 ?",
                    new string[] {"Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"},
                    new (int, int)[] {(0, 5), (5, 6), (7, 8), (8, 9), (9, 12), (12, 13), (14, 17), (18, 21), (22, 25), (26, 28), (29, 30)}
                };
            }
        }

        private const string Gpt2VocabUrl = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json";
        private const string Gpt2MergesUrl = "https://huggingface.co/openai-community/gpt2/raw/main/merges.txt";

        [Fact]
        public async void TestGpt2Vocab()
        {
            using HttpClient httpClient = new HttpClient();
            using Stream vocabStream = await httpClient.GetStreamAsync(Gpt2VocabUrl);
            using Stream mergesStream = await httpClient.GetStreamAsync(Gpt2MergesUrl);

            Bpe bpe = new Bpe(vocabStream, mergesStream);
            Tokenizer tokenizer = new Tokenizer(bpe);

            string text = "The quick brown fox jumps over the lazy dog!";

            EncodingResult encoding = tokenizer.Encode(text);
            IReadOnlyList<int> ids = tokenizer.EncodeToIds(text);

            Assert.Equal(12, encoding.Tokens.Count);
            Assert.Equal(12, encoding.Offsets.Count);
            Assert.Equal(12, encoding.Ids.Count);
            Assert.Equal(encoding.Ids, ids);
            Assert.Equal(12, tokenizer.CountTokens(text));
        }

        private static string WriteToMergeFile((string, string)[] mergeEntries)
        {
            string fileName = Utils.CreateTemporaryFile("txt");
            using StreamWriter file = new(fileName);
            foreach ((string s1, string s2) in mergeEntries)
            {
                file.WriteLine($"{s1} {s2}");
            }

            return fileName;
        }

        private static string WriteToVocabFile(Dictionary<string, int> dic)
        {
            string fileName = Utils.CreateTemporaryFile("json");
            File.WriteAllText(fileName, JsonSerializer.Serialize<Dictionary<string, int>>(dic), Encoding.UTF8);
            return fileName;
        }

        internal static Bpe CreateEmptyBpe()
        {
            using MemoryStream emptyVocabStream = new MemoryStream();
            using StreamWriter writer = new StreamWriter(emptyVocabStream);
            writer.Write("{}");
            writer.Flush();
            emptyVocabStream.Position = 0;

            return new Bpe(vocabStream: emptyVocabStream, mergesStream: null, UnknownToken);
        }
    }
}

