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

        private readonly static Dictionary<string, int> _vocabDataWithWordPrefixAndEndOfWordSuffix =
            new Dictionary<string, int>() { { UnknownToken, 0 }, { "!", 5 }, { ",", 6 }, { ".", 7 }, { "B", 8 }, { "H", 9 }, { "T", 10 }, { "W", 11 }, { "a", 12 }, { "b", 13 }, { "c", 14 }, { "d", 15 }, { "e", 16 },
                        { "f", 17 }, { "g", 18 }, { "h", 19 }, { "i", 20 }, { "k", 21 }, { "l", 22 }, { "m", 23 }, { "n", 24 }, { "o", 25 }, { "p", 26 }, { "r", 27 }, { "s", 28 }, { "t", 29 }, { "u", 30 }, { "v", 31 },
                        { "z", 32 }, { ".</w>", 33 }, { "##o", 34 }, { "##r", 35 }, { "##l", 36 }, { "##d</w>", 37 }, { "##h", 38 }, { "##i", 39 }, { "##s</w>", 40 }, { "##s", 41 }, { "##e</w>", 42 }, { "a</w>", 43 },
                        { "##a", 44 }, { "##n</w>", 45 }, { "##e", 46 }, { "##n", 47 }, { "##t", 48 }, { "##k", 49 }, { "##z", 50 }, { "##r</w>", 51 }, { "##c", 52 }, { "##b</w>", 53 }, { "##u", 54 }, { "##m", 55 },
                        { "##t</w>", 56 }, { "##p", 57 }, { "##o</w>", 58 }, { ",</w>", 59 }, { "!</w>", 60 }, { "##g", 61 }, { "to</w>", 62 }, { "##en", 63 }, { "##oc", 64 }, { "##ra", 65 }, { "Bp", 66 }, { "He", 67 },
                        { "Th", 68 }, { "Wo", 69 }, { "an", 70 }, { "doc", 71 }, { "fi", 72 }, { "gen", 73 }, { "is</w>", 74 }, { "me", 75 }, { "to", 76 }, { "th", 77 }, { "tra", 78 }, { "us", 79 }, { "voc", 80 },
                        { "##rl", 81 }, { "##rg", 82 }, { "##ll", 83 }, { "##le", 84 }, { "##is</w>", 85 }, { "##in</w>", 86 }, { "##iz", 87 }, { "##ab</w>", 88 }, { "##er</w>", 89 }, { "##era", 90 }, { "##te</w>", 91 },
                        { "##ken", 92 }, { "##um", 93 }, { "##ent</w>", 94 }, { "Bpe</w>", 95 }, { "Hell", 96 }, { "This</w>", 97 }, { "Worl", 98 }, { "and</w>", 99 }, { "docum", 100 }, { "file", 101 }, { "genera", 102 },
                        { "merg", 103 }, { "token", 104 }, { "the</w>", 105 }, { "train</w>", 106 }, { "use</w>", 107 }, { "vocab</w>", 108 }, { "##izer</w>", 109 }, { "Hello</w>", 110 }, { "World</w>", 111 },
                        { "document</w>", 112 }, { "files</w>", 113 }, { "generate</w>", 114 }, { "merge</w>", 115 }, { "tokenizer</w>", 116 } };
        private readonly static (string, string)[] _mergeDataWithWordPrefixAndEndOfWordSuffix =
                    new (string, string)[] {  ("t", "##o</w>"), ("##e", "##n"), ("##o", "##c"), ("##r", "##a"), ("B", "##p"), ("H", "##e"), ("T", "##h"), ("W", "##o"), ("a", "##n"),
                        ("d", "##oc"), ("f", "##i"), ("g", "##en"), ("i", "##s</w>"), ("m", "##e"), ("t", "##o"), ("t", "##h"), ("t", "##ra"), ("u", "##s"), ("v", "##oc"), ("##r", "##l"), ("##r", "##g"), ("##l", "##l"),
                        ("##l", "##e"), ("##i", "##s</w>"), ("##i", "##n</w>"), ("##i", "##z"), ("##a", "##b</w>"), ("##e", "##r</w>"), ("##e", "##ra"), ("##t", "##e</w>"), ("##k", "##en"), ("##u", "##m"), ("##en", "##t</w>"),
                        ("Bp", "##e</w>"), ("He", "##ll"), ("Th", "##is</w>"), ("Wo", "##rl"), ("an", "##d</w>"), ("doc", "##um"), ("fi", "##le"), ("gen", "##era"), ("me", "##rg"), ("to", "##ken"), ("th", "##e</w>"),
                        ("tra", "##in</w>"), ("us", "##e</w>"), ("voc", "##ab</w>"), ("##iz", "##er</w>"), ("Hell", "##o</w>"), ("Worl", "##d</w>"), ("docum", "##ent</w>"), ("file", "##s</w>"), ("genera", "##te</w>"),
                        ("merg", "##e</w>"), ("token", "##izer</w>") };


        public static IEnumerable<object?[]> BpeData
        {
            get
            {
                // vocab, merges, sentence, unknownToken, continuingSubwordPrefix , endOfWordSuffix, offsets, ids, expectedTokens, fuseUnknownToken, decodedTokens, decodedTokensWithoutUnknownToken
                yield return new object?[]
                {
                    new Dictionary<string, int>() { { "a", 1 }, { "b", 2 }, { UnknownToken, 3} },
                    null,
                    "c",
                    UnknownToken,
                    null,
                    null,
                    new (int, int)[] { (0, 1) },
                    new int[] { 3 },
                    new string[] { UnknownToken },
                    false,
                    UnknownToken,
                    ""
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>() { { "a", 1 }, { "b", 2 } },
                    null,
                    "a",
                    null,
                    null,
                    null,
                    new (int, int)[] { (0, 1) },
                    new int[] { 1 },
                    new string[] { "a" },
                    false,
                    "a",
                    "a"
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>() { { "a", 1 }, { "b", 2 } },
                    null,
                    "b",
                    null,
                    null,
                    null,
                    new (int, int)[] { (0, 1) },
                    new int[] { 2 },
                    new string[] { "b" },
                    false,
                    "b",
                    "b"
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>() { { "a", 1 }, { "b", 2 }, { UnknownToken, 3} },
                    null,
                    "abc",
                    UnknownToken,
                    null,
                    null,
                    new (int, int)[] { (0, 1), (1, 1), (2, 1) },
                    new int[] { 1, 2, 3 },
                    new string[] { "a", "b", UnknownToken },
                    false,
                    $"ab{UnknownToken}",
                    "ab"
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>() { { "a", 1 }, { "b", 2 }, { UnknownToken, 3} },
                    null,
                    "a b c",
                    UnknownToken,
                    null,
                    null,
                    new (int, int)[] { (0, 1), (2, 1), (4, 1) },
                    new int[] { 1, 2, 3 },
                    new string[] { "a", "b", UnknownToken },
                    false,
                    $"ab{UnknownToken}",
                    "ab"
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>() { { "a", 1 }, { "b", 2 }, { "ab", 3 }, { UnknownToken, 4} },
                    new (string, string)[] { ("a", "b") },
                    "ab c",
                    UnknownToken,
                    null,
                    null,
                    new (int, int)[] { (0, 2), (3, 1) },
                    new int[] { 3, 4 },
                    new string[] { "ab", UnknownToken },
                    false,
                    $"ab{UnknownToken}",
                    "ab"
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>() { { "a", 1 }, { "b", 2 }, { "c", 3 }, { "ab", 4 }, { "abc", 5 } },
                    new (string, string)[] { ("a", "b"), ("ab", "c") },
                    "abc",
                    null,
                    null,
                    null,
                    new (int, int)[] { (0, 3) },
                    new int[] { 5 },
                    new string[] { "abc" },
                    false,
                    "abc",
                    "abc"
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>(){ { UnknownToken, 0} },
                    null,
                    "abc",
                    UnknownToken,
                    null,
                    null,
                    new (int, int)[] { (0, 1), (1, 1), (2, 1) },
                    new int[] { 0, 0, 0 },
                    new string[] { UnknownToken, UnknownToken, UnknownToken },
                    false,
                    $"{UnknownToken}{UnknownToken}{UnknownToken}",
                    ""
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>(){ { UnknownToken, 0} },
                    null,
                    "abc",
                    UnknownToken,
                    null,
                    null,
                    new (int, int)[] { (0, 3) },
                    new int[] { 0 },
                    new string[] { UnknownToken },
                    true,
                    $"{UnknownToken}",
                    ""
                };
                yield return new object?[]
                {
                    new Dictionary<string, int>() { { UnknownToken, 0}, { "H", 1 }, { "e", 2 }, { "l", 3 }, { "o", 4 }, { "!", 5 }, { "He", 6 }, { "lo", 7}, { "llo", 8 },
                                                    { "w", 9 }, { "r", 10 }, { "d", 11 }, { "wo", 12 }, { "rl", 13 }, { "rld", 14 }, {",", 15} },
                    new (string, string)[] { ("H", "e"), ("l", "o"), ("l", "lo"), ("w", "o"), ("r", "l"), ("rl", "d") },
                    "Hello, world!",
                    UnknownToken,
                    null,
                    null,
                    new (int, int)[] { (0, 2), (2, 3), (5, 1), (7, 2), (9, 3), (12, 1) },
                    new int[] { 6, 8, 15, 12, 14, 5 },
                    new string[] { "He", "llo", ",", "wo", "rld", "!" },
                    false,
                    $"Hello,world!",
                    $"Hello,world!"
                };
                yield return new object?[]
                {
                    _vocabDataWithWordPrefixAndEndOfWordSuffix,
                    _mergeDataWithWordPrefixAndEndOfWordSuffix,
                    "Hello, World!",
                    UnknownToken,
                    "##",
                    "</w>",
                    new (int, int)[] { (0, 5), (5, 1), (7, 5), (12, 1) },
                    new int[] { 110, 59, 111, 60 },
                    new string[] { "Hello</w>", ",</w>", "World</w>", "!</w>" },
                    false,
                    $"Hello , World !",
                    $"Hello , World !"
                };
                yield return new object?[]
                {
                    _vocabDataWithWordPrefixAndEndOfWordSuffix,
                    _mergeDataWithWordPrefixAndEndOfWordSuffix,
                    "This is a generalizer to tokenize!",
                    UnknownToken,
                    "##",
                    "</w>",
                    new (int, int)[] { (0, 4), (5, 2), (8, 1), (10, 6), (16, 1), (17, 4), (22, 2), (25, 5), (30, 2), (32, 1), (33, 1) },
                    new int[] { 97, 74, 43, 102, 36, 109, 62, 104, 87, 42, 60 },
                    new string[] { "This</w>", "is</w>", "a</w>", "genera", "##l", "##izer</w>", "to</w>", "token", "##iz", "##e</w>", "!</w>" },
                    false,
                    $"This is a generalizer to tokenize !",
                    $"This is a generalizer to tokenize !"
                };
            }
        }

        [Theory]
        [MemberData(nameof(BpeData))]
        public void SimpleTestWithUnknownToken(
                        Dictionary<string, int> vocab,
                        (string, string)[]? merges,
                        string sentence,
                        string unknownToken,
                        string? continuingSubwordPrefix,
                        string? endOfWordSuffix,
                        (int, int)[] offsets,
                        int[] ids,
                        string[] expectedTokens,
                        bool fuseUnknownToken,
                        string decodedTokens,
                        string decodedTokensWithoutUnknownToken)
        {
            string vocabFile = WriteToVocabFile(vocab);
            string? mergesFile = merges is null ? null : WriteToMergeFile(merges);

            try
            {
                Bpe bpe = new Bpe(vocabFile, mergesFile, unknownToken, continuingSubwordPrefix, endOfWordSuffix, fuseUnknownToken);
                Tokenizer tokenizer = new Tokenizer(bpe);
                EncodingResult encoding = tokenizer.Encode(sentence);
                IReadOnlyList<int> idsList = tokenizer.EncodeToIds(sentence);

                Assert.Equal(expectedTokens.Length, encoding.Tokens.Count);
                Assert.Equal(offsets.Length, encoding.Offsets.Count);
                Assert.Equal(ids.Length, encoding.Ids.Count);
                Assert.Equal(ids.Length, idsList.Count);
                Assert.Equal(ids.Length, tokenizer.CountTokens(sentence));
                Assert.Equal(decodedTokens, tokenizer.Decode(encoding.Ids));
                Assert.Equal(decodedTokensWithoutUnknownToken, bpe.Decode(encoding.Ids, considerSpecialTokens: false));

                for (int i = 0; i < encoding.Tokens.Count; i++)
                {
                    Assert.Equal(expectedTokens[i], encoding.Tokens[i]);
                    Assert.Equal(offsets[i], encoding.Offsets[i]);
                    Assert.Equal(ids[i], encoding.Ids[i]);
                    Assert.Equal(ids[i], idsList[i]);
                    Assert.Equal(encoding.Tokens[i], tokenizer.Model.MapIdToToken(encoding.Ids[i]));
                    Assert.Equal(encoding.Ids[i], tokenizer.Model.MapTokenToId(encoding.Tokens[i].AsSpan()));
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

            TokenizerTests.TestTokenLimits(tokenizer);
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
            writer.Write("{ \"Ukn\": 0 }");
            writer.Flush();
            emptyVocabStream.Position = 0;

            return new Bpe(vocabStream: emptyVocabStream, mergesStream: null, unknownToken: "Ukn");
        }
    }
}

