// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
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
                BpeTokenizer bpe = BpeTokenizer.Create(vocabFile: vocabFile, mergesFile: mergesFile, preTokenizer: WhiteSpacePreTokenizer.Instance, normalizer: null, unknownToken: unknownToken,
                                    continuingSubwordPrefix: continuingSubwordPrefix, endOfWordSuffix: endOfWordSuffix, fuseUnknownTokens: fuseUnknownToken);
                Tokenizer tokenizer = bpe;
                IReadOnlyList<EncodedToken> encoding = tokenizer.EncodeToTokens(sentence, out _);
                int[] encodingIds = encoding.Select(t => t.Id).ToArray();
                IReadOnlyList<int> idsList = tokenizer.EncodeToIds(sentence);

                Assert.Equal(expectedTokens.Length, encoding.Count);
                Assert.Equal(offsets.Length, encoding.Count);
                Assert.Equal(ids.Length, encoding.Count);
                Assert.Equal(ids.Length, idsList.Count);
                Assert.Equal(ids.Length, tokenizer.CountTokens(sentence));
                Assert.Equal(decodedTokens, tokenizer.Decode(encodingIds));
                Assert.Equal(decodedTokensWithoutUnknownToken, bpe.Decode(encodingIds, considerSpecialTokens: false));

                TestDecodingWithSpan(bpe, encodingIds, considerSpecialTokens: true, decodedTokens);
                TestDecodingWithSpan(bpe, encodingIds, considerSpecialTokens: false, decodedTokensWithoutUnknownToken);

                var reverseVocabulary = bpe.Vocabulary.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

                for (int i = 0; i < encoding.Count; i++)
                {
                    Assert.Equal(expectedTokens[i], encoding[i].Value);
                    Assert.Equal(offsets[i], encoding[i].Offset);
                    Assert.Equal(ids[i], encoding[i].Id);
                    Assert.Equal(ids[i], idsList[i]);
                    Assert.Equal(encoding[i].Value, reverseVocabulary[encodingIds[i]]);
                    Assert.Equal(encodingIds[i], bpe.Vocabulary[encoding[i].Value]);
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

        private void TestDecodingWithSpan(BpeTokenizer bpe, int[] ids, bool considerSpecialTokens, string expectedDecoded)
        {
            char[] destinationBuffer = new char[expectedDecoded.Length];

            OperationStatus status;
            int lastIdsConsumed = 0;
            int lastCharactersWritten = 0;
            int idsConsumed;
            int charactersWritten;

            for (int i = 1; i < destinationBuffer.Length - 1; i += Math.Max(1, destinationBuffer.Length - 3)) // enough to test length 1, and destinationBuffer.Length - 2 only.
            {
                status = bpe.Decode(ids, destinationBuffer.AsSpan().Slice(0, i), considerSpecialTokens, out idsConsumed, out charactersWritten);
                Assert.Equal(OperationStatus.DestinationTooSmall, status);
                Assert.True(idsConsumed < ids.Length);
                Assert.True(idsConsumed >= lastIdsConsumed);
                Assert.True(charactersWritten < expectedDecoded.Length);
                Assert.True(charactersWritten >= lastCharactersWritten);
                lastIdsConsumed = idsConsumed;
                lastCharactersWritten = charactersWritten;
            }

            status = bpe.Decode(ids, destinationBuffer.AsSpan(), considerSpecialTokens, out idsConsumed, out charactersWritten);
            Assert.Equal(OperationStatus.Done, status);
            Assert.Equal(ids.Length, idsConsumed);
            Assert.Equal(expectedDecoded.Length, charactersWritten);
            Assert.Equal(expectedDecoded, destinationBuffer.AsSpan().ToString());
        }

        private static Tokenizer? _gpt2Tokenizer = null;

        private static Tokenizer GetGpt2Tokenizer()
        {
            if (_gpt2Tokenizer is null)
            {
                // "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json";
                // "https://huggingface.co/openai-community/gpt2/raw/main/merges.txt";
                using Stream vocabStream = File.OpenRead(Path.Combine(@"Gpt-2", "vocab.json"));
                using Stream mergesStream = File.OpenRead(Path.Combine(@"Gpt-2", "merges.txt"));

                _gpt2Tokenizer = BpeTokenizer.Create(vocabStream, mergesStream);
            }

            return _gpt2Tokenizer;
        }

        [Fact]
        public async void TestBpeCreation()
        {
            // "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json";
            // "https://huggingface.co/openai-community/gpt2/raw/main/merges.txt";
            string vocabFile = Path.Combine(@"Gpt-2", "vocab.json");
            string mergesFile = Path.Combine(@"Gpt-2", "merges.txt");

            BpeTokenizer bpe = BpeTokenizer.Create(vocabFile, mergesFile);
            ValidateTokenizer(bpe);

            using Stream vocabStream = File.OpenRead(vocabFile);
            using Stream mergesStream = File.OpenRead(mergesFile);

            bpe = BpeTokenizer.Create(vocabStream, mergesStream);
            ValidateTokenizer(bpe);

            // Reset the streams for reusing and ensuring the stream are not disposed too.
            vocabStream.Position = 0;
            mergesStream.Position = 0;

            bpe = await BpeTokenizer.CreateAsync(vocabStream, mergesStream);
            ValidateTokenizer(bpe);
        }

        [Fact]
        public void TestGpt2Vocab()
        {
            Tokenizer tokenizer = GetGpt2Tokenizer();
            ValidateTokenizer(tokenizer);
        }

        private void ValidateTokenizer(Tokenizer tokenizer)
        {
            string text = "The quick brown fox jumps over the lazy dog!";

            IReadOnlyList<EncodedToken> encoding = tokenizer.EncodeToTokens(text, out _);
            IReadOnlyList<int> ids = tokenizer.EncodeToIds(text);

            Assert.Equal(12, encoding.Count);
            Assert.Equal(encoding.Select(t => t.Id).ToArray(), ids);
            Assert.Equal(12, tokenizer.CountTokens(text));

            TokenizerTests.TestTokenLimits(tokenizer);
        }


        public static IEnumerable<object?[]> BpeTestData
        {
            get
            {
                // string to tokenize, produced tokens, the token offsets
                yield return new object?[]
                {
                    "the brown fox jumped over the lazy dog!",
                    new string[] { "the", "brown", "fox", "j", "umped", "over", "the", "l", "azy", "dog", "!" },
                    new (int Index, int Length)[] { (0, 3), (4, 5), (10, 3), (14, 1), (15, 5), (21, 4), (26, 3), (30, 1), (31, 3), (35, 3), (38, 1) },
                    new int[] { 1169, 33282, 12792, 73, 27073, 2502, 1169, 75, 12582, 9703, 0 }
                };
                yield return new object?[]
                {
                    "he traveled to Egypt during the summer, the weather was hot and ammunition." ,
                    new string[] { "he", "travel", "ed", "to", "Egypt", "during", "the", "sum", "mer", ",", "the", "weather", "was", "hot", "and", "am", "munition", "." },
                    new (int Index, int Length)[] { (0, 2), (3, 6), (9, 2), (12, 2), (15, 5), (21, 6), (28, 3), (32, 3), (35, 3), (38, 1), (40, 3), (44, 7), (52, 3), (56, 3), (60, 3), (64, 2), (66, 8), (74, 1) },
                    new int[] { 258, 35927, 276, 1462, 39299, 42122, 1169, 16345, 647, 11, 1169, 23563, 9776, 8940, 392, 321, 12640, 13 }
                };
                yield return new object?[]
                {
                    "She played many games and she felt exhausted afterward",
                    new string[] { "She", "played", "many", "games", "and", "she", "felt", "ex", "ha", "usted", "after", "ward" },
                    new (int Index, int Length)[] { (0, 3), (4, 6), (11, 4), (16, 5), (22, 3), (26, 3), (30, 4), (35, 2), (37, 2), (39, 5), (45, 5), (50, 4) },
                    new int[] { 3347, 21542, 21834, 19966, 392, 7091, 31985, 1069, 3099, 8459, 8499, 904 }
                };
                yield return new object?[]
                {
                    "Hello, y'all! How are you 😁 ?",
                    new string[] { "Hello", ",", "y", "'", "all", "!", "How", "are", "you", "?" },
                    new (int Index, int Length)[] { (0, 5), (5, 1), (7, 1), (8, 1), (9, 3), (12, 1), (14, 3), (18, 3), (22, 3), (29, 1) },
                    new int[] { 15496, 11, 88, 6, 439, 0, 2437, 533, 5832, 30 }
                };
            }
        }

        [Theory]
        [MemberData(nameof(BpeTestData))]
        public void TestBpeTokenizer(string text, string[] expectedTokens, (int Index, int Length)[] expectedOffsets, int[] expectedIds)
        {
            Tokenizer tokenizer = GetGpt2Tokenizer();

            IReadOnlyList<EncodedToken> encoding = tokenizer.EncodeToTokens(text, out _);
            IReadOnlyList<EncodedToken> encoding1 = tokenizer.EncodeToTokens(text.AsSpan(), out _);

            Assert.Equal(expectedTokens, encoding.Select(t => t.Value).ToArray());
            Assert.Equal(expectedOffsets, encoding.Select(t => t.Offset).ToArray());
            Assert.Equal(expectedIds, encoding.Select(t => t.Id).ToArray());

            Assert.Equal(expectedTokens, encoding1.Select(t => t.Value).ToArray());
            Assert.Equal(expectedOffsets, encoding1.Select(t => t.Offset).ToArray());
            Assert.Equal(expectedIds, encoding1.Select(t => t.Id).ToArray());

            Assert.Equal(expectedIds, tokenizer.EncodeToIds(text));
            Assert.Equal(expectedIds, tokenizer.EncodeToIds(text.AsSpan()));
            Assert.Equal(expectedIds, tokenizer.EncodeToIds(text, expectedIds.Length, out string? normalizedString, out int length));
            Assert.Null(normalizedString);
            Assert.Equal(text.Length, length);
            Assert.Equal(expectedIds, tokenizer.EncodeToIds(text.AsSpan(), expectedIds.Length, out normalizedString, out length));
            Assert.Null(normalizedString);
            Assert.Equal(text.Length, length);

            Assert.Equal(expectedIds.Take(expectedIds.Length - 2), tokenizer.EncodeToIds(text, expectedIds.Length - 2, out normalizedString, out length));
            Assert.Null(normalizedString);
            int expectedLength = expectedOffsets[expectedOffsets.Length - 3].Index + expectedOffsets[expectedOffsets.Length - 3].Length;
            Assert.Equal(expectedLength, length);
            Assert.Equal(expectedIds.Take(expectedIds.Length - 2), tokenizer.EncodeToIds(text.AsSpan(), expectedIds.Length - 2, out normalizedString, out length));
            Assert.Null(normalizedString);
            Assert.Equal(expectedLength, length);

            Assert.Equal(expectedIds.Length, tokenizer.CountTokens(text));
            Assert.Equal(expectedIds.Length, tokenizer.CountTokens(text.AsSpan()));

            Assert.Equal(expectedOffsets[expectedOffsets.Length - 4].Index + expectedOffsets[expectedOffsets.Length - 4].Length, tokenizer.GetIndexByTokenCount(text, expectedIds.Length - 3, out normalizedString, out int tokenCount));
            Assert.Null(normalizedString);
            Assert.Equal(expectedIds.Length - 3, tokenCount);
            Assert.Equal(expectedOffsets[expectedOffsets.Length - 4].Index + expectedOffsets[expectedOffsets.Length - 4].Length, tokenizer.GetIndexByTokenCount(text.AsSpan(), expectedIds.Length - 3, out normalizedString, out tokenCount));
            Assert.Null(normalizedString);
            Assert.Equal(expectedIds.Length - 3, tokenCount);

            Assert.Equal(expectedOffsets[expectedOffsets.Length - 3].Index, tokenizer.GetIndexByTokenCountFromEnd(text, 3, out normalizedString, out tokenCount));
            Assert.Null(normalizedString);
            Assert.Equal(3, tokenCount);
            Assert.Equal(expectedOffsets[expectedOffsets.Length - 3].Index, tokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), 3, out normalizedString, out tokenCount));
            Assert.Null(normalizedString);
            Assert.Equal(3, tokenCount);
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

        internal static BpeTokenizer CreateEmptyBpe(PreTokenizer? preTokenizer = null, Normalizer? normalizer = null)
        {
            using MemoryStream emptyVocabStream = new MemoryStream();
            using StreamWriter writer = new StreamWriter(emptyVocabStream);
            writer.Write("{ \"Ukn\": 0 }");
            writer.Flush();
            emptyVocabStream.Position = 0;

            return BpeTokenizer.Create(
                        vocabStream: emptyVocabStream, mergesStream: null, preTokenizer: preTokenizer ?? WhiteSpacePreTokenizer.Instance, normalizer: normalizer, unknownToken: "Ukn");
        }
    }
}

