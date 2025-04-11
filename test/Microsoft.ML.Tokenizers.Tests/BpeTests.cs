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
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Xunit;
using static System.Net.Mime.MediaTypeNames;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class BpeTests
    {
        private const string UnknownToken = "[unk]";

        private static readonly Dictionary<string, int> _vocabDataWithWordPrefixAndEndOfWordSuffix =
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
        private static readonly (string, string)[] _mergeDataWithWordPrefixAndEndOfWordSuffix =
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
                BpeTokenizer bpe = BpeTokenizer.Create(vocabFile: vocabFile, mergesFile: mergesFile, preTokenizer: PreTokenizer.CreateWordOrNonWord(), normalizer: null, unknownToken: unknownToken,
                                    continuingSubwordPrefix: continuingSubwordPrefix, endOfWordSuffix: endOfWordSuffix, fuseUnknownTokens: fuseUnknownToken);

                SimpleWithUnknownTokenTest(bpe, sentence, offsets, ids, expectedTokens, decodedTokens, decodedTokensWithoutUnknownToken);
            }
            finally
            {
                Utils.DeleteFile(vocabFile);
                if (mergesFile is not null)
                {
                    Utils.DeleteFile(mergesFile);
                }
            }

            BpeOptions bpeOptions = new BpeOptions(vocab.Select(kvp => (kvp.Key, kvp.Value)))
            {
                Merges = merges?.Select(kvp => $"{kvp.Item1} {kvp.Item2}"),
                PreTokenizer = PreTokenizer.CreateWordOrNonWord(),
                Normalizer = null,
                UnknownToken = unknownToken,
                ContinuingSubwordPrefix = continuingSubwordPrefix,
                EndOfWordSuffix = endOfWordSuffix,
                FuseUnknownTokens = fuseUnknownToken
            };

            BpeTokenizer bpe1 = BpeTokenizer.Create(bpeOptions);
            SimpleWithUnknownTokenTest(bpe1, sentence, offsets, ids, expectedTokens, decodedTokens, decodedTokensWithoutUnknownToken);
        }

        private void SimpleWithUnknownTokenTest(BpeTokenizer bpe, string sentence, (int, int)[] offsets, int[] ids, string[] expectedTokens, string decodedTokens, string decodedTokensWithoutUnknownToken)
        {
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
                Assert.Equal(offsets[i], (encoding[i].Offset.Start.Value, encoding[i].Offset.End.Value - encoding[i].Offset.Start.Value));
                Assert.Equal(ids[i], encoding[i].Id);
                Assert.Equal(ids[i], idsList[i]);
                Assert.Equal(encoding[i].Value, reverseVocabulary[encodingIds[i]]);
                Assert.Equal(encodingIds[i], bpe.Vocabulary[encoding[i].Value]);
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
        public async Task TestBpeCreation()
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

            string jsonString = File.ReadAllText(vocabFile);
            Dictionary<string, int>? dictionary = JsonSerializer.Deserialize<Dictionary<string, int>>(jsonString);

            bpe = BpeTokenizer.Create(
                new BpeOptions(dictionary!.Select(kvp => (kvp.Key, kvp.Value)))
                {
                    Merges = File.ReadAllLines(mergesFile).Skip(1).ToArray() // Skip the first line which is the header "#version".
                });

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
            Assert.Equal(expectedOffsets, encoding.Select(t => (t.Offset.Start.Value, t.Offset.End.Value - t.Offset.Start.Value)).ToArray());
            Assert.Equal(expectedIds, encoding.Select(t => t.Id).ToArray());

            Assert.Equal(expectedTokens, encoding1.Select(t => t.Value).ToArray());
            Assert.Equal(expectedOffsets, encoding1.Select(t => (t.Offset.Start.Value, t.Offset.End.Value - t.Offset.Start.Value)).ToArray());
            Assert.Equal(expectedIds, encoding1.Select(t => t.Id).ToArray());

            Assert.Equal(expectedIds, tokenizer.EncodeToIds(text));
            Assert.Equal(expectedIds, tokenizer.EncodeToIds(text.AsSpan()));
            Assert.Equal(expectedIds, tokenizer.EncodeToIds(text, expectedIds.Length, out string? normalizedText, out int length));
            Assert.Null(normalizedText);
            Assert.Equal(text.Length, length);
            Assert.Equal(expectedIds, tokenizer.EncodeToIds(text.AsSpan(), expectedIds.Length, out normalizedText, out length));
            Assert.Null(normalizedText);
            Assert.Equal(text.Length, length);

            Assert.Equal(expectedIds.Take(expectedIds.Length - 2), tokenizer.EncodeToIds(text, expectedIds.Length - 2, out normalizedText, out length));
            Assert.Null(normalizedText);
            int expectedLength = expectedOffsets[expectedOffsets.Length - 3].Index + expectedOffsets[expectedOffsets.Length - 3].Length;
            Assert.Equal(expectedLength, length);
            Assert.Equal(expectedIds.Take(expectedIds.Length - 2), tokenizer.EncodeToIds(text.AsSpan(), expectedIds.Length - 2, out normalizedText, out length));
            Assert.Null(normalizedText);
            Assert.Equal(expectedLength, length);

            Assert.Equal(expectedIds.Length, tokenizer.CountTokens(text));
            Assert.Equal(expectedIds.Length, tokenizer.CountTokens(text.AsSpan()));

            Assert.Equal(expectedOffsets[expectedOffsets.Length - 4].Index + expectedOffsets[expectedOffsets.Length - 4].Length, tokenizer.GetIndexByTokenCount(text, expectedIds.Length - 3, out normalizedText, out int tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(expectedIds.Length - 3, tokenCount);
            Assert.Equal(expectedOffsets[expectedOffsets.Length - 4].Index + expectedOffsets[expectedOffsets.Length - 4].Length, tokenizer.GetIndexByTokenCount(text.AsSpan(), expectedIds.Length - 3, out normalizedText, out tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(expectedIds.Length - 3, tokenCount);

            Assert.Equal(expectedOffsets[expectedOffsets.Length - 3].Index, tokenizer.GetIndexByTokenCountFromEnd(text, 3, out normalizedText, out tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(3, tokenCount);
            Assert.Equal(expectedOffsets[expectedOffsets.Length - 3].Index, tokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), 3, out normalizedText, out tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(3, tokenCount);
        }

        [Fact]
        public void TestWithSpecialTokens()
        {
            // Picked from https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct/raw/main/tokenizer.json
            IReadOnlyDictionary<string, int> specialTokens = new Dictionary<string, int>()
            {
                {"<|endoftext|>",     0 },
                {"<|im_start|>",      1 },
                {"<|im_end|>",        2 },
                {"<repo_name>",       3 },
                {"<reponame>",        4 },
                {"<file_sep>",        5 },
                {"<filename>",        6 },
                {"<gh_stars>",        7 },
                {"<issue_start>",     8 },
                {"<issue_comment>",   9 },
                {"<issue_closed>",   10 },
                {"<jupyter_start>",  11 },
                {"<jupyter_text>",   12 },
                {"<jupyter_code>",   13 },
                {"<jupyter_output>", 14 },
                {"<jupyter_script>", 15 },
                {"<empty_output>",   16 },
            };

            using Stream vocabStream = File.OpenRead(Path.Combine(@"Gpt-2", "vocab.json"));
            using Stream mergesStream = File.OpenRead(Path.Combine(@"Gpt-2", "merges.txt"));

            var bpeTokenizer = BpeTokenizer.Create(vocabStream, mergesStream, PreTokenizer.CreateWordOrNonWord(specialTokens), normalizer: null, specialTokens: specialTokens, unknownToken: "<|endoftext|>");

            string input = "Hello, y'all! <issue_comment>How are you 😁 ?<|endoftext|>";

            IReadOnlyList<EncodedToken> tokens = bpeTokenizer.EncodeToTokens(input, out _);

            EncodedToken[] expectedTokens = [
                new EncodedToken(15496, "Hello",            new Range(0, 5)),
                new EncodedToken(11,    ",",                new Range(5, 6)),
                new EncodedToken(88,    "y",                new Range(7, 8)),
                new EncodedToken(6,     "'",                new Range(8, 9)),
                new EncodedToken(439,   "all",              new Range(9, 12)),
                new EncodedToken(0,     "!",                new Range(12, 13)),
                new EncodedToken(9,     "<issue_comment>",  new Range(14, 29)),
                new EncodedToken(2437,  "How",              new Range(29, 32)),
                new EncodedToken(533,   "are",              new Range(33, 36)),
                new EncodedToken(5832,  "you",              new Range(37, 40)),
                new EncodedToken(50256, "<|endoftext|>",    new Range(41, 43)),
                new EncodedToken(30,    "?",                new Range(44, 45)),
                new EncodedToken(0,     "<|endoftext|>",    new Range(45, 58))
            ];

            Assert.Equal(expectedTokens, tokens);

            IReadOnlyList<int> ids = bpeTokenizer.EncodeToIds(input);
            Assert.Equal(expectedTokens.Select(t => t.Id).ToArray(), ids);
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
                        vocabStream: emptyVocabStream, mergesStream: null, preTokenizer: preTokenizer ?? PreTokenizer.CreateWordOrNonWord(), normalizer: normalizer, unknownToken: "Ukn");
        }

        //
        // DeepSeek tests
        //

        private static string DumpEncodingTokens(IReadOnlyList<EncodedToken> encoding)
        {
            string result = $"[{string.Join(", ", encoding.Select(t => $"{t.Id}"))} ] {Environment.NewLine}";
            result += $"[{string.Join(", ", encoding.Select(t => $"\"{t.Value}\""))} ] {Environment.NewLine}";
            result += $"[{string.Join(", ", encoding.Select(t => $"({t.Offset.Start}, {t.Offset.End.Value})"))} ]";
            return result;
        }

        public static IEnumerable<object?[]> DeepSeekData
        {
            get
            {
                // text, ids, tokens, offsets
                yield return new object?[]
                {
                    @"\sqrt{3x-1}+(1+x)^2", // Math expression
                    new int[] { 0, 52765, 93, 21, 90, 15, 19, 14419, 10, 19, 34705, 21590, 20 },
                    new string[] { "<｜begin▁of▁sentence｜>", "\\sqrt", "{", "3", "x", "-", "1", "}+", "(", "1", "+x", ")^", "2" },
                    new (int, int)[] { (0, 0), (0, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 12), (12, 13), (13, 14), (14, 16), (16, 18), (18, 19) },
                };

                yield return new object?[]
                {
                    "def add(a, b):\n    return a + b", // Python function
                    new int[] { 0, 3465, 1258, 6036, 14, 291, 3395, 361, 1354, 260, 940, 291 },
                    new string[] { "<｜begin▁of▁sentence｜>", "def", "Ġadd", "(a", ",", "Ġb", "):Ċ", "ĠĠĠ", "Ġreturn", "Ġa", "Ġ+", "Ġb" },
                    new (int, int)[] { (0, 0), (0, 3), (3, 7), (7, 9), (9, 10), (10, 12), (12, 15), (15, 18), (18, 25), (25, 27), (27, 29), (29, 31) },
                };

                yield return new object?[]
                {
                    "function greet(name) {\n    return a + b", // JavaScript function
                    new int[] { 0, 8701, 49166, 17984, 11, 875, 361, 1354, 260, 940, 291 },
                    new string[] { "<｜begin▁of▁sentence｜>", "function", "Ġgreet", "(name", ")", "Ġ{Ċ", "ĠĠĠ", "Ġreturn", "Ġa", "Ġ+", "Ġb" },
                    new (int, int)[] { (0, 0), (0, 8), (8, 14), (14, 19), (19, 20), (20, 23), (23, 26), (26, 33), (33, 35), (35, 37), (37, 39) },
                };

                yield return new object?[]
                {
                    "Hello, how are you?\nBonjour, comment ça va?\n你好，你怎么样？\n", // Multilingual text
                    new int[] { 0, 19923, 14, 1192, 477, 440, 2755, 52015, 46703, 14, 7006, 65750, 10879, 2755, 30594, 303, 804, 19602, 6692 },
                    new string[] { "<｜begin▁of▁sentence｜>", "Hello", ",", "Ġhow", "Ġare", "Ġyou", "?Ċ", "Bon", "jour", ",", "Ġcomment", "ĠÃ§a", "Ġva", "?Ċ", "ä½łå¥½", "ï¼Į", "ä½ł", "æĢİä¹Īæł·", "ï¼ŁĊ" },
                    new (int, int)[] { (0, 0), (0, 5), (5, 6), (6, 10), (10, 14), (14, 18), (18, 20), (20, 23), (23, 27), (27, 28), (28, 36), (36, 39), (39, 42), (42, 44), (44, 46), (46, 47), (47, 48), (48, 51), (51, 53) },
                };

                yield return new object?[]
                {
                    "カタカナ", // Japanese text
                    new int[] { 0, 15961, 11767, 15961, 27071 },
                    new string[] { "<｜begin▁of▁sentence｜>", "ãĤ«", "ãĤ¿", "ãĤ«", "ãĥĬ" },
                    new (int, int)[] { (0, 0), (0, 1), (1, 2), (2, 3), (3, 4) },
                };

                yield return new object?[]
                {
                    "1234567890", // numbers to ensure splitting on 3-digits boundary
                    new int[] { 0, 6895, 18009, 25744, 18 },
                    new string[] { "<｜begin▁of▁sentence｜>", "123", "456", "789", "0" },
                    new (int, int)[] { (0, 0), (0, 3), (3, 6), (6, 9), (9, 10) },
                };

                yield return new object?[]
                {
                    "This is a test 1234567890カタカナ日本語 for us.", // multilingual text with numbers
                    new int[] { 0, 2337, 344, 260, 1950, 223, 6895, 18009, 25744, 18, 15961, 11767, 15961, 27071, 88768, 362, 550, 16 },
                    new string[] { "<｜begin▁of▁sentence｜>", "This", "Ġis", "Ġa", "Ġtest", "Ġ", "123", "456", "789", "0", "ãĤ«", "ãĤ¿", "ãĤ«", "ãĥĬ", "æĹ¥æľ¬èªŀ", "Ġfor", "Ġus", "." },
                    new (int, int)[] { (0, 0), (0, 4), (4, 7), (7, 9), (9, 14), (14, 15), (15, 18), (18, 21), (21, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 32), (32, 36), (36, 39), (39, 40) },
                };

                yield return new object?[]
                {
                    // DeepSeek-R1 tutorial example long text
                    "In modern software development, comprehensive testing remains a critical challenge. This tutorial demonstrates how to combine DeepSeek's intelligence with pytest's testing " +
                    "framework to generate robust unit tests automatically. Automated Unit Test Generation with DeepSeek-R1 and pytest Environment Configuration Set up your testing environment with " +
                    "these essential packages: # Install core dependencies !pip install pytest deepseek-ai transformers pytest-cov mock # Verify installation import pytest print(f\"pytest version: " +
                    "{pytest.version}\") Pro Tip: Create a dedicated virtual environment for testing: python -m venv testenv && source testenv/bin/activate Basic Test Generation Workflow Create a " +
                    "test generator module: # test_generator.py from transformers import AutoTokenizer, AutoModelForCausalLM class TestGenerator: def init(self): self.tokenizer = " +
                    "AutoTokenizer.from_pretrained(\"deepseek-ai/deepseek-r1\") self.model = AutoModelForCausalLM.from_pretrained(\"deepseek-ai/deepseek-r1\") " +
                    "def generate_test(self, function_code: str) -> str: prompt = f \"\"\"Generate pytest unit tests for this Python function: {function_code} \\nFollow these requirements: 1. " +
                    "Use parameterized testing 2. Include edge cases 3. Add type annotations 4. Include descriptive docstrings\"\"\" inputs = self.tokenizer(prompt, return_tensors= \"pt\") " +
                    "outputs = self.model.generate(**inputs, max_length= 1024) return self.tokenizer.decode(outputs, skip_special_tokens=True) Testing Sample Application Code Create a simple " +
                    "calculator module to test: # calculator.py def add(a: float, b: float) -> float: return a + b def subtract(a: float, b: float) -> float: return a - b Generating and Executing " +
                    "Tests Automate the test lifecycle: # Generate tests generator = TestGenerator() with open (\"calculator.py\") as f: tests = generator.generate_test(f.read()) # Save generated " +
                    "tests with open (\"test_calculator.py\", \"w\") as f: f.write(tests) # Run tests with pytest !pytest test_calculator.py -v --cov=calculator Advanced Test Patterns Implement " +
                    "mock testing with AI-generated scenarios: # test_database.py from unittest.mock import Mock import pytest def test_db_connection(): # AI-generated test scenario " +
                    "mock_db = Mock() mock_db.connect.return_value = True assert mock_db.connect(\"localhost:5432\") is True mock_db.connect.assert_called_once_with(\"localhost:5432\") " +
                    "CI/CD Integration Add automated testing to GitHub Actions: # .github/workflows/tests.yml name: AI-Powered Tests on: [push] jobs: test: runs-on: ubuntu-latest steps: - " +
                    "uses: actions/checkout@v3 - name: Set up Python uses: actions/setup-python@v4 with: python-version: '3.10' - name: Install dependencies run: | pip install -r requirements.txt - " +
                    "name: Generate tests run: python generate_tests.py - name: Run tests run: pytest --cov=src/ --cov-report=xml",
                    new int[]
                    {
                        0, 1124, 5970, 6460, 2934, 14, 10501, 8273, 7926, 260, 6490, 8851, 16, 1162, 24038, 23723, 1192, 304, 20036, 22651, 4374, 1465, 734, 12967, 418, 105659, 734, 8273, 10303, 304,
                        10559, 16064, 4761, 8499, 15400, 16, 79763, 14749, 6205, 32036, 418, 22651, 4374, 1465, 8555, 19, 305, 105659, 10651, 44894, 8269, 890, 782, 8273, 3431, 418, 1305, 4930, 26607,
                        28, 1823, 31788, 8668, 38937, 4050, 92530, 6564, 105659, 5212, 121559, 15, 2238, 98918, 105659, 2846, 757, 25330, 1823, 78419, 19544, 1662, 105659, 2777, 5123, 4, 82, 99297,
                        6013, 28, 680, 82, 99297, 121719, 95, 5925, 1317, 42542, 28, 12722, 260, 14381, 10814, 3431, 362, 8273, 28, 24847, 565, 79, 12487, 88, 1950, 17821, 6546, 4688, 1950, 17821,
                        32702, 17, 76117, 15824, 6205, 32036, 7194, 10107, 12722, 260, 1950, 23794, 12124, 28, 1823, 1950, 15810, 95223, 23042, 538, 98918, 1662, 22416, 63241, 14, 22416, 8449, 3870,
                        37, 51935, 39851, 1312, 6205, 46223, 28, 1351, 3447, 6666, 2605, 2280, 99885, 9160, 438, 22416, 63241, 38425, 5224, 2101, 17021, 1698, 60861, 121559, 15, 2238, 41545, 755,
                        121559, 9954, 19, 5925, 2280, 25824, 438, 22416, 8449, 3870, 37, 51935, 39851, 38425, 5224, 2101, 17021, 1698, 60861, 121559, 15, 2238, 41545, 755, 121559, 9954, 19, 5925,
                        1351, 10559, 21525, 6666, 14, 2019, 17636, 28, 1691, 11, 6248, 1691, 28, 12275, 438, 285, 26495, 99732, 105659, 4761, 8499, 362, 566, 15255, 2019, 28, 680, 8701, 17636, 95,
                        874, 80, 20676, 1305, 7172, 28, 223, 19, 16, 6948, 10767, 1766, 8273, 223, 20, 16, 52480, 9449, 4599, 223, 21, 16, 7043, 2613, 71134, 223, 22, 16, 52480, 35984, 3696, 56638, 63356,
                        21102, 438, 2280, 99885, 9160, 7093, 79078, 14, 1354, 3682, 63248, 31, 582, 529, 5925, 25798, 438, 2280, 25824, 119629, 109464, 111082, 14, 3979, 29079, 31, 223, 5769, 22, 11, 1354,
                        2280, 99885, 9160, 104065, 61327, 85, 14, 21429, 4731, 43549, 3682, 58878, 19552, 11, 27445, 28454, 13591, 9909, 12722, 260, 4654, 17612, 12124, 304, 1950, 28, 1823, 17612, 23042,
                        1351, 1258, 6036, 28, 12249, 14, 291, 28, 12249, 11, 6248, 12249, 28, 1354, 260, 940, 291, 1351, 24522, 6036, 28, 12249, 14, 291, 28, 12249, 11, 6248, 12249, 28, 1354, 260, 565,
                        291, 115825, 305, 14136, 10063, 34291, 23284, 434, 270, 1950, 72487, 28, 1823, 62754, 8499, 23794, 438, 6205, 46223, 1393, 418, 2526, 12951, 77401, 23042, 5925, 412, 285, 28, 8499,
                        438, 23794, 119629, 21525, 5123, 17627, 14042, 1823, 27473, 9846, 8499, 418, 2526, 12951, 7958, 4941, 50452, 1741, 23042, 1760, 582, 89, 5925, 412, 285, 28, 285, 26214, 4665, 7333,
                        11, 1823, 19633, 8499, 418, 105659, 4050, 82, 99297, 1950, 4941, 50452, 1741, 23042, 565, 88, 3820, 83671, 31, 77401, 21555, 6205, 49885, 23639, 25330, 8273, 418, 7703, 38762, 21805,
                        28, 1823, 1950, 4084, 11444, 23042, 538, 82332, 100170, 1662, 51705, 1662, 105659, 1351, 1950, 77563, 65, 30000, 24590, 1823, 7703, 38762, 1950, 18553, 25330, 77563, 438, 51705, 1393,
                        25330, 77563, 80383, 16, 3916, 23028, 438, 11485, 8719, 25330, 77563, 80383, 1698, 38017, 28, 25498, 20, 5925, 344, 11485, 25330, 77563, 80383, 21498, 4941, 10546, 72710, 29784, 1698,
                        38017, 28, 25498, 20, 5925, 19415, 100949, 36845, 7043, 27818, 8273, 304, 56720, 31817, 28, 1823, 1204, 14765, 124907, 44672, 9543, 7333, 105169, 2329, 28, 7703, 6351, 99434, 34291,
                        377, 28, 764, 38312, 63, 11193, 28, 1950, 28, 12122, 8909, 28, 21342, 37094, 2800, 66469, 6531, 28, 565, 6623, 28, 8102, 17, 9547, 606, 34, 88, 21, 565, 2329, 28, 8269, 890, 15255, 6623,
                        28, 8102, 2283, 319, 1425, 3095, 8150, 34, 88, 22, 418, 28, 24847, 15, 9713, 28, 905, 21, 16, 553, 9, 565, 2329, 28, 31788, 38937, 2632, 28, 369, 14220, 6564, 565, 84, 7172, 16415, 565,
                        2329, 28, 62754, 8499, 2632, 28, 24847, 10559, 3682, 7333, 23042, 565, 2329, 28, 19633, 8499, 2632, 28, 105659, 3820, 83671, 31, 24483, 17, 3820, 83671, 77002, 33577, 3957
                    },
                    new string[]
                    {
                        "<｜begin▁of▁sentence｜>", "In", "Ġmodern", "Ġsoftware", "Ġdevelopment", ",", "Ġcomprehensive", "Ġtesting", "Ġremains", "Ġa", "Ġcritical", "Ġchallenge", ".", "ĠThis",
                        "Ġtutorial", "Ġdemonstrates", "Ġhow", "Ġto", "Ġcombine", "ĠDeep", "Se", "ek", "'s", "Ġintelligence", "Ġwith", "Ġpytest", "'s", "Ġtesting", "Ġframework", "Ġto",
                        "Ġgenerate", "Ġrobust", "Ġunit", "Ġtests", "Ġautomatically", ".", "ĠAutomated", "ĠUnit", "ĠTest", "ĠGeneration", "Ġwith", "ĠDeep", "Se", "ek", "-R", "1", "Ġand",
                        "Ġpytest", "ĠEnvironment", "ĠConfiguration", "ĠSet", "Ġup", "Ġyour", "Ġtesting", "Ġenvironment", "Ġwith", "Ġthese", "Ġessential", "Ġpackages", ":", "Ġ#", "ĠInstall",
                        "Ġcore", "Ġdependencies", "Ġ!", "pip", "Ġinstall", "Ġpytest", "Ġdeep", "seek", "-", "ai", "Ġtransformers", "Ġpytest", "-c", "ov", "Ġmock", "Ġ#", "ĠVerify", "Ġinstallation",
                        "Ġimport", "Ġpytest", "Ġprint", "(f", "\"", "p", "ytest", "Ġversion", ":", "Ġ{", "p", "ytest", ".version", "}", "\")", "ĠPro", "ĠTip", ":", "ĠCreate", "Ġa", "Ġdedicated",
                        "Ġvirtual", "Ġenvironment", "Ġfor", "Ġtesting", ":", "Ġpython", "Ġ-", "m", "Ġven", "v", "Ġtest", "env", "Ġ&&", "Ġsource", "Ġtest", "env", "/bin", "/", "activate", "ĠBasic",
                        "ĠTest", "ĠGeneration", "ĠWork", "flow", "ĠCreate", "Ġa", "Ġtest", "Ġgenerator", "Ġmodule", ":", "Ġ#", "Ġtest", "_g", "enerator", ".py", "Ġfrom", "Ġtransformers", "Ġimport",
                        "ĠAuto", "Tokenizer", ",", "ĠAuto", "Model", "For", "C", "ausal", "LM", "Ġclass", "ĠTest", "Generator", ":", "Ġdef", "Ġinit", "(self", "):", "Ġself", ".token", "izer", "Ġ=",
                        "ĠAuto", "Tokenizer", ".from", "_p", "ret", "rained", "(\"", "deep", "seek", "-", "ai", "/de", "ep", "seek", "-r", "1", "\")", "Ġself", ".model", "Ġ=", "ĠAuto", "Model", "For",
                        "C", "ausal", "LM", ".from", "_p", "ret", "rained", "(\"", "deep", "seek", "-", "ai", "/de", "ep", "seek", "-r", "1", "\")", "Ġdef", "Ġgenerate", "_test", "(self", ",",
                        "Ġfunction", "_code", ":", "Ġstr", ")", "Ġ->", "Ġstr", ":", "Ġprompt", "Ġ=", "Ġf", "Ġ\"\"\"", "Generate", "Ġpytest", "Ġunit", "Ġtests", "Ġfor", "Ġthis", "ĠPython", "Ġfunction",
                        ":", "Ġ{", "function", "_code", "}", "Ġ\\", "n", "Follow", "Ġthese", "Ġrequirements", ":", "Ġ", "1", ".", "ĠUse", "Ġparameter", "ized", "Ġtesting", "Ġ", "2", ".", "ĠInclude",
                        "Ġedge", "Ġcases", "Ġ", "3", ".", "ĠAdd", "Ġtype", "Ġannotations", "Ġ", "4", ".", "ĠInclude", "Ġdescriptive", "Ġdoc", "strings", "\"\"\"", "Ġinputs", "Ġ=", "Ġself", ".token",
                        "izer", "(p", "rompt", ",", "Ġreturn", "_t", "ensors", "=", "Ġ\"", "pt", "\")", "Ġoutputs", "Ġ=", "Ġself", ".model", ".generate", "(**", "inputs", ",", "Ġmax", "_length",
                        "=", "Ġ", "102", "4", ")", "Ġreturn", "Ġself", ".token", "izer", ".decode", "(output", "s", ",", "Ġskip", "_s", "pecial", "_t", "okens", "=True", ")", "ĠTesting", "ĠSample",
                        "ĠApplication", "ĠCode", "ĠCreate", "Ġa", "Ġsimple", "Ġcalculator", "Ġmodule", "Ġto", "Ġtest", ":", "Ġ#", "Ġcalculator", ".py", "Ġdef", "Ġadd", "(a", ":", "Ġfloat", ",",
                        "Ġb", ":", "Ġfloat", ")", "Ġ->", "Ġfloat", ":", "Ġreturn", "Ġa", "Ġ+", "Ġb", "Ġdef", "Ġsubtract", "(a", ":", "Ġfloat", ",", "Ġb", ":", "Ġfloat", ")", "Ġ->", "Ġfloat", ":",
                        "Ġreturn", "Ġa", "Ġ-", "Ġb", "ĠGenerating", "Ġand", "ĠExec", "uting", "ĠTests", "ĠAutom", "ate", "Ġthe", "Ġtest", "Ġlifecycle", ":", "Ġ#", "ĠGenerate", "Ġtests", "Ġgenerator",
                        "Ġ=", "ĠTest", "Generator", "()", "Ġwith", "Ġopen", "Ġ(\"", "calculator", ".py", "\")", "Ġas", "Ġf", ":", "Ġtests", "Ġ=", "Ġgenerator", ".generate", "_test", "(f", ".read", "())",
                        "Ġ#", "ĠSave", "Ġgenerated", "Ġtests", "Ġwith", "Ġopen", "Ġ(\"", "test", "_c", "alcul", "ator", ".py", "\",", "Ġ\"", "w", "\")", "Ġas", "Ġf", ":", "Ġf", ".write", "(t", "ests", ")",
                        "Ġ#", "ĠRun", "Ġtests", "Ġwith", "Ġpytest", "Ġ!", "p", "ytest", "Ġtest", "_c", "alcul", "ator", ".py", "Ġ-", "v", "Ġ--", "cov", "=", "calculator", "ĠAdvanced", "ĠTest", "ĠPatterns",
                        "ĠImplement", "Ġmock", "Ġtesting", "Ġwith", "ĠAI", "-generated", "Ġscenarios", ":", "Ġ#", "Ġtest", "_d", "atabase", ".py", "Ġfrom", "Ġunittest", ".mock", "Ġimport", "ĠMock",
                        "Ġimport", "Ġpytest", "Ġdef", "Ġtest", "_db", "_", "connection", "():", "Ġ#", "ĠAI", "-generated", "Ġtest", "Ġscenario", "Ġmock", "_db", "Ġ=", "ĠMock", "()", "Ġmock", "_db",
                        ".connect", ".", "return", "_value", "Ġ=", "ĠTrue", "Ġassert", "Ġmock", "_db", ".connect", "(\"", "localhost", ":", "543", "2", "\")", "Ġis", "ĠTrue", "Ġmock", "_db", ".connect",
                        ".assert", "_c", "alled", "_once", "_with", "(\"", "localhost", ":", "543", "2", "\")", "ĠCI", "/CD", "ĠIntegration", "ĠAdd", "Ġautomated", "Ġtesting", "Ġto", "ĠGitHub", "ĠActions",
                        ":", "Ġ#", "Ġ.", "github", "/work", "flows", "/t", "ests", ".yml", "Ġname", ":", "ĠAI", "-P", "owered", "ĠTests", "Ġon", ":", "Ġ[", "push", "]", "Ġjobs", ":", "Ġtest", ":", "Ġruns",
                        "-on", ":", "Ġub", "untu", "-l", "atest", "Ġsteps", ":", "Ġ-", "Ġuses", ":", "Ġactions", "/", "check", "out", "@", "v", "3", "Ġ-", "Ġname", ":", "ĠSet", "Ġup", "ĠPython", "Ġuses",
                        ":", "Ġactions", "/s", "et", "up", "-p", "ython", "@", "v", "4", "Ġwith", ":", "Ġpython", "-", "version", ":", "Ġ'", "3", ".", "10", "'", "Ġ-", "Ġname", ":", "ĠInstall",
                        "Ġdependencies", "Ġrun", ":", "Ġ|", "Ġpip", "Ġinstall", "Ġ-", "r", "Ġrequirements", ".txt", "Ġ-", "Ġname", ":", "ĠGenerate", "Ġtests", "Ġrun", ":", "Ġpython", "Ġgenerate", "_t",
                        "ests", ".py", "Ġ-", "Ġname", ":", "ĠRun", "Ġtests", "Ġrun", ":", "Ġpytest", "Ġ--", "cov", "=", "src", "/", "Ġ--", "cov", "-report", "=x", "ml"
                    },
                    new (int, int)[]
                    {
                        (0, 0), (0, 2), (2, 9), (9, 18), (18, 30), (30, 31), (31, 45), (45, 53), (53, 61), (61, 63), (63, 72), (72, 82), (82, 83), (83, 88), (88, 97), (97, 110), (110, 114), (114, 117),
                        (117, 125), (125, 130), (130, 132), (132, 134), (134, 136), (136, 149), (149, 154), (154, 161), (161, 163), (163, 171), (171, 181), (181, 184), (184, 193), (193, 200), (200, 205),
                        (205, 211), (211, 225), (225, 226), (226, 236), (236, 241), (241, 246), (246, 257), (257, 262), (262, 267), (267, 269), (269, 271), (271, 273), (273, 274), (274, 278), (278, 285),
                        (285, 297), (297, 311), (311, 315), (315, 318), (318, 323), (323, 331), (331, 343), (343, 348), (348, 354), (354, 364), (364, 373), (373, 374), (374, 376), (376, 384), (384, 389),
                        (389, 402), (402, 404), (404, 407), (407, 415), (415, 422), (422, 427), (427, 431), (431, 432), (432, 434), (434, 447), (447, 454), (454, 456), (456, 458), (458, 463), (463, 465),
                        (465, 472), (472, 485), (485, 492), (492, 499), (499, 505), (505, 507), (507, 508), (508, 509), (509, 514), (514, 522), (522, 523), (523, 525), (525, 526), (526, 531), (531, 539),
                        (539, 540), (540, 542), (542, 546), (546, 550), (550, 551), (551, 558), (558, 560), (560, 570), (570, 578), (578, 590), (590, 594), (594, 602), (602, 603), (603, 610), (610, 612),
                        (612, 613), (613, 617), (617, 618), (618, 623), (623, 626), (626, 629), (629, 636), (636, 641), (641, 644), (644, 648), (648, 649), (649, 657), (657, 663), (663, 668), (668, 679),
                        (679, 684), (684, 688), (688, 695), (695, 697), (697, 702), (702, 712), (712, 719), (719, 720), (720, 722), (722, 727), (727, 729), (729, 737), (737, 740), (740, 745), (745, 758),
                        (758, 765), (765, 770), (770, 779), (779, 780), (780, 785), (785, 790), (790, 793), (793, 794), (794, 799), (799, 801), (801, 807), (807, 812), (812, 821), (821, 822), (822, 826),
                        (826, 831), (831, 836), (836, 838), (838, 843), (843, 849), (849, 853), (853, 855), (855, 860), (860, 869), (869, 874), (874, 876), (876, 879), (879, 885), (885, 887), (887, 891),
                        (891, 895), (895, 896), (896, 898), (898, 901), (901, 903), (903, 907), (907, 909), (909, 910), (910, 912), (912, 917), (917, 923), (923, 925), (925, 930), (930, 935), (935, 938),
                        (938, 939), (939, 944), (944, 946), (946, 951), (951, 953), (953, 956), (956, 962), (962, 964), (964, 968), (968, 972), (972, 973), (973, 975), (975, 978), (978, 980), (980, 984),
                        (984, 986), (986, 987), (987, 989), (989, 993), (993, 1002), (1002, 1007), (1007, 1012), (1012, 1013), (1013, 1022), (1022, 1027), (1027, 1028), (1028, 1032), (1032, 1033),
                        (1033, 1036), (1036, 1040), (1040, 1041), (1041, 1048), (1048, 1050), (1050, 1052), (1052, 1056), (1056, 1064), (1064, 1071), (1071, 1076), (1076, 1082), (1082, 1086), (1086, 1091),
                        (1091, 1098), (1098, 1107), (1107, 1108), (1108, 1110), (1110, 1118), (1118, 1123), (1123, 1124), (1124, 1126), (1126, 1127), (1127, 1133), (1133, 1139), (1139, 1152), (1152, 1153),
                        (1153, 1154), (1154, 1155), (1155, 1156), (1156, 1160), (1160, 1170), (1170, 1174), (1174, 1182), (1182, 1183), (1183, 1184), (1184, 1185), (1185, 1193), (1193, 1198), (1198, 1204),
                        (1204, 1205), (1205, 1206), (1206, 1207), (1207, 1211), (1211, 1216), (1216, 1228), (1228, 1229), (1229, 1230), (1230, 1231), (1231, 1239), (1239, 1251), (1251, 1255), (1255, 1262),
                        (1262, 1265), (1265, 1272), (1272, 1274), (1274, 1279), (1279, 1285), (1285, 1289), (1289, 1291), (1291, 1296), (1296, 1297), (1297, 1304), (1304, 1306), (1306, 1312), (1312, 1313),
                        (1313, 1315), (1315, 1317), (1317, 1319), (1319, 1327), (1327, 1329), (1329, 1334), (1334, 1340), (1340, 1349), (1349, 1352), (1352, 1358), (1358, 1359), (1359, 1363), (1363, 1370),
                        (1370, 1371), (1371, 1372), (1372, 1375), (1375, 1376), (1376, 1377), (1377, 1384), (1384, 1389), (1389, 1395), (1395, 1399), (1399, 1406), (1406, 1413), (1413, 1414), (1414, 1415),
                        (1415, 1420), (1420, 1422), (1422, 1428), (1428, 1430), (1430, 1435), (1435, 1440), (1440, 1441), (1441, 1449), (1449, 1456), (1456, 1468), (1468, 1473), (1473, 1480), (1480, 1482),
                        (1482, 1489), (1489, 1500), (1500, 1507), (1507, 1510), (1510, 1515), (1515, 1516), (1516, 1518), (1518, 1529), (1529, 1532), (1532, 1536), (1536, 1540), (1540, 1542), (1542, 1543),
                        (1543, 1549), (1549, 1550), (1550, 1552), (1552, 1553), (1553, 1559), (1559, 1560), (1560, 1563), (1563, 1569), (1569, 1570), (1570, 1577), (1577, 1579), (1579, 1581), (1581, 1583),
                        (1583, 1587), (1587, 1596), (1596, 1598), (1598, 1599), (1599, 1605), (1605, 1606), (1606, 1608), (1608, 1609), (1609, 1615), (1615, 1616), (1616, 1619), (1619, 1625), (1625, 1626),
                        (1626, 1633), (1633, 1635), (1635, 1637), (1637, 1639), (1639, 1650), (1650, 1654), (1654, 1659), (1659, 1664), (1664, 1670), (1670, 1676), (1676, 1679), (1679, 1683), (1683, 1688),
                        (1688, 1698), (1698, 1699), (1699, 1701), (1701, 1710), (1710, 1716), (1716, 1726), (1726, 1728), (1728, 1733), (1733, 1742), (1742, 1744), (1744, 1749), (1749, 1754), (1754, 1757),
                        (1757, 1767), (1767, 1770), (1770, 1772), (1772, 1775), (1775, 1777), (1777, 1778), (1778, 1784), (1784, 1786), (1786, 1796), (1796, 1805), (1805, 1810), (1810, 1812), (1812, 1817),
                        (1817, 1820), (1820, 1822), (1822, 1827), (1827, 1837), (1837, 1843), (1843, 1848), (1848, 1853), (1853, 1856), (1856, 1860), (1860, 1862), (1862, 1867), (1867, 1871), (1871, 1874),
                        (1874, 1876), (1876, 1878), (1878, 1879), (1879, 1881), (1881, 1884), (1884, 1886), (1886, 1887), (1887, 1889), (1889, 1895), (1895, 1897), (1897, 1901), (1901, 1902), (1902, 1904),
                        (1904, 1908), (1908, 1914), (1914, 1919), (1919, 1926), (1926, 1928), (1928, 1929), (1929, 1934), (1934, 1939), (1939, 1941), (1941, 1946), (1946, 1950), (1950, 1953), (1953, 1955),
                        (1955, 1956), (1956, 1959), (1959, 1962), (1962, 1963), (1963, 1973), (1973, 1982), (1982, 1987), (1987, 1996), (1996, 2006), (2006, 2011), (2011, 2019), (2019, 2024), (2024, 2027),
                        (2027, 2037), (2037, 2047), (2047, 2048), (2048, 2050), (2050, 2055), (2055, 2057), (2057, 2064), (2064, 2067), (2067, 2072), (2072, 2081), (2081, 2086), (2086, 2093), (2093, 2098),
                        (2098, 2105), (2105, 2112), (2112, 2116), (2116, 2121), (2121, 2124), (2124, 2125), (2125, 2135), (2135, 2138), (2138, 2140), (2140, 2143), (2143, 2153), (2153, 2158), (2158, 2167),
                        (2167, 2172), (2172, 2175), (2175, 2177), (2177, 2182), (2182, 2184), (2184, 2189), (2189, 2192), (2192, 2200), (2200, 2201), (2201, 2207), (2207, 2213), (2213, 2215), (2215, 2220),
                        (2220, 2227), (2227, 2232), (2232, 2235), (2235, 2243), (2243, 2245), (2245, 2254), (2254, 2255), (2255, 2258), (2258, 2259), (2259, 2261), (2261, 2264), (2264, 2269), (2269, 2274),
                        (2274, 2277), (2277, 2285), (2285, 2292), (2292, 2294), (2294, 2299), (2299, 2304), (2304, 2309), (2309, 2311), (2311, 2320), (2320, 2321), (2321, 2324), (2324, 2325), (2325, 2327),
                        (2327, 2330), (2330, 2333), (2333, 2345), (2345, 2349), (2349, 2359), (2359, 2367), (2367, 2370), (2370, 2377), (2377, 2385), (2385, 2386), (2386, 2388), (2388, 2390), (2390, 2396),
                        (2396, 2401), (2401, 2406), (2406, 2408), (2408, 2412), (2412, 2416), (2416, 2421), (2421, 2422), (2422, 2425), (2425, 2427), (2427, 2433), (2433, 2439), (2439, 2442), (2442, 2443),
                        (2443, 2445), (2445, 2449), (2449, 2450), (2450, 2455), (2455, 2456), (2456, 2461), (2461, 2462), (2462, 2467), (2467, 2470), (2470, 2471), (2471, 2474), (2474, 2478), (2478, 2480),
                        (2480, 2485), (2485, 2491), (2491, 2492), (2492, 2494), (2494, 2499), (2499, 2500), (2500, 2508), (2508, 2509), (2509, 2514), (2514, 2517), (2517, 2518), (2518, 2519), (2519, 2520),
                        (2520, 2522), (2522, 2527), (2527, 2528), (2528, 2532), (2532, 2535), (2535, 2542), (2542, 2547), (2547, 2548), (2548, 2556), (2556, 2558), (2558, 2560), (2560, 2562), (2562, 2564),
                        (2564, 2569), (2569, 2570), (2570, 2571), (2571, 2572), (2572, 2577), (2577, 2578), (2578, 2585), (2585, 2586), (2586, 2593), (2593, 2594), (2594, 2596), (2596, 2597), (2597, 2598),
                        (2598, 2600), (2600, 2601), (2601, 2603), (2603, 2608), (2608, 2609), (2609, 2617), (2617, 2630), (2630, 2634), (2634, 2635), (2635, 2637), (2637, 2641), (2641, 2649), (2649, 2651),
                        (2651, 2652), (2652, 2665), (2665, 2669), (2669, 2671), (2671, 2676), (2676, 2677), (2677, 2686), (2686, 2692), (2692, 2696), (2696, 2697), (2697, 2704), (2704, 2713), (2713, 2715),
                        (2715, 2719), (2719, 2722), (2722, 2724), (2724, 2729), (2729, 2730), (2730, 2734), (2734, 2740), (2740, 2744), (2744, 2745), (2745, 2752), (2752, 2755), (2755, 2758), (2758, 2759),
                        (2759, 2762), (2762, 2763), (2763, 2766), (2766, 2769), (2769, 2776), (2776, 2778), (2778, 2780)
                    },
                };
            }
        }

        private static BpeTokenizer _deepSeekR1Tokenizer = CreateBpeTokenizerFromJson();
        private static IReadOnlyDictionary<int, string> _vocabReverse = _deepSeekR1Tokenizer.Vocabulary.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

        [Theory]
        [MemberData(nameof(DeepSeekData))]
        public void TestDeepSeekR1Tokenizer(string text, int[] ids, string[] tokens, (int, int)[] offsets)
        {
            BpeTokenizer tokenizer = _deepSeekR1Tokenizer;
            IReadOnlyList<EncodedToken> encoding = tokenizer.EncodeToTokens(text, out _);

            Assert.Equal(ids, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokens, encoding.Select(t => t.Value).ToArray());
            Assert.Equal(offsets, encoding.Select(t => (t.Offset.Start.Value, t.Offset.End.Value)).ToArray());
            Assert.Equal(ids, tokenizer.EncodeToIds(text));
            Assert.Equal(ids.Count(), tokenizer.CountTokens(text));

            for (int i = 1; i < encoding.Count; i++)
            {
                IReadOnlyList<int> subIds = tokenizer.EncodeToIds(text, maxTokenCount: i, out _, out int charConsumed);
                Assert.Equal(i, subIds.Count);
                Assert.Equal(ids.Take(i), subIds);

                int index = tokenizer.GetIndexByTokenCount(text, i, out _, out int tokenCount);
                Assert.Equal(encoding[i].Offset.Start.Value, index);
                Assert.Equal(i, tokenCount);
                index = tokenizer.GetIndexByTokenCountFromEnd(text, i, out _, out tokenCount);
                Assert.Equal(encoding[encoding.Count - i].Offset.Start.Value, index);
                Assert.Equal(i, tokenCount);
            }

            int beginningOfSentenceId = ids[0];
            Assert.True(_vocabReverse.TryGetValue(beginningOfSentenceId, out string? beginningOfSentenceToken));
            Assert.Equal(beginningOfSentenceToken + text, tokenizer.Decode(ids));
            Assert.Equal(text, tokenizer.Decode(ids, considerSpecialTokens: false));

            char[] destinationBuffer = new char[beginningOfSentenceToken.Length + text.Length];
            Assert.Equal(OperationStatus.Done, tokenizer.Decode(ids, destinationBuffer.AsSpan(), considerSpecialTokens: true, out int idsConsumed, out int charsWritten));
            Assert.Equal(beginningOfSentenceToken + text, destinationBuffer.AsSpan(0, charsWritten).ToString());
            Assert.Equal(ids.Length, idsConsumed);
            Assert.Equal(OperationStatus.Done, tokenizer.Decode(ids, destinationBuffer.AsSpan(), considerSpecialTokens: false, out idsConsumed, out charsWritten));
            Assert.Equal(text, destinationBuffer.AsSpan(0, charsWritten).ToString());
            Assert.Equal(ids.Length, idsConsumed);
            Assert.Equal(OperationStatus.DestinationTooSmall, tokenizer.Decode(ids, destinationBuffer.AsSpan(0, text.Length - 1), considerSpecialTokens: false, out idsConsumed, out charsWritten));
            Assert.True(idsConsumed < ids.Length);
            Assert.True(charsWritten < text.Length);
            Assert.Equal(OperationStatus.DestinationTooSmall, tokenizer.Decode(ids, destinationBuffer.AsSpan(0, text.Length), considerSpecialTokens: true, out idsConsumed, out charsWritten));
            Assert.True(idsConsumed < ids.Length);
            Assert.True(charsWritten <= text.Length);

            //
            // Test special tokens
            //

            string thinkStartSpecialToken = "<think>";
            string thinkEndSpecialToken = "<think>";
            Assert.NotNull(tokenizer.SpecialTokens);
            Assert.True(tokenizer.SpecialTokens.TryGetValue(thinkStartSpecialToken, out int thinkStartId));
            Assert.True(tokenizer.SpecialTokens.TryGetValue(thinkEndSpecialToken, out int thinkEndId));

            string textWithSpecialTokens = $"{thinkStartSpecialToken}{text}{thinkEndSpecialToken}";
            encoding = tokenizer.EncodeToTokens(textWithSpecialTokens, out _);
            ids = encoding.Select(e => e.Id).ToArray();
            Assert.Equal(ids, tokenizer.EncodeToIds(textWithSpecialTokens));

            Assert.Equal(beginningOfSentenceId, encoding[0].Id);
            Assert.Equal(thinkStartId, encoding[1].Id);
            Assert.Equal(thinkEndId, encoding[^1].Id);
            Assert.Equal(beginningOfSentenceToken, encoding[0].Value);
            Assert.Equal(thinkStartSpecialToken, encoding[1].Value);
            Assert.Equal(thinkEndSpecialToken, encoding[^1].Value);

            Assert.Equal(beginningOfSentenceToken + textWithSpecialTokens, tokenizer.Decode(ids));
            Assert.Equal(text, tokenizer.Decode(ids, considerSpecialTokens: false));
        }

        private static BpeTokenizer CreateBpeTokenizerFromJson()
        {
            // @"https://huggingface.co/deepseek-ai/DeepSeek-R1/resolve/main/tokenizer.json?download=true"
            using Stream jsonModelStream = File.OpenRead(Path.Combine(@"DeepSeek", "tokenizer-DeepSeek-R1.json"));
            using var reader = new StreamReader(jsonModelStream, Encoding.UTF8);
            string json = reader.ReadToEnd();

            using JsonDocument doc = JsonDocument.Parse(json);
            JsonElement root = doc.RootElement;
            if (!root.TryGetProperty("model", out JsonElement modelElement) ||
                modelElement.ValueKind != JsonValueKind.Object ||
                !modelElement.TryGetProperty("type", out JsonElement typeElement) ||
                !"BPE".Equals(typeElement.GetString(), StringComparison.OrdinalIgnoreCase) ||
                !modelElement.TryGetProperty("vocab", out JsonElement vocabElement) ||
                vocabElement.ValueKind != JsonValueKind.Object)
            {
                throw new InvalidOperationException("Invalid model format");
            }

            BpeOptions bpeOptions = new BpeOptions(GetVocabulary(vocabElement));

            if (modelElement.TryGetProperty("unk_token", out JsonElement unKnownElement))
            {
                bpeOptions.UnknownToken = unKnownElement.GetString();
            }

            if (modelElement.TryGetProperty("continuing_subword_prefix", out JsonElement continuingSubwordPrefixElement))
            {
                bpeOptions.ContinuingSubwordPrefix = continuingSubwordPrefixElement.GetString();
            }

            if (modelElement.TryGetProperty("end_of_word_suffix", out JsonElement endOfWordSuffixElement))
            {
                bpeOptions.EndOfWordSuffix = endOfWordSuffixElement.GetString();
            }

            if (modelElement.TryGetProperty("fuse_unknown_tokens", out JsonElement fuseUnknownTokensElement))
            {
                bpeOptions.FuseUnknownTokens = fuseUnknownTokensElement.GetBoolean();
            }

            bpeOptions.SpecialTokens = GetSpecialTokens(root);
            bpeOptions.Merges = GetMerges(modelElement);
            IReadOnlyList<PreTokenizer>? preTokenizers = GetPreTokenizer(root, out bool byteLevel);
            bpeOptions.ByteLevel = byteLevel;

            if (preTokenizers is not null)
            {
                bpeOptions.PreTokenizer = new CompositePreTokenizer(preTokenizers, bpeOptions.SpecialTokens);
            }

            bpeOptions.BeginningOfSentenceToken = "<｜begin▁of▁sentence｜>";

            return BpeTokenizer.Create(bpeOptions);
        }

        private static IEnumerable<(string Token, int Id)> GetVocabulary(JsonElement vocabElement)
        {
            foreach (JsonProperty token in vocabElement.EnumerateObject())
            {
                yield return (token.Name, token.Value.GetInt32());
            }
        }

        private static IEnumerable<string> GetMerges(JsonElement modelElement)
        {
            if (modelElement.TryGetProperty("merges", out JsonElement mergesElement) && mergesElement.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement merge in mergesElement.EnumerateArray())
                {
                    if (merge.ValueKind == JsonValueKind.String)
                    {
                        yield return merge.GetString()!;
                    }
                }
            }
        }

        internal const int DefaultTimeOutInMilliseconds = 30_000;

        private static IReadOnlyList<PreTokenizer>? GetPreTokenizer(JsonElement root, out bool byteLevel)
        {
            byteLevel = false;
            List<PreTokenizer> preTokenizers = new List<PreTokenizer>();

            if (root.TryGetProperty("pre_tokenizer", out JsonElement preTokenizerElement) &&
                preTokenizerElement.ValueKind == JsonValueKind.Object &&
                preTokenizerElement.TryGetProperty("type", out JsonElement typeElement) &&
                typeElement.ValueKind == JsonValueKind.String &&
                "Sequence".Equals(typeElement.GetString(), StringComparison.OrdinalIgnoreCase) &&
                preTokenizerElement.TryGetProperty("pretokenizers", out JsonElement preTokenizersElement) &&
                preTokenizersElement.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement preTokenizer in preTokenizersElement.EnumerateArray())
                {
                    if (preTokenizer.ValueKind == JsonValueKind.Object &&
                        preTokenizer.TryGetProperty("type", out JsonElement preTokenizerTypeElement) &&
                        preTokenizerTypeElement.ValueKind == JsonValueKind.String)
                    {
                        string preTokenizerType = preTokenizerTypeElement.GetString()!;
                        if ("Split".Equals(preTokenizerType, StringComparison.OrdinalIgnoreCase))
                        {
                            if (preTokenizer.TryGetProperty("pattern", out JsonElement patternElement) &&
                                patternElement.ValueKind == JsonValueKind.Object &&
                                patternElement.TryGetProperty("Regex", out JsonElement regexElement) &&
                                regexElement.ValueKind == JsonValueKind.String)
                            {
                                string pattern = regexElement.GetString()!;

                                preTokenizers.Add(new RegexPreTokenizer(new Regex(pattern, RegexOptions.Compiled, TimeSpan.FromMilliseconds(DefaultTimeOutInMilliseconds)), null));
                            }
                        }
                        else if ("ByteLevel".Equals(preTokenizerType, StringComparison.OrdinalIgnoreCase))
                        {
                            byteLevel = true;
                        }
                    }
                }

                return preTokenizers;
            }

            return null;
        }

        private static Dictionary<string, int>? GetSpecialTokens(JsonElement root)
        {
            if (root.TryGetProperty("added_tokens", out JsonElement modelElement) && modelElement.ValueKind == JsonValueKind.Array)
            {
                Dictionary<string, int> specialTokens = new Dictionary<string, int>();
                foreach (JsonElement token in modelElement.EnumerateArray())
                {
                    if (token.TryGetProperty("content", out JsonElement contentElement) &&
                        contentElement.ValueKind == JsonValueKind.String &&
                        token.TryGetProperty("id", out JsonElement idElement) && idElement.ValueKind == JsonValueKind.Number)
                    {
                        string content = contentElement.GetString()!;
                        if (content is not null && !content.StartsWith("<｜place▁holder", StringComparison.OrdinalIgnoreCase))
                        {
                            specialTokens[content] = idElement.GetInt32();
                        }
                    }
                }

                return specialTokens;
            }

            return null;
        }
    }
}

