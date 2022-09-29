// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
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
                    new (int, int)[] { (0, 1), (1, 2), (2, 3) },
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
                    new (int, int)[] { (0, 1), (2, 3), (4, 5) },
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
                    new (int, int)[] { (0, 2), (3, 4) },
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
                    new (int, int)[] { (0, 1), (1, 2), (2, 3) },
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
                Bpe bpe = new Bpe(vocabFile, mergesFile, UnknownToken);
                bpe.FuseUnknownTokens = fuseUnknownToken;

                Assert.Equal(vocab.Count + 1u, bpe.GetVocabSize());
                Tokenizer tokenizer = new Tokenizer(bpe);

                TokenizerResult encoding = tokenizer.Encode(sentence);

                Assert.Equal(expectedTokens.Length, encoding.Tokens.Count);
                Assert.Equal(offsets.Length, encoding.Offsets.Count);
                Assert.Equal(ids.Length, encoding.Ids.Count);
                Assert.Equal(decodedTokens, tokenizer.Decode(encoding.Ids));

                for (int i = 0; i < encoding.Tokens.Count; i++)
                {
                    Assert.Equal(expectedTokens[i], encoding.Tokens[i]);
                    Assert.Equal(offsets[i], encoding.Offsets[i]);
                    Assert.Equal(ids[i], encoding.Ids[i]);
                    Assert.Equal(encoding.Tokens[i], tokenizer.Model.IdToToken(encoding.Ids[i]));
                    Assert.Equal(encoding.Ids[i], tokenizer.Model.TokenToId(encoding.Tokens[i]));
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

        [Fact]
        public void TestTrainingLoadingVocabFile()
        {
            string trainingFilePath = Utils.SaveEmbeddedResourceFile("wiki.test.raw");
            string prefix = Guid.NewGuid().ToString();
            string vocabFilePath = Path.GetTempPath() + Path.DirectorySeparatorChar + prefix + "-vocab.json";
            string mergeFilePath = Path.GetTempPath() + Path.DirectorySeparatorChar + prefix + "-merges.txt";

            try
            {
                //
                // Training
                //

                Tokenizer tokenizer = new Tokenizer(new Bpe());
                Trainer bpeTrainer = new BpeTrainer(specialTokens: new List<AddedToken>() { "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]" }, minFrequency: 0, vocabSize: 50_000);
                tokenizer.TrainFromFiles(bpeTrainer, null, trainingFilePath);

                tokenizer.Model.Save(Path.GetTempPath(), prefix);

                Assert.True(File.Exists(mergeFilePath));
                Assert.True(File.Exists(vocabFilePath));

                //
                // Create the tokenizer using the generated vocab and merges files
                //

                tokenizer = new Tokenizer(new Bpe(vocabFilePath, mergeFilePath));
                Assert.True(tokenizer.Model.GetVocab().TryGetValue("[UNK]", out int unkId));
                Assert.Equal(0, unkId);

                foreach (object?[] arguments in BpeTestData)
                {
                    TokenizerResult enc = tokenizer.Encode((string)arguments[0]!);
                    Assert.Equal((string)arguments[0]!, enc.OriginalString);
                    Assert.Equal((string[])arguments[1]!, enc.Tokens);
                    (int, int)[] offsets = ((int, int)[])arguments[2]!;
                    for (int i = 0; i < offsets.Length; i++)
                        Assert.Equal(offsets[i], enc.Offsets[i]);

                    Assert.Equal(enc.Tokens.Count, enc.Ids.Count);

                    IReadOnlyDictionary<string, int> vocab = tokenizer.Model.GetVocab();
                    for (int i = 0; i < enc.Ids.Count; i++)
                    {
                        Assert.Equal(vocab[enc.Tokens[i]], enc.Ids[i]);
                    }
                }
            }
            finally
            {
                Utils.DeleteFile(trainingFilePath);
                Utils.DeleteFile(mergeFilePath);
                Utils.DeleteFile(vocabFilePath);
            }
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
    }
}

