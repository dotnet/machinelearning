﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;

using Xunit;
using System.Diagnostics;
using System.Threading.Tasks;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class EnglishRobertaTests
    {
        private static readonly string _vocabUrl = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json";
        private static readonly string _mergeUrl = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe";
        private static readonly string _dictUrl = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt";

        public static IEnumerable<object[]> BertaData
        {
            get
            {
                // Sentence, Expected Ids, Expected Tokens, Expected Offsets, Decoded Tokens, Token occurrence values
                yield return new object[]
                {
                    "Hello Berta",
                    new int[] { 15496, 22108, 64 },
                    new string[] { "Hello", "\u0120Bert", "a" },
                    new (int, int)[] { (0, 5), (5, 5), (10, 1) },
                    new int[] { 35245, 144292, 18759122 },
                    new string[] { "Hello", " Bert", "a" },
                };

                // Intentionally repeating the same case data to test caching.
                // Sentence, Expected Ids, Expected Tokens, Expected Offsets, Decoded Tokens, Token occurrence values
                yield return new object[]
                {
                    "Hello Berta",
                    new int[] { 15496, 22108, 64 },
                    new string[] { "Hello", "\u0120Bert", "a" },
                    new (int, int)[] { (0, 5), (5, 5), (10, 1) },
                    new int[] { 35245, 144292, 18759122 },
                    new string[] { "Hello", " Bert", "a" },
                };

                // Sentence, Expected Ids, Expected Tokens, Expected Offsets, Decoded Tokens, Token occurrence values
                yield return new object[]
                {
                    "In the night.", // Highest occurrence tokens
                    new int[] { 818, 262, 1755, 13 },
                    new string[] { "In", "\u0120the", "\u0120night", "." },
                    new (int, int)[] { (0, 2), (2, 4), (6, 6), (12, 1) },
                    new int[] { 2224123, 800385005, 6062347, 850314647 },
                    new string[] { "In", " the", " night", "." },
                };

                // Sentence, Expected Ids, Expected Tokens, Expected Offsets, Decoded Tokens, Token occurrence values
                yield return new object[]
                {
                    "He\U0001F601llo Ber\U0001F601ta", // Non-Latin characters should be ignored
                    new int[] { 1544, 18798, 4312, 8326 },
                    new string[] { "He", "llo", "ĠBer", "ta" },
                    new (int, int)[] { (0, 2), (4, 3), (7, 4), (13, 2) },
                    new int[] { 2759525, 207306, 565286, 560191 },
                    new string[] { "He", "llo", " Ber", "ta" },
                };

                // Sentence, Expected Ids, Expected Tokens, Expected Offsets, Decoded Tokens, Token occurrence values
                yield return new object[]
                {
                    "\U0001F601\U0001F601\u0660\u0340", // Full Non-Latin string
                    new int[] { },
                    new string[] {  },
                    new (int, int)[] { },
                    new int[] {  },
                    new string[] {  },
                };
            }
        }

        private static Tokenizer? _robertaTokenizer = null;
        private async static Task<Tokenizer> GetRobertaTokenizer()
        {
            if (_robertaTokenizer is null)
            {
                string vocabFile = Utils.CreateTemporaryFile("json");
                string mergeFile = Utils.CreateTemporaryFile("txt");
                string translationFile = Utils.CreateTemporaryFile("txt");

                try
                {
                    await Utils.DownloadFile(_vocabUrl, vocabFile);
                    await Utils.DownloadFile(_mergeUrl, mergeFile);
                    await Utils.DownloadFile(_dictUrl, translationFile);

                    _robertaTokenizer = new EnglishRoberta(vocabFile, mergeFile, translationFile, RobertaPreTokenizer.Instance);
                }
                finally
                {
                    Utils.DeleteFile(vocabFile);
                    Utils.DeleteFile(mergeFile);
                    Utils.DeleteFile(translationFile);
                }
            }

            return _robertaTokenizer;
        }

        [Fact]
        public async void TokenizationTest()
        {
            string vocabFile = Utils.CreateTemporaryFile("json");
            string mergeFile = Utils.CreateTemporaryFile("txt");
            string translationFile = Utils.CreateTemporaryFile("txt");
            string[]? paths = null; ;

            try
            {
                await Utils.DownloadFile(_vocabUrl, vocabFile);
                await Utils.DownloadFile(_mergeUrl, mergeFile);
                await Utils.DownloadFile(_dictUrl, translationFile);

                Tokenizer tokenizer = new EnglishRoberta(vocabFile, mergeFile, translationFile, RobertaPreTokenizer.Instance);
                TestTokenizer(tokenizer);
                TokenizerTests.TestTokenLimits(tokenizer);

                tokenizer = new EnglishRoberta(vocabFile, mergeFile, translationFile, RobertaPreTokenizer.Instance, filterUnsupportedChars: false);
                TestTokenizer(tokenizer);

                using Stream vocabStream = File.OpenRead(vocabFile);
                using Stream mergeStream = File.OpenRead(mergeFile);
                using Stream translationStream = File.OpenRead(translationFile);
                tokenizer = new EnglishRoberta(vocabStream, mergeStream, translationStream, RobertaPreTokenizer.Instance);
                TestTokenizer(tokenizer);

                // Ensure caching works regardless of which method is called first.
                for (CallingOrder order = CallingOrder.Encode; order <= CallingOrder.CountTokens; order++)
                {
                    tokenizer = new EnglishRoberta(vocabFile, mergeFile, translationFile, RobertaPreTokenizer.Instance);
                    TestTokenizer(tokenizer, order);

                    tokenizer = new EnglishRoberta(vocabFile, mergeFile, translationFile, RobertaPreTokenizer.Instance, filterUnsupportedChars: false);
                    TestTokenizer(tokenizer, order);
                }
            }
            finally
            {
                Utils.DeleteFile(vocabFile);
                Utils.DeleteFile(mergeFile);
                Utils.DeleteFile(translationFile);

                if (paths is not null)
                {
                    Utils.DeleteFile(paths[0]);
                    Utils.DeleteFile(paths[1]);
                    Utils.DeleteFile(paths[2]);
                }
            }
        }

        public static IEnumerable<object?[]> RobertaTestData
        {
            get
            {
                // string to tokenize, produced tokens, the token offsets
                yield return new object?[]
                {
                    "the brown fox jumped over the lazy dog!",
                    new string[] { "the", "Ġbrown", "Ġfox", "Ġjumped", "Ġover", "Ġthe", "Ġlazy", "Ġdog", "!" },
                    new (int Index, int Length)[] { (0, 3), (3, 6), (9, 4), (13, 7), (20, 5), (25, 4), (29, 5), (34, 4), (38, 1) },
                    new int[] { 1169, 7586, 21831, 11687, 625, 262, 16931, 3290, 0 }
                };
                yield return new object?[]
                {
                    "he traveled to Egypt during the summer, the weather was hot and ammunition." ,
                    new string[] { "he", "Ġtraveled", "Ġto", "ĠEgypt", "Ġduring", "Ġthe", "Ġsummer", ",", "Ġthe", "Ġweather", "Ġwas", "Ġhot", "Ġand", "Ġammunition", "." },
                    new (int Index, int Length)[] { (0, 2), (2, 9), (11, 3), (14, 6), (20, 7), (27, 4), (31, 7), (38, 1), (39, 4), (43, 8), (51, 4), (55, 4), (59, 4), (63, 11), (74, 1) },
                    new int[] { 258, 14113, 284, 6365, 1141, 262, 3931, 11, 262, 6193, 373, 3024, 290, 14271, 13 }
                };
                yield return new object?[]
                {
                    "She played many games and she felt exhausted afterward",
                    new string[] { "She", "Ġplayed", "Ġmany", "Ġgames", "Ġand", "Ġshe", "Ġfelt", "Ġexhausted", "Ġafterward" },
                    new (int Index, int Length)[] { (0, 3), (3, 7), (10, 5), (15, 6), (21, 4), (25, 4), (29, 5), (34, 10), (44, 10) },
                    new int[] { 3347, 2826, 867, 1830, 290, 673, 2936, 19064, 20875 }
                };
                yield return new object?[]
                {
                    "Hello, y'all! How are you 😁 ?",
                    new string[] { "Hello", ",", "Ġy", "'", "all", "!", "ĠHow", "Ġare", "Ġyou", "Ġ", "Ġ?" },
                    new (int Index, int Length)[] { (0, 5), (5, 1), (6, 2), (8, 1), (9, 3), (12, 1), (13, 4), (17, 4), (21, 4), (25, 1), (28, 2) },
                    new int[] { 15496, 11, 331, 6, 439, 0, 1374, 389, 345, 220, 5633 }
                };
            }
        }

        [Theory]
        [MemberData(nameof(RobertaTestData))]
        public async void TestTokenizerEncoding(string text, string[] expectedTokens, (int Index, int Length)[] expectedOffsets, int[] expectedIds)
        {
            Tokenizer tokenizer = await GetRobertaTokenizer();

            IReadOnlyList<Token> encoding = tokenizer.Encode(text, out _);
            IReadOnlyList<Token> encoding1 = tokenizer.Encode(text.AsSpan(), out _);

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

            Assert.Equal(expectedOffsets[expectedOffsets.Length - 4].Index + expectedOffsets[expectedOffsets.Length - 4].Length, tokenizer.IndexOfTokenCount(text, expectedIds.Length - 3, out normalizedString, out int tokenCount));
            Assert.Null(normalizedString);
            Assert.Equal(expectedIds.Length - 3, tokenCount);
            Assert.Equal(expectedOffsets[expectedOffsets.Length - 4].Index + expectedOffsets[expectedOffsets.Length - 4].Length, tokenizer.IndexOfTokenCount(text.AsSpan(), expectedIds.Length - 3, out normalizedString, out tokenCount));
            Assert.Null(normalizedString);
            Assert.Equal(expectedIds.Length - 3, tokenCount);

            Assert.Equal(expectedOffsets[expectedOffsets.Length - 3].Index, tokenizer.LastIndexOfTokenCount(text, 3, out normalizedString, out tokenCount));
            Assert.Null(normalizedString);
            Assert.Equal(3, tokenCount);
            Assert.Equal(expectedOffsets[expectedOffsets.Length - 3].Index, tokenizer.LastIndexOfTokenCount(text.AsSpan(), 3, out normalizedString, out tokenCount));
            Assert.Null(normalizedString);
            Assert.Equal(3, tokenCount);
        }

        private enum CallingOrder
        {
            Encode,
            EncodeToIds,
            CountTokens
        }

        // Calling EncodeToIds after calling Encode will cause EncodeToIds uses the cached data from the previous Encode call.
        // Calling with callIdsFirst = true will test the other way around.
        private void TestTokenizer(Tokenizer tokenizer, CallingOrder callingOrder = CallingOrder.Encode)
        {
            Assert.True(tokenizer is EnglishRoberta);
            Assert.True(tokenizer.PreTokenizer is RobertaPreTokenizer);

            foreach (object[] p in BertaData)
            {
                IReadOnlyList<int> ids;
                IReadOnlyList<Token> encoding;
                int idsCount;

                if (callingOrder == CallingOrder.Encode)
                {
                    encoding = tokenizer.Encode((string)p[0], out _);
                    ids = tokenizer.EncodeToIds((string)p[0]);
                    idsCount = tokenizer.CountTokens((string)p[0]);
                }
                else if (callingOrder == CallingOrder.EncodeToIds)
                {
                    ids = tokenizer.EncodeToIds((string)p[0]);
                    encoding = tokenizer.Encode((string)p[0], out _);
                    idsCount = tokenizer.CountTokens((string)p[0]);
                }
                else // CountTokens
                {
                    idsCount = tokenizer.CountTokens((string)p[0]);
                    ids = tokenizer.EncodeToIds((string)p[0]);
                    encoding = tokenizer.Encode((string)p[0], out _);
                }

                int[] encodingIds = encoding.Select(t => t.Id).ToArray();
                (int, int)[] offsets = encoding.Select(t => t.Offset).ToArray();
                string[] tokens = encoding.Select(t => t.Value).ToArray();

                Assert.Equal(p[1], encodingIds);
                Assert.Equal(p[1], ids);
                Assert.Equal(((int[])p[1]).Length, idsCount);
                Assert.Equal(p[3], offsets);

                EnglishRoberta? robertaModel = tokenizer as EnglishRoberta;
                Assert.Equal(p[2], tokens);

                Assert.Equal(string.Concat((string[])(p[robertaModel!.FilterUnsupportedChars ? 5 : 2])), tokenizer.Decode(encodingIds));

                Assert.NotNull(robertaModel);
                Assert.Equal(encodingIds, robertaModel!.ConvertOccurrenceRanksToIds(robertaModel!.ConvertIdsToOccurrenceRanks(encodingIds)));
                Assert.Equal(p[4], robertaModel.ConvertIdsToOccurrenceValues(encodingIds));

                for (int i = 0; i < tokens.Length; i++)
                {
                    if (robertaModel.FilterUnsupportedChars)
                    {
                        string[]? filteredToken = p[5] as string[];
                        Assert.Equal(filteredToken![i], tokenizer.MapIdToToken(encodingIds[i]));
                    }
                    else
                    {
                        Assert.Equal(tokens[i], tokenizer.MapIdToToken(encodingIds[i]));
                        string[]? unfilteredToken = p[2] as string[];
                        Assert.Equal(unfilteredToken![i], tokenizer.MapIdToToken(encodingIds[i]));
                    }

                    Assert.Equal(encodingIds[i], tokenizer.MapTokenToId(tokens[i].AsSpan()));
                }
            }
        }
    }
}
