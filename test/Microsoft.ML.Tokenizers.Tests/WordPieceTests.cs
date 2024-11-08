// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class WordPieceTests
    {
        static string[] _vocabTokens = ["[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn", "##ing"];

        internal static string CreateVocabFile(string[] vocabTokens)
        {
            string vocabFile = Path.GetTempFileName();
            File.WriteAllLines(vocabFile, vocabTokens);
            return vocabFile;
        }

        [Fact]
        public void TestCreation()
        {
            string vocabFile = CreateVocabFile(_vocabTokens);

            try
            {
                using Stream vocabStream = File.OpenRead(vocabFile);
                WordPieceTokenizer[] wordPieceTokenizers = [WordPieceTokenizer.Create(vocabFile), WordPieceTokenizer.Create(vocabStream)];

                foreach (var tokenizer in wordPieceTokenizers)
                {
                    Assert.NotNull(tokenizer.PreTokenizer);
                    Assert.Equal("[UNK]", tokenizer.UnknownToken);
                    Assert.Equal(0, tokenizer.UnknownTokenId);
                    Assert.Null(tokenizer.Normalizer);
                    Assert.Equal(100, tokenizer.MaxInputCharsPerWord);
                    Assert.Equal("##", tokenizer.ContinuingSubwordPrefix);
                }
            }
            finally
            {
                File.Delete(vocabFile);
            }
        }

        [Fact]
        public void TestTokenization()
        {
            string vocabFile = CreateVocabFile(_vocabTokens);

            try
            {
                WordPieceTokenizer tokenizer = WordPieceTokenizer.Create(vocabFile);

                Assert.Null(tokenizer.SpecialTokens);

                IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens("", out _);
                Assert.Empty(tokens);
                Assert.Equal(0, tokenizer.CountTokens(""));
                IReadOnlyList<int> ids = tokenizer.EncodeToIds("");
                Assert.Empty(ids);
                int index = tokenizer.GetIndexByTokenCount("", maxTokenCount: 10, normalizedText: out _, tokenCount: out int tokenCount);
                Assert.Equal(0, index);
                Assert.Equal(0, tokenCount);
                index = tokenizer.GetIndexByTokenCountFromEnd("", maxTokenCount: 10, normalizedText: out _, tokenCount: out tokenCount);
                Assert.Equal(0, index);
                Assert.Equal(0, tokenCount);

                string text = "unwanted running";
                tokens = tokenizer.EncodeToTokens(text, out _);
                Assert.Equal(
                    [
                        new EncodedToken(7, "un", new Range(0, 2)),
                        new EncodedToken(4, "##want", new Range(2, 6)),
                        new EncodedToken(5, "##ed", new Range(6, 8)),
                        new EncodedToken(8, "runn", new Range(9, 13)),
                        new EncodedToken(9, "##ing", new Range(13, 16))
                    ],
                    tokens
                );

                ids = tokenizer.EncodeToIds(text);
                Assert.Equal([7, 4, 5, 8, 9], ids);

                int[] expectedTokenCount = [0, 0, 3, 3, 5];
                for (int i = 1; i <= 5; i++)
                {
                    Assert.Equal(ids.Take(expectedTokenCount[i - 1]).ToArray(), tokenizer.EncodeToIds(text, maxTokenCount: i, normalizedText: out _, out tokenCount));
                }

                Assert.Equal(text, tokenizer.Decode(ids));

                Span<char> buffer = stackalloc char[text.Length];
                for (int i = 0; i < text.Length - 1; i++)
                {
                    Span<char> bufferSlice = buffer.Slice(0, i);
                    OperationStatus result = tokenizer.Decode(ids, bufferSlice, out int idsConsumed, out int charsWritten);
                    Assert.Equal(OperationStatus.DestinationTooSmall, result);

                    int j = 0;

                    while (i >= tokens[j].Offset.End.Value)
                    {
                        j++;
                    }

                    Assert.Equal(j, idsConsumed);
                    Assert.Equal(j == 0 ? 0 : tokens[j - 1].Offset.End.Value, charsWritten);
                    Assert.Equal(j == 0 ? "" : text.Substring(0, tokens[j - 1].Offset.End.Value), bufferSlice.Slice(0, charsWritten).ToString());
                }

                Assert.Equal(5, tokenizer.CountTokens(text));

                int[] expectedIndexes = [0, 0, 8, 9, 16];
                expectedTokenCount = [0, 0, 3, 3, 5];

                for (int i = 1; i <= 5; i++)
                {
                    index = tokenizer.GetIndexByTokenCount(text, maxTokenCount: i, normalizedText: out _, out tokenCount);
                    Assert.Equal(expectedTokenCount[i - 1], tokenCount);
                    Assert.Equal(expectedIndexes[i - 1], index);
                }

                expectedIndexes = [16, 9, 8, 8, 0];
                expectedTokenCount = [0, 2, 2, 2, 5];

                for (int i = 1; i <= 5; i++)
                {
                    index = tokenizer.GetIndexByTokenCountFromEnd(text, maxTokenCount: i, normalizedText: out _, out tokenCount);
                    Assert.Equal(expectedTokenCount[i - 1], tokenCount);
                    Assert.Equal(expectedIndexes[i - 1], index);
                }
            }
            finally
            {
                File.Delete(vocabFile);
            }
        }

        [Fact]
        public void TestTokenizationWithUnknownTokens()
        {
            string vocabFile = CreateVocabFile(_vocabTokens);

            try
            {
                WordPieceTokenizer tokenizer = WordPieceTokenizer.Create(vocabFile);

                string text = "unwantedX running";

                IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens(text, out _);
                Assert.Equal(
                    [
                        new EncodedToken(0, "[UNK]", new Range(0, 9)),
                        new EncodedToken(8, "runn",  new Range(10, 14)),
                        new EncodedToken(9, "##ing", new Range(14, 17))
                    ],
                    tokens
                );

                IReadOnlyList<int> ids = tokenizer.EncodeToIds(text);
                Assert.Equal([0, 8, 9], ids);

                Assert.Equal("[UNK] running", tokenizer.Decode(ids));
            }
            finally
            {
                File.Delete(vocabFile);
            }
        }

        [Fact]
        public void TestTokenizationWithSpecialTokens()
        {
            string vocabFile = CreateVocabFile(_vocabTokens);

            try
            {
                Dictionary<string, int> specialTokens = new Dictionary<string, int>
                {
                    { "[UNK]", 0 }, { "[CLS]", 1 }, { "[SEP]", 2 }
                };
                WordPieceTokenizer tokenizer = WordPieceTokenizer.Create(vocabFile, new WordPieceOptions { SpecialTokens = specialTokens });

                Assert.Equal(specialTokens, tokenizer.SpecialTokens);

                string text = "[UNK] unwanted [SEP][CLS] running [CLS]";

                IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens(text, out _);
                Assert.Equal(
                    [
                        new EncodedToken(0, "[UNK]", new Range(0, 5)),
                        new EncodedToken(7, "un", new Range(6, 8)),
                        new EncodedToken(4, "##want", new Range(8, 12)),
                        new EncodedToken(5, "##ed", new Range(12, 14)),
                        new EncodedToken(2, "[SEP]", new Range(15, 20)),
                        new EncodedToken(1, "[CLS]", new Range(20, 25)),
                        new EncodedToken(8, "runn", new Range(26, 30)),
                        new EncodedToken(9, "##ing", new Range(30, 33)),
                        new EncodedToken(1, "[CLS]", new Range(34, 39)),
                    ],
                    tokens
                );

                IReadOnlyList<int> ids = tokenizer.EncodeToIds(text);
                Assert.Equal([0, 7, 4, 5, 2, 1, 8, 9, 1], ids);

                Assert.Equal("[UNK] unwanted [SEP] [CLS] running [CLS]", tokenizer.Decode(ids));
            }
            finally
            {
                File.Delete(vocabFile);
            }
        }
    }
}
