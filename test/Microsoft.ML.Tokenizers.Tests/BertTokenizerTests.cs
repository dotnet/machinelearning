// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics.Tracing;
using System.IO;
using System.Threading.Tasks;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class BertTokenizerTests
    {
        [Fact]
        public void TestWithLowerCasing()
        {
            //                     Ids: 0        1        2        3        4       5    6    7      8        9      10      11     12
            string[] vocabTokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "!", ",", "?", "hello", "world", "how", "are", "you"];

            string vocabFile = WordPieceTests.CreateVocabFile(vocabTokens);

            try
            {
                using Stream vocabStream = File.OpenRead(vocabFile);
                BertTokenizer[] bertTokenizers = [BertTokenizer.Create(vocabFile), BertTokenizer.Create(vocabStream)];

                foreach (var tokenizer in bertTokenizers)
                {
                    Assert.NotNull(tokenizer.PreTokenizer);
                    Assert.Equal("[UNK]", tokenizer.UnknownToken);
                    Assert.Equal(1, tokenizer.UnknownTokenId);
                    Assert.NotNull(tokenizer.Normalizer);
                    Assert.NotNull(tokenizer.PreTokenizer);

                    string text = "Hello, How are you?";
                    var tokens = tokenizer.EncodeToTokens(text, out string? normalizedText);
                    Assert.Equal("hello, how are you?", normalizedText);

                    Assert.Equal(
                        [
                            new EncodedToken(8, "hello", new Range(0, 5)),
                            new EncodedToken(6, ",", new Range(5, 6)),
                            new EncodedToken(10, "how", new Range(7, 10)),
                            new EncodedToken(11, "are", new Range(11, 14)),
                            new EncodedToken(12, "you", new Range(15, 18)),
                            new EncodedToken(7, "?", new Range(18, 19))
                        ],
                        tokens);

                    var ids = tokenizer.EncodeToIds(text);
                    Assert.Equal([tokenizer.ClassificationTokenId, 8, 6, 10, 11, 12, 7, tokenizer.SeparatorTokenId], ids);

                    Assert.Equal("[CLS] hello, how are you? [SEP]", tokenizer.Decode(ids));
                    Assert.Equal("hello, how are you?", tokenizer.Decode(ids, skipSpecialTokens: true));

                    tokens = tokenizer.EncodeToTokens(tokenizer.Decode(ids), out normalizedText);
                    Assert.Equal("[cls] hello, how are you? [sep]", normalizedText);
                    Assert.Equal(
                        [
                            new EncodedToken(2, "[CLS]", new Range(0, 5)),
                            new EncodedToken(8, "hello", new Range(6, 11)),
                            new EncodedToken(6, ",", new Range(11, 12)),
                            new EncodedToken(10, "how", new Range(13, 16)),
                            new EncodedToken(11, "are", new Range(17, 20)),
                            new EncodedToken(12, "you", new Range(21, 24)),
                            new EncodedToken(7, "?", new Range(24, 25)),
                            new EncodedToken(3, "[SEP]", new Range(26, 31))
                        ],
                        tokens);

                    ids = tokenizer.EncodeToIds(normalizedText!);
                    Assert.Equal([tokenizer.ClassificationTokenId, tokenizer.ClassificationTokenId, 8, 6, 10, 11, 12, 7, tokenizer.SeparatorTokenId, tokenizer.SeparatorTokenId], ids);
                }
            }
            finally
            {
                File.Delete(vocabFile);
            }
        }

        [Fact]
        public void TestWithNoLowerCasing()
        {
            //                   Ids: 0        1        2        3        4       5    6    7      8        9      10      11     12
            string[] vocabTokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "!", ",", "?", "hello", "world", "how", "are", "you"];

            string vocabFile = WordPieceTests.CreateVocabFile(vocabTokens);

            try
            {
                using Stream vocabStream = File.OpenRead(vocabFile);
                BertTokenizer[] bertTokenizers = [BertTokenizer.Create(vocabFile, new BertOptions { LowerCaseBeforeTokenization = false }),
                                                  BertTokenizer.Create(vocabStream, new BertOptions { LowerCaseBeforeTokenization = false })];

                foreach (var tokenizer in bertTokenizers)
                {
                    Assert.NotNull(tokenizer.PreTokenizer);
                    Assert.Equal("[UNK]", tokenizer.UnknownToken);
                    Assert.Equal(1, tokenizer.UnknownTokenId);
                    Assert.NotNull(tokenizer.Normalizer);
                    Assert.NotNull(tokenizer.PreTokenizer);

                    string text = "Hello, How are you?";
                    var tokens = tokenizer.EncodeToTokens(text, out string? normalizedText);
                    Assert.Equal("Hello, How are you?", normalizedText);

                    Assert.Equal(
                        [
                            new EncodedToken(1, "[UNK]", new Range(0, 5)),
                            new EncodedToken(6, ",", new Range(5, 6)),
                            new EncodedToken(1, "[UNK]", new Range(7, 10)),
                            new EncodedToken(11, "are", new Range(11, 14)),
                            new EncodedToken(12, "you", new Range(15, 18)),
                            new EncodedToken(7, "?", new Range(18, 19))
                        ],
                        tokens);

                    var ids = tokenizer.EncodeToIds(text);
                    Assert.Equal([tokenizer.ClassificationTokenId, 1, 6, 1, 11, 12, 7, tokenizer.SeparatorTokenId], ids);

                    Assert.Equal("[CLS] [UNK], [UNK] are you? [SEP]", tokenizer.Decode(ids));
                    Assert.Equal(", are you?", tokenizer.Decode(ids, skipSpecialTokens: true));
                }
            }
            finally
            {
                File.Delete(vocabFile);
            }
        }

        [Fact]
        public async Task TestWithAccentMarks()
        {
            //                  Ids:   0        1        2        3        4       5    6    7      8       9      10      11     12       13       14           15          16         17        18       19
            string[] vocabTokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "!", ",", "?", "Café", "cafe", "café", "Über", "über", "uber", "Ångström", "ångström", "angstrom", "Résumé", "résumé", "resume",
            //                  Ids:  20     21        22         23
                                    "Cafe", "Uber", "Angstrom", "Resume"];
            string vocabFile = WordPieceTests.CreateVocabFile(vocabTokens);

            try
            {
                using Stream vocabStream = File.OpenRead(vocabFile);
                BertTokenizer bertTokenizer = await BertTokenizer.CreateAsync(vocabStream); // lowercasing and no accent stripping

                string text = "Café Über Ångström Résumé!";
                var tokens = bertTokenizer.EncodeToTokens(text, out string? normalizedText);
                Assert.Equal(
                    [
                        new EncodedToken(10, "café", new Range(0, 4)),
                        new EncodedToken(12, "über", new Range(5, 9)),
                        new EncodedToken(15, "ångström", new Range(10, 18)),
                        new EncodedToken(18, "résumé", new Range(19, 25)),
                        new EncodedToken(5, "!", new Range(25, 26)),
                    ],
                    tokens);

                Assert.Equal("café über ångström résumé!", normalizedText);

                vocabStream.Position = 0;
                bertTokenizer = await BertTokenizer.CreateAsync(vocabStream, new BertOptions { LowerCaseBeforeTokenization = false }); // no lowercasing and no accent stripping
                tokens = bertTokenizer.EncodeToTokens(text, out normalizedText);
                Assert.Equal(
                    [
                        new EncodedToken(8, "Café", new Range(0, 4)),
                        new EncodedToken(11, "Über", new Range(5, 9)),
                        new EncodedToken(14, "Ångström", new Range(10, 18)),
                        new EncodedToken(17, "Résumé", new Range(19, 25)),
                        new EncodedToken(5, "!", new Range(25, 26)),
                    ],
                    tokens);

                Assert.Equal("Café Über Ångström Résumé!", normalizedText);

                vocabStream.Position = 0;
                bertTokenizer = await BertTokenizer.CreateAsync(vocabStream, new BertOptions { RemoveNonSpacingMarks = true }); // lowercasing and accent stripping
                tokens = bertTokenizer.EncodeToTokens(text, out normalizedText);
                Assert.Equal("cafe uber angstrom resume!", normalizedText);
                Assert.Equal(
                    [
                        new EncodedToken(9, "cafe", new Range(0, 4)),
                        new EncodedToken(13, "uber", new Range(5, 9)),
                        new EncodedToken(16, "angstrom", new Range(10, 18)),
                        new EncodedToken(19, "resume", new Range(19, 25)),
                        new EncodedToken(5, "!", new Range(25, 26)),
                    ],
                    tokens);

                vocabStream.Position = 0;
                bertTokenizer = await BertTokenizer.CreateAsync(vocabStream, new BertOptions { LowerCaseBeforeTokenization = false, RemoveNonSpacingMarks = true }); // no lowercasing and accent stripping
                tokens = bertTokenizer.EncodeToTokens(text, out normalizedText);
                Assert.Equal("Cafe Uber Angstrom Resume!", normalizedText);
                Assert.Equal(
                    [
                        new EncodedToken(20, "Cafe", new Range(0, 4)),
                        new EncodedToken(21, "Uber", new Range(5, 9)),
                        new EncodedToken(22, "Angstrom", new Range(10, 18)),
                        new EncodedToken(23, "Resume", new Range(19, 25)),
                        new EncodedToken(5, "!", new Range(25, 26)),
                    ],
                    tokens);
            }
            finally
            {
                File.Delete(vocabFile);
            }
        }

        [Fact]
        public async Task TestChineseCharacters()
        {
            //                 Ids:    0        1        2        3        4       5     6       7      8     9    10    11    12
            string[] vocabTokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "!", "##驷", "##驸", "受", "叟", "叢", "驷", "驸"];
            string vocabFile = WordPieceTests.CreateVocabFile(vocabTokens);

            try
            {
                using Stream vocabStream = File.OpenRead(vocabFile);
                BertTokenizer bertTokenizer = await BertTokenizer.CreateAsync(vocabStream); // tokenize Chinese characters
                string text = "叟驷 叢驸!";

                var tokens = bertTokenizer.EncodeToTokens(text, out string? normalizedText);
                Assert.Equal(" 叟  驷   叢  驸 !", normalizedText);
                Assert.Equal(
                    [
                        new EncodedToken(9, "叟", new Range(1, 2)),
                        new EncodedToken(11, "驷", new Range(4, 5)),
                        new EncodedToken(10, "叢", new Range(8, 9)),
                        new EncodedToken(12, "驸", new Range(11, 12)),
                        new EncodedToken(5, "!", new Range(13, 14))
                    ],
                    tokens);
                IReadOnlyList<int> ids = bertTokenizer.EncodeToIds(text);
                Assert.Equal("[CLS] 叟 驷 叢 驸! [SEP]", bertTokenizer.Decode(bertTokenizer.EncodeToIds(text)));
                Assert.Equal("叟 驷 叢 驸!", bertTokenizer.Decode(bertTokenizer.EncodeToIds(text), skipSpecialTokens: true));

                vocabStream.Position = 0;
                bertTokenizer = await BertTokenizer.CreateAsync(vocabStream, new BertOptions { IndividuallyTokenizeCjk = false }); // do not tokenize Chinese characters
                tokens = bertTokenizer.EncodeToTokens(text, out normalizedText);
                Assert.Equal("叟驷 叢驸!", normalizedText);

                Assert.Equal(
                    [
                        new EncodedToken(9, "叟", new Range(0, 1)),
                        new EncodedToken(6, "##驷", new Range(1, 2)),
                        new EncodedToken(10, "叢", new Range(3, 4)),
                        new EncodedToken(7, "##驸", new Range(4, 5)),
                        new EncodedToken(5, "!", new Range(5, 6))
                    ],
                    tokens);
                ids = bertTokenizer.EncodeToIds(text);
                Assert.Equal("[CLS] 叟驷 叢驸! [SEP]", bertTokenizer.Decode(bertTokenizer.EncodeToIds(text)));
                Assert.Equal("叟驷 叢驸!", bertTokenizer.Decode(bertTokenizer.EncodeToIds(text), skipSpecialTokens: true));
            }
            finally
            {
                File.Delete(vocabFile);
            }
        }

        [Fact]
        public void TestBuildInputsWithSpecialTokens()
        {
            //                   Ids: 0        1        2        3        4        5    6    7      8       9       10     11     12    13    14     15
            string[] vocabTokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "!", ",", "?", "hello", "world", "how", "are", "you", "i", "am", "fine"];

            string vocabFile = WordPieceTests.CreateVocabFile(vocabTokens);

            try
            {
                using Stream vocabStream = File.OpenRead(vocabFile);
                BertTokenizer bertTokenizer = BertTokenizer.Create(vocabFile);

                string text1 = "Hello, How are you?";
                string text2 = "I am fine!";

                var ids1 = bertTokenizer.EncodeToIds(text1);
                Assert.Equal([bertTokenizer.ClassificationTokenId, 8, 6, 10, 11, 12, 7, bertTokenizer.SeparatorTokenId], ids1);

                var ids2 = bertTokenizer.EncodeToIds(text2);
                Assert.Equal([bertTokenizer.ClassificationTokenId, 13, 14, 15, 5, bertTokenizer.SeparatorTokenId], ids2);

                Assert.Equal(
                    [bertTokenizer.ClassificationTokenId, bertTokenizer.ClassificationTokenId, 8, 6, 10, 11, 12, 7, bertTokenizer.SeparatorTokenId, bertTokenizer.SeparatorTokenId],
                    bertTokenizer.BuildInputsWithSpecialTokens(ids1));

                Span<int> ids1Span = stackalloc int[1];
                OperationStatus status = bertTokenizer.BuildInputsWithSpecialTokens(ids1, ids1Span, out int written);
                Assert.Equal(OperationStatus.DestinationTooSmall, status);
                Assert.Equal(0, written);

                ids1Span = stackalloc int[ids1.Count + 2];
                status = bertTokenizer.BuildInputsWithSpecialTokens(ids1, ids1Span, out written);
                Assert.Equal(OperationStatus.Done, status);
                Assert.Equal(ids1.Count + 2, written);
                Assert.Equal(new int[] { bertTokenizer.ClassificationTokenId, bertTokenizer.ClassificationTokenId, 8, 6, 10, 11, 12, 7, bertTokenizer.SeparatorTokenId, bertTokenizer.SeparatorTokenId }, ids1Span.ToArray());

                Assert.Equal(
                    [bertTokenizer.ClassificationTokenId, bertTokenizer.ClassificationTokenId, 8, 6, 10, 11, 12, 7, bertTokenizer.SeparatorTokenId, bertTokenizer.SeparatorTokenId, bertTokenizer.ClassificationTokenId, 13, 14, 15, 5, bertTokenizer.SeparatorTokenId, bertTokenizer.SeparatorTokenId],
                    bertTokenizer.BuildInputsWithSpecialTokens(ids1, ids2));

                ids1Span = stackalloc int[1];
                status = bertTokenizer.BuildInputsWithSpecialTokens(ids1, ids1Span, out written, ids2);
                Assert.Equal(OperationStatus.DestinationTooSmall, status);
                Assert.Equal(0, written);

                ids1Span = stackalloc int[ids1.Count + ids2.Count + 3];
                status = bertTokenizer.BuildInputsWithSpecialTokens(ids1, ids1Span, out written, ids2);
                Assert.Equal(OperationStatus.Done, status);
                Assert.Equal(ids1Span.Length, written);
                Assert.Equal(
                        new int[] { bertTokenizer.ClassificationTokenId, bertTokenizer.ClassificationTokenId, 8, 6, 10, 11, 12, 7, bertTokenizer.SeparatorTokenId, bertTokenizer.SeparatorTokenId, bertTokenizer.ClassificationTokenId, 13, 14, 15, 5, bertTokenizer.SeparatorTokenId, bertTokenizer.SeparatorTokenId },
                        ids1Span.ToArray());

                ids1 = bertTokenizer.EncodeToIds(text1, addSpecialTokens: false);
                Assert.Equal([8, 6, 10, 11, 12, 7], ids1);

                ids2 = bertTokenizer.EncodeToIds(text2, addSpecialTokens: false);
                Assert.Equal([13, 14, 15, 5], ids2);

                Assert.Equal(
                    [bertTokenizer.ClassificationTokenId, 8, 6, 10, 11, 12, 7, bertTokenizer.SeparatorTokenId],
                    bertTokenizer.BuildInputsWithSpecialTokens(ids1));

                ids1Span = stackalloc int[1];
                status = bertTokenizer.BuildInputsWithSpecialTokens(ids1, ids1Span, out written);
                Assert.Equal(OperationStatus.DestinationTooSmall, status);
                Assert.Equal(0, written);

                ids1Span = stackalloc int[ids1.Count + 2];
                status = bertTokenizer.BuildInputsWithSpecialTokens(ids1, ids1Span, out written);
                Assert.Equal(OperationStatus.Done, status);
                Assert.Equal(ids1Span.Length, written);
                Assert.Equal(
                        new int[] { bertTokenizer.ClassificationTokenId, 8, 6, 10, 11, 12, 7, bertTokenizer.SeparatorTokenId },
                        ids1Span.ToArray());

                Assert.Equal(
                    [bertTokenizer.ClassificationTokenId, 8, 6, 10, 11, 12, 7, bertTokenizer.SeparatorTokenId, 13, 14, 15, 5, bertTokenizer.SeparatorTokenId],
                    bertTokenizer.BuildInputsWithSpecialTokens(ids1, ids2));

                ids1Span = stackalloc int[1];
                status = bertTokenizer.BuildInputsWithSpecialTokens(ids1, ids1Span, out written, ids2);
                Assert.Equal(OperationStatus.DestinationTooSmall, status);
                Assert.Equal(0, written);

                ids1Span = stackalloc int[ids1.Count + ids2.Count + 3];
                status = bertTokenizer.BuildInputsWithSpecialTokens(ids1, ids1Span, out written, ids2);
                Assert.Equal(OperationStatus.Done, status);
                Assert.Equal(ids1Span.Length, written);
                Assert.Equal(
                        new int[] { bertTokenizer.ClassificationTokenId, 8, 6, 10, 11, 12, 7, bertTokenizer.SeparatorTokenId, 13, 14, 15, 5, bertTokenizer.SeparatorTokenId },
                        ids1Span.ToArray());
            }
            finally
            {
                File.Delete(vocabFile);
            }
        }

        [Fact]
        public void TestGetSpecialTokensMask()
        {
            //                   Ids: 0        1        2        3        4        5    6    7      8       9       10     11     12    13    14     15
            string[] vocabTokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "!", ",", "?", "hello", "world", "how", "are", "you", "i", "am", "fine"];

            string vocabFile = WordPieceTests.CreateVocabFile(vocabTokens);

            try
            {
                using Stream vocabStream = File.OpenRead(vocabFile);
                BertTokenizer bertTokenizer = BertTokenizer.Create(vocabFile);

                string text1 = "Hello, How are you?";
                string text2 = "I am fine!";

                var ids1 = bertTokenizer.EncodeToIds(text1);
                Assert.Equal([bertTokenizer.ClassificationTokenId, 8, 6, 10, 11, 12, 7, bertTokenizer.SeparatorTokenId], ids1);

                var ids2 = bertTokenizer.EncodeToIds(text2);
                Assert.Equal([bertTokenizer.ClassificationTokenId, 13, 14, 15, 5, bertTokenizer.SeparatorTokenId], ids2);

                Assert.Equal(
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    bertTokenizer.GetSpecialTokensMask(ids1, additionalTokenIds: null, alreadyHasSpecialTokens: true));

                Span<int> ids1Span = stackalloc int[1];
                OperationStatus status = bertTokenizer.GetSpecialTokensMask(ids1, ids1Span, out int written, alreadyHasSpecialTokens: true);
                Assert.Equal(OperationStatus.DestinationTooSmall, status);
                Assert.Equal(0, written);

                ids1Span = stackalloc int[ids1.Count];
                status = bertTokenizer.GetSpecialTokensMask(ids1, ids1Span, out written, alreadyHasSpecialTokens: true);
                Assert.Equal(OperationStatus.Done, status);
                Assert.Equal(ids1.Count, written);
                Assert.Equal(new int[] { 1, 0, 0, 0, 0, 0, 0, 1 }, ids1Span.ToArray());

                Assert.Equal(
                    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                    bertTokenizer.GetSpecialTokensMask(ids1, additionalTokenIds: ids2, alreadyHasSpecialTokens: true));

                ids1Span = stackalloc int[1];
                status = bertTokenizer.GetSpecialTokensMask(ids1, ids1Span, out written, ids2, alreadyHasSpecialTokens: true);
                Assert.Equal(OperationStatus.DestinationTooSmall, status);
                Assert.Equal(0, written);

                ids1Span = stackalloc int[ids1.Count + ids2.Count];
                status = bertTokenizer.GetSpecialTokensMask(ids1, ids1Span, out written, ids2, alreadyHasSpecialTokens: true);
                Assert.Equal(OperationStatus.Done, status);
                Assert.Equal(ids1.Count + ids2.Count, written);
                Assert.Equal(new int[] { 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1 }, ids1Span.ToArray());

                ids1 = bertTokenizer.EncodeToIds(text1, addSpecialTokens: false);
                Assert.Equal([8, 6, 10, 11, 12, 7], ids1);

                ids2 = bertTokenizer.EncodeToIds(text2, addSpecialTokens: false);
                Assert.Equal([13, 14, 15, 5], ids2);
                Assert.Equal(
                    [1, 0, 0, 0, 0, 0, 0, 1],
                    bertTokenizer.GetSpecialTokensMask(ids1, additionalTokenIds: null, alreadyHasSpecialTokens: false));

                ids1Span = stackalloc int[1];
                status = bertTokenizer.GetSpecialTokensMask(ids1, ids1Span, out written, alreadyHasSpecialTokens: false);
                Assert.Equal(OperationStatus.DestinationTooSmall, status);
                Assert.Equal(0, written);

                ids1Span = stackalloc int[ids1.Count + 2];
                status = bertTokenizer.GetSpecialTokensMask(ids1, ids1Span, out written, alreadyHasSpecialTokens: false);
                Assert.Equal(OperationStatus.Done, status);
                Assert.Equal(ids1.Count + 2, written);
                Assert.Equal(new int[] { 1, 0, 0, 0, 0, 0, 0, 1 }, ids1Span.ToArray());

                Assert.Equal(
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                    bertTokenizer.GetSpecialTokensMask(ids1, additionalTokenIds: ids2, alreadyHasSpecialTokens: false));

                ids1Span = stackalloc int[1];
                status = bertTokenizer.GetSpecialTokensMask(ids1, ids1Span, out written, ids2, alreadyHasSpecialTokens: false);
                Assert.Equal(OperationStatus.DestinationTooSmall, status);
                Assert.Equal(0, written);

                ids1Span = stackalloc int[ids1.Count + ids2.Count + 3];
                status = bertTokenizer.GetSpecialTokensMask(ids1, ids1Span, out written, ids2, alreadyHasSpecialTokens: false);
                Assert.Equal(OperationStatus.Done, status);
                Assert.Equal(ids1.Count + ids2.Count + 3, written);
                Assert.Equal(new int[] { 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }, ids1Span.ToArray());
            }
            finally
            {
                File.Delete(vocabFile);
            }
        }

        [Fact]
        public void TestCreateTokenTypeIdsFromSequences()
        {
            //                   Ids: 0        1        2        3        4        5    6    7      8       9       10     11     12    13    14     15
            string[] vocabTokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "!", ",", "?", "hello", "world", "how", "are", "you", "i", "am", "fine"];

            string vocabFile = WordPieceTests.CreateVocabFile(vocabTokens);

            try
            {
                using Stream vocabStream = File.OpenRead(vocabFile);
                BertTokenizer bertTokenizer = BertTokenizer.Create(vocabFile);

                string text1 = "Hello, How are you?";
                string text2 = "I am fine!";

                var ids1 = bertTokenizer.EncodeToIds(text1, addSpecialTokens: false);
                Assert.Equal([8, 6, 10, 11, 12, 7], ids1);

                var ids2 = bertTokenizer.EncodeToIds(text2, addSpecialTokens: false);
                Assert.Equal([13, 14, 15, 5], ids2);

                Assert.Equal(
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    bertTokenizer.CreateTokenTypeIdsFromSequences(ids1));

                Span<int> ids1Span = stackalloc int[1];
                OperationStatus status = bertTokenizer.CreateTokenTypeIdsFromSequences(ids1, ids1Span, out int written);
                Assert.Equal(OperationStatus.DestinationTooSmall, status);
                Assert.Equal(0, written);

                ids1Span = stackalloc int[ids1.Count + 2];
                status = bertTokenizer.CreateTokenTypeIdsFromSequences(ids1, ids1Span, out written);
                Assert.Equal(OperationStatus.Done, status);
                Assert.Equal(ids1.Count + 2, written);
                Assert.Equal(new int[] { 0, 0, 0, 0, 0, 0, 0, 0 }, ids1Span.ToArray());

                Assert.Equal(
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                    bertTokenizer.CreateTokenTypeIdsFromSequences(ids1, ids2));

                ids1Span = stackalloc int[1];
                status = bertTokenizer.CreateTokenTypeIdsFromSequences(ids1, ids1Span, out written, ids2);
                Assert.Equal(OperationStatus.DestinationTooSmall, status);
                Assert.Equal(0, written);

                ids1Span = stackalloc int[ids1.Count + ids2.Count + 3];
                status = bertTokenizer.CreateTokenTypeIdsFromSequences(ids1, ids1Span, out written, ids2);
                Assert.Equal(OperationStatus.Done, status);
                Assert.Equal(ids1Span.Length, written);
                Assert.Equal(new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 }, ids1Span.ToArray());
            }
            finally
            {
                File.Delete(vocabFile);
            }
        }
    }
}