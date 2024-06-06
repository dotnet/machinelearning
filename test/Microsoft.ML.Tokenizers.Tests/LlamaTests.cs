// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text.Json;
using System.Linq;
using System.IO;
using System.Threading.Tasks;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class LlamaTests
    {
        private static readonly HttpClient _httpClient = new HttpClient() { Timeout = TimeSpan.FromMinutes(5) };
        private static Tokenizer _llamaTokenizer = CreateLlamaTokenizer();
        private static Tokenizer _llamaMistralTokenizer = CreateLMistralTokenizer();

        private static Tokenizer CreateLlamaTokenizer()
        {
            // @"https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/resolve/main/tokenizer.model?download=true";
            // @"https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model";
            using Stream remoteStream = File.OpenRead(Path.Combine(@"Llama", "tokenizer.model"));
            return Tokenizer.CreateLlama(remoteStream);
        }

        private static Tokenizer CreateLMistralTokenizer()
        {
            // @"https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.model?download=true";
            using Stream remoteStream = File.OpenRead(Path.Combine(@"Mistral", "tokenizer.model"));
            return Tokenizer.CreateLlama(remoteStream);
        }

        public static IEnumerable<object[]> LlamaTestData()
        {
            // input text, ids, tokens, offsets
            yield return new object[]
            {
                _llamaTokenizer,
                "Hello, world!",
                new int[] { 1, 15043, 29892, 3186, 29991 },
                new string[] { "<s>", "▁Hello", ",", "▁world", "!" },
                new (int Index, int Length)[] { (0, 0), (0, 6), (6, 1), (7, 6), (13, 1) }
            };

            yield return new object[]
            {
                _llamaMistralTokenizer,
                "Hello, world!",
                new int[] { 1, 22557, 28725, 1526, 28808 },
                new string[] { "<s>", "▁Hello", ",", "▁world", "!" },
                new (int Index, int Length)[] { (0, 0), (0, 6), (6, 1), (7, 6), (13, 1) }
            };

            yield return new object[]
            {
                _llamaTokenizer,
                "",
                new int[0],
                new string[0],
                new (int Index, int Length)[0]
            };

            yield return new object[]
            {
                _llamaMistralTokenizer,
                "",
                new int[0],
                new string[0],
                new (int Index, int Length)[0]
            };

            yield return new object[]
            {
                _llamaTokenizer,
                @"The sun dipped below the horizon, casting a warm golden hue across the tranquil meadow. Birds fluttered from "        +
                "tree to tree, their melodic songs filling the air. A gentle breeze rustled the leaves, carrying with it the scent of " +
                "blooming flowers. In the distance, the silhouette of a lone figure stood atop a hill, gazing out at the vast expanse " +
                "before them. It was a moment frozen in time, where nature and solitude merged into something magical.",
                new int[] { 1, 450, 6575, 652, 2986, 2400, 278, 28205, 29892, 23013, 263, 14294, 22843, 298, 434, 4822, 278, 22024,
                            339, 309, 592, 6986, 29889, 17777, 29879, 20287, 287, 515, 5447, 304, 5447, 29892, 1009, 9232, 397, 293,
                            12516, 27523, 278, 4799, 29889, 319, 9914, 289, 929, 911, 21580, 839, 278, 11308, 29892, 19436, 411,
                            372, 278, 885, 296, 310, 6668, 28826, 18281, 29889, 512, 278, 5418, 29892, 278, 4047, 10774, 2353, 310,
                            263, 301, 650, 4377, 8389, 472, 459, 263, 17306, 29892, 12642, 292, 714, 472, 278, 13426, 1518, 12350,
                            1434, 963, 29889, 739, 471, 263, 3256, 14671, 2256, 297, 931, 29892, 988, 5469, 322, 899, 4279, 19412,
                            964, 1554, 2320, 936, 29889},
                new string[] { "<s>", "▁The", "▁sun", "▁di", "pped", "▁below", "▁the", "▁horizon", ",", "▁casting", "▁a", "▁warm",
                            "▁golden", "▁h", "ue", "▁across", "▁the", "▁tran", "qu", "il", "▁me", "adow", ".", "▁Bird", "s", "▁flutter",
                            "ed", "▁from", "▁tree", "▁to", "▁tree", ",", "▁their", "▁mel", "od", "ic", "▁songs", "▁filling", "▁the", "▁air",
                            ".", "▁A", "▁gentle", "▁b", "ree", "ze", "▁rust", "led", "▁the", "▁leaves", ",", "▁carrying", "▁with", "▁it",
                            "▁the", "▁sc", "ent", "▁of", "▁blo", "oming", "▁flowers", ".", "▁In", "▁the", "▁distance", ",", "▁the", "▁sil",
                            "hou", "ette", "▁of", "▁a", "▁l", "one", "▁figure", "▁stood", "▁at", "op", "▁a", "▁hill", ",", "▁gaz", "ing",
                            "▁out", "▁at", "▁the", "▁vast", "▁exp", "anse", "▁before", "▁them", ".", "▁It", "▁was", "▁a", "▁moment", "▁fro",
                            "zen", "▁in", "▁time", ",", "▁where", "▁nature", "▁and", "▁sol", "itude", "▁merged", "▁into", "▁something", "▁mag",
                            "ical", "." },
                new (int Index, int Length)[] { (0, 0), (0, 4), (4, 4), (8, 3), (11, 4), (15, 6), (21, 4), (25, 8), (33, 1), (34, 8), (42, 2),
                            (44, 5), (49, 7), (56, 2), (58, 2), (60, 7), (67, 4), (71, 5), (76, 2), (78, 2), (80, 3), (83, 4),
                            (87, 1), (88, 5), (93, 1), (94, 8), (102, 2), (104, 5), (109, 5), (114, 3), (117, 5), (122, 1),
                            (123, 6), (129, 4), (133, 2), (135, 2), (137, 6), (143, 8), (151, 4), (155, 4), (159, 1), (160, 2),
                            (162, 7), (169, 2), (171, 3), (174, 2), (176, 5), (181, 3), (184, 4), (188, 7), (195, 1), (196, 9),
                            (205, 5), (210, 3), (213, 4), (217, 3), (220, 3), (223, 3), (226, 4), (230, 5), (235, 8), (243, 1),
                            (244, 3), (247, 4), (251, 9), (260, 1), (261, 4), (265, 4), (269, 3), (272, 4), (276, 3), (279, 2),
                            (281, 2), (283, 3), (286, 7), (293, 6), (299, 3), (302, 2), (304, 2), (306, 5), (311, 1), (312, 4),
                            (316, 3), (319, 4), (323, 3), (326, 4), (330, 5), (335, 4), (339, 4), (343, 7), (350, 5), (355, 1),
                            (356, 3), (359, 4), (363, 2), (365, 7), (372, 4), (376, 3), (379, 3), (382, 5), (387, 1), (388, 6),
                            (394, 7), (401, 4), (405, 4), (409, 5), (414, 7), (421, 5), (426, 10), (436, 4), (440, 4), (444, 1) }
            };

            yield return new object[]
            {
                _llamaMistralTokenizer,
                @"The sun dipped below the horizon, casting a warm golden hue across the tranquil meadow. Birds fluttered from "        +
                "tree to tree, their melodic songs filling the air. A gentle breeze rustled the leaves, carrying with it the scent of " +
                "blooming flowers. In the distance, the silhouette of a lone figure stood atop a hill, gazing out at the vast expanse " +
                "before them. It was a moment frozen in time, where nature and solitude merged into something magical.",
                new int[] { 1, 415, 4376, 281, 5885, 3624, 272, 18259, 28725, 24668, 264, 6100, 13863, 295, 441, 2673, 272, 467, 20668, 309,
                            528, 5547, 28723, 18213, 28713, 972, 329, 8308, 477, 4718, 298, 4718, 28725, 652, 27043, 294, 9184, 15990, 272,
                            2423, 28723, 330, 10434, 24284, 14912, 1006, 272, 8049, 28725, 10839, 395, 378, 272, 21535, 302, 3449, 17846,
                            11888, 28723, 560, 272, 5328, 28725, 272, 2958, 11331, 3186, 302, 264, 305, 538, 5248, 4857, 438, 410, 264,
                            12254, 28725, 14961, 288, 575, 438, 272, 9555, 2365, 14788, 1159, 706, 28723, 661, 403, 264, 2470, 15199, 297,
                            727, 28725, 970, 4735, 304, 2128, 4484, 22750, 778, 1545, 20927, 28723 },
                new string[] { "<s>", "▁The", "▁sun", "▁d", "ipped", "▁below", "▁the", "▁horizon", ",", "▁casting", "▁a", "▁warm", "▁golden",
                            "▁h", "ue", "▁across", "▁the", "▁tr", "anqu", "il", "▁me", "adow", ".", "▁Bird", "s", "▁fl", "ut", "tered", "▁from",
                            "▁tree", "▁to", "▁tree", ",", "▁their", "▁melod", "ic", "▁songs", "▁filling", "▁the", "▁air", ".", "▁A", "▁gentle",
                            "▁breeze", "▁rust", "led", "▁the", "▁leaves", ",", "▁carrying", "▁with", "▁it", "▁the", "▁scent", "▁of", "▁blo",
                            "oming", "▁flowers", ".", "▁In", "▁the", "▁distance", ",", "▁the", "▁sil", "hou", "ette", "▁of", "▁a", "▁l", "one",
                            "▁figure", "▁stood", "▁at", "op", "▁a", "▁hill", ",", "▁gaz", "ing", "▁out", "▁at", "▁the", "▁vast", "▁exp", "anse",
                            "▁before", "▁them", ".", "▁It", "▁was", "▁a", "▁moment", "▁frozen", "▁in", "▁time", ",", "▁where", "▁nature", "▁and",
                            "▁sol", "itude", "▁merged", "▁into", "▁something", "▁magical", "." },
                new (int Index, int Length)[] { (0, 0), (0, 4), (4, 4), (8, 2), (10, 5), (15, 6), (21, 4), (25, 8), (33, 1), (34, 8), (42, 2), (44, 5),
                            (49, 7), (56, 2), (58, 2), (60, 7), (67, 4), (71, 3), (74, 4), (78, 2), (80, 3), (83, 4), (87, 1), (88, 5), (93, 1), (94, 3),
                            (97, 2), (99, 5), (104, 5), (109, 5), (114, 3), (117, 5), (122, 1), (123, 6), (129, 6), (135, 2), (137, 6), (143, 8), (151, 4),
                            (155, 4), (159, 1), (160, 2), (162, 7), (169, 7), (176, 5), (181, 3), (184, 4), (188, 7), (195, 1), (196, 9), (205, 5), (210, 3),
                            (213, 4), (217, 6), (223, 3), (226, 4), (230, 5), (235, 8), (243, 1), (244, 3), (247, 4), (251, 9), (260, 1), (261, 4), (265, 4),
                            (269, 3), (272, 4), (276, 3), (279, 2), (281, 2), (283, 3), (286, 7), (293, 6), (299, 3), (302, 2), (304, 2), (306, 5), (311, 1),
                            (312, 4), (316, 3), (319, 4), (323, 3), (326, 4), (330, 5), (335, 4), (339, 4), (343, 7), (350, 5), (355, 1), (356, 3), (359, 4),
                            (363, 2), (365, 7), (372, 7), (379, 3), (382, 5), (387, 1), (388, 6), (394, 7), (401, 4), (405, 4), (409, 5), (414, 7), (421, 5),
                            (426, 10), (436, 8), (444, 1) }
            };

            // byte encoding with ASCII range
            yield return new object[]
            {
                _llamaTokenizer,
                "\nHello\n\rWorld!\n",
                new int[] { 1, 29871, 13, 10994, 13, 30004, 14058, 29991, 13 },
                new string[] { "<s>", "▁", "<0x0A>", "Hello", "<0x0A>", "\r", "World", "!", "<0x0A>" },
                new (int Index, int Length)[] { (0, 0), (0, 1), (1, 1), (2, 5), (7, 1), (8, 1), (9, 5), (14, 1), (15, 1) }
            };

            // byte encoding with ASCII range
            yield return new object[]
            {
                _llamaMistralTokenizer,
                "\nHello\n\rWorld!\n",
                new int[] { 1, 28705, 13, 16230, 13, 28801, 11978, 28808, 13 },
                new string[] { "<s>", "▁", "<0x0A>", "Hello", "<0x0A>", "\r", "World", "!", "<0x0A>" },
                new (int Index, int Length)[] { (0, 0), (0, 1), (1, 1), (2, 5), (7, 1), (8, 1), (9, 5), (14, 1), (15, 1) }
            };

            // byte encoding with unknown tokens
            yield return new object[]
            {
                _llamaTokenizer,
                "This is 👍, an emoji.", // 👍is U+0001F44D (\ud83d\ude4f) surrogate pair which will be encode to the utf-8 bytes 0xF0, 0x9F, 0x91, 0x8D.
                new int[] { 1, 910, 338, 29871, 243, 162, 148, 144, 29892, 385, 953, 29877, 2397, 29889 },
                new string[] { "<s>", "▁This", "▁is", "▁", "<0xF0>", "<0x9F>", "<0x91>", "<0x8D>", ",", "▁an", "▁em", "o", "ji", "." },
                new (int Index, int Length)[] { (0, 0), (0, 5), (5, 3), (8, 1), (9, 2), (9, 0), (9, 0), (9, 0), (11, 1), (12, 3), (15, 3), (18, 1), (19, 2), (21, 1) }
            };

            yield return new object[]
            {
                _llamaMistralTokenizer,
                "This is 👍, an emoji.", // 👍is U+0001F44D (\ud83d\ude4f) surrogate pair Mistral tokenizer include this surrogate in its vocabulary.
                new int[] { 1, 851, 349, 28705, 30195, 28725, 396, 877, 27813, 28723 },
                new string[] { "<s>", "▁This", "▁is", "▁", "👍", ",", "▁an", "▁em", "oji", "." },
                new (int Index, int Length)[] { (0, 0), (0, 5), (5, 3), (8, 1), (9, 2), (11, 1), (12, 3), (15, 3), (18, 3), (21, 1) }
            };
        }

        [Theory]
        [MemberData(nameof(LlamaTestData))]
        public void TestLlamaTokenizer(Tokenizer llamaTokenizer, string input, int[] ids, string[] tokens, (int Index, int Length)[] offsets)
        {
            SentencePieceBpe? bpe = llamaTokenizer as SentencePieceBpe;
            Assert.NotNull(bpe);

            IReadOnlyList<Token> result = llamaTokenizer.Encode(input, out _);
            Assert.Equal(ids, result.Select(t => t.Id).ToArray());
            Assert.Equal(tokens, result.Select(t => t.Value).ToArray());
            Assert.Equal(offsets, result.Select(t => t.Offset).ToArray());
            Assert.Equal(input, llamaTokenizer.Decode(ids));
            Assert.Equal(ids, llamaTokenizer.EncodeToIds(input));
            Assert.Equal(ids.Length, llamaTokenizer.CountTokens(input));

            for (int i = 0; i < tokens.Length; i++)
            {
                Assert.Equal(tokens[i], bpe!.MapIdToToken(ids[i]));
                Assert.Equal(ids[i], bpe!.MapTokenToId(tokens[i].AsSpan()));
                Assert.Equal(ids[i], bpe!.Vocab[tokens[i]]);
            }

            Assert.NotNull(llamaTokenizer.Normalizer);
            string normalizedInput = llamaTokenizer.Normalizer!.Normalize(input);

            bool isEmptyInput = string.IsNullOrEmpty(input);

            IReadOnlyList<Token> bpeTokens = bpe.Encode(normalizedInput.AsSpan(), out _, addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false);
            Assert.Equal(ids.Skip(1), bpeTokens.Select(token => token.Id));
            Assert.Equal(tokens.Skip(1), bpeTokens.Select(token => token.Value));
            Assert.Equal(input, llamaTokenizer.Decode(bpeTokens.Select(token => token.Id)));
            IReadOnlyList<int> encodedIds = bpe.EncodeToIds(normalizedInput.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false);
            Assert.Equal(ids.Skip(1), encodedIds);
            Assert.Equal(isEmptyInput ? 0 : ids.Length - 1, bpe.CountTokens(normalizedInput.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false));

            bpeTokens = bpe.Encode(normalizedInput.AsSpan(), out _, addBeginningOfSentence: false, addEndOfSentence: true, considerNormalization: false);
            Assert.Equal(isEmptyInput ? Array.Empty<int>() : ids.Skip(1).Concat(new[] { bpe.EndOfSentenceId }), bpeTokens.Select(token => token.Id));
            Assert.Equal(isEmptyInput ? Array.Empty<string>() : tokens.Skip(1).Concat(new[] { bpe.EndOfSentenceToken }), bpeTokens.Select(token => token.Value));
            Assert.Equal(input, llamaTokenizer.Decode(bpeTokens.Select(token => token.Id)));
            encodedIds = bpe.EncodeToIds(normalizedInput.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: true, considerNormalization: false);
            Assert.Equal(isEmptyInput ? Array.Empty<int>() : ids.Skip(1).Concat(new[] { bpe.EndOfSentenceId }), encodedIds);
            Assert.Equal(isEmptyInput ? 0 : ids.Length, bpe.CountTokens(normalizedInput.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: true, considerNormalization: false));

            bpeTokens = bpe.Encode(normalizedInput.AsSpan(), out _, addBeginningOfSentence: true, addEndOfSentence: true, considerNormalization: false);
            Assert.Equal(isEmptyInput ? Array.Empty<int>() : ids.Concat(new[] { bpe.EndOfSentenceId }), bpeTokens.Select(token => token.Id));
            Assert.Equal(isEmptyInput ? Array.Empty<string>() : tokens.Concat(new[] { bpe.EndOfSentenceToken }), bpeTokens.Select(token => token.Value));
            Assert.Equal(input, llamaTokenizer.Decode(bpeTokens.Select(token => token.Id)));
            encodedIds = bpe.EncodeToIds(normalizedInput.AsSpan(), addBeginningOfSentence: true, addEndOfSentence: true, considerNormalization: false);
            Assert.Equal(isEmptyInput ? Array.Empty<int>() : ids.Concat(new[] { bpe.EndOfSentenceId }), encodedIds);
            Assert.Equal(isEmptyInput ? 0 : ids.Length + 1, bpe.CountTokens(normalizedInput.AsSpan(), addBeginningOfSentence: true, addEndOfSentence: true, considerNormalization: false));
        }

        public static IEnumerable<object[]> LlamaTokenizersListData()
        {
            yield return new object[] { _llamaTokenizer };
            yield return new object[] { _llamaMistralTokenizer };
        }

        [Theory]
        [MemberData(nameof(LlamaTokenizersListData))]
        public void TestLlamaTokenizerWithEmptyInput(Tokenizer llamaTokenizer)
        {
            Assert.Equal([], llamaTokenizer.Encode((string)null!, out _));
            Assert.Equal([], llamaTokenizer.Encode(Span<char>.Empty, out _));

            Assert.Equal([], llamaTokenizer.EncodeToIds((string)null!));
            Assert.Equal([], llamaTokenizer.EncodeToIds(Span<char>.Empty));

            Assert.Equal(0, llamaTokenizer.CountTokens((string)null!));
            Assert.Equal(0, llamaTokenizer.CountTokens(Span<char>.Empty));

            Assert.Throws<ArgumentNullException>(() => llamaTokenizer.Decode(null!));
        }

        [Theory]
        [MemberData(nameof(LlamaTokenizersListData))]
        public void TestLlamaTokenizerProperties(Tokenizer llamaTokenizer)
        {
            SentencePieceBpe? bpe = llamaTokenizer as SentencePieceBpe;
            Assert.NotNull(bpe);
            Assert.NotNull(llamaTokenizer.Normalizer);

            Assert.Equal("▁Hello,▁World!", llamaTokenizer.Normalizer.Normalize("Hello, World!"));

            Assert.True(bpe.Vocab.Count > 0);
            Assert.True(bpe.Vocab.TryGetValue("▁", out _));

            Assert.Equal(0, bpe.UnknownId);
            Assert.Equal("<unk>", bpe.UnknownToken);
            Assert.Equal(1, bpe.BeginningOfSentenceId);
            Assert.Equal("<s>", bpe.BeginningOfSentenceToken);
            Assert.Equal(2, bpe.EndOfSentenceId);
            Assert.Equal("</s>", bpe.EndOfSentenceToken);

            Assert.Equal(bpe.Vocab["▁"], bpe.MapTokenToId("▁".AsSpan()));
            Assert.Equal("▁", bpe.MapIdToToken(bpe.Vocab["▁"]));

            Assert.True(bpe.ByteFallback);
            Assert.True(bpe.AddDummyPrefix);
            Assert.True(bpe.EscapeWhiteSpaces);
            Assert.False(bpe.TreatWhitespaceAsSuffix);

            TokenizerTests.TestTokenLimits(llamaTokenizer);
        }

        [Fact]
        public void TestSentencePieceNormalizer()
        {
            SentencePieceNormalizer normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: false, addDummyPrefix: false, escapeWhiteSpaces: false, treatWhitespaceAsSuffix: false);
            Assert.Equal("Hello,      World!", normalizer.Normalize("Hello,      World!"));
            Assert.Equal("Hello,      World!", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: true, addDummyPrefix: false, escapeWhiteSpaces: false, treatWhitespaceAsSuffix: false);
            Assert.Equal("Hello, World!", normalizer.Normalize("Hello,      World!"));
            Assert.Equal("Hello, World!", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: true, addDummyPrefix: true, escapeWhiteSpaces: false, treatWhitespaceAsSuffix: false);
            Assert.Equal(" Hello, World!", normalizer.Normalize("Hello,      World!"));
            Assert.Equal(" Hello, World!", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: true, addDummyPrefix: true, escapeWhiteSpaces: true, treatWhitespaceAsSuffix: false);
            Assert.Equal("▁Hello,▁World!", normalizer.Normalize("Hello,      World!"));
            Assert.Equal("▁Hello,▁World!", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: false, addDummyPrefix: true, escapeWhiteSpaces: true, treatWhitespaceAsSuffix: false);
            Assert.Equal("▁Hello,▁▁▁▁▁▁World!", normalizer.Normalize("Hello,      World!"));
            Assert.Equal("▁Hello,▁▁▁▁▁▁World!", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: true, addDummyPrefix: true, escapeWhiteSpaces: true, treatWhitespaceAsSuffix: true);
            Assert.Equal("Hello,▁World!▁", normalizer.Normalize("Hello,      World!"));
            Assert.Equal("Hello,▁World!▁", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: true, addDummyPrefix: false, escapeWhiteSpaces: true, treatWhitespaceAsSuffix: true);
            Assert.Equal("Hello,▁World!", normalizer.Normalize("Hello,      World!"));
            Assert.Equal("Hello,▁World!", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: false, addDummyPrefix: true, escapeWhiteSpaces: true, treatWhitespaceAsSuffix: true);
            Assert.Equal("Hello,▁▁▁▁▁▁World!▁", normalizer.Normalize("Hello,      World!"));
            Assert.Equal("Hello,▁▁▁▁▁▁World!▁", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: false, addDummyPrefix: true, escapeWhiteSpaces: false, treatWhitespaceAsSuffix: true);
            Assert.Equal("Hello,      World! ", normalizer.Normalize("Hello,      World!"));
            Assert.Equal("Hello,      World! ", normalizer.Normalize("Hello,      World!".AsSpan()));
        }

        public static IEnumerable<object?[]> TokenizerTestData
        {
            get
            {
                // string to tokenize, produced tokens, the token offsets
                yield return new object?[]
                {
                    "the brown fox jumped over the lazy dog!",
                    "▁the▁brown▁fox▁jumped▁over▁the▁lazy▁dog!",
                    new string[] { "<s>", "▁the", "▁brown", "▁fo", "x", "▁jump", "ed", "▁over", "▁the", "▁lazy", "▁dog", "!" },
                    new (int Index, int Length)[] { (0, 0), (0, 4), (4, 6), (10, 3), (13, 1), (14, 5), (19, 2), (21, 5), (26, 4), (30, 5), (35, 4), (39, 1) },
                    new int[] { 1, 278, 17354, 1701, 29916, 12500, 287, 975, 278, 17366, 11203, 29991 }
                };
                yield return new object?[]
                {
                    "he traveled to Egypt during the summer, the weather was hot and ammunition." ,
                    "▁he▁traveled▁to▁Egypt▁during▁the▁summer,▁the▁weather▁was▁hot▁and▁ammunition." ,
                    new string[] { "<s>", "▁he", "▁tra", "ve", "led", "▁to", "▁Egypt", "▁during", "▁the", "▁summer", ",", "▁the", "▁weather", "▁was", "▁hot", "▁and", "▁am", "mun", "ition", "." },
                    new (int Index, int Length)[] { (0, 0), (0, 3), (3, 4), (7, 2), (9, 3), (12, 3), (15, 6), (21, 7), (28, 4), (32, 7), (39, 1), (40, 4), (44, 8), (52, 4), (56, 4), (60, 4), (64, 3), (67, 3), (70, 5), (75, 1) },
                    new int[] { 1, 540, 1020, 345, 839, 304, 12892, 2645, 278, 11801, 29892, 278, 14826, 471, 7375, 322, 626, 24579, 654, 29889 }
                };
                yield return new object?[]
                {
                    "She played many games and she felt exhausted afterward",
                    "▁She▁played▁many▁games▁and▁she▁felt▁exhausted▁afterward",
                    new string[] { "<s>", "▁She", "▁played", "▁many", "▁games", "▁and", "▁she", "▁felt", "▁exha", "usted", "▁after", "ward" },
                    new (int Index, int Length)[] { (0, 0), (0, 4), (4, 7), (11, 5), (16, 6), (22, 4), (26, 4), (30, 5), (35, 5), (40, 5), (45, 6), (51, 4) },
                    new int[] { 1, 2296, 5318, 1784, 8090, 322, 1183, 7091, 18782, 16656, 1156, 1328 }
                };
                yield return new object?[]
                {
                    "Hello, y'all! How are you 😁 ?",
                    "▁Hello,▁y'all!▁How▁are▁you▁😁▁?",
                    new string[] { "<s>", "▁Hello", ",", "▁y", "'", "all", "!", "▁How", "▁are", "▁you", "▁", "<0xF0>", "<0x9F>", "<0x98>", "<0x81>", "▁?" },
                    new (int Index, int Length)[] { (0, 0), (0, 6), (6, 1), (7, 2), (9, 1), (10, 3), (13, 1), (14, 4), (18, 4), (22, 4), (26, 1), (27, 2), (27, 0), (27, 0), (27, 0), (29, 2) },
                    new int[] { 1, 15043, 29892, 343, 29915, 497, 29991, 1128, 526, 366, 29871, 243, 162, 155, 132, 1577 }
                };
            }
        }

        [Theory]
        [MemberData(nameof(TokenizerTestData))]
        public void TestTokenizerEncoding(string text, string normalizedText, string[] expectedTokens, (int Index, int Length)[] expectedOffsets, int[] expectedIds)
        {
            Tokenizer tokenizer = _llamaTokenizer;

            Assert.NotNull(tokenizer.Normalizer);
            Assert.Null(tokenizer.PreTokenizer);

            IReadOnlyList<Token> encoding = tokenizer.Encode(text, out _);
            IReadOnlyList<Token> encoding1 = tokenizer.Encode(text.AsSpan(), out _);

            Assert.Equal(expectedTokens, encoding.Select(t => t.Value).ToArray());
            Assert.Equal(expectedOffsets, encoding.Select(t => t.Offset).ToArray());
            Assert.Equal(expectedIds, encoding.Select(t => t.Id).ToArray());

            Assert.Equal(expectedTokens, encoding1.Select(t => t.Value).ToArray());
            Assert.Equal(expectedOffsets, encoding1.Select(t => t.Offset).ToArray());
            Assert.Equal(expectedIds, encoding1.Select(t => t.Id).ToArray());

            SentencePieceBpe sentencePieceBpe = (tokenizer as SentencePieceBpe)!;
            foreach (bool considerNormalization in new[] { true, false })
                foreach (bool addBeginningOfSentence in new[] { true, false })
                    foreach (bool addEndOfSentence in new[] { true, false })
                    {
                        encoding = sentencePieceBpe.Encode(
                                        considerNormalization ? text : normalizedText,
                                        out _,
                                        addBeginningOfSentence: addBeginningOfSentence,
                                        addEndOfSentence: addEndOfSentence,
                                        considerPreTokenization: false,
                                        considerNormalization: considerNormalization);

                        encoding1 = sentencePieceBpe.Encode(
                                        considerNormalization ? text.AsSpan() : normalizedText.AsSpan(),
                                        out _,
                                        addBeginningOfSentence: addBeginningOfSentence,
                                        addEndOfSentence: addEndOfSentence,
                                        considerPreTokenization: false,
                                        considerNormalization: considerNormalization);

                        string[] expectedTokens1 = addBeginningOfSentence ? expectedTokens : expectedTokens.Skip(1).ToArray();
                        expectedTokens1 = addEndOfSentence ? expectedTokens1.Concat(new[] { sentencePieceBpe.EndOfSentenceToken }).ToArray() : expectedTokens1;

                        (int Index, int Length)[] expectedOffsets1 = addBeginningOfSentence ? expectedOffsets : expectedOffsets.Skip(1).ToArray();
                        expectedOffsets1 = addEndOfSentence ? expectedOffsets1.Concat(new[] { (normalizedText.Length, 0) }).ToArray() : expectedOffsets1;

                        int[] expectedIds1 = addBeginningOfSentence ? expectedIds : expectedIds.Skip(1).ToArray();
                        expectedIds1 = addEndOfSentence ? expectedIds1.Concat(new[] { sentencePieceBpe.EndOfSentenceId }).ToArray() : expectedIds1;

                        Assert.Equal(expectedTokens1, encoding.Select(t => t.Value).ToArray());
                        Assert.Equal(expectedOffsets1, encoding.Select(t => t.Offset).ToArray());
                        Assert.Equal(expectedIds1, encoding.Select(t => t.Id).ToArray());
                    }
        }

        [Theory]
        [MemberData(nameof(TokenizerTestData))]
        public void TestTokenizerEncodingToIds(string text, string normalizedText, string[] expectedTokens, (int Index, int Length)[] expectedOffsets, int[] expectedIds)
        {
            Tokenizer tokenizer = _llamaTokenizer;

            Assert.NotNull(expectedTokens);
            Assert.NotNull(expectedOffsets);

            Assert.Equal(expectedIds, tokenizer.EncodeToIds(text));
            Assert.Equal(expectedIds, tokenizer.EncodeToIds(text.AsSpan()));
            Assert.Equal(expectedIds, tokenizer.EncodeToIds(text, expectedIds.Length, out string? normalizedString, out int length));
            Assert.Equal(normalizedText, normalizedString);
            Assert.Equal(normalizedText.Length, length);
            Assert.Equal(expectedIds, tokenizer.EncodeToIds(text.AsSpan(), expectedIds.Length, out normalizedString, out length));
            Assert.Equal(normalizedText, normalizedString);
            Assert.Equal(normalizedText.Length, length);

            SentencePieceBpe sentencePieceBpe = (tokenizer as SentencePieceBpe)!;
            foreach (bool considerNormalization in new[] { true, false })
                foreach (bool addBeginningOfSentence in new[] { true, false })
                    foreach (bool addEndOfSentence in new[] { true, false })
                    {
                        // (string text, bool addBeginningOfSentence, bool addEndOfSentence, int maxTokenCount, out string? normalizedString, out int textLength, bool considerPreTokenization = true, bool considerNormalization = true)

                        int[] expectedIds1 = addBeginningOfSentence ? expectedIds : expectedIds.Skip(1).ToArray();
                        expectedIds1 = addEndOfSentence ? expectedIds1.Concat(new[] { sentencePieceBpe.EndOfSentenceId }).ToArray() : expectedIds1;

                        Assert.Equal(expectedIds1, sentencePieceBpe.EncodeToIds(
                                                        considerNormalization ? text : normalizedText,
                                                        addBeginningOfSentence: addBeginningOfSentence,
                                                        addEndOfSentence: addEndOfSentence,
                                                        expectedIds1.Length,
                                                        out normalizedString,
                                                        out length,
                                                        considerNormalization: considerNormalization));

                        Assert.Equal(expectedIds1, sentencePieceBpe.EncodeToIds(
                                                        considerNormalization ? text.AsSpan() : normalizedText.AsSpan(),
                                                        addBeginningOfSentence: addBeginningOfSentence,
                                                        addEndOfSentence: addEndOfSentence,
                                                        expectedIds1.Length,
                                                        out normalizedString,
                                                        out length,
                                                        considerNormalization: considerNormalization));

                        Assert.Equal(considerNormalization ? normalizedText : null, normalizedString);
                        Assert.Equal(normalizedText.Length, length);

                        Assert.Equal(expectedIds1.Take(expectedIds1.Length - 6), sentencePieceBpe.EncodeToIds(
                                                                                    considerNormalization ? text : normalizedText,
                                                                                    addBeginningOfSentence: addBeginningOfSentence,
                                                                                    addEndOfSentence: addEndOfSentence,
                                                                                    expectedIds1.Length - 6,
                                                                                    out normalizedString,
                                                                                    out length,
                                                                                    considerNormalization: considerNormalization));
                        Assert.Equal(considerNormalization ? normalizedText : null, normalizedString);

                        (int Index, int Length)[] expectedOffsets1 = addBeginningOfSentence ? expectedOffsets.Take(expectedIds1.Length - 6).ToArray() : expectedOffsets.Skip(1).Take(expectedIds1.Length - 6).ToArray();

                        int expectedLength = expectedOffsets1[expectedOffsets1.Length - 1].Index + expectedOffsets1[expectedOffsets1.Length - 1].Length;
                        Assert.Equal(expectedLength, length);

                        Assert.Equal(expectedIds1.Take(expectedIds1.Length - 6), sentencePieceBpe.EncodeToIds(
                                                                                    considerNormalization ? text.AsSpan() : normalizedText.AsSpan(),
                                                                                    addBeginningOfSentence: addBeginningOfSentence,
                                                                                    addEndOfSentence: addEndOfSentence,
                                                                                    expectedIds1.Length - 6,
                                                                                    out normalizedString,
                                                                                    out length,
                                                                                    considerNormalization: considerNormalization));
                        Assert.Equal(expectedLength, length);
                    }
        }


        [Theory]
        [MemberData(nameof(TokenizerTestData))]
        public void TestTokenizerCountTokens(string text, string normalizedText, string[] expectedTokens, (int Index, int Length)[] expectedOffsets, int[] expectedIds)
        {
            Tokenizer tokenizer = _llamaTokenizer;

            Assert.NotNull(expectedTokens);

            Assert.Equal(expectedIds.Length, tokenizer.CountTokens(text));
            Assert.Equal(expectedIds.Length, tokenizer.CountTokens(text.AsSpan()));

            Assert.Equal(expectedOffsets[expectedOffsets.Length - 7].Index + expectedOffsets[expectedOffsets.Length - 7].Length, tokenizer.IndexOfTokenCount(text, expectedIds.Length - 6, out string? normalizedString, out int tokenCount));
            Assert.Equal(normalizedText, normalizedString);
            Assert.Equal(expectedIds.Length - 6, tokenCount);
            Assert.Equal(expectedOffsets[expectedOffsets.Length - 7].Index + expectedOffsets[expectedOffsets.Length - 7].Length, tokenizer.IndexOfTokenCount(text.AsSpan(), expectedIds.Length - 6, out normalizedString, out tokenCount));
            Assert.Equal(normalizedText, normalizedString);
            Assert.Equal(expectedIds.Length - 6, tokenCount);

            Assert.Equal(expectedOffsets[expectedOffsets.Length - 7].Index, tokenizer.LastIndexOfTokenCount(text, 7, out normalizedString, out tokenCount));
            Assert.Equal(normalizedText, normalizedString);
            Assert.Equal(7, tokenCount);
            Assert.Equal(expectedOffsets[expectedOffsets.Length - 7].Index, tokenizer.LastIndexOfTokenCount(text.AsSpan(), 7, out normalizedString, out tokenCount));
            Assert.Equal(normalizedText, normalizedString);
            Assert.Equal(7, tokenCount);
        }
    }
}
