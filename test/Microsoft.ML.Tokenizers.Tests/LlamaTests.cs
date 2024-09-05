// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Tokenizers;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class LlamaTests
    {
        private static readonly HttpClient _httpClient = new HttpClient() { Timeout = TimeSpan.FromMinutes(5) };
        private static Tokenizer _llamaTokenizer = CreateLlamaTokenizer();
        private static Tokenizer _llamaMistralTokenizer = CreateLMistralTokenizer();
        private static Tokenizer _llamaPhi3Tokenizer = CreateLPhi3Tokenizer();
        private static Tokenizer _llamaPhi3TokenizerWithTreatSpaceSuffix = CreateLPhi3Tokenizer(treatWhitespaceAsSuffix: true);
        internal const string DummyPrefix = "\u2581"; // '▁' (LOWER ONE EIGHT BLOCK)

        private static Tokenizer CreateLlamaTokenizer()
        {
            // @"https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/resolve/main/tokenizer.model?download=true";
            // @"https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model";
            using Stream remoteStream = File.OpenRead(Path.Combine(@"Llama", "tokenizer.model"));
            return LlamaTokenizer.Create(remoteStream);
        }

        private static Tokenizer CreateLMistralTokenizer()
        {
            // @"https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.model?download=true";
            using Stream remoteStream = File.OpenRead(Path.Combine(@"Mistral", "tokenizer.model"));
            return LlamaTokenizer.Create(remoteStream);
        }

        private static Tokenizer CreateLPhi3Tokenizer(bool treatWhitespaceAsSuffix = false)
        {
            // Phi3 is using the same tokenizer.model used by Llama. https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/tree/main
            using Stream remoteStream = File.OpenRead(Path.Combine(@"Llama", "tokenizer.model"));
            LlamaTokenizer tokenizer = LlamaTokenizer.Create(remoteStream, addBeginOfSentence: true, addEndOfSentence: false,
                                            specialTokens: new Dictionary<string, int>
                                            {
                                                // added tokens are picked up from https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/tokenizer_config.json
                                                { "<unk>",                 0 },
                                                { "<s>",                   1 },
                                                { "</s>",                  2 },
                                                { "<|endoftext|>" ,    32000 },
                                                { "<|assistant|>",     32001 },
                                                { "<|placeholder1|>",  32002 },
                                                { "<|placeholder2|>",  32003 },
                                                { "<|placeholder3|>",  32004 },
                                                { "<|placeholder4|>",  32005 },
                                                { "<|system|>",        32006 },
                                                { "<|end|>",           32007 },
                                                { "<|placeholder5|>",  32008 },
                                                { "<|placeholder6|>",  32009 },
                                                { "<|user|>",          32010 },
                                            });

            if (treatWhitespaceAsSuffix)
            {
                PropertyInfo? propertyInfo = typeof(SentencePieceBpeTokenizer).GetProperty("TreatWhitespaceAsSuffix", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
                if (propertyInfo != null)
                {
                    propertyInfo.SetValue(tokenizer, true);
                }

                propertyInfo = typeof(SentencePieceNormalizer).GetProperty("TreatWhitespaceAsSuffix", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
                if (propertyInfo != null)
                {
                    propertyInfo.SetValue(tokenizer.Normalizer, true);
                }
            }

            return tokenizer;
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
        public void TestLlamaTokenizer(Tokenizer tokenizer, string input, int[] ids, string[] tokens, (int Index, int Length)[] offsets)
        {
            // Phi-3 and Llama are using the same tokenizer.model, so we can test both with the same data as long as we are not using added tokens which behave differently for Phi-3.
            Tokenizer[] tokenizers = tokenizer == _llamaTokenizer ? new[] { tokenizer, _llamaPhi3Tokenizer } : new[] { tokenizer };

            foreach (Tokenizer llamaTokenizer in tokenizers)
            {
                LlamaTokenizer bpe = (llamaTokenizer as LlamaTokenizer)!;
                Assert.NotNull(bpe);

                IReadOnlyList<EncodedToken> result = llamaTokenizer.EncodeToTokens(input, out _);
                Assert.Equal(ids, result.Select(t => t.Id).ToArray());
                Assert.Equal(tokens, result.Select(t => t.Value).ToArray());
                Assert.Equal(offsets, result.Select(t => t.Offset).ToArray());
                Assert.Equal(input, llamaTokenizer.Decode(ids));
                TestDecodingWithSpan(bpe, ids, input);
                Assert.Equal(ids, llamaTokenizer.EncodeToIds(input));
                Assert.Equal(ids.Length, llamaTokenizer.CountTokens(input));

                var reverseVocabulary = bpe.Vocabulary.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

                for (int i = 0; i < tokens.Length; i++)
                {
                    Assert.Equal(tokens[i], reverseVocabulary[ids[i]]);
                    Assert.Equal(ids[i], bpe.Vocabulary[tokens[i]]);
                }

                Assert.NotNull(llamaTokenizer.Normalizer);
                string normalizedInput = llamaTokenizer.Normalizer!.Normalize(input);

                bool isEmptyInput = string.IsNullOrEmpty(input);

                IReadOnlyList<EncodedToken> bpeTokens = bpe.EncodeToTokens(normalizedInput.AsSpan(), out _, addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false);
                Assert.Equal(ids.Skip(1), bpeTokens.Select(token => token.Id));
                Assert.Equal(tokens.Skip(1), bpeTokens.Select(token => token.Value));
                int[] extractedIds = bpeTokens.Select(token => token.Id).ToArray();
                Assert.Equal(input, llamaTokenizer.Decode(extractedIds));
                TestDecodingWithSpan(bpe, extractedIds, input);
                IReadOnlyList<int> encodedIds = bpe.EncodeToIds(normalizedInput.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false);
                Assert.Equal(ids.Skip(1), encodedIds);
                Assert.Equal(isEmptyInput ? 0 : ids.Length - 1, bpe.CountTokens(normalizedInput.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false));

                bpeTokens = bpe.EncodeToTokens(normalizedInput.AsSpan(), out _, addBeginningOfSentence: false, addEndOfSentence: true, considerNormalization: false);
                Assert.Equal(isEmptyInput ? Array.Empty<int>() : ids.Skip(1).Concat(new[] { bpe.EndOfSentenceId }), bpeTokens.Select(token => token.Id));
                Assert.Equal(isEmptyInput ? Array.Empty<string>() : tokens.Skip(1).Concat(new[] { bpe.EndOfSentenceToken }), bpeTokens.Select(token => token.Value));
                extractedIds = bpeTokens.Select(token => token.Id).ToArray();
                Assert.Equal(input, llamaTokenizer.Decode(extractedIds));
                TestDecodingWithSpan(bpe, extractedIds, input);
                encodedIds = bpe.EncodeToIds(normalizedInput.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: true, considerNormalization: false);
                Assert.Equal(isEmptyInput ? Array.Empty<int>() : ids.Skip(1).Concat(new[] { bpe.EndOfSentenceId }), encodedIds);
                Assert.Equal(isEmptyInput ? 0 : ids.Length, bpe.CountTokens(normalizedInput.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: true, considerNormalization: false));

                bpeTokens = bpe.EncodeToTokens(normalizedInput.AsSpan(), out _, addBeginningOfSentence: true, addEndOfSentence: true, considerNormalization: false);
                Assert.Equal(isEmptyInput ? Array.Empty<int>() : ids.Concat(new[] { bpe.EndOfSentenceId }), bpeTokens.Select(token => token.Id));
                Assert.Equal(isEmptyInput ? Array.Empty<string>() : tokens.Concat(new[] { bpe.EndOfSentenceToken }), bpeTokens.Select(token => token.Value));
                extractedIds = bpeTokens.Select(token => token.Id).ToArray();
                Assert.Equal(input, llamaTokenizer.Decode(extractedIds));
                TestDecodingWithSpan(bpe, extractedIds, input);
                encodedIds = bpe.EncodeToIds(normalizedInput.AsSpan(), addBeginningOfSentence: true, addEndOfSentence: true, considerNormalization: false);
                Assert.Equal(isEmptyInput ? Array.Empty<int>() : ids.Concat(new[] { bpe.EndOfSentenceId }), encodedIds);
                Assert.Equal(isEmptyInput ? 0 : ids.Length + 1, bpe.CountTokens(normalizedInput.AsSpan(), addBeginningOfSentence: true, addEndOfSentence: true, considerNormalization: false));
            }
        }

        private void TestDecodingWithSpan(LlamaTokenizer tokenizer, int[] ids, string expectedDecoded)
        {
            char[] destinationBuffer = new char[expectedDecoded.Length];

            OperationStatus status;
            int lastIdsConsumed = 0;
            int lastCharactersWritten = 0;
            int idsConsumed;
            int charactersWritten;

            for (int i = 1; i < destinationBuffer.Length - 1; i += Math.Max(1, destinationBuffer.Length - 3)) // enough to test length 1, and destinationBuffer.Length - 2 only.
            {
                status = tokenizer.Decode(ids, destinationBuffer.AsSpan().Slice(0, i), out idsConsumed, out charactersWritten);
                Assert.Equal(OperationStatus.DestinationTooSmall, status);
                Assert.True(idsConsumed < ids.Length);
                Assert.True(idsConsumed >= lastIdsConsumed);
                Assert.True(charactersWritten < expectedDecoded.Length);
                Assert.True(charactersWritten >= lastCharactersWritten);
                lastIdsConsumed = idsConsumed;
                lastCharactersWritten = charactersWritten;
            }

            status = tokenizer.Decode(ids, destinationBuffer.AsSpan(), out idsConsumed, out charactersWritten);
            Assert.Equal(OperationStatus.Done, status);
            Assert.Equal(ids.Length, idsConsumed);
            Assert.Equal(expectedDecoded.Length, charactersWritten);
            Assert.Equal(expectedDecoded, destinationBuffer.AsSpan().ToString());
        }

        public static IEnumerable<object[]> LlamaTokenizersListData()
        {
            yield return new object[] { _llamaTokenizer };
            yield return new object[] { _llamaMistralTokenizer };
            yield return new object[] { _llamaPhi3Tokenizer };
        }

        [Theory]
        [MemberData(nameof(LlamaTokenizersListData))]
        public void TestLlamaTokenizerWithEmptyInput(Tokenizer llamaTokenizer)
        {
            Assert.Equal([], llamaTokenizer.EncodeToTokens((string)null!, out _));
            Assert.Equal([], llamaTokenizer.EncodeToTokens(Span<char>.Empty, out _));

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
            LlamaTokenizer? bpe = llamaTokenizer as LlamaTokenizer;
            Assert.NotNull(bpe);
            Assert.NotNull(llamaTokenizer.Normalizer);

            Assert.Equal("▁Hello,▁World!", llamaTokenizer.Normalizer.Normalize("Hello, World!"));

            Assert.True(bpe.Vocabulary.Count > 0);
            Assert.True(bpe.Vocabulary.TryGetValue("▁", out _));

            Assert.Equal(0, bpe.UnknownId);
            Assert.Equal("<unk>", bpe.UnknownToken);
            Assert.Equal(1, bpe.BeginningOfSentenceId);
            Assert.Equal("<s>", bpe.BeginningOfSentenceToken);
            Assert.Equal(2, bpe.EndOfSentenceId);
            Assert.Equal("</s>", bpe.EndOfSentenceToken);


            Assert.True(bpe.ByteFallback);
            Assert.True(bpe.AddDummyPrefix);
            Assert.True(bpe.EscapeWhiteSpaces);
            Assert.False(bpe.TreatWhitespaceAsSuffix);

            TokenizerTests.TestTokenLimits(llamaTokenizer);
        }

        /// <summary>
        /// Test that the special token with a small id is decoded correctly.
        /// </summary>
        [Theory]
        [MemberData(nameof(LlamaTokenizersListData))]
        public void TestDecodeSpecialTokenWithSmallId(LlamaTokenizer llamaTokenizer)
        {
            Assert.Equal(llamaTokenizer.EndOfSentenceToken, llamaTokenizer.Decode([llamaTokenizer.EndOfSentenceId], considerSpecialTokens: true));
        }

        [Fact]
        public void TestSentencePieceNormalizer()
        {
            SentencePieceNormalizer normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: false, addDummyPrefix: false, escapeWhiteSpaces: false, treatWhitespaceAsSuffix: false, specialTokens: null);
            Assert.Equal("Hello,      World!", normalizer.Normalize("Hello,      World!"));
            Assert.Equal("Hello,      World!", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: true, addDummyPrefix: false, escapeWhiteSpaces: false, treatWhitespaceAsSuffix: false, specialTokens: null);
            Assert.Equal("Hello, World!", normalizer.Normalize("Hello,      World!"));
            Assert.Equal("Hello, World!", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: true, addDummyPrefix: true, escapeWhiteSpaces: false, treatWhitespaceAsSuffix: false, specialTokens: null);
            Assert.Equal(" Hello, World!", normalizer.Normalize("Hello,      World!"));
            Assert.Equal(" Hello, World!", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: true, addDummyPrefix: true, escapeWhiteSpaces: true, treatWhitespaceAsSuffix: false, specialTokens: null);
            Assert.Equal("▁Hello,▁World!", normalizer.Normalize("Hello,      World!"));
            Assert.Equal("▁Hello,▁World!", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: false, addDummyPrefix: true, escapeWhiteSpaces: true, treatWhitespaceAsSuffix: false, specialTokens: null);
            Assert.Equal("▁Hello,▁▁▁▁▁▁World!", normalizer.Normalize("Hello,      World!"));
            Assert.Equal("▁Hello,▁▁▁▁▁▁World!", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: true, addDummyPrefix: true, escapeWhiteSpaces: true, treatWhitespaceAsSuffix: true, specialTokens: null);
            Assert.Equal("Hello,▁World!▁", normalizer.Normalize("Hello,      World!"));
            Assert.Equal("Hello,▁World!▁", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: true, addDummyPrefix: false, escapeWhiteSpaces: true, treatWhitespaceAsSuffix: true, specialTokens: null);
            Assert.Equal("Hello,▁World!", normalizer.Normalize("Hello,      World!"));
            Assert.Equal("Hello,▁World!", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: false, addDummyPrefix: true, escapeWhiteSpaces: true, treatWhitespaceAsSuffix: true, specialTokens: null);
            Assert.Equal("Hello,▁▁▁▁▁▁World!▁", normalizer.Normalize("Hello,      World!"));
            Assert.Equal("Hello,▁▁▁▁▁▁World!▁", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: false, addDummyPrefix: true, escapeWhiteSpaces: false, treatWhitespaceAsSuffix: true, specialTokens: null);
            Assert.Equal("Hello,      World! ", normalizer.Normalize("Hello,      World!"));
            Assert.Equal("Hello,      World! ", normalizer.Normalize("Hello,      World!".AsSpan()));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: false, addDummyPrefix: true, escapeWhiteSpaces: true, treatWhitespaceAsSuffix: false, specialTokens: (_llamaPhi3Tokenizer as LlamaTokenizer)!.SpecialTokens);
            Assert.Equal("<|user|>", normalizer.Normalize("<|user|>"));
            Assert.Equal("<|user|><|system|><|assistant|><|endoftext|>", normalizer.Normalize("<|user|><|system|><|assistant|><|endoftext|>"));
            Assert.Equal("▁Hello<|user|>", normalizer.Normalize("Hello<|user|>"));
            Assert.Equal("▁Hello,▁<|user|>World", normalizer.Normalize("Hello, <|user|>World"));
            Assert.Equal("<|endoftext|>▁Hello<|user|>", normalizer.Normalize("<|endoftext|>Hello<|user|>"));
            Assert.Equal("", normalizer.Normalize(""));

            normalizer = new SentencePieceNormalizer(removeExtraWhiteSpaces: false, addDummyPrefix: true, escapeWhiteSpaces: true, treatWhitespaceAsSuffix: true, specialTokens: (_llamaPhi3Tokenizer as LlamaTokenizer)!.SpecialTokens);
            Assert.Equal("<|user|>", normalizer.Normalize("<|user|>"));
            Assert.Equal("<|user|><|system|><|assistant|><|endoftext|>", normalizer.Normalize("<|user|><|system|><|assistant|><|endoftext|>"));
            Assert.Equal("Hello▁<|user|>", normalizer.Normalize("Hello<|user|>"));
            Assert.Equal("Hello,▁<|user|>World▁", normalizer.Normalize("Hello, <|user|>World"));
            Assert.Equal("<|endoftext|>Hello▁<|user|>", normalizer.Normalize("<|endoftext|>Hello<|user|>"));
            Assert.Equal("", normalizer.Normalize(""));

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

            IReadOnlyList<EncodedToken> encoding = tokenizer.EncodeToTokens(text, out _);
            IReadOnlyList<EncodedToken> encoding1 = tokenizer.EncodeToTokens(text.AsSpan(), out _);

            Assert.Equal(expectedTokens, encoding.Select(t => t.Value).ToArray());
            Assert.Equal(expectedOffsets, encoding.Select(t => t.Offset).ToArray());
            Assert.Equal(expectedIds, encoding.Select(t => t.Id).ToArray());

            Assert.Equal(expectedTokens, encoding1.Select(t => t.Value).ToArray());
            Assert.Equal(expectedOffsets, encoding1.Select(t => t.Offset).ToArray());
            Assert.Equal(expectedIds, encoding1.Select(t => t.Id).ToArray());

            SentencePieceBpeTokenizer sentencePieceBpe = (tokenizer as SentencePieceBpeTokenizer)!;
            foreach (bool considerNormalization in new[] { true, false })
                foreach (bool addBeginningOfSentence in new[] { true, false })
                    foreach (bool addEndOfSentence in new[] { true, false })
                    {
                        encoding = sentencePieceBpe.EncodeToTokens(
                                        considerNormalization ? text : normalizedText,
                                        out _,
                                        addBeginningOfSentence: addBeginningOfSentence,
                                        addEndOfSentence: addEndOfSentence,
                                        considerPreTokenization: false,
                                        considerNormalization: considerNormalization);

                        encoding1 = sentencePieceBpe.EncodeToTokens(
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

            SentencePieceBpeTokenizer sentencePieceBpe = (tokenizer as SentencePieceBpeTokenizer)!;
            foreach (bool considerNormalization in new[] { true, false })
                foreach (bool addBeginningOfSentence in new[] { true, false })
                    foreach (bool addEndOfSentence in new[] { true, false })
                    {
                        // (string text, bool addBeginningOfSentence, bool addEndOfSentence, int maxTokenCount, out string? normalizedString, out int charsConsumed, bool considerPreTokenization = true, bool considerNormalization = true)

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

            Assert.Equal(expectedOffsets[expectedOffsets.Length - 7].Index + expectedOffsets[expectedOffsets.Length - 7].Length, tokenizer.GetIndexByTokenCount(text, expectedIds.Length - 6, out string? normalizedString, out int tokenCount));
            Assert.Equal(normalizedText, normalizedString);
            Assert.Equal(expectedIds.Length - 6, tokenCount);
            Assert.Equal(expectedOffsets[expectedOffsets.Length - 7].Index + expectedOffsets[expectedOffsets.Length - 7].Length, tokenizer.GetIndexByTokenCount(text.AsSpan(), expectedIds.Length - 6, out normalizedString, out tokenCount));
            Assert.Equal(normalizedText, normalizedString);
            Assert.Equal(expectedIds.Length - 6, tokenCount);

            Assert.Equal(expectedOffsets[expectedOffsets.Length - 7].Index, tokenizer.GetIndexByTokenCountFromEnd(text, 7, out normalizedString, out tokenCount));
            Assert.Equal(normalizedText, normalizedString);
            Assert.Equal(7, tokenCount);
            Assert.Equal(expectedOffsets[expectedOffsets.Length - 7].Index, tokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), 7, out normalizedString, out tokenCount));
            Assert.Equal(normalizedText, normalizedString);
            Assert.Equal(7, tokenCount);
        }

        [Fact]
        public void TestPhi3Tokenizer()
        {
            LlamaTokenizer tokenizer = (_llamaPhi3Tokenizer as LlamaTokenizer)!;
            Assert.True(tokenizer.SpecialTokens is not null);

            StringBuilder sb = new(); // Create bigger string containing all Added Tokens
            IReadOnlyList<EncodedToken> encodedTokens;
            IReadOnlyList<int> encodedIds;
            int tokenCount;
            string? normalizedString;

            foreach (var kvp in tokenizer.SpecialTokens)
            {
                encodedTokens = tokenizer.EncodeToTokens(kvp.Key, out normalizedString);
                Assert.Equal(new[] { tokenizer.BeginningOfSentenceToken, kvp.Key }, encodedTokens.Select(et => et.Value).ToArray());
                Assert.Equal(new[] { tokenizer.BeginningOfSentenceId, kvp.Value }, encodedTokens.Select(et => et.Id).ToArray());
                Assert.Equal($"{kvp.Key}", normalizedString);

                encodedIds = tokenizer.EncodeToIds(kvp.Key);
                Assert.Equal(encodedIds, encodedTokens.Select(et => et.Id).ToArray());

                tokenCount = tokenizer.CountTokens(kvp.Key);
                Assert.Equal(tokenCount, encodedTokens.Count);

                sb.Append($" Hello{kvp.Key}");
            }

            string s = sb.ToString();
            string expectedNormalizedString = $"{DummyPrefix}{s.Replace(' ', DummyPrefix[0])}";

            encodedTokens = tokenizer.EncodeToTokens(s, out normalizedString, addBeginningOfSentence: false, addEndOfSentence: false);
            Assert.Equal(expectedNormalizedString, normalizedString);

            string[] specialTokens = tokenizer.SpecialTokens.Keys.ToArray();

            string accumulatedString = DummyPrefix;
            string accumulatedStringFromEnd = "";

            for (int i = 1; i <= encodedTokens.Count; i++)
            {
                int index = tokenizer.GetIndexByTokenCount(s, addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: i, out normalizedString, out tokenCount);
                Assert.Equal(index, accumulatedString.Length);
                Assert.Equal(i, tokenCount);

                accumulatedString += i % 2 != 0 ? $"Hello{DummyPrefix}" : specialTokens[i / 2 - 1];

                accumulatedStringFromEnd = (encodedTokens.Count == i ? DummyPrefix : (i % 2 == 0 ? $"{DummyPrefix}Hello" : specialTokens[specialTokens.Length - 1 - (i / 2)])) + accumulatedStringFromEnd;

                index = tokenizer.GetIndexByTokenCountFromEnd(s, addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: i, considerNormalization: true, out normalizedString, out tokenCount);
                Assert.Equal(i, tokenCount);
                Assert.Equal(index, normalizedString!.Length - accumulatedStringFromEnd.Length);
            }
        }

        public static IEnumerable<object[]> Phi3TestData()
        {
            // text to tokenize,
            // Decode text without special tokens,
            // expected ids
            // expected ids when using space suffix
            yield return new object[]
            {
                "Can you provide ways to eat combinations of bananas and dragonfruits?",
                "Can you provide ways to eat combinations of bananas and dragonfruits?",
                new int[]
                {
                    1, 1815, 366, 3867, 5837, 304, 17545, 18240, 310, 9892, 16397, 322, 8338, 265, 29888, 21211, 29973
                },
                new int[]
                {
                    1, 6028, 366, 3867, 5837, 304, 17545, 18240, 310, 9892, 16397, 322, 8338, 265, 29888, 21211, 29973, 29871
                }
            };

            yield return new object[]
            {
                "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2." +
                " Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey.",
                "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2." +
                " Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey.",
                new int[]
                {
                    1, 18585, 29991, 2266, 526, 777, 5837, 304, 17545, 9892, 16397, 322, 8338, 265, 29888, 21211, 4208, 29901, 29871, 29896, 29889, 10765, 1648, 322, 8338, 265, 29888,
                    9216, 10597, 347, 29901, 3164, 355, 9892, 16397, 322, 8338, 265, 29888, 21211, 4208, 411, 777, 27274, 322, 298, 4992, 29889, 29871, 29906, 29889, 10765, 1648, 322,
                    8338, 265, 29888, 9216, 4497, 328, 29901, 23478, 269, 506, 287, 9892, 16397, 322, 8338, 265, 29888, 21211, 4208, 411, 777, 454, 3712, 3623, 625, 322, 298, 4992, 29889
                },
                new int[]
                {
                    1, 29903, 545, 29991, 2266, 526, 777, 5837, 304, 17545, 9892, 16397, 322, 8338, 265, 29888, 21211, 4208, 29901, 29871, 29896, 29889, 10765, 1648, 322, 8338, 265, 29888,
                    9216, 10597, 347, 29901, 3164, 355, 9892, 16397, 322, 8338, 265, 29888, 21211, 4208, 411, 777, 27274, 322, 298, 4992, 29889, 29871, 29906, 29889, 10765, 1648, 322, 8338,
                    265, 29888, 9216, 4497, 328, 29901, 23478, 269, 506, 287, 9892, 16397, 322, 8338, 265, 29888, 21211, 4208, 411, 777, 454, 3712, 3623, 625, 322, 298, 4992, 29889, 29871
                }
            };

            yield return new object[]
            {
                "What about solving an 2x + 3 = 7 equation?",
                "What about solving an 2x + 3 = 7 equation?",
                new int[]
                {
                    1, 1724, 1048, 17069, 385, 29871, 29906, 29916, 718, 29871, 29941, 353, 29871, 29955, 6306, 29973
                },
                new int[]
                {
                    1, 5618, 1048, 17069, 385, 29871, 29906, 29916, 718, 29871, 29941, 353, 29871, 29955, 6306, 29973, 29871
                }
            };

            yield return new object[]
            {
                "\nCount to 3\n",
                "\nCount to 3\n",
                new int[]
                {
                    1, 29871, 13, 3981, 304, 29871, 29941, 13
                },
                new int[]
                {
                    1, 13, 3981, 304, 29871, 29941, 13, 29871
                }
            };

            yield return new object[]
            {
                "<|user|>",
                "",
                new int[]
                {
                    1, 32010
                },
                new int[]
                {
                    1, 32010
                }
            };

            yield return new object[]
            {
                "<|end|>",
                "",
                new int[]
                {
                    1, 32007
                },
                new int[]
                {
                    1, 32007
                }
            };

            yield return new object[]
            {
                "<|assistant|>",
                "",
                new int[]
                {
                    1, 32001
                },
                new int[]
                {
                    1, 32001
                }
            };

            yield return new object[]
            {
                "<|user|>\nCount to 3<|end|>\n<|assistant|>",
                "\nCount to 3\n",
                new int[]
                {
                    1, 32010, 29871, 13, 3981, 304, 29871, 29941, 32007, 13, 32001
                },
                new int[]
                {
                    1, 32010, 13, 3981, 304, 29871, 29941, 32007, 13, 29871, 32001
                }
            };
        }

        [Theory]
        [MemberData(nameof(Phi3TestData))]
        public void TestPhi3TokenizerIdEncoding(string text, string decodedWithNoSpecialTokens, int[] expectedIds, int[] expectedIdsWithSuffix)
        {
            LlamaTokenizer tokenizer = (_llamaPhi3Tokenizer as LlamaTokenizer)!;
            var ids = tokenizer.EncodeToIds(text);
            Assert.Equal(expectedIds, ids);
            Assert.Equal(decodedWithNoSpecialTokens, tokenizer.Decode(expectedIds));
            string textWithSpecialTokens = $"{tokenizer.BeginningOfSentenceToken}{text}";
            Assert.Equal(textWithSpecialTokens, tokenizer.Decode(expectedIds, considerSpecialTokens: true));

            char[] destinationBuffer = new char[decodedWithNoSpecialTokens.Length];

            int idsConsumed;
            int charactersWritten;

            for (int i = 1; i < destinationBuffer.Length - 1; i += Math.Max(1, destinationBuffer.Length - 3)) // enough to test length 1, and destinationBuffer.Length - 2 only.
            {
                Assert.Equal(OperationStatus.DestinationTooSmall, tokenizer.Decode(ids, destinationBuffer.AsSpan().Slice(0, i), out idsConsumed, out charactersWritten));
                Assert.True(idsConsumed < ids.Count);
                Assert.True(decodedWithNoSpecialTokens.AsSpan().StartsWith(destinationBuffer.AsSpan().Slice(0, charactersWritten)));
            }

            Assert.Equal(OperationStatus.Done, tokenizer.Decode(ids, destinationBuffer.AsSpan(), out idsConsumed, out charactersWritten));
            Assert.Equal(ids.Count, idsConsumed);
            Assert.Equal(decodedWithNoSpecialTokens.Length, charactersWritten);
            Assert.Equal(decodedWithNoSpecialTokens, destinationBuffer.AsSpan().ToString());

            destinationBuffer = new char[textWithSpecialTokens.Length];

            for (int i = 1; i < destinationBuffer.Length - 1; i += Math.Max(1, destinationBuffer.Length - 3)) // enough to test length 1, and destinationBuffer.Length - 2 only.
            {
                Assert.Equal(OperationStatus.DestinationTooSmall, tokenizer.Decode(ids, destinationBuffer.AsSpan().Slice(0, i), considerSpecialTokens: true, out idsConsumed, out charactersWritten));
                Assert.True(idsConsumed < ids.Count);
                Assert.True(textWithSpecialTokens.AsSpan().StartsWith(destinationBuffer.AsSpan().Slice(0, charactersWritten)));
            }

            Assert.Equal(OperationStatus.Done, tokenizer.Decode(ids, destinationBuffer.AsSpan(), considerSpecialTokens: true, out idsConsumed, out charactersWritten));
            Assert.Equal(ids.Count, idsConsumed);
            Assert.Equal(textWithSpecialTokens.Length, charactersWritten);
            Assert.Equal(textWithSpecialTokens, destinationBuffer.AsSpan().ToString());

            LlamaTokenizer tokenizerWithSuffix = (_llamaPhi3TokenizerWithTreatSpaceSuffix as LlamaTokenizer)!;
            Assert.True(tokenizerWithSuffix.TreatWhitespaceAsSuffix);
            ids = tokenizerWithSuffix.EncodeToIds(text);
            Assert.Equal(expectedIdsWithSuffix, ids);
            Assert.Equal(decodedWithNoSpecialTokens, tokenizerWithSuffix.Decode(expectedIdsWithSuffix));
            Assert.Equal(textWithSpecialTokens, tokenizerWithSuffix.Decode(expectedIdsWithSuffix, considerSpecialTokens: true));

            //
            // Test with suffix instead of prefix
            //

            destinationBuffer = new char[decodedWithNoSpecialTokens.Length + 1]; // one extra for suffix

            for (int i = 1; i < destinationBuffer.Length - 1; i += Math.Max(1, destinationBuffer.Length - 3)) // enough to test length 1, and destinationBuffer.Length - 2 only.
            {
                Assert.Equal(OperationStatus.DestinationTooSmall, tokenizerWithSuffix.Decode(ids, destinationBuffer.AsSpan().Slice(0, i), out idsConsumed, out charactersWritten));
                Assert.True(idsConsumed < ids.Count);
                Assert.True(decodedWithNoSpecialTokens.AsSpan().StartsWith(destinationBuffer.AsSpan().Slice(0, charactersWritten)));
            }

            Assert.Equal(OperationStatus.Done, tokenizerWithSuffix.Decode(ids, destinationBuffer.AsSpan(), out idsConsumed, out charactersWritten));
            Assert.Equal(ids.Count, idsConsumed);
            Assert.Equal(decodedWithNoSpecialTokens.Length, charactersWritten);
            Assert.Equal(decodedWithNoSpecialTokens, destinationBuffer.AsSpan().Slice(0, charactersWritten).ToString());

            destinationBuffer = new char[textWithSpecialTokens.Length + 1];

            for (int i = 1; i < destinationBuffer.Length - 1; i += Math.Max(1, destinationBuffer.Length - 3)) // enough to test length 1, and destinationBuffer.Length - 2 only.
            {
                Assert.Equal(OperationStatus.DestinationTooSmall, tokenizerWithSuffix.Decode(ids, destinationBuffer.AsSpan().Slice(0, i), considerSpecialTokens: true, out idsConsumed, out charactersWritten));
                Assert.True(idsConsumed < ids.Count);
                var sp = destinationBuffer.AsSpan().Slice(0, charactersWritten);
                if (sp.Length > 0 && sp[sp.Length - 1] == ' ')
                {
                    sp = sp.Slice(0, sp.Length - 1);
                }
                Assert.True(textWithSpecialTokens.AsSpan().StartsWith(sp));
            }

            Assert.Equal(OperationStatus.Done, tokenizerWithSuffix.Decode(ids, destinationBuffer.AsSpan(), considerSpecialTokens: true, out idsConsumed, out charactersWritten));
            Assert.Equal(ids.Count, idsConsumed);
            Assert.Equal(textWithSpecialTokens.Length, charactersWritten);
            Assert.Equal(textWithSpecialTokens, destinationBuffer.AsSpan(0, charactersWritten).ToString());
        }
    }
}
