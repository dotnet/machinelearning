// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection.Metadata;
using System.Text;
using System.Text.Json;
using Microsoft.ML.Tokenizers;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class UnigramTests
    {
        private static SentencePieceTokenizer _unigramTokenizer = CreateUnigramTokenizer();
        private static SentencePieceTokenizer _unigramTokenizerWithSpecialTokens = CreateUnigramTokenizerWithSpecialTokens();
        private static SentencePieceTokenizer _unigramTokenizerFromJson = CreateUnigramTokenizerFromJson();

        private static SentencePieceTokenizer CreateUnigramTokenizer()
        {
            // @"https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/sentencepiece.bpe.model?download=true";
            using Stream remoteStream = File.OpenRead(Path.Combine(@"Paraphrase-multilingual-MiniLM-L12-v2", "sentencepiece.bpe.model"));
            return SentencePieceTokenizer.Create(remoteStream);
        }

        private static SentencePieceTokenizer CreateUnigramTokenizerFromJson()
        {
            // @"https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/tokenizer.json?download=true";
            using Stream jsonModelStream = File.OpenRead(Path.Combine(@"Paraphrase-multilingual-MiniLM-L12-v2", "tokenizer.json"));
            using var reader = new StreamReader(jsonModelStream, Encoding.UTF8);
            string json = reader.ReadToEnd();
            using JsonDocument doc = JsonDocument.Parse(json);
            JsonElement root = doc.RootElement;

            SentencePieceOptions options = new SentencePieceOptions();
            options.ModelType = SentencePieceModelType.Unigram;
            options.EscapeWhiteSpaces = true;
            options.AddDummyPrefix = true;

            options.BeginningOfSentenceToken = "<s>";
            options.EndOfSentenceToken = "</s>";
            options.UnknownToken = "<unk>";

            options.SpecialTokens = new Dictionary<string, int>
            {
                { "<s>",    0       },
                { "<pad>",  1       },
                { "</s>",   2       },
                { "<unk>",  3       },
                { "<mask>", 250001  }
            };

            if (root.TryGetProperty("normalizer", out JsonElement normalizerElement) && normalizerElement.GetProperty("type").GetString() == "Precompiled")
            {
                string? precompiledCharsMap = normalizerElement.GetProperty("precompiled_charsmap").GetString();
                if (precompiledCharsMap is not null)
                {
                    byte[] bytes = Convert.FromBase64String(precompiledCharsMap);
                    options.PrecompiledNormalizationData = bytes;
                }
            }

            options.Vocabulary = GetVocabulary(root);
            return SentencePieceTokenizer.Create(options);
        }

        private static IEnumerable<KeyValuePair<string, float>> GetVocabulary(JsonElement root)
        {
            if (root.TryGetProperty("model", out JsonElement modelElement) &&
                modelElement.TryGetProperty("vocab", out JsonElement vocabElement) &&
                vocabElement.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement token in vocabElement.EnumerateArray())
                {
                    if (token.ValueKind == JsonValueKind.Array && token.GetArrayLength() == 2)
                    {
                        string? tokenString = token[0].GetString();
                        if (tokenString is null)
                        {
                            throw new InvalidOperationException("Invalid model vocabulary format");
                        }
                        yield return new KeyValuePair<string, float>(tokenString, token[1].GetSingle());
                    }
                }
            }
            else
            {
                throw new InvalidOperationException("Invalid model vocabulary format");
            }
        }

        private static SentencePieceTokenizer CreateUnigramTokenizerWithSpecialTokens()
        {
            // @"https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/sentencepiece.bpe.model?download=true";
            using Stream remoteStream = File.OpenRead(Path.Combine(@"Paraphrase-multilingual-MiniLM-L12-v2", "sentencepiece.bpe.model"));
            return SentencePieceTokenizer.Create(remoteStream, specialTokens:
                                                                new Dictionary<string, int>
                                                                {
                                                                    { "<unk>",                 0 },
                                                                    { "<s>",                   1 },
                                                                    { "</s>",                  2 },
                                                                    { "<pad>",                 7 },
                                                                    { "<mask>",                8 },
                                                                });
        }

        public static IEnumerable<object[]> UnigramTestData()
        {
            // tokenizer, input text, normalized text, decoded text, ids, tokens, offsets
            yield return new object[]
            {
                "Hello, world!",
                "▁Hello,▁world!",
                "Hello, world!",
                new int[] { 35377, 3, 8998, 37 },
                new string[] { "▁Hello", ",", "▁world", "!" },
                new Range[] { new Range(0, 6), new Range(6, 7), new Range(7, 13), new Range(13, 14) }
            };

            yield return new object[]
            {
                "Hello, ①ｶﾀｶﾅﬀ⁰⅓Ⅳ \U00010200 world! \uD800\uDE00", // include normalization and unknown characters
                "▁Hello,▁1カタカナff01⁄3IV▁\U00010200▁world!▁\U00010200",
                "Hello, 1カタカナff01⁄3IV  world! ",
                new int[] { 35377, 3, 105, 10044, 10792, 10044, 17455, 4901, 6745, 244258, 362, 15582, 5, 0, 8998, 37, 5, 0 }, // Unknown Id is 0
                new string[] { "▁Hello", ",", "▁1", "カ", "タ", "カ", "ナ", "ff", "01", "⁄", "3", "IV", "▁", "\U00010200", "▁world", "!", "▁", "\U00010200" },
                new Range[]
                {
                    new Range(0, 6), new Range(6, 7), new Range(7, 9), new Range(9, 10), new Range(10, 11), new Range(11, 12),
                    new Range(12, 13), new Range(13, 15), new Range(15, 17), new Range(17, 18), new Range(18, 19), new Range(19, 21),
                    new Range(21, 22), new Range(22, 24), new Range(24, 30), new Range(30, 31), new Range(31, 32), new Range(32, 34)
                }
            };

            yield return new object[]
            {
                "",
                "",
                "",
                new int[0],
                new string[0],
                new Range[0]
            };

            yield return new object[]
            {
                @"The sun dipped below the horizon, casting a warm golden hue across the tranquil meadow. Birds fluttered from " +
                "tree to tree, their melodic songs filling the air. A gentle breeze rustled the leaves, carrying with it the scent of " +
                "blooming flowers. In the distance, the silhouette of a lone figure stood atop a hill, gazing out at the vast expanse " +
                "before them. It was a moment frozen in time, where nature and solitude merged into something magical.",

                "▁The▁sun▁dipped▁below▁the▁horizon,▁casting▁a▁warm▁golden▁hue▁across▁the▁tranquil▁meadow.▁Birds▁fluttered▁from▁tree▁to▁tree,▁their" +
                "▁melodic▁songs▁filling▁the▁air.▁A▁gentle▁breeze▁rustled▁the▁leaves,▁carrying▁with▁it▁the▁scent▁of▁blooming▁flowers.▁In▁the▁distance" +
                ",▁the▁silhouette▁of▁a▁lone▁figure▁stood▁atop▁a▁hill,▁gazing▁out▁at▁the▁vast▁expanse▁before▁them.▁It▁was▁a▁moment▁frozen▁in▁time,▁" +
                "where▁nature▁and▁solitude▁merged▁into▁something▁magical.",

                @"The sun dipped below the horizon, casting a warm golden hue across the tranquil meadow. Birds fluttered from " +
                "tree to tree, their melodic songs filling the air. A gentle breeze rustled the leaves, carrying with it the scent of " +
                "blooming flowers. In the distance, the silhouette of a lone figure stood atop a hill, gazing out at the vast expanse " +
                "before them. It was a moment frozen in time, where nature and solitude merged into something magical.",

                new int[]
                {
                    580, 4261, 44, 48397, 35063, 69, 5, 156633, 3, 176049, 9, 24813, 158043, 78023, 36879, 69, 46193, 10547, 24292, 4,
                    72606, 6, 139099, 55, 296, 1294, 53200, 46, 53200, 3, 2362, 43670, 237, 52335, 26291, 213, 69, 1830, 4, 61, 21506,
                    132, 12561, 6658, 52647, 6258, 69, 31357, 6, 3, 85357, 213, 677, 441, 69, 25453, 17, 110, 29694, 305, 213, 189066,
                    4, 359, 69, 62487, 3, 69, 5794, 13884, 8675, 110, 9, 458, 85, 26365, 192941, 9, 13783, 9, 130472, 3, 13958, 213,
                    1809, 98, 69, 18409, 14699, 20539, 8107, 2855, 4, 1649, 508, 9, 3094, 1237, 70462, 22, 1732, 3, 7439, 31424, 135,
                    3114, 21752, 12, 42563, 70, 3933, 9843, 49845, 288, 4
                },

                new string[]
                {
                    "▁The", "▁sun", "▁di", "pped", "▁below", "▁the", "▁", "horizon", ",", "▁casting", "▁a", "▁warm", "▁golden", "▁hue",
                    "▁across", "▁the", "▁tranquil", "▁mea", "dow", ".", "▁Bird", "s", "▁flutt", "er", "ed", "▁from", "▁tree", "▁to", "▁tree",
                    ",", "▁their", "▁melodi", "c", "▁songs", "▁fill", "ing", "▁the", "▁air", ".", "▁A", "▁gent", "le", "▁bre", "eze", "▁rust",
                    "led", "▁the", "▁leave", "s", ",", "▁carry", "ing", "▁with", "▁it", "▁the", "▁scen", "t", "▁of", "▁blo", "om", "ing",
                    "▁flowers", ".", "▁In", "▁the", "▁distance", ",", "▁the", "▁sil", "hou", "ette", "▁of", "▁a", "▁lo", "ne", "▁figure",
                    "▁stood", "▁a", "top", "▁a", "▁hill", ",", "▁gaz", "ing", "▁out", "▁at", "▁the", "▁vast", "▁exp", "anse", "▁before",
                    "▁them", ".", "▁It", "▁was", "▁a", "▁moment", "▁f", "rozen", "▁in", "▁time", ",", "▁where", "▁nature", "▁and", "▁sol",
                    "itud", "e", "▁merge", "d", "▁into", "▁something", "▁magic", "al", "."
                },

                new Range[]
                {
                    new Range(0, 4), new Range(4, 8), new Range(8, 11), new Range(11, 15), new Range(15, 21), new Range(21, 25),
                    new Range(25, 26), new Range(26, 33), new Range(33, 34), new Range(34, 42), new Range(42, 44), new Range(44, 49), new Range(49, 56),
                    new Range(56, 60), new Range(60, 67), new Range(67, 71), new Range(71, 80), new Range(80, 84), new Range(84, 87), new Range(87, 88),
                    new Range(88, 93), new Range(93, 94), new Range(94, 100), new Range(100, 102), new Range(102, 104), new Range(104, 109), new Range(109, 114),
                    new Range(114, 117), new Range(117, 122), new Range(122, 123), new Range(123, 129), new Range(129, 136), new Range(136, 137),
                    new Range(137, 143), new Range(143, 148), new Range(148, 151), new Range(151, 155), new Range(155, 159), new Range(159, 160),
                    new Range(160, 162), new Range(162, 167), new Range(167, 169), new Range(169, 173), new Range(173, 176), new Range(176, 181),
                    new Range(181, 184), new Range(184, 188), new Range(188, 194), new Range(194, 195), new Range(195, 196), new Range(196, 202),
                    new Range(202, 205), new Range(205, 210), new Range(210, 213), new Range(213, 217), new Range(217, 222), new Range(222, 223),
                    new Range(223, 226), new Range(226, 230), new Range(230, 232), new Range(232, 235), new Range(235, 243), new Range(243, 244),
                    new Range(244, 247), new Range(247, 251), new Range(251, 260), new Range(260, 261), new Range(261, 265), new Range(265, 269),
                    new Range(269, 272), new Range(272, 276), new Range(276, 279), new Range(279, 281), new Range(281, 284), new Range(284, 286),
                    new Range(286, 293), new Range(293, 299), new Range(299, 301), new Range(301, 304), new Range(304, 306), new Range(306, 311),
                    new Range(311, 312), new Range(312, 316), new Range(316, 319), new Range(319, 323), new Range(323, 326), new Range(326, 330),
                    new Range(330, 335), new Range(335, 339), new Range(339, 343), new Range(343, 350), new Range(350, 355), new Range(355, 356),
                    new Range(356, 359), new Range(359, 363), new Range(363, 365), new Range(365, 372), new Range(372, 374), new Range(374, 379),
                    new Range(379, 382), new Range(382, 387), new Range(387, 388), new Range(388, 394), new Range(394, 401), new Range(401, 405),
                    new Range(405, 409), new Range(409, 413), new Range(413, 414), new Range(414, 420), new Range(420, 421), new Range(421, 426),
                    new Range(426, 436), new Range(436, 442), new Range(442, 444), new Range(444, 445)
                }
            };

            yield return new object[]
            {
                "This is 👍, an emoji.",
                "▁This▁is▁👍,▁an▁emoji.",
                "This is 👍, an emoji.",
                new int[] { 3292, 82, 5, 118279, 3, 141, 27, 121504, 4 },
                new string[] { "▁This", "▁is", "▁", "👍", ",", "▁an", "▁e", "moji", "." },
                new Range[] { new Range(0, 5), new Range(5, 8), new Range(8, 9), new Range(9, 11), new Range(11, 12), new Range(12, 15), new Range(15, 17), new Range(17, 21), new Range(21, 22) }
            };

            yield return new object[]
            {
                "清水寺は京都にある。", // Japanese
                "▁清水寺は京都にある。",
                "清水寺は京都にある。",
                new int[] { 5, 177585, 32566, 341, 60423, 24432, 29 },
                new string[] { "▁", "清水", "寺", "は", "京都", "にある", "。" },
                new Range[] { new Range(0, 1), new Range(1, 3), new Range(3, 4), new Range(4, 5), new Range(5, 7), new Range(7, 10), new Range(10, 11) }
            };

            yield return new object[]
            {
                "xyz東京", // Latin-Japanese
                "▁xyz東京",
                "xyz東京",
                new int[] { 1021, 32188, 22887 },
                new string[] { "▁x", "yz", "東京" },
                new Range[] { new Range(0, 2), new Range(2, 4), new Range(4, 6) }
            };

            yield return new object[]
            {
                "㍻",        // Japanese with normalization
                "▁平成",
                "平成",
                new int[] { 5, 44405 },
                new string[] { "▁", "平成" },
                new Range[] { new Range(0, 1), new Range(1, 3) }
            };

            yield return new object[]
            {
                "ＫＡＤＯＫＡＷＡABC", // Full-width Latin to normalize to normal width
                "▁KADOKAWAABC",
                "KADOKAWAABC",
                new int[] { 340, 41387, 218268, 186943 },
                new string[] { "▁K", "ADO", "KAWA", "ABC" },
                new Range[] { new Range(0, 2), new Range(2, 5), new Range(5, 9), new Range(9, 12) }
            };

            yield return new object[]
            {
                "ℌ𝔢𝔩𝔩𝔬 𝔚𝔬𝔯𝔩𝔡!", // Gothic script
                "▁Hello▁World!",
                "Hello World!",
                new int[] { 35377, 6660, 37 },
                new string[] { "▁Hello", "▁World", "!" },
                new Range[] { new Range(0, 6), new Range(6, 12), new Range(12, 13) }
            };

            yield return new object[]
            {
                "𝛢𝛷𝛢𝛪𝛯𝛪", // Greek script
                "▁ΑΦΑΙΞΙ",
                "ΑΦΑΙΞΙ",
                new int[] { 3866, 203768, 15470, 72125, 15470 },
                new string[] { "▁Α", "ΦΑ", "Ι", "Ξ", "Ι" },
                new Range[] { new Range(0, 2), new Range(2, 4), new Range(4, 5), new Range(5, 6), new Range(6, 7) }
            };

            yield return new object[]
            {
                "𝖘𝖙𝖗𝖆𝖓𝖎𝖈𝖆", // Russian script
                "▁stranica",
                "stranica",
                new int[] { 60133 },
                new string[] { "▁stranica" },
                new Range[] { new Range(0, 9) }
            };

            yield return new object[]
            {
                "老師", // Chinese
                "▁老師",
                "老師",
                new int[] { 5, 25924 },
                new string[] { "▁", "老師" },
                new Range[] { new Range(0, 1), new Range(1, 3) }
            };
        }

        private (IEnumerable<int> Ids, IEnumerable<string> Tokens, IEnumerable<Range> Offsets) ExtractedIds(
                                                                                                SentencePieceTokenizer tokenizer,
                                                                                                IReadOnlyList<EncodedToken> tokens,
                                                                                                string? normalized,
                                                                                                bool addBeginningOfSentence,
                                                                                                bool addEndOfSentence)
        {
            List<EncodedToken> writableTokens = tokens.ToList();
            if (addBeginningOfSentence && writableTokens.Count > 0)
            {
                Assert.True(writableTokens[0].Id == tokenizer.BeginningOfSentenceId);
                Assert.True(writableTokens[0].Value == tokenizer.BeginningOfSentenceToken);
                Assert.True(writableTokens[0].Offset.Equals(new Range(0, 0)));
                writableTokens.RemoveAt(0);
            }

            if (addEndOfSentence && writableTokens.Count > 0)
            {
                Assert.True(writableTokens[writableTokens.Count - 1].Id == tokenizer.EndOfSentenceId);
                Assert.True(writableTokens[writableTokens.Count - 1].Value == tokenizer.EndOfSentenceToken);

                if (normalized is not null)
                {
                    Assert.True(writableTokens[writableTokens.Count - 1].Offset.Equals(new Range(normalized.Length, normalized.Length)));
                }
                writableTokens.RemoveAt(writableTokens.Count - 1);
            }

            return (
                writableTokens.Select(t => t.Id),
                writableTokens.Select(t => t.Value),
                writableTokens.Select(t => t.Offset)
            );
        }

        private void Validate((IEnumerable<int> Ids, IEnumerable<string> Tokens, IEnumerable<Range> Offsets) extracted, int[] ids, string[] tokens, Range[] offsets)
        {
            Assert.Equal(ids, extracted.Ids);
            Assert.Equal(tokens, extracted.Tokens);
            Assert.Equal(offsets, extracted.Offsets);
        }

        /// <summary>
        /// _unigramTokenizerFromJson, the tokenizer created from the json file has the ids shifted by 1 compared to the tokenizer created from tokenizer.bpe.model file.
        /// </summary>
        /// <param name="ids"></param>
        /// <returns></returns>
        private int[] GetShiftedIds(int[] ids)
        {
            int[] shiftedIds = new int[ids.Length];
            foreach (int i in Enumerable.Range(0, ids.Length))
            {
                if (ids[i] == _unigramTokenizer.UnknownId)
                {
                    shiftedIds[i] = _unigramTokenizerFromJson.UnknownId;
                }
                else if (ids[i] == _unigramTokenizer.BeginningOfSentenceId)
                {
                    shiftedIds[i] = _unigramTokenizerFromJson.BeginningOfSentenceId;
                }
                else if (ids[i] == _unigramTokenizer.EndOfSentenceId)
                {
                    shiftedIds[i] = _unigramTokenizerFromJson.EndOfSentenceId;
                }
                else
                {
                    shiftedIds[i] = ids[i] + 1;
                }
            }

            return shiftedIds;
        }

        [Theory]
        [MemberData(nameof(UnigramTestData))]
        public void EncodeToTokensTest(string inputText, string normalizedText, string decodedString, int[] ids, string[] tokens, Range[] offsets)
        {
            int[] shiftedIds = GetShiftedIds(ids);

            Assert.True(decodedString is not null);  // to make the compiler happy
            IReadOnlyList<EncodedToken> result = _unigramTokenizer.EncodeToTokens(inputText, out string? normalized);
            (IEnumerable<int> Ids, IEnumerable<string> Tokens, IEnumerable<Range> Offsets) extracted = ExtractedIds(_unigramTokenizer, result, normalizedText, _unigramTokenizer.AddBeginningOfSentence, _unigramTokenizer.AddEndOfSentence);
            Validate(extracted, ids, tokens, offsets);

            result = _unigramTokenizerFromJson.EncodeToTokens(inputText, out normalized);
            extracted = ExtractedIds(_unigramTokenizerFromJson, result, normalizedText, _unigramTokenizerFromJson.AddBeginningOfSentence, _unigramTokenizerFromJson.AddEndOfSentence);
            Validate(extracted, shiftedIds, tokens, offsets);

            result = _unigramTokenizer.EncodeToTokens(inputText.AsSpan(), out normalized);
            extracted = ExtractedIds(_unigramTokenizer, result, normalizedText, _unigramTokenizer.AddBeginningOfSentence, _unigramTokenizer.AddEndOfSentence);
            Validate(extracted, ids, tokens, offsets);

            result = _unigramTokenizerFromJson.EncodeToTokens(inputText.AsSpan(), out normalized);
            extracted = ExtractedIds(_unigramTokenizerFromJson, result, normalizedText, _unigramTokenizerFromJson.AddBeginningOfSentence, _unigramTokenizerFromJson.AddEndOfSentence);
            Validate(extracted, shiftedIds, tokens, offsets);

            result = _unigramTokenizer.EncodeToTokens(inputText, out normalized, addBeginningOfSentence: true, addEndOfSentence: false);
            extracted = ExtractedIds(_unigramTokenizer, result, normalizedText, true, false);
            Validate(extracted, ids, tokens, offsets);

            result = _unigramTokenizerFromJson.EncodeToTokens(inputText, out normalized, addBeginningOfSentence: true, addEndOfSentence: false);
            extracted = ExtractedIds(_unigramTokenizerFromJson, result, normalizedText, true, false);
            Validate(extracted, shiftedIds, tokens, offsets);

            result = _unigramTokenizer.EncodeToTokens(inputText.AsSpan(), out normalized, addBeginningOfSentence: true, addEndOfSentence: false);
            extracted = ExtractedIds(_unigramTokenizer, result, normalizedText, true, false);
            Validate(extracted, ids, tokens, offsets);

            result = _unigramTokenizerFromJson.EncodeToTokens(inputText.AsSpan(), out normalized, addBeginningOfSentence: true, addEndOfSentence: false);
            extracted = ExtractedIds(_unigramTokenizerFromJson, result, normalizedText, true, false);
            Validate(extracted, shiftedIds, tokens, offsets);

            result = _unigramTokenizer.EncodeToTokens(inputText, out normalized, addBeginningOfSentence: true, addEndOfSentence: true);
            extracted = ExtractedIds(_unigramTokenizer, result, normalizedText, true, true);
            Validate(extracted, ids, tokens, offsets);

            result = _unigramTokenizerFromJson.EncodeToTokens(inputText, out normalized, addBeginningOfSentence: true, addEndOfSentence: true);
            extracted = ExtractedIds(_unigramTokenizerFromJson, result, normalizedText, true, true);
            Validate(extracted, shiftedIds, tokens, offsets);

            result = _unigramTokenizer.EncodeToTokens(inputText.AsSpan(), out normalized, addBeginningOfSentence: true, addEndOfSentence: true);
            extracted = ExtractedIds(_unigramTokenizer, result, normalizedText, true, true);
            Validate(extracted, ids, tokens, offsets);

            result = _unigramTokenizerFromJson.EncodeToTokens(inputText.AsSpan(), out normalized, addBeginningOfSentence: true, addEndOfSentence: true);
            extracted = ExtractedIds(_unigramTokenizerFromJson, result, normalizedText, true, true);
            Validate(extracted, shiftedIds, tokens, offsets);

            string newString = $"{_unigramTokenizer.BeginningOfSentenceToken}{inputText}<pad>{inputText}{_unigramTokenizer.EndOfSentenceToken}";
            result = _unigramTokenizerWithSpecialTokens.EncodeToTokens(newString, out normalized, addBeginningOfSentence: false, addEndOfSentence: false);
            extracted = ExtractedIds(_unigramTokenizerWithSpecialTokens, result, normalizedText, false, false);

            int[] expectedIds = new int[ids.Length * 2 + 3];
            expectedIds[0] = _unigramTokenizerWithSpecialTokens.BeginningOfSentenceId;
            Array.Copy(ids, 0, expectedIds, 1, ids.Length);
            expectedIds[ids.Length + 1] = _unigramTokenizerWithSpecialTokens.SpecialTokens!["<pad>"];
            Array.Copy(ids, 0, expectedIds, ids.Length + 2, ids.Length);
            expectedIds[ids.Length * 2 + 2] = _unigramTokenizerWithSpecialTokens.EndOfSentenceId;
            Assert.Equal(expectedIds, extracted.Ids);

            string[] expectedTokens = new string[tokens.Length * 2 + 3];
            expectedTokens[0] = _unigramTokenizerWithSpecialTokens.BeginningOfSentenceToken;
            Array.Copy(tokens, 0, expectedTokens, 1, tokens.Length);
            expectedTokens[tokens.Length + 1] = "<pad>";
            Array.Copy(tokens, 0, expectedTokens, tokens.Length + 2, tokens.Length);
            expectedTokens[tokens.Length * 2 + 2] = _unigramTokenizerWithSpecialTokens.EndOfSentenceToken;
            Assert.Equal(expectedTokens, extracted.Tokens);

            newString = $"{_unigramTokenizerFromJson.BeginningOfSentenceToken}{inputText}<pad>{inputText}{_unigramTokenizerFromJson.EndOfSentenceToken}";
            result = _unigramTokenizerFromJson.EncodeToTokens(newString, out normalized, addBeginningOfSentence: false, addEndOfSentence: false);
            extracted = ExtractedIds(_unigramTokenizerFromJson, result, normalizedText, false, false);

            expectedIds = new int[ids.Length * 2 + 3];
            expectedIds[0] = _unigramTokenizerFromJson.BeginningOfSentenceId;
            Array.Copy(shiftedIds, 0, expectedIds, 1, shiftedIds.Length);
            expectedIds[shiftedIds.Length + 1] = _unigramTokenizerFromJson.SpecialTokens!["<pad>"];
            Array.Copy(shiftedIds, 0, expectedIds, shiftedIds.Length + 2, shiftedIds.Length);
            expectedIds[shiftedIds.Length * 2 + 2] = _unigramTokenizerFromJson.EndOfSentenceId;
            Assert.Equal(expectedIds, extracted.Ids);

            expectedTokens = new string[tokens.Length * 2 + 3];
            expectedTokens[0] = _unigramTokenizerFromJson.BeginningOfSentenceToken;
            Array.Copy(tokens, 0, expectedTokens, 1, tokens.Length);
            expectedTokens[tokens.Length + 1] = "<pad>";
            Array.Copy(tokens, 0, expectedTokens, tokens.Length + 2, tokens.Length);
            expectedTokens[tokens.Length * 2 + 2] = _unigramTokenizerFromJson.EndOfSentenceToken;
            Assert.Equal(expectedTokens, extracted.Tokens);
        }

        [Theory]
        [MemberData(nameof(UnigramTestData))]
        public void EncodeToIdsTest(string inputText, string normalizedText, string decodedString, int[] ids, string[] tokens, Range[] offsets)
        {
            int[] shiftedIds = GetShiftedIds(ids);

            Assert.True(decodedString is not null);  // to make the compiler happy
            Assert.True(tokens is not null);  // to make the compiler happy
            Assert.True(offsets is not null); // to make the compiler happy

            IReadOnlyList<int> result = _unigramTokenizer.EncodeToIds(inputText, addBeginningOfSentence: false, addEndOfSentence: false);
            Assert.Equal(ids, result);
            result = _unigramTokenizer.EncodeToIds(inputText.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false);
            Assert.Equal(ids, result);

            result = _unigramTokenizerFromJson.EncodeToIds(inputText, addBeginningOfSentence: false, addEndOfSentence: false);
            Assert.Equal(shiftedIds, result);
            result = _unigramTokenizerFromJson.EncodeToIds(inputText.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false);
            Assert.Equal(shiftedIds, result);

            result = _unigramTokenizer.EncodeToIds(inputText, addBeginningOfSentence: true, addEndOfSentence: false);
            List<int> ints = result is List<int> list ? list : result.ToList();
            if (ints.Count > 0)
            {
                ints.RemoveAt(0);
            }
            Assert.Equal(ids, ints);

            result = _unigramTokenizerFromJson.EncodeToIds(inputText, addBeginningOfSentence: true, addEndOfSentence: false);
            ints = result is List<int> list1 ? list1 : result.ToList();
            if (ints.Count > 0)
            {
                ints.RemoveAt(0);
            }
            Assert.Equal(shiftedIds, ints);

            result = _unigramTokenizer.EncodeToIds(inputText.AsSpan(), addBeginningOfSentence: true, addEndOfSentence: false);
            ints = result is List<int> ? (List<int>)result : result.ToList();
            if (ints.Count > 0)
            {
                ints.RemoveAt(0);
            }
            Assert.Equal(ids, ints);

            result = _unigramTokenizerFromJson.EncodeToIds(inputText.AsSpan(), addBeginningOfSentence: true, addEndOfSentence: false);
            ints = result is List<int> ? (List<int>)result : result.ToList();
            if (ints.Count > 0)
            {
                ints.RemoveAt(0);
            }
            Assert.Equal(shiftedIds, ints);

            result = _unigramTokenizer.EncodeToIds(inputText, addBeginningOfSentence: true, addEndOfSentence: true);
            ints = result is List<int> ? (List<int>)result : result.ToList();
            if (ints.Count > 0)
            {
                ints.RemoveAt(0);
                ints.RemoveAt(ints.Count - 1);
            }
            Assert.Equal(ids, ints);

            result = _unigramTokenizerFromJson.EncodeToIds(inputText, addBeginningOfSentence: true, addEndOfSentence: true);
            ints = result is List<int> ? (List<int>)result : result.ToList();
            if (ints.Count > 0)
            {
                ints.RemoveAt(0);
                ints.RemoveAt(ints.Count - 1);
            }
            Assert.Equal(shiftedIds, ints);

            result = _unigramTokenizer.EncodeToIds(inputText.AsSpan(), addBeginningOfSentence: true, addEndOfSentence: true);
            ints = result is List<int> ? (List<int>)result : result.ToList();
            if (ints.Count > 0)
            {
                ints.RemoveAt(0);
                ints.RemoveAt(ints.Count - 1);
            }
            Assert.Equal(ids, ints);

            result = _unigramTokenizerFromJson.EncodeToIds(inputText.AsSpan(), addBeginningOfSentence: true, addEndOfSentence: true);
            ints = result is List<int> ? (List<int>)result : result.ToList();
            if (ints.Count > 0)
            {
                ints.RemoveAt(0);
                ints.RemoveAt(ints.Count - 1);
            }
            Assert.Equal(shiftedIds, ints);

            for (int i = 1; i <= ids.Length; i++)
            {
                result = _unigramTokenizer.EncodeToIds(inputText, addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: i, out string? normalized, out int charConsumed);
                Assert.Equal(ids.Take(i), result);
                Assert.Equal(normalizedText, normalized);

                result = _unigramTokenizerFromJson.EncodeToIds(inputText, addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: i, out normalized, out charConsumed);
                Assert.Equal(shiftedIds.Take(i), result);
                Assert.Equal(normalizedText, normalized);

                result = _unigramTokenizer.EncodeToIds(inputText.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: i, out normalized, out charConsumed);
                Assert.Equal(ids.Take(i), result);
                Assert.Equal(normalizedText, normalized);

                result = _unigramTokenizerFromJson.EncodeToIds(inputText.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: i, out normalized, out charConsumed);
                Assert.Equal(shiftedIds.Take(i), result);
                Assert.Equal(normalizedText, normalized);

                result = _unigramTokenizer.EncodeToIds(inputText, addBeginningOfSentence: true, addEndOfSentence: true, maxTokenCount: i, out normalized, out charConsumed);
                ints = result is List<int> ? (List<int>)result : result.ToList();
                if (ints.Count > 0)
                {
                    ints.RemoveAt(0);
                }
                if (ints.Count > ids.Length)
                {
                    ints.RemoveAt(ints.Count - 1);
                }
                Assert.Equal(ids.Take(i - 1), ints); // Exclude the counted BoS token
                if (normalized is not null)
                {
                    Assert.Equal(normalizedText, normalized);
                }

                result = _unigramTokenizerFromJson.EncodeToIds(inputText, addBeginningOfSentence: true, addEndOfSentence: true, maxTokenCount: i, out normalized, out charConsumed);
                ints = result is List<int> ? (List<int>)result : result.ToList();
                if (ints.Count > 0)
                {
                    ints.RemoveAt(0);
                }
                if (ints.Count > shiftedIds.Length)
                {
                    ints.RemoveAt(ints.Count - 1);
                }
                Assert.Equal(shiftedIds.Take(i - 1), ints); // Exclude the counted BoS token
                if (normalized is not null)
                {
                    Assert.Equal(normalizedText, normalized);
                }

                result = _unigramTokenizer.EncodeToIds(inputText.AsSpan(), addBeginningOfSentence: true, addEndOfSentence: true, maxTokenCount: i, out normalized, out charConsumed);
                ints = result is List<int> ? (List<int>)result : result.ToList();
                if (ints.Count > 0)
                {
                    ints.RemoveAt(0);
                }
                if (ints.Count > ids.Length)
                {
                    ints.RemoveAt(ints.Count - 1);
                }
                Assert.Equal(ids.Take(i - 1), ints); // Exclude the counted BoS token
                if (normalized is not null)
                {
                    Assert.Equal(normalizedText, normalized);
                }

                result = _unigramTokenizerFromJson.EncodeToIds(inputText.AsSpan(), addBeginningOfSentence: true, addEndOfSentence: true, maxTokenCount: i, out normalized, out charConsumed);
                ints = result is List<int> ? (List<int>)result : result.ToList();
                if (ints.Count > 0)
                {
                    ints.RemoveAt(0);
                }
                if (ints.Count > shiftedIds.Length)
                {
                    ints.RemoveAt(ints.Count - 1);
                }
                Assert.Equal(shiftedIds.Take(i - 1), ints); // Exclude the counted BoS token
                if (normalized is not null)
                {
                    Assert.Equal(normalizedText, normalized);
                }
            }

            inputText = $"{_unigramTokenizerWithSpecialTokens.BeginningOfSentenceToken}{inputText}<pad>{inputText}{_unigramTokenizerWithSpecialTokens.EndOfSentenceToken}";
            int[] expectedIds = new int[ids.Length * 2 + 3];
            expectedIds[0] = _unigramTokenizerWithSpecialTokens.BeginningOfSentenceId;
            Array.Copy(ids, 0, expectedIds, 1, ids.Length);
            expectedIds[ids.Length + 1] = _unigramTokenizerWithSpecialTokens.SpecialTokens!["<pad>"];
            Array.Copy(ids, 0, expectedIds, ids.Length + 2, ids.Length);
            expectedIds[ids.Length * 2 + 2] = _unigramTokenizerWithSpecialTokens.EndOfSentenceId;
            string expectedNormalized = $"{_unigramTokenizerWithSpecialTokens.BeginningOfSentenceToken}{normalizedText}<pad>{normalizedText}{_unigramTokenizerWithSpecialTokens.EndOfSentenceToken}";

            for (int i = 1; i <= expectedIds.Length; i++)
            {
                result = _unigramTokenizerWithSpecialTokens.EncodeToIds(inputText, addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: i, out string? normalized, out int charConsumed);
                Assert.Equal(expectedIds.Take(i), result);
                Assert.Equal(expectedNormalized, normalized);

                result = _unigramTokenizerWithSpecialTokens.EncodeToIds(inputText.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: i, out normalized, out charConsumed);
                Assert.Equal(expectedIds.Take(i), result);
                Assert.Equal(expectedNormalized, normalized);
            }

            expectedIds = new int[shiftedIds.Length * 2 + 3];
            expectedIds[0] = _unigramTokenizerFromJson.BeginningOfSentenceId;
            Array.Copy(shiftedIds, 0, expectedIds, 1, shiftedIds.Length);
            expectedIds[shiftedIds.Length + 1] = _unigramTokenizerFromJson.SpecialTokens!["<pad>"];
            Array.Copy(shiftedIds, 0, expectedIds, shiftedIds.Length + 2, shiftedIds.Length);
            expectedIds[shiftedIds.Length * 2 + 2] = _unigramTokenizerFromJson.EndOfSentenceId;
            expectedNormalized = $"{_unigramTokenizerFromJson.BeginningOfSentenceToken}{normalizedText}<pad>{normalizedText}{_unigramTokenizerFromJson.EndOfSentenceToken}";

            for (int i = 1; i <= expectedIds.Length; i++)
            {
                result = _unigramTokenizerFromJson.EncodeToIds(inputText, addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: i, out string? normalized, out int charConsumed);
                Assert.Equal(expectedIds.Take(i), result);
                Assert.Equal(expectedNormalized, normalized);

                result = _unigramTokenizerFromJson.EncodeToIds(inputText.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: i, out normalized, out charConsumed);
                Assert.Equal(expectedIds.Take(i), result);
                Assert.Equal(expectedNormalized, normalized);
            }
        }

        [Theory]
        [MemberData(nameof(UnigramTestData))]
        public void GetIndexByTokenCountTest(string inputText, string normalizedText, string decodedString, int[] ids, string[] tokens, Range[] offsets)
        {
            Assert.True(decodedString is not null);  // to make the compiler happy
            Assert.True(tokens is not null);  // to make the compiler happy
            Assert.True(offsets is not null); // to make the compiler happy

            int[] shiftedIds = GetShiftedIds(ids);
            int totalTokens = ids.Length;

            for (int i = 1; i <= totalTokens; i++)
            {
                int index = _unigramTokenizer.GetIndexByTokenCount(inputText, addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: 1, out string? normalized, out int charConsumed);
                Assert.Equal(normalizedText, normalized);
                IReadOnlyList<int> ids1 = _unigramTokenizer.EncodeToIds(normalized!.Substring(0, index), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false);
                IReadOnlyList<int> ids2 = index < normalized.Length ? _unigramTokenizer.EncodeToIds(normalized!.Substring(index), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false) : new List<int>();
                Assert.Equal(ids, ids1.Concat(ids2).ToList());

                index = _unigramTokenizerFromJson.GetIndexByTokenCount(inputText, addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: 1, out normalized, out charConsumed);
                Assert.Equal(normalizedText, normalized);
                ids1 = _unigramTokenizerFromJson.EncodeToIds(normalized!.Substring(0, index), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false);
                ids2 = index < normalized.Length ? _unigramTokenizerFromJson.EncodeToIds(normalized!.Substring(index), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false) : new List<int>();
                Assert.Equal(shiftedIds, ids1.Concat(ids2).ToList());

                index = _unigramTokenizer.GetIndexByTokenCount(inputText.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: 1, out normalized, out charConsumed);
                Assert.Equal(normalizedText, normalized);
                ids1 = _unigramTokenizer.EncodeToIds(normalized!.Substring(0, index).AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false);
                ids2 = index < normalized.Length ? _unigramTokenizer.EncodeToIds(normalized!.Substring(index).AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false) : new List<int>();
                Assert.Equal(ids, ids1.Concat(ids2).ToList());

                index = _unigramTokenizerFromJson.GetIndexByTokenCount(inputText.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: 1, out normalized, out charConsumed);
                Assert.Equal(normalizedText, normalized);
                ids1 = _unigramTokenizerFromJson.EncodeToIds(normalized!.Substring(0, index).AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false);
                ids2 = index < normalized.Length ? _unigramTokenizerFromJson.EncodeToIds(normalized!.Substring(index).AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false) : new List<int>();
                Assert.Equal(shiftedIds, ids1.Concat(ids2).ToList());

                index = _unigramTokenizer.GetIndexByTokenCountFromEnd(inputText, addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: 1, considerNormalization: true, out normalized, out charConsumed);
                Assert.Equal(normalizedText, normalized);
                ids1 = _unigramTokenizer.EncodeToIds(normalized!.Substring(0, index), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false);
                ids2 = index < normalized.Length ? _unigramTokenizer.EncodeToIds(normalized!.Substring(index), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false) : new List<int>();
                Assert.Equal(ids, ids1.Concat(ids2).ToList());

                index = _unigramTokenizerFromJson.GetIndexByTokenCountFromEnd(inputText, addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: 1, considerNormalization: true, out normalized, out charConsumed);
                Assert.Equal(normalizedText, normalized);
                ids1 = _unigramTokenizerFromJson.EncodeToIds(normalized!.Substring(0, index), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false);
                ids2 = index < normalized.Length ? _unigramTokenizerFromJson.EncodeToIds(normalized!.Substring(index), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false) : new List<int>();
                Assert.Equal(shiftedIds, ids1.Concat(ids2).ToList());

                index = _unigramTokenizer.GetIndexByTokenCountFromEnd(inputText.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: 1, considerNormalization: true, out normalized, out charConsumed);
                Assert.Equal(normalizedText, normalized);
                ids1 = _unigramTokenizer.EncodeToIds(normalized!.Substring(0, index).AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false);
                ids2 = index < normalized.Length ? _unigramTokenizer.EncodeToIds(normalized!.Substring(index).AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false) : new List<int>();
                Assert.Equal(ids, ids1.Concat(ids2).ToList());

                index = _unigramTokenizerFromJson.GetIndexByTokenCountFromEnd(inputText.AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, maxTokenCount: 1, considerNormalization: true, out normalized, out charConsumed);
                Assert.Equal(normalizedText, normalized);
                ids1 = _unigramTokenizerFromJson.EncodeToIds(normalized!.Substring(0, index).AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false);
                ids2 = index < normalized.Length ? _unigramTokenizerFromJson.EncodeToIds(normalized!.Substring(index).AsSpan(), addBeginningOfSentence: false, addEndOfSentence: false, considerNormalization: false) : new List<int>();
                Assert.Equal(shiftedIds, ids1.Concat(ids2).ToList());
            }
        }

        [Theory]
        [MemberData(nameof(UnigramTestData))]
        public void DecodeTest(string inputText, string normalizedText, string decodedString, int[] ids, string[] tokens, Range[] offsets)
        {
            Assert.True(tokens is not null);  // to make the compiler happy
            Assert.True(offsets is not null); // to make the compiler happy
            Assert.True(inputText is not null);  // to make the compiler happy
            Assert.True(normalizedText is not null);  // to make the compiler happy

            DecodeWithTokenizerTest(_unigramTokenizer, decodedString, ids);
            DecodeWithTokenizerTest(_unigramTokenizerFromJson, decodedString, GetShiftedIds(ids));
        }

        private static void DecodeWithTokenizerTest(SentencePieceTokenizer tokenizer, string decodedString, int[] ids)
        {
            string result = tokenizer.Decode(ids, considerSpecialTokens: false);
            Assert.Equal(decodedString, result);

            char[] buffer = new char[decodedString.Length];

            OperationStatus status = tokenizer.Decode(ids, buffer, considerSpecialTokens: false, out int idsConsumed, out int charsWritten);
            Assert.Equal(OperationStatus.Done, status);
            Assert.Equal(ids.Length, idsConsumed);
            Assert.Equal(decodedString, buffer.AsSpan().Slice(0, charsWritten).ToString());

            for (int i = 0; i < decodedString.Length - 1; i++)
            {
                status = tokenizer.Decode(ids, buffer.AsSpan().Slice(0, i), considerSpecialTokens: false, out idsConsumed, out charsWritten);
                Assert.Equal(OperationStatus.DestinationTooSmall, status);
                Assert.Equal(decodedString.AsSpan().Slice(0, charsWritten).ToString(), buffer.AsSpan().Slice(0, charsWritten).ToString());
            }
        }

        [Fact]
        public void SpecialTokensTest()
        {
            Assert.Equal("<unk>", _unigramTokenizer.UnknownToken);
            Assert.Equal(0, _unigramTokenizer.UnknownId);
            Assert.Equal("<s>", _unigramTokenizer.BeginningOfSentenceToken);
            Assert.Equal(1, _unigramTokenizer.BeginningOfSentenceId);
            Assert.Equal("</s>", _unigramTokenizer.EndOfSentenceToken);
            Assert.Equal(2, _unigramTokenizer.EndOfSentenceId);
        }

        [Fact]
        public void JsonTokenizerSpecialTokensTest()
        {
            Assert.Equal("<unk>", _unigramTokenizerFromJson.UnknownToken);
            Assert.Equal(3, _unigramTokenizerFromJson.UnknownId);
            Assert.Equal("<s>", _unigramTokenizerFromJson.BeginningOfSentenceToken);
            Assert.Equal(0, _unigramTokenizerFromJson.BeginningOfSentenceId);
            Assert.Equal("</s>", _unigramTokenizerFromJson.EndOfSentenceToken);
            Assert.Equal(2, _unigramTokenizerFromJson.EndOfSentenceId);

            var specialTokens = new Dictionary<string, int>
            {
                { "<s>",    0       },
                { "<pad>",  1       },
                { "</s>",   2       },
                { "<unk>",  3       },
                { "<mask>", 250001  }
            };

            Assert.Equal(specialTokens, _unigramTokenizerFromJson.SpecialTokens);
            Assert.Equal(0, _unigramTokenizerFromJson.Vocabulary["<s>"]);
            Assert.Equal(1, _unigramTokenizerFromJson.Vocabulary["<pad>"]);
            Assert.Equal(2, _unigramTokenizerFromJson.Vocabulary["</s>"]);
            Assert.Equal(3, _unigramTokenizerFromJson.Vocabulary["<unk>"]);
            Assert.Equal(250001, _unigramTokenizerFromJson.Vocabulary["<mask>"]);
        }
    }
}
