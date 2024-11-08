// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class CodeGenTests
    {
        private static Tokenizer _codegen350MMonoTokenizer = CreateCodegen350MMonoTokenizer();
        private static Tokenizer _codegen350MMonoTokenizerWithSpace = CreateCodegen350MMonoTokenizer(addPrefixSpace: true);
        private static Tokenizer _codegen350MMonoTokenizerWithBeginningOfSentence = CreateCodegen350MMonoTokenizer(bos: true);
        private static Tokenizer _codegen350MMonoTokenizerWithEndOfSentence = CreateCodegen350MMonoTokenizer(eos: true);
        private static Tokenizer _codegen350MMonoTokenizerWithBeginningAndEndOfSentence = CreateCodegen350MMonoTokenizer(bos: true, eos: true);

        private static Tokenizer CreateCodegen350MMonoTokenizer(bool addPrefixSpace = false, bool bos = false, bool eos = false)
        {
            // @"https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/vocab.json?download=true";
            // @"https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/merges.txt?download=true";

            using Stream vocabStream = File.OpenRead(Path.Combine(@"Codegen-350M-mono", "vocab.json"));
            using Stream mergesStream = File.OpenRead(Path.Combine(@"Codegen-350M-mono", "merges.txt"));

            return CodeGenTokenizer.Create(vocabStream, mergesStream, addPrefixSpace, bos, eos);
        }

        private static Tokenizer CreateCodegenPhi2Tokenizer()
        {
            // https://huggingface.co/microsoft/phi-2/resolve/main/vocab.json?download=true
            // https://huggingface.co/microsoft/phi-2/resolve/main/merges.txt?download=true

            using Stream vocabStream = File.OpenRead(Path.Combine(@"Phi-2", "vocab.json"));
            using Stream mergesStream = File.OpenRead(Path.Combine(@"Phi-2", "merges.txt"));

            return CodeGenTokenizer.Create(vocabStream, mergesStream);
        }

        public static IEnumerable<object?[]> CodeGenTestData
        {
            get
            {
                // string to tokenize,
                // produced tokens,
                // the token offsets,
                // the tokens ids, produced tokens when AddPrefixSpace is enabled,
                // the token offsets when AddPrefixSpace is enabled,
                // the tokens ids when AddPrefixSpace is enabled
                yield return new object?[]
                {
                    "Hello World",
                    new string[] { "Hello", "ĠWorld" },
                    new (int Index, int Length)[] { (0, 5), (5, 6) },
                    new int[] { 15496, 2159 },
                    new string[] { "ĠHello", "ĠWorld" },
                    new (int Index, int Length)[] { (0, 5), (5, 6) },
                    new int[] { 18435, 2159 },
                };

                yield return new object?[]
                {
                    " Hello World", // with space prefix this depends on the SpecialTokens
                    new string[] { "ĠHello", "ĠWorld" },
                    new (int Index, int Length)[] { (0, 6), (6, 6) },
                    new int[] { 18435, 2159 },
                    new string[] { "  ", "Hello", "ĠWorld" },
                    new (int Index, int Length)[] { (0, 1), (1, 5), (6, 6) },
                    new int[] { 50286, 15496, 2159 },
                };

                yield return new object?[]
                {
                    "the brown fox jumped over the lazy dog!\r\n", // text in range 0 ~ FF
                    new string[] { "the", "Ġbrown", "Ġfox", "Ġjumped", "Ġover", "Ġthe", "Ġlazy", "Ġdog", "!", "č", "Ċ" },
                    new (int Index, int Length)[] { (0, 3), (3, 6), (9, 4), (13, 7), (20, 5), (25, 4), (29, 5), (34, 4), (38, 1), (39, 1), (40, 1) },
                    new int[] { 1169, 7586, 21831, 11687, 625, 262, 16931, 3290, 0, 201, 198 },
                    new string[] { "Ġthe", "Ġbrown", "Ġfox", "Ġjumped", "Ġover", "Ġthe", "Ġlazy", "Ġdog", "!", "č", "Ċ" },
                    new (int Index, int Length)[] { (0, 3), (3, 6), (9, 4), (13, 7), (20, 5), (25, 4), (29, 5), (34, 4), (38, 1), (39, 1), (40, 1) },
                    new int[] { 262, 7586, 21831, 11687, 625, 262, 16931, 3290, 0, 201, 198 }
                };

                yield return new object?[]
                {
                    "\u0924\u1009\u1129\u1241\uE860\u3438.", // text greater than 7FF Devanagari, Myanmar, Hangul, Ethiopic, Palmyrene, CJK तဉᄩቁ㐸.
                    new string[] { "à¤", "¤", "á", "Ģ", "ī", "á", "Ħ", "©", "á", "ī", "ģ", "î", "¡", "ł", "ã", "Ĳ", "¸", "." },
                    new (int Index, int Length)[] { (0, 0), (0, 1), (1, 0), (1, 0), (1, 1), (2, 0), (2, 0), (2, 1), (3, 0), (3, 0), (3, 1), (4, 0), (4, 0), (4, 1), (5, 0), (5, 0), (5, 1), (6, 1) },
                    new int[] { 11976, 97, 157, 222, 231, 157, 226, 102, 157, 231, 223, 170, 94, 254, 159, 238, 116, 13 },
                    new string[] { "Ġà¤", "¤", "á", "Ģ", "ī", "á", "Ħ", "©", "á", "ī", "ģ", "î", "¡", "ł", "ã", "Ĳ", "¸", "." },
                    new (int Index, int Length)[] { (0, 0), (0, 1), (1, 0), (1, 0), (1, 1), (2, 0), (2, 0), (2, 1), (3, 0), (3, 0), (3, 1), (4, 0), (4, 0), (4, 1), (5, 0), (5, 0), (5, 1), (6, 1) },
                    new int[] { 28225, 97, 157, 222, 231, 157, 226, 102, 157, 231, 223, 170, 94, 254, 159, 238, 116, 13 }
                };

                yield return new object?[]
                {
                    "Some Greek letters ΣΦΩ αβγδε.", // text in range 100 ~ 7FF
                    new string[] { "Some", "ĠGreek", "Ġletters", "ĠÎ", "£", "Î", "¦", "Î", "©", "ĠÎ±", "Î²", "Î³", "Î", "´",  "Îµ", "." },
                    new (int Index, int Length)[] { (0, 4), (4, 6), (10, 8), (18, 1), (19, 1), (20, 0), (20, 1), (21, 0), (21, 1), (22, 2), (24, 1), (25, 1), (26, 0), (26, 1), (27, 1), (28, 1) },
                    new int[] { 4366, 8312, 7475, 7377, 96, 138, 99, 138, 102, 26367, 26638, 42063, 138, 112, 30950, 13 },
                    new string[] { "ĠSome", "ĠGreek", "Ġletters", "ĠÎ", "£", "Î", "¦", "Î", "©", "ĠÎ±", "Î²", "Î³", "Î", "´",  "Îµ", "." },
                    new (int Index, int Length)[] { (0, 4), (4, 6), (10, 8), (18, 1), (19, 1), (20, 0), (20, 1), (21, 0), (21, 1), (22, 2), (24, 1), (25, 1), (26, 0), (26, 1), (27, 1), (28, 1) },
                    new int[] { 2773, 8312, 7475, 7377, 96, 138, 99, 138, 102, 26367, 26638, 42063, 138, 112, 30950, 13 }
                };

                yield return new object?[]
                {
                    "αβγδε", // no spaces
                    new string[] { "Î±", "Î²", "Î³", "Î", "´",  "Îµ" },
                    new (int Index, int Length)[] { (0, 1), (1, 1), (2, 1), (3, 0), (3, 1), (4, 1) },
                    new int[] { 17394, 26638, 42063, 138, 112, 30950 },
                    new string[] { "ĠÎ±", "Î²", "Î³", "Î", "´",  "Îµ" },
                    new (int Index, int Length)[] { (0, 1), (1, 1), (2, 1), (3, 0), (3, 1), (4, 1) },
                    new int[] { 26367, 26638, 42063, 138, 112, 30950 }
                };

                yield return new object?[]
                {
                    "Surrogates: 😀😂😍😘",
                    new string[] { "Sur", "rog", "ates", ":", "ĠðŁĺ", "Ģ", "ðŁĺ", "Ĥ", "ðŁĺ", "į", "ðŁĺ", "ĺ" },
                    new (int Index, int Length)[] { (0, 3), (3, 3), (6, 4), (10, 1), (11, 1), (12, 2), (14, 0), (14, 2), (16, 0), (16, 2), (18, 0), (18, 2) },
                    new int[] { 14214, 3828, 689, 25, 30325, 222, 47249, 224, 47249, 235, 47249, 246 },
                    new string[] { "ĠSur", "rog", "ates", ":", "ĠðŁĺ", "Ģ", "ðŁĺ", "Ĥ", "ðŁĺ", "į", "ðŁĺ", "ĺ" },
                    new (int Index, int Length)[] { (0, 3), (3, 3), (6, 4), (10, 1), (11, 1), (12, 2), (14, 0), (14, 2), (16, 0), (16, 2), (18, 0), (18, 2) },
                    new int[] { 4198, 3828, 689, 25, 30325, 222, 47249, 224, 47249, 235, 47249, 246 }
                };

                yield return new object?[]
                {
                    "Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides " +
                    "general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet...) for Natural " +
                    "Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained " +
                    "models in 100+ languages and deep interoperability between Jax, PyTorch and TensorFlow.",
                    new string[] { "Transform", "ers", "Ġ(", "formerly", "Ġknown", "Ġas", "Ġpy", "tor", "ch", "-", "transform", "ers", "Ġand", "Ġpy", "tor", "ch", "-",
                                    "pret", "rained", "-", "bert", ")", "Ġprovides", "Ġgeneral", "-", "purpose", "Ġarchitectures", "Ġ(", "BER", "T", ",", "ĠG", "PT",
                                    "-", "2", ",", "ĠRo", "BER", "Ta", ",", "ĠXL", "M", ",", "ĠDist", "il", "B", "ert", ",", "ĠXL", "Net", "...)", "Ġfor", "ĠNatural",
                                    "ĠLanguage", "ĠUnderstanding", "Ġ(", "NL", "U", ")", "Ġand", "ĠNatural", "ĠLanguage", "ĠGeneration", "Ġ(", "NL", "G", ")", "Ġwith",
                                    "Ġover", "Ġ32", "+", "Ġpret", "rained", "Ġmodels", "Ġin", "Ġ100", "+", "Ġlanguages", "Ġand", "Ġdeep", "Ġinteroper", "ability",
                                    "Ġbetween", "ĠJ", "ax", ",", "ĠPy", "Tor", "ch", "Ġand", "ĠT", "ensor", "Flow", "." },
                    new (int Index, int Length)[] { (0, 9), (9, 3), (12, 2), (14, 8), (22, 6), (28, 3), (31, 3), (34, 3), (37, 2), (39, 1), (40, 9), (49, 3), (52, 4),
                                    (56, 3), (59, 3), (62, 2), (64, 1), (65, 4), (69, 6), (75, 1), (76, 4), (80, 1), (81, 9), (90, 8), (98, 1), (99, 7), (106, 14),
                                    (120, 2), (122, 3), (125, 1), (126, 1), (127, 2), (129, 2), (131, 1), (132, 1), (133, 1), (134, 3), (137, 3), (140, 2), (142, 1),
                                    (143, 3), (146, 1), (147, 1), (148, 5), (153, 2), (155, 1), (156, 3), (159, 1), (160, 3), (163, 3), (166, 4), (170, 4), (174, 8),
                                    (182, 9), (191, 14), (205, 2), (207, 2), (209, 1), (210, 1), (211, 4), (215, 8), (223, 9), (232, 11), (243, 2), (245, 2), (247, 1),
                                    (248, 1), (249, 5), (254, 5), (259, 3), (262, 1), (263, 5), (268, 6), (274, 7), (281, 3), (284, 4), (288, 1), (289, 10), (299, 4),
                                    (303, 5), (308, 10), (318, 7), (325, 8), (333, 2), (335, 2), (337, 1), (338, 3), (341, 3), (344, 2), (346, 4), (350, 2), (352, 5),
                                    (357, 4), (361, 1) },
                    new int[] { 41762, 364, 357, 36234, 1900, 355, 12972, 13165, 354, 12, 35636, 364, 290, 12972, 13165, 354, 12, 5310, 13363, 12, 4835, 8, 3769, 2276,
                                    12, 29983, 45619, 357, 13246, 51, 11, 402, 11571, 12, 17, 11, 5564, 13246, 38586, 11, 16276, 44, 11, 4307, 346, 33, 861, 11, 16276,
                                    7934, 23029, 329, 12068, 15417, 28491, 357, 32572, 52, 8, 290, 12068, 15417, 16588, 357, 32572, 38, 8, 351, 625, 3933, 10, 2181, 13363,
                                    4981, 287, 1802, 10, 8950, 290, 2769, 48817, 1799, 1022, 449, 897, 11, 9485, 15884, 354, 290, 309, 22854, 37535, 13 },
                    new string[] { "ĠTransformers", "Ġ(", "formerly", "Ġknown", "Ġas", "Ġpy", "tor", "ch", "-", "transform", "ers", "Ġand", "Ġpy", "tor", "ch", "-",
                                    "pret", "rained", "-", "bert", ")", "Ġprovides", "Ġgeneral", "-", "purpose", "Ġarchitectures", "Ġ(", "BER", "T", ",", "ĠG", "PT",
                                    "-", "2", ",", "ĠRo", "BER", "Ta", ",", "ĠXL", "M", ",", "ĠDist", "il", "B", "ert", ",", "ĠXL", "Net", "...)", "Ġfor", "ĠNatural",
                                    "ĠLanguage", "ĠUnderstanding", "Ġ(", "NL", "U", ")", "Ġand", "ĠNatural", "ĠLanguage", "ĠGeneration", "Ġ(", "NL", "G", ")", "Ġwith",
                                    "Ġover", "Ġ32", "+", "Ġpret", "rained", "Ġmodels", "Ġin", "Ġ100", "+", "Ġlanguages", "Ġand", "Ġdeep", "Ġinteroper", "ability",
                                    "Ġbetween", "ĠJ", "ax", ",", "ĠPy", "Tor", "ch", "Ġand", "ĠT", "ensor", "Flow", "." },
                    new (int Index, int Length)[] { (0, 12), (12, 2), (14, 8), (22, 6), (28, 3), (31, 3), (34, 3), (37, 2), (39, 1), (40, 9), (49, 3), (52, 4),
                                    (56, 3), (59, 3), (62, 2), (64, 1), (65, 4), (69, 6), (75, 1), (76, 4), (80, 1), (81, 9), (90, 8), (98, 1), (99, 7), (106, 14),
                                    (120, 2), (122, 3), (125, 1), (126, 1), (127, 2), (129, 2), (131, 1), (132, 1), (133, 1), (134, 3), (137, 3), (140, 2), (142, 1),
                                    (143, 3), (146, 1), (147, 1), (148, 5), (153, 2), (155, 1), (156, 3), (159, 1), (160, 3), (163, 3), (166, 4), (170, 4), (174, 8),
                                    (182, 9), (191, 14), (205, 2), (207, 2), (209, 1), (210, 1), (211, 4), (215, 8), (223, 9), (232, 11), (243, 2), (245, 2), (247, 1),
                                    (248, 1), (249, 5), (254, 5), (259, 3), (262, 1), (263, 5), (268, 6), (274, 7), (281, 3), (284, 4), (288, 1), (289, 10), (299, 4),
                                    (303, 5), (308, 10), (318, 7), (325, 8), (333, 2), (335, 2), (337, 1), (338, 3), (341, 3), (344, 2), (346, 4), (350, 2), (352, 5),
                                    (357, 4), (361, 1) },
                    new int[] { 39185, 357, 36234, 1900, 355, 12972, 13165, 354, 12, 35636, 364, 290, 12972, 13165, 354, 12, 5310, 13363, 12, 4835, 8, 3769, 2276,
                                    12, 29983, 45619, 357, 13246, 51, 11, 402, 11571, 12, 17, 11, 5564, 13246, 38586, 11, 16276, 44, 11, 4307, 346, 33, 861, 11, 16276,
                                    7934, 23029, 329, 12068, 15417, 28491, 357, 32572, 52, 8, 290, 12068, 15417, 16588, 357, 32572, 38, 8, 351, 625, 3933, 10, 2181, 13363,
                                    4981, 287, 1802, 10, 8950, 290, 2769, 48817, 1799, 1022, 449, 897, 11, 9485, 15884, 354, 290, 309, 22854, 37535, 13 }
                };

                yield return new object?[]
                {
                    "BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly " +
                    "conditioning on both left and right context in all layers.",
                    new string[] { "BER", "T", "Ġis", "Ġdesigned", "Ġto", "Ġpre", "-", "train", "Ġdeep", "Ġbid", "irection", "al", "Ġrepresentations", "Ġfrom", "Ġunl",
                                    "abel", "ed", "Ġtext", "Ġby", "Ġjointly", "Ġconditioning", "Ġon", "Ġboth", "Ġleft", "Ġand", "Ġright", "Ġcontext", "Ġin", "Ġall",
                                    "Ġlayers", "." },
                    new (int Index, int Length)[] { (0, 3), (3, 1), (4, 3), (7, 9), (16, 3), (19, 4), (23, 1), (24, 5), (29, 5), (34, 4), (38, 8), (46, 2), (48, 16),
                                    (64, 5), (69, 4), (73, 4), (77, 2), (79, 5), (84, 3), (87, 8), (95, 13), (108, 3), (111, 5), (116, 5), (121, 4), (125, 6), (131, 8),
                                    (139, 3), (142, 4), (146, 7), (153, 1) },
                    new int[] { 13246, 51, 318, 3562, 284, 662, 12, 27432, 2769, 8406, 4154, 282, 24612, 422, 9642, 9608, 276, 2420, 416, 26913, 21143, 319, 1111, 1364, 290, 826, 4732, 287, 477, 11685, 13 },
                    new string[] { "ĠB", "ERT", "Ġis", "Ġdesigned", "Ġto", "Ġpre", "-", "train", "Ġdeep", "Ġbid", "irection", "al", "Ġrepresentations", "Ġfrom", "Ġunl",
                                    "abel", "ed", "Ġtext", "Ġby", "Ġjointly", "Ġconditioning", "Ġon", "Ġboth", "Ġleft", "Ġand", "Ġright", "Ġcontext", "Ġin", "Ġall",
                                    "Ġlayers", "." },
                    new (int Index, int Length)[] { (0, 1), (1, 3), (4, 3), (7, 9), (16, 3), (19, 4), (23, 1), (24, 5), (29, 5), (34, 4), (38, 8), (46, 2), (48, 16),
                                    (64, 5), (69, 4), (73, 4), (77, 2), (79, 5), (84, 3), (87, 8), (95, 13), (108, 3), (111, 5), (116, 5), (121, 4), (125, 6), (131, 8),
                                    (139, 3), (142, 4), (146, 7), (153, 1) },
                    new int[] { 347, 17395, 318, 3562, 284, 662, 12, 27432, 2769, 8406, 4154, 282, 24612, 422, 9642, 9608, 276, 2420, 416, 26913, 21143, 319, 1111, 1364, 290, 826, 4732, 287, 477, 11685, 13 }
                };

                yield return new object?[]
                {
                    "The quick brown fox jumps over the lazy dog.",
                    new string[] { "The", "Ġquick", "Ġbrown", "Ġfox", "Ġjumps", "Ġover", "Ġthe", "Ġlazy", "Ġdog", "." },
                    new (int Index, int Length)[] { (0, 3), (3, 6), (9, 6), (15, 4), (19, 6), (25, 5), (30, 4), (34, 5), (39, 4), (43, 1) },
                    new int[] { 464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13 },
                    new string[] { "ĠThe", "Ġquick", "Ġbrown", "Ġfox", "Ġjumps", "Ġover", "Ġthe", "Ġlazy", "Ġdog", "." },
                    new (int Index, int Length)[] { (0, 3), (3, 6), (9, 6), (15, 4), (19, 6), (25, 5), (30, 4), (34, 5), (39, 4), (43, 1) },
                    new int[] { 383, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13 }
                };
            }
        }

        [Theory]
        [MemberData(nameof(CodeGenTestData))]
        public void TestTokenizerEncoding(
                        string text,
                        string[] expectedTokens,
                        (int Index, int Length)[] expectedOffsets,
                        int[] expectedIds,
                        string[] expectedTokensWithSpace,
                        (int Index, int Length)[] expectedOffsetsWithSpace,
                        int[] expectedIdsWithSpace)
        {
            TestTokenizer(_codegen350MMonoTokenizer, text, expectedTokens, expectedOffsets, expectedIds, expectedTokensWithSpace, expectedOffsetsWithSpace, expectedIdsWithSpace);
            TestTokenizer(_codegen350MMonoTokenizerWithSpace, text, expectedTokens, expectedOffsets, expectedIds, expectedTokensWithSpace, expectedOffsetsWithSpace, expectedIdsWithSpace);

            Tokenizer phi2Tokenizer = CreateCodegenPhi2Tokenizer();
            TestTokenizer(phi2Tokenizer, text, expectedTokens, expectedOffsets, expectedIds, expectedTokensWithSpace, expectedOffsetsWithSpace, expectedIdsWithSpace);

            TestDecoding(_codegen350MMonoTokenizer, text);
            TestDecoding(_codegen350MMonoTokenizerWithSpace, text);
            TestDecoding(phi2Tokenizer, text);
        }

        private void ValidateEncoding(IReadOnlyList<EncodedToken> encoding, bool addPrefixSpace, string[] expectedTokens, (int Index, int Length)[] expectedOffsets, int[] expectedIds,
                                    string[] expectedTokensWithSpace, (int Index, int Length)[] expectedOffsetsWithSpace, int[] expectedIdsWithSpace)
        {
            if (addPrefixSpace)
            {
                Assert.Equal(expectedIdsWithSpace, encoding.Select(t => t.Id).ToArray());
                Assert.Equal(expectedTokensWithSpace, encoding.Select(t => t.Value).ToArray());
                Assert.Equal(expectedOffsetsWithSpace, encoding.Select(t => (t.Offset.Start.Value, t.Offset.End.Value - t.Offset.Start.Value)).ToArray());
            }
            else
            {
                Assert.Equal(expectedIds, encoding.Select(t => t.Id).ToArray());
                Assert.Equal(expectedTokens, encoding.Select(t => t.Value).ToArray());
                Assert.Equal(expectedOffsets, encoding.Select(t => (t.Offset.Start.Value, t.Offset.End.Value - t.Offset.Start.Value)).ToArray());
            }
        }

        private void TestDecoding(Tokenizer tokenizer, string text)
        {
            CodeGenTokenizer codeGenTokenizer = (tokenizer as CodeGenTokenizer)!;

            IReadOnlyList<EncodedToken> encoding = tokenizer.EncodeToTokens(text, out _);
            int[] ids = encoding.Select(t => t.Id).ToArray();
            Assert.Equal(text, tokenizer.Decode(ids));
            encoding = tokenizer.EncodeToTokens(text.AsSpan(), out _);
            ids = encoding.Select(t => t.Id).ToArray();
            Assert.Equal(text, tokenizer.Decode(ids));
            TestDecodingWithSpan(codeGenTokenizer, ids, codeGenTokenizer.AddPrefixSpace, considerSpecialTokens: false, text);

            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: codeGenTokenizer.AddPrefixSpace, addBeginningOfSentence: true, addEndOfSentence: false, out _);
            ids = encoding.Select(t => t.Id).ToArray();
            Assert.Equal(text, codeGenTokenizer.Decode(ids));
            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: codeGenTokenizer.AddPrefixSpace, addBeginningOfSentence: true, addEndOfSentence: false, out _);
            ids = encoding.Select(t => t.Id).ToArray();
            Assert.Equal(text, codeGenTokenizer.Decode(ids));
            TestDecodingWithSpan(codeGenTokenizer, ids, codeGenTokenizer.AddPrefixSpace, considerSpecialTokens: false, text);

            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: codeGenTokenizer.AddPrefixSpace, addBeginningOfSentence: false, addEndOfSentence: true, out _);
            ids = encoding.Select(t => t.Id).ToArray();
            Assert.Equal(text, codeGenTokenizer.Decode(ids));
            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: codeGenTokenizer.AddPrefixSpace, addBeginningOfSentence: false, addEndOfSentence: true, out _);
            ids = encoding.Select(t => t.Id).ToArray();
            Assert.Equal(text, codeGenTokenizer.Decode(ids));
            TestDecodingWithSpan(codeGenTokenizer, ids, codeGenTokenizer.AddPrefixSpace, considerSpecialTokens: false, text);

            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: codeGenTokenizer.AddPrefixSpace, addBeginningOfSentence: true, addEndOfSentence: true, out _);
            ids = encoding.Select(t => t.Id).ToArray();
            Assert.Equal(text, codeGenTokenizer.Decode(ids));
            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: codeGenTokenizer.AddPrefixSpace, addBeginningOfSentence: true, addEndOfSentence: true, out _);
            ids = encoding.Select(t => t.Id).ToArray();
            Assert.Equal(text, codeGenTokenizer.Decode(ids));
            TestDecodingWithSpan(codeGenTokenizer, ids, codeGenTokenizer.AddPrefixSpace, considerSpecialTokens: false, text);

            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: true, addBeginningOfSentence: true, addEndOfSentence: true, out _);
            ids = encoding.Select(t => t.Id).ToArray();
            Assert.Equal(text, codeGenTokenizer.Decode(ids, hasPrefixSpace: true, considerSpecialTokens: false));
            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: true, addBeginningOfSentence: true, addEndOfSentence: true, out _);
            ids = encoding.Select(t => t.Id).ToArray();
            Assert.Equal(text, codeGenTokenizer.Decode(ids, hasPrefixSpace: true, considerSpecialTokens: false));
            TestDecodingWithSpan(codeGenTokenizer, ids, hasPrefixSpace: true, considerSpecialTokens: false, text);

            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: true, addBeginningOfSentence: true, addEndOfSentence: true, out _);
            ids = encoding.Select(t => t.Id).ToArray();
            string targetText = $"{codeGenTokenizer.BeginningOfSentenceToken}{text}{codeGenTokenizer.EndOfSentenceToken}";
            Assert.Equal(targetText, codeGenTokenizer.Decode(ids, hasPrefixSpace: true, considerSpecialTokens: true));
            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: true, addBeginningOfSentence: true, addEndOfSentence: true, out _);
            ids = encoding.Select(t => t.Id).ToArray();
            Assert.Equal(targetText, codeGenTokenizer.Decode(ids, hasPrefixSpace: true, considerSpecialTokens: true));
            TestDecodingWithSpan(codeGenTokenizer, ids, hasPrefixSpace: true, considerSpecialTokens: true, targetText);
        }

        private void TestDecodingWithSpan(CodeGenTokenizer tokenizer, int[] ids, bool hasPrefixSpace, bool considerSpecialTokens, string expectedDecoded)
        {
            char[] destinationBuffer = new char[expectedDecoded.Length];

            OperationStatus status;
            int lastIdsConsumed = 0;
            int lastCharactersWritten = 0;
            int idsConsumed;
            int charactersWritten;

            for (int i = 1; i < destinationBuffer.Length - 1; i += Math.Max(1, destinationBuffer.Length - 3)) // enough to test length 1, and destinationBuffer.Length - 2 only.
            {
                status = tokenizer.Decode(ids, destinationBuffer.AsSpan().Slice(0, i), hasPrefixSpace, considerSpecialTokens, out idsConsumed, out charactersWritten);
                Assert.Equal(OperationStatus.DestinationTooSmall, status);
                Assert.True(idsConsumed < ids.Length);
                Assert.True(idsConsumed >= lastIdsConsumed);
                Assert.True(charactersWritten < expectedDecoded.Length);
                Assert.True(charactersWritten >= lastCharactersWritten);
                lastIdsConsumed = idsConsumed;
                lastCharactersWritten = charactersWritten;
            }

            status = tokenizer.Decode(ids, destinationBuffer.AsSpan(), hasPrefixSpace, considerSpecialTokens, out idsConsumed, out charactersWritten);
            Assert.Equal(OperationStatus.Done, status);
            Assert.Equal(ids.Length, idsConsumed);
            Assert.Equal(expectedDecoded.Length, charactersWritten);
            Assert.Equal(expectedDecoded, destinationBuffer.AsSpan().ToString());
        }

        private void TestTokenizer(
                        Tokenizer tokenizer,
                        string text,
                        string[] expectedTokens,
                        (int Index, int Length)[] expectedOffsets,
                        int[] expectedIds,
                        string[] expectedTokensWithSpace,
                        (int Index, int Length)[] expectedOffsetsWithSpace,
                        int[] expectedIdsWithSpace)
        {
            CodeGenTokenizer codeGenTokenizer = (tokenizer as CodeGenTokenizer)!;

            //
            // Full Encoding
            //

            IReadOnlyList<EncodedToken> encoding = tokenizer.EncodeToTokens(text, out _);
            ValidateEncoding(encoding, codeGenTokenizer.AddPrefixSpace, expectedTokens, expectedOffsets, expectedIds, expectedTokensWithSpace, expectedOffsetsWithSpace, expectedIdsWithSpace);

            encoding = tokenizer.EncodeToTokens(text.AsSpan(), out _);
            ValidateEncoding(encoding, codeGenTokenizer.AddPrefixSpace, expectedTokens, expectedOffsets, expectedIds, expectedTokensWithSpace, expectedOffsetsWithSpace, expectedIdsWithSpace);

            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out _);
            ValidateEncoding(encoding, addPrefixSpace: false, expectedTokens, expectedOffsets, expectedIds, expectedTokensWithSpace, expectedOffsetsWithSpace, expectedIdsWithSpace);

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out _);
            ValidateEncoding(encoding, addPrefixSpace: false, expectedTokens, expectedOffsets, expectedIds, expectedTokensWithSpace, expectedOffsetsWithSpace, expectedIdsWithSpace);

            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false, out _);
            ValidateEncoding(encoding, addPrefixSpace: true, expectedTokens, expectedOffsets, expectedIds, expectedTokensWithSpace, expectedOffsetsWithSpace, expectedIdsWithSpace);

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false, out _);
            ValidateEncoding(encoding, addPrefixSpace: true, expectedTokens, expectedOffsets, expectedIds, expectedTokensWithSpace, expectedOffsetsWithSpace, expectedIdsWithSpace);

            //
            // Encode To Ids
            //

            var ids = codeGenTokenizer.AddPrefixSpace ? expectedIdsWithSpace : expectedIds;

            Assert.Equal(ids, tokenizer.EncodeToIds(text));
            Assert.Equal(ids, tokenizer.EncodeToIds(text.AsSpan()));

            Assert.Equal(expectedIdsWithSpace, codeGenTokenizer.EncodeToIds(text, addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false));
            Assert.Equal(expectedIdsWithSpace, codeGenTokenizer.EncodeToIds(text.AsSpan(), addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false));
            Assert.Equal(expectedIds, codeGenTokenizer.EncodeToIds(text, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false));
            Assert.Equal(expectedIds, codeGenTokenizer.EncodeToIds(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false));

            Assert.Equal(ids, codeGenTokenizer.EncodeToIds(text, ids.Length, out string? normalizedText, out int length));
            Assert.Null(normalizedText);
            Assert.Equal(text.Length, length);
            Assert.Equal(ids, codeGenTokenizer.EncodeToIds(text.AsSpan(), ids.Length, out normalizedText, out length));
            Assert.Null(normalizedText);
            Assert.Equal(text.Length, length);

            Assert.Equal(expectedIds, codeGenTokenizer.EncodeToIds(text, expectedIds.Length, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out length));
            Assert.Null(normalizedText);
            Assert.Equal(text.Length, length);
            Assert.Equal(expectedIds, codeGenTokenizer.EncodeToIds(text.AsSpan(), expectedIds.Length, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out length));
            Assert.Null(normalizedText);
            Assert.Equal(text.Length, length);

            Assert.Equal(expectedIdsWithSpace, codeGenTokenizer.EncodeToIds(text, expectedIdsWithSpace.Length, addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out length));
            Assert.Null(normalizedText);
            Assert.Equal(text.Length, length);
            Assert.Equal(expectedIdsWithSpace, codeGenTokenizer.EncodeToIds(text.AsSpan(), expectedIdsWithSpace.Length, addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out length));
            Assert.Null(normalizedText);
            Assert.Equal(text.Length, length);

            int expectedTokensToExclude = expectedOffsets.Length > 1 && expectedOffsets[expectedOffsets.Length - 1].Index == expectedOffsets[expectedOffsets.Length - 2].Index ? 2 : 1;
            Assert.Equal(ids.Take(ids.Length - expectedTokensToExclude), codeGenTokenizer.EncodeToIds(text, ids.Length - 1, out normalizedText, out length));
            Assert.Null(normalizedText);
            var offsets = codeGenTokenizer.AddPrefixSpace ? expectedOffsetsWithSpace : expectedOffsets;
            int expectedLength = offsets.Length > expectedTokensToExclude ? offsets[offsets.Length - expectedTokensToExclude - 1].Index + offsets[offsets.Length - expectedTokensToExclude - 1].Length : 0;
            Assert.Equal(expectedLength, length);
            Assert.Equal(ids.Take(ids.Length - expectedTokensToExclude), codeGenTokenizer.EncodeToIds(text.AsSpan(), ids.Length - 1, out normalizedText, out length));
            Assert.Null(normalizedText);
            Assert.Equal(expectedLength, length);

            Assert.Equal(expectedIds.Take(expectedIds.Length - expectedTokensToExclude), codeGenTokenizer.EncodeToIds(text, expectedIds.Length - 1, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out length));
            Assert.Null(normalizedText);
            Assert.Equal(expectedLength, length);
            Assert.Equal(expectedIds.Take(expectedIds.Length - expectedTokensToExclude), codeGenTokenizer.EncodeToIds(text.AsSpan(), expectedIds.Length - 1, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out length));
            Assert.Null(normalizedText);
            Assert.Equal(expectedLength, length);

            Assert.Equal(expectedIdsWithSpace.Take(expectedIdsWithSpace.Length - expectedTokensToExclude), codeGenTokenizer.EncodeToIds(text, expectedIdsWithSpace.Length - 1, addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out length));
            Assert.Null(normalizedText);
            Assert.Equal(expectedLength, length);
            Assert.Equal(expectedIdsWithSpace.Take(expectedIdsWithSpace.Length - expectedTokensToExclude), codeGenTokenizer.EncodeToIds(text.AsSpan(), expectedIdsWithSpace.Length - 1, addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out length));
            Assert.Null(normalizedText);
            Assert.Equal(expectedLength, length);

            //
            // CountTokens
            //

            Assert.Equal(ids.Length, codeGenTokenizer.CountTokens(text));
            Assert.Equal(ids.Length, codeGenTokenizer.CountTokens(text.AsSpan()));

            Assert.Equal(expectedIds.Length, codeGenTokenizer.CountTokens(text, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false));
            Assert.Equal(expectedIds.Length, codeGenTokenizer.CountTokens(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false));

            Assert.Equal(expectedIdsWithSpace.Length, codeGenTokenizer.CountTokens(text, addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false));
            Assert.Equal(expectedIdsWithSpace.Length, codeGenTokenizer.CountTokens(text.AsSpan(), addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false));

            //
            // IndexOf
            //

            offsets = codeGenTokenizer.AddPrefixSpace ? expectedOffsetsWithSpace : expectedOffsets;

            Assert.Equal(offsets[offsets.Length - 1].Index + offsets[offsets.Length - 1].Length, codeGenTokenizer.GetIndexByTokenCount(text, ids.Length, out normalizedText, out int tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(ids.Length, tokenCount);
            Assert.Equal(offsets[offsets.Length - 1].Index + offsets[offsets.Length - 1].Length, codeGenTokenizer.GetIndexByTokenCount(text.AsSpan(), ids.Length, out normalizedText, out tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(ids.Length, tokenCount);

            Assert.Equal(expectedOffsets[expectedOffsets.Length - 1].Index + expectedOffsets[expectedOffsets.Length - 1].Length, codeGenTokenizer.GetIndexByTokenCount(text, expectedIds.Length, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(expectedIds.Length, tokenCount);
            Assert.Equal(expectedOffsets[expectedOffsets.Length - 1].Index + expectedOffsets[expectedOffsets.Length - 1].Length, codeGenTokenizer.GetIndexByTokenCount(text.AsSpan(), expectedIds.Length, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(expectedIds.Length, tokenCount);

            Assert.Equal(expectedOffsetsWithSpace[expectedOffsetsWithSpace.Length - 1].Index + expectedOffsetsWithSpace[expectedOffsetsWithSpace.Length - 1].Length, codeGenTokenizer.GetIndexByTokenCount(text, expectedIdsWithSpace.Length, addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(expectedIdsWithSpace.Length, tokenCount);
            Assert.Equal(expectedOffsetsWithSpace[expectedOffsetsWithSpace.Length - 1].Index + expectedOffsetsWithSpace[expectedOffsetsWithSpace.Length - 1].Length, codeGenTokenizer.GetIndexByTokenCount(text.AsSpan(), expectedIdsWithSpace.Length, addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(expectedIdsWithSpace.Length, tokenCount);

            //
            // LastIndexOf
            //

            int expectedIndex = offsets.Length > 1 && offsets[offsets.Length - 1].Index == offsets[offsets.Length - 2].Index ? text.Length : offsets[offsets.Length - 1].Index;
            int expectedTokenCount = expectedIndex == text.Length ? 0 : 1;
            Assert.Equal(expectedIndex, codeGenTokenizer.GetIndexByTokenCountFromEnd(text, 1, out normalizedText, out tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(expectedTokenCount, tokenCount);
            Assert.Equal(expectedIndex, codeGenTokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), 1, out normalizedText, out tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(expectedTokenCount, tokenCount);

            Assert.Equal(expectedIndex, codeGenTokenizer.GetIndexByTokenCountFromEnd(text, 1, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(expectedTokenCount, tokenCount);
            Assert.Equal(expectedIndex, codeGenTokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), 1, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(expectedTokenCount, tokenCount);

            expectedIndex = offsets.Length > 1 && expectedOffsetsWithSpace[expectedOffsetsWithSpace.Length - 1].Index == expectedOffsetsWithSpace[expectedOffsetsWithSpace.Length - 2].Index ? text.Length : expectedOffsetsWithSpace[expectedOffsetsWithSpace.Length - 1].Index;
            expectedTokenCount = expectedIndex == text.Length ? 0 : 1;
            Assert.Equal(expectedIndex, codeGenTokenizer.GetIndexByTokenCountFromEnd(text, 1, addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(expectedTokenCount, tokenCount);
            Assert.Equal(expectedIndex, codeGenTokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), 1, addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(expectedTokenCount, tokenCount);

            //
            // Id to Token and Token to Id mapping
            //
            var tokens = codeGenTokenizer.AddPrefixSpace ? expectedTokensWithSpace : expectedTokens;

            var reverseVocab = codeGenTokenizer.Vocabulary.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
            var reverseSpecialTokens = codeGenTokenizer.SpecialTokens?.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

            for (int i = 0; i < tokens.Length; i++)
            {
                Assert.Equal(tokens[i], MapIdToToken(ids[i]));
                Assert.Equal(ids[i], MapTokenId(tokens[i]));
            }

            string MapIdToToken(int id)
            {
                if (reverseVocab.TryGetValue(id, out string? token))
                {
                    return token;
                }

                return reverseSpecialTokens![id];
            }

            int MapTokenId(string token)
            {
                if (codeGenTokenizer.Vocabulary.TryGetValue(token, out int id))
                {
                    return id;
                }

                return codeGenTokenizer.SpecialTokens![token];
            }
        }

        [Theory]
        [MemberData(nameof(CodeGenTestData))]
        public void TestBegginingAndEndOfSentenceEncoding(
                        string text,
                        string[] expectedTokens,
                        (int Index, int Length)[] expectedOffsets,
                        int[] expectedIds,
                        string[] expectedTokensWithSpace,
                        (int Index, int Length)[] expectedOffsetsWithSpace,
                        int[] expectedIdsWithSpace)

        {
            Assert.NotNull(expectedOffsets);
            Assert.NotNull(expectedOffsetsWithSpace);

            //
            // Beginning of Sentence
            //

            CodeGenTokenizer codeGenTokenizer = (_codegen350MMonoTokenizerWithBeginningOfSentence as CodeGenTokenizer)!;

            IReadOnlyList<EncodedToken> encoding = codeGenTokenizer.EncodeToTokens(text, out _);
            Assert.True(codeGenTokenizer.BeginningOfSentenceToken is not null);
            Assert.True(codeGenTokenizer.BeginningOfSentenceId.HasValue);
            var idList = new List<int>(expectedIds);
            idList.Insert(0, codeGenTokenizer.BeginningOfSentenceId!.Value);
            var tokensList = new List<string>(expectedTokens);
            tokensList.Insert(0, codeGenTokenizer.BeginningOfSentenceToken!);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal((0, 0), (encoding[0].Offset.Start.Value, encoding[0].Offset.End.Value));

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), out _);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal((0, 0), (encoding[0].Offset.Start.Value, encoding[0].Offset.End.Value));

            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: false, out _);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal((0, 0), (encoding[0].Offset.Start.Value, encoding[0].Offset.End.Value));

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: false, out _);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal((0, 0), (encoding[0].Offset.Start.Value, encoding[0].Offset.End.Value));

            idList = new List<int>(expectedIdsWithSpace);
            idList.Insert(0, codeGenTokenizer.BeginningOfSentenceId!.Value);
            tokensList = new List<string>(expectedTokensWithSpace);
            tokensList.Insert(0, codeGenTokenizer.BeginningOfSentenceToken!);
            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: true, addBeginningOfSentence: true, addEndOfSentence: false, out _);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal((0, 0), (encoding[0].Offset.Start.Value, encoding[0].Offset.End.Value));

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: true, addBeginningOfSentence: true, addEndOfSentence: false, out _);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal((0, 0), (encoding[0].Offset.Start.Value, encoding[0].Offset.End.Value));

            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out _);
            Assert.Equal(expectedIds, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(expectedTokens, encoding.Select(t => t.Value).ToArray());
            Assert.True(!encoding[0].Offset.Equals(new Range(0, 0)) || !encoding[1].Offset.Equals(new Range(0, 0)));

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out _);
            Assert.Equal(expectedIds, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(expectedTokens, encoding.Select(t => t.Value).ToArray());
            Assert.True(!encoding[0].Offset.Equals(new Range(0, 0)) || !encoding[1].Offset.Equals(new Range(0, 0)));

            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false, out _);
            Assert.Equal(expectedIdsWithSpace, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(expectedTokensWithSpace, encoding.Select(t => t.Value).ToArray());
            Assert.True(!encoding[0].Offset.Equals(new Range(0, 0)) || !encoding[1].Offset.Equals(new Range(0, 0)));

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false, out _);
            Assert.Equal(expectedIdsWithSpace, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(expectedTokensWithSpace, encoding.Select(t => t.Value).ToArray());
            Assert.True(!encoding[0].Offset.Equals(new Range(0, 0)) || !encoding[1].Offset.Equals(new Range(0, 0)));

            IReadOnlyList<int> ids = codeGenTokenizer.EncodeToIds(text);
            Assert.Equal(codeGenTokenizer.BeginningOfSentenceId.Value, ids[0]);
            ids = codeGenTokenizer.EncodeToIds(text.AsSpan());
            Assert.Equal(codeGenTokenizer.BeginningOfSentenceId.Value, ids[0]);
            ids = codeGenTokenizer.EncodeToIds(text, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: false);
            Assert.Equal(codeGenTokenizer.BeginningOfSentenceId.Value, ids[0]);
            ids = codeGenTokenizer.EncodeToIds(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: false);
            Assert.Equal(codeGenTokenizer.BeginningOfSentenceId.Value, ids[0]);
            ids = codeGenTokenizer.EncodeToIds(text, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false);
            Assert.NotEqual(codeGenTokenizer.BeginningOfSentenceId.Value, ids[0]);
            ids = codeGenTokenizer.EncodeToIds(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false);
            Assert.NotEqual(codeGenTokenizer.BeginningOfSentenceId.Value, ids[0]);
            ids = codeGenTokenizer.EncodeToIds(text, maxTokenCount: 5, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: false, out string? normalizedText, out int charsConsumed);
            Assert.Equal(codeGenTokenizer.BeginningOfSentenceId.Value, ids[0]);
            ids = codeGenTokenizer.EncodeToIds(text.AsSpan(), maxTokenCount: 5, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: false, out normalizedText, out charsConsumed);
            Assert.Equal(codeGenTokenizer.BeginningOfSentenceId.Value, ids[0]);

            int tokenCount = codeGenTokenizer.CountTokens(text, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false);
            int count = codeGenTokenizer.CountTokens(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false);
            Assert.Equal(tokenCount, count);
            count = codeGenTokenizer.CountTokens(text);
            Assert.Equal(tokenCount + 1, count);
            count = codeGenTokenizer.CountTokens(text.AsSpan());
            Assert.Equal(tokenCount + 1, count);
            count = codeGenTokenizer.CountTokens(text, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: false);
            Assert.Equal(tokenCount + 1, count);
            count = codeGenTokenizer.CountTokens(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: false);
            Assert.Equal(tokenCount + 1, count);

            int length = codeGenTokenizer.GetIndexByTokenCount(text, maxTokenCount: 500, out normalizedText, out count);
            Assert.Equal(tokenCount + 1, count);
            Assert.Equal(text.Length, length);
            length = codeGenTokenizer.GetIndexByTokenCount(text.AsSpan(), maxTokenCount: 500, out normalizedText, out count);
            Assert.Equal(tokenCount + 1, count);
            Assert.Equal(text.Length, length);
            length = codeGenTokenizer.GetIndexByTokenCount(text, maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: false, out normalizedText, out count);
            Assert.Equal(tokenCount + 1, count);
            Assert.Equal(text.Length, length);
            length = codeGenTokenizer.GetIndexByTokenCount(text.AsSpan(), maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: false, out normalizedText, out count);
            Assert.Equal(tokenCount + 1, count);
            Assert.Equal(text.Length, length);
            length = codeGenTokenizer.GetIndexByTokenCount(text, maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out count);
            Assert.Equal(tokenCount, count);
            Assert.Equal(text.Length, length);
            length = codeGenTokenizer.GetIndexByTokenCount(text.AsSpan(), maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out count);
            Assert.Equal(tokenCount, count);
            Assert.Equal(text.Length, length);

            int index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text, maxTokenCount: 500, out normalizedText, out count);
            Assert.Equal(tokenCount + 1, count);
            Assert.Equal(0, index);
            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), maxTokenCount: 500, out normalizedText, out count);
            Assert.Equal(tokenCount + 1, count);
            Assert.Equal(0, index);
            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text, maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: false, out normalizedText, out count);
            Assert.Equal(tokenCount + 1, count);
            Assert.Equal(0, index);
            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: false, out normalizedText, out count);
            Assert.Equal(tokenCount + 1, count);
            Assert.Equal(0, index);
            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text, maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out count);
            Assert.Equal(tokenCount, count);
            Assert.Equal(0, index);
            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out count);
            Assert.Equal(tokenCount, count);
            Assert.Equal(0, index);

            //
            // End of Sentence
            //

            codeGenTokenizer = (_codegen350MMonoTokenizerWithEndOfSentence as CodeGenTokenizer)!;

            encoding = codeGenTokenizer.EncodeToTokens(text, out _);
            Assert.True(codeGenTokenizer.EndOfSentenceToken is not null);
            Assert.True(codeGenTokenizer.EndOfSentenceId.HasValue);
            idList = new List<int>(expectedIds);
            idList.Add(codeGenTokenizer.EndOfSentenceId!.Value);
            tokensList = new List<string>(expectedTokens);
            tokensList.Add(codeGenTokenizer.EndOfSentenceToken!);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), out _);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: true, out _);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: true, out _);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            idList = new List<int>(expectedIdsWithSpace);
            idList.Add(codeGenTokenizer.EndOfSentenceId!.Value);
            tokensList = new List<string>(expectedTokensWithSpace);
            tokensList.Add(codeGenTokenizer.EndOfSentenceToken!);
            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: true, out _);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: true, out _);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out _);
            Assert.Equal(expectedIds, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(expectedTokens, encoding.Select(t => t.Value).ToArray());
            Assert.NotEqual(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out _);
            Assert.Equal(expectedIds, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(expectedTokens, encoding.Select(t => t.Value).ToArray());
            Assert.NotEqual(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false, out _);
            Assert.Equal(expectedIdsWithSpace, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(expectedTokensWithSpace, encoding.Select(t => t.Value).ToArray());
            Assert.NotEqual(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false, out _);
            Assert.Equal(expectedIdsWithSpace, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(expectedTokensWithSpace, encoding.Select(t => t.Value).ToArray());
            Assert.NotEqual(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            ids = codeGenTokenizer.EncodeToIds(text);
            Assert.Equal(codeGenTokenizer.EndOfSentenceId.Value, ids[ids.Count - 1]);
            ids = codeGenTokenizer.EncodeToIds(text.AsSpan());
            Assert.Equal(codeGenTokenizer.EndOfSentenceId.Value, ids[ids.Count - 1]);
            ids = codeGenTokenizer.EncodeToIds(text, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: true);
            Assert.Equal(codeGenTokenizer.EndOfSentenceId.Value, ids[ids.Count - 1]);
            ids = codeGenTokenizer.EncodeToIds(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: true);
            Assert.Equal(codeGenTokenizer.EndOfSentenceId.Value, ids[ids.Count - 1]);
            ids = codeGenTokenizer.EncodeToIds(text, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false);
            Assert.NotEqual(codeGenTokenizer.EndOfSentenceId.Value, ids[ids.Count - 1]);
            ids = codeGenTokenizer.EncodeToIds(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false);
            Assert.NotEqual(codeGenTokenizer.EndOfSentenceId.Value, ids[ids.Count - 1]);
            ids = codeGenTokenizer.EncodeToIds(text, maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: true, out normalizedText, out charsConsumed);
            Assert.Equal(codeGenTokenizer.EndOfSentenceId.Value, ids[ids.Count - 1]);
            ids = codeGenTokenizer.EncodeToIds(text.AsSpan(), maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: true, out normalizedText, out charsConsumed);
            Assert.Equal(codeGenTokenizer.EndOfSentenceId.Value, ids[ids.Count - 1]);

            tokenCount = codeGenTokenizer.CountTokens(text, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false);
            count = codeGenTokenizer.CountTokens(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false);
            Assert.Equal(tokenCount, count);
            count = codeGenTokenizer.CountTokens(text);
            Assert.Equal(tokenCount + 1, count);
            count = codeGenTokenizer.CountTokens(text.AsSpan());
            Assert.Equal(tokenCount + 1, count);
            count = codeGenTokenizer.CountTokens(text, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: true);
            Assert.Equal(tokenCount + 1, count);
            count = codeGenTokenizer.CountTokens(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: true);
            Assert.Equal(tokenCount + 1, count);

            length = codeGenTokenizer.GetIndexByTokenCount(text, maxTokenCount: 500, out normalizedText, out count);
            Assert.Equal(tokenCount + 1, count);
            Assert.Equal(text.Length, length);
            length = codeGenTokenizer.GetIndexByTokenCount(text.AsSpan(), maxTokenCount: 500, out normalizedText, out count);
            Assert.Equal(tokenCount + 1, count);
            Assert.Equal(text.Length, length);
            length = codeGenTokenizer.GetIndexByTokenCount(text, maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: true, out normalizedText, out count);
            Assert.Equal(tokenCount + 1, count);
            Assert.Equal(text.Length, length);
            length = codeGenTokenizer.GetIndexByTokenCount(text.AsSpan(), maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: true, out normalizedText, out count);
            Assert.Equal(tokenCount + 1, count);
            Assert.Equal(text.Length, length);
            length = codeGenTokenizer.GetIndexByTokenCount(text, maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out count);
            Assert.Equal(tokenCount, count);
            Assert.Equal(text.Length, length);
            length = codeGenTokenizer.GetIndexByTokenCount(text.AsSpan(), maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out count);
            Assert.Equal(tokenCount, count);
            Assert.Equal(text.Length, length);

            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text, maxTokenCount: 500, out normalizedText, out count);
            Assert.Equal(tokenCount + 1, count);
            Assert.Equal(0, index);
            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), maxTokenCount: 500, out normalizedText, out count);
            Assert.Equal(tokenCount + 1, count);
            Assert.Equal(0, index);
            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text, maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: true, out normalizedText, out count);
            Assert.Equal(tokenCount + 1, count);
            Assert.Equal(0, index);
            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: true, out normalizedText, out count);
            Assert.Equal(tokenCount + 1, count);
            Assert.Equal(0, index);
            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text, maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out count);
            Assert.Equal(tokenCount, count);
            Assert.Equal(0, index);
            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out count);
            Assert.Equal(tokenCount, count);
            Assert.Equal(0, index);

            //
            // Beginning & End of Sentence
            //

            codeGenTokenizer = (_codegen350MMonoTokenizerWithBeginningAndEndOfSentence as CodeGenTokenizer)!;

            encoding = codeGenTokenizer.EncodeToTokens(text, out _);
            Assert.True(codeGenTokenizer.BeginningOfSentenceToken is not null);
            Assert.True(codeGenTokenizer.BeginningOfSentenceId.HasValue);
            idList = new List<int>(expectedIds);
            idList.Insert(0, codeGenTokenizer.BeginningOfSentenceId!.Value);
            idList.Add(codeGenTokenizer.EndOfSentenceId!.Value);
            tokensList = new List<string>(expectedTokens);
            tokensList.Insert(0, codeGenTokenizer.BeginningOfSentenceToken!);
            tokensList.Add(codeGenTokenizer.EndOfSentenceToken!);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal(new Range(0, 0), encoding[0].Offset);
            Assert.Equal(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), out _);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal(new Range(0, 0), encoding[0].Offset);
            Assert.Equal(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: true, out _);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal(new Range(0, 0), encoding[0].Offset);
            Assert.Equal(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: true, out _);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal(new Range(0, 0), encoding[0].Offset);
            Assert.Equal(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            idList = new List<int>(expectedIdsWithSpace);
            idList.Insert(0, codeGenTokenizer.BeginningOfSentenceId!.Value);
            idList.Add(codeGenTokenizer.EndOfSentenceId!.Value);
            tokensList = new List<string>(expectedTokensWithSpace);
            tokensList.Insert(0, codeGenTokenizer.BeginningOfSentenceToken!);
            tokensList.Add(codeGenTokenizer.EndOfSentenceToken!);
            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: true, addBeginningOfSentence: true, addEndOfSentence: true, out _);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal(new Range(0, 0), encoding[0].Offset);
            Assert.Equal(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: true, addBeginningOfSentence: true, addEndOfSentence: true, out _);
            Assert.Equal(idList, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(tokensList, encoding.Select(t => t.Value).ToArray());
            Assert.Equal(new Range(0, 0), encoding[0].Offset);
            Assert.Equal(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out _);
            Assert.Equal(expectedIds, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(expectedTokens, encoding.Select(t => t.Value).ToArray());
            Assert.True(!encoding[0].Offset.Equals(new Range(0, 0)) || !encoding[1].Offset.Equals(new Range(0, 0)));
            Assert.NotEqual(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out _);
            Assert.Equal(expectedIds, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(expectedTokens, encoding.Select(t => t.Value).ToArray());
            Assert.True(!encoding[0].Offset.Equals(new Range(0, 0)) || !encoding[1].Offset.Equals(new Range(0, 0)));
            Assert.NotEqual(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            encoding = codeGenTokenizer.EncodeToTokens(text, addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false, out _);
            Assert.Equal(expectedIdsWithSpace, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(expectedTokensWithSpace, encoding.Select(t => t.Value).ToArray());
            Assert.True(!encoding[0].Offset.Equals(new Range(0, 0)) || !encoding[1].Offset.Equals(new Range(0, 0)));
            Assert.NotEqual(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            encoding = codeGenTokenizer.EncodeToTokens(text.AsSpan(), addPrefixSpace: true, addBeginningOfSentence: false, addEndOfSentence: false, out _);
            Assert.Equal(expectedIdsWithSpace, encoding.Select(t => t.Id).ToArray());
            Assert.Equal(expectedTokensWithSpace, encoding.Select(t => t.Value).ToArray());
            Assert.True(!encoding[0].Offset.Equals(new Range(0, 0)) || !encoding[1].Offset.Equals(new Range(0, 0)));
            Assert.NotEqual(new Range(text.Length, text.Length), encoding[encoding.Count - 1].Offset);

            ids = codeGenTokenizer.EncodeToIds(text);
            Assert.Equal(codeGenTokenizer.BeginningOfSentenceId.Value, ids[0]);
            Assert.Equal(codeGenTokenizer.EndOfSentenceId.Value, ids[ids.Count - 1]);
            ids = codeGenTokenizer.EncodeToIds(text.AsSpan());
            Assert.Equal(codeGenTokenizer.BeginningOfSentenceId.Value, ids[0]);
            Assert.Equal(codeGenTokenizer.EndOfSentenceId.Value, ids[ids.Count - 1]);
            ids = codeGenTokenizer.EncodeToIds(text, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: true);
            Assert.Equal(codeGenTokenizer.BeginningOfSentenceId.Value, ids[0]);
            Assert.Equal(codeGenTokenizer.EndOfSentenceId.Value, ids[ids.Count - 1]);
            ids = codeGenTokenizer.EncodeToIds(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: true);
            Assert.Equal(codeGenTokenizer.BeginningOfSentenceId.Value, ids[0]);
            Assert.Equal(codeGenTokenizer.EndOfSentenceId.Value, ids[ids.Count - 1]);
            ids = codeGenTokenizer.EncodeToIds(text, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false);
            Assert.NotEqual(codeGenTokenizer.BeginningOfSentenceId.Value, ids[0]);
            Assert.NotEqual(codeGenTokenizer.EndOfSentenceId.Value, ids[ids.Count - 1]);
            ids = codeGenTokenizer.EncodeToIds(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false);
            Assert.NotEqual(codeGenTokenizer.BeginningOfSentenceId.Value, ids[0]);
            Assert.NotEqual(codeGenTokenizer.EndOfSentenceId.Value, ids[ids.Count - 1]);
            ids = codeGenTokenizer.EncodeToIds(text, maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: true, out normalizedText, out charsConsumed);
            Assert.Equal(codeGenTokenizer.BeginningOfSentenceId.Value, ids[0]);
            Assert.Equal(codeGenTokenizer.EndOfSentenceId.Value, ids[ids.Count - 1]);
            ids = codeGenTokenizer.EncodeToIds(text.AsSpan(), maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: true, out normalizedText, out charsConsumed);
            Assert.Equal(codeGenTokenizer.BeginningOfSentenceId.Value, ids[0]);
            Assert.Equal(codeGenTokenizer.EndOfSentenceId.Value, ids[ids.Count - 1]);

            tokenCount = codeGenTokenizer.CountTokens(text, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false);
            count = codeGenTokenizer.CountTokens(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false);
            Assert.Equal(tokenCount, count);
            count = codeGenTokenizer.CountTokens(text);
            Assert.Equal(tokenCount + 2, count);
            count = codeGenTokenizer.CountTokens(text.AsSpan());
            Assert.Equal(tokenCount + 2, count);
            count = codeGenTokenizer.CountTokens(text, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: true);
            Assert.Equal(tokenCount + 2, count);
            count = codeGenTokenizer.CountTokens(text.AsSpan(), addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: true);
            Assert.Equal(tokenCount + 2, count);
            length = codeGenTokenizer.GetIndexByTokenCount(text, maxTokenCount: 500, out normalizedText, out count);
            Assert.Equal(tokenCount + 2, count);
            Assert.Equal(text.Length, length);
            length = codeGenTokenizer.GetIndexByTokenCount(text.AsSpan(), maxTokenCount: 500, out normalizedText, out count);
            Assert.Equal(tokenCount + 2, count);
            Assert.Equal(text.Length, length);
            length = codeGenTokenizer.GetIndexByTokenCount(text, maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: true, out normalizedText, out count);
            Assert.Equal(tokenCount + 2, count);
            Assert.Equal(text.Length, length);
            length = codeGenTokenizer.GetIndexByTokenCount(text.AsSpan(), maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: true, out normalizedText, out count);
            Assert.Equal(tokenCount + 2, count);
            Assert.Equal(text.Length, length);
            length = codeGenTokenizer.GetIndexByTokenCount(text, maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out count);
            Assert.Equal(tokenCount, count);
            Assert.Equal(text.Length, length);
            length = codeGenTokenizer.GetIndexByTokenCount(text.AsSpan(), maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out count);
            Assert.Equal(tokenCount, count);
            Assert.Equal(text.Length, length);

            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text, maxTokenCount: 500, out normalizedText, out count);
            Assert.Equal(tokenCount + 2, count);
            Assert.Equal(0, index);
            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), maxTokenCount: 500, out normalizedText, out count);
            Assert.Equal(tokenCount + 2, count);
            Assert.Equal(0, index);
            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text, maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: true, out normalizedText, out count);
            Assert.Equal(tokenCount + 2, count);
            Assert.Equal(0, index);
            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: true, addEndOfSentence: true, out normalizedText, out count);
            Assert.Equal(tokenCount + 2, count);
            Assert.Equal(0, index);
            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text, maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out count);
            Assert.Equal(tokenCount, count);
            Assert.Equal(0, index);
            index = codeGenTokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), maxTokenCount: 500, addPrefixSpace: false, addBeginningOfSentence: false, addEndOfSentence: false, out normalizedText, out count);
            Assert.Equal(tokenCount, count);
            Assert.Equal(0, index);
        }

        private const string DefaultSpecialToken = "<|endoftext|>";

        [Fact]
        public void TestDefaultValues()
        {
            CodeGenTokenizer codeGenTokenizer = (_codegen350MMonoTokenizer as CodeGenTokenizer)!;
            Assert.False(codeGenTokenizer.AddPrefixSpace);
            Assert.False(codeGenTokenizer.AddBeginningOfSentence);
            Assert.False(codeGenTokenizer.AddEndOfSentence);

            Assert.Equal(codeGenTokenizer.EncodeToIds(DefaultSpecialToken)[0], codeGenTokenizer.BeginningOfSentenceId!.Value);
            Assert.Equal(codeGenTokenizer.EncodeToIds(DefaultSpecialToken)[0], codeGenTokenizer.EndOfSentenceId!.Value);
            Assert.Equal(codeGenTokenizer.EncodeToIds(DefaultSpecialToken)[0], codeGenTokenizer.UnknownTokenId!.Value);

            Assert.Equal(DefaultSpecialToken, codeGenTokenizer.BeginningOfSentenceToken);
            Assert.Equal(DefaultSpecialToken, codeGenTokenizer.EndOfSentenceToken);
            Assert.Equal(DefaultSpecialToken, codeGenTokenizer.UnknownToken);
        }

        [Theory]
        [InlineData(1, 0, 0, 0, 3)]
        [InlineData(2, 2, 1, 2, 1)]
        [InlineData(3, 2, 1, 2, 1)]
        [InlineData(4, 4, 3, 4, 0)]
        [InlineData(5, 4, 3, 4, 0)]
        public void TestTokenLimits(int maxTokenCount, int expectedTokenCount, int expectedTextLength, int expectedTokenCountFromEnd, int expectedTextIndexFromEnd)
        {
            // cannot split between the first two tokens nor last two tokens
            string input = "δ😀";
            int[] encodingIds = [138, 112, 47249, 222];
            (int Index, int Length)[] offsets = [(0, 0), (0, 1), (1, 0), (1, 2)];
            int calculatedLengthUsingOffsets = expectedTokenCount > 0 ? offsets[expectedTokenCount - 1].Index + offsets[expectedTokenCount - 1].Length : 0;

            IReadOnlyList<int> ids = _codegen350MMonoTokenizer.EncodeToIds(input, maxTokenCount, out _, out int charsConsumed);
            Assert.Equal(expectedTokenCount, ids.Count);
            Assert.Equal(expectedTextLength, charsConsumed);
            Assert.Equal(encodingIds.Take(expectedTokenCount), ids);
            Assert.Equal(calculatedLengthUsingOffsets, charsConsumed);
            ids = _codegen350MMonoTokenizer.EncodeToIds(input.AsSpan(), maxTokenCount, out _, out charsConsumed);
            Assert.Equal(expectedTokenCount, ids.Count);
            Assert.Equal(expectedTextLength, charsConsumed);
            Assert.Equal(encodingIds.Take(expectedTokenCount), ids);
            Assert.Equal(calculatedLengthUsingOffsets, charsConsumed);

            charsConsumed = _codegen350MMonoTokenizer.GetIndexByTokenCount(input, maxTokenCount, out _, out int tokenCount);
            Assert.Equal(expectedTokenCount, tokenCount);
            Assert.Equal(expectedTextLength, charsConsumed);
            Assert.Equal(calculatedLengthUsingOffsets, charsConsumed);
            charsConsumed = _codegen350MMonoTokenizer.GetIndexByTokenCount(input.AsSpan(), maxTokenCount, out _, out tokenCount);
            Assert.Equal(expectedTokenCount, tokenCount);
            Assert.Equal(expectedTextLength, charsConsumed);
            Assert.Equal(calculatedLengthUsingOffsets, charsConsumed);

            calculatedLengthUsingOffsets = expectedTokenCountFromEnd > 0 ? offsets[offsets.Length - expectedTokenCountFromEnd].Index : input.Length;
            charsConsumed = _codegen350MMonoTokenizer.GetIndexByTokenCountFromEnd(input, maxTokenCount, out _, out tokenCount);
            Assert.Equal(expectedTokenCountFromEnd, tokenCount);
            Assert.Equal(expectedTextIndexFromEnd, charsConsumed);
            Assert.Equal(calculatedLengthUsingOffsets, charsConsumed);
            charsConsumed = _codegen350MMonoTokenizer.GetIndexByTokenCountFromEnd(input.AsSpan(), maxTokenCount, out _, out tokenCount);
            Assert.Equal(expectedTokenCountFromEnd, tokenCount);
            Assert.Equal(expectedTextIndexFromEnd, charsConsumed);
            Assert.Equal(calculatedLengthUsingOffsets, charsConsumed);
        }
    }
}
