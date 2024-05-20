﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.DotNet.RemoteExecutor;
using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class TiktokenTests
    {
        const string IMStart = "<|im_start|>";
        const string IMEnd = "<|im_end|>";

        private static readonly Dictionary<string, int> _specialTokens = new Dictionary<string, int>
                                                {
                                                    { IMStart, 100264},
                                                    { IMEnd, 100265},
                                                };

        public static Tokenizer GPT4 { get; } = Tokenizer.CreateTiktokenForModel("gpt-4", _specialTokens);
        public static Tokenizer GPT2 { get; } = Tokenizer.CreateTiktokenForModel("gpt2");
        public static Tokenizer P50kBase { get; } = Tokenizer.CreateTiktokenForModel("text-davinci-003");
        public static Tokenizer R50kBase { get; } = Tokenizer.CreateTiktokenForModel("ada");
        public static Tokenizer P50kEdit { get; } = Tokenizer.CreateTiktokenForModel("text-davinci-edit-001");
        public static Tokenizer GPT4o { get; } = Tokenizer.CreateTiktokenForModel("gpt-4o");

        [Fact]
        public async void TestTokenizerCreation()
        {
            TestGPT4TokenizationEncoding(GPT4);

            Assert.True(GPT4 is Tiktoken);
            IReadOnlyDictionary<string, int>? specialTokensEncoder = (GPT4 as Tiktoken)!.SpecialTokens;

            string tokenizerDataFileName = Utils.CreateTemporaryFile("tiktoken");

            using Stream compressedStream = typeof(Tokenizer).Assembly.GetManifestResourceStream("cl100k_base.tiktoken.deflate")!;
            using Stream deflateStream = new DeflateStream(compressedStream, CompressionMode.Decompress);

            using (Stream fileStream = File.OpenWrite(tokenizerDataFileName))
            {
                deflateStream.CopyTo(fileStream);
            }

            try
            {
                Tokenizer tokenizer = new Tiktoken(tokenizerDataFileName, GPT4.PreTokenizer, specialTokensEncoder);
                TestGPT4TokenizationEncoding(tokenizer);

                using (Stream stream = File.OpenRead(tokenizerDataFileName))
                {
                    tokenizer = new Tiktoken(stream, GPT4.PreTokenizer, specialTokensEncoder);
                }
                TestGPT4TokenizationEncoding(tokenizer);

                tokenizer = await Tokenizer.CreateTiktokenAsync(tokenizerDataFileName, GPT4.PreTokenizer, normalizer: null, specialTokensEncoder);
                TestGPT4TokenizationEncoding(tokenizer);

                using (Stream stream = File.OpenRead(tokenizerDataFileName))
                {
                    tokenizer = await Tokenizer.CreateTiktokenAsync(stream, GPT4.PreTokenizer, normalizer: null, specialTokensEncoder);
                }
                TestGPT4TokenizationEncoding(tokenizer);

                using (Stream stream = File.OpenRead(tokenizerDataFileName))
                {
                    tokenizer = Tokenizer.CreateTiktokenForModel("gpt-4", stream);
                }
                TestGPT4TokenizationEncoding(tokenizer);

                using (Stream stream = File.OpenRead(tokenizerDataFileName))
                {
                    tokenizer = await Tokenizer.CreateTiktokenForModelAsync("gpt-3.5-turbo", stream);
                }
                TestGPT4TokenizationEncoding(tokenizer);

                tokenizer = Tokenizer.CreateTiktokenForModel("gpt-4");
                TestGPT4TokenizationEncoding(tokenizer);
            }
            finally
            {
                Utils.DeleteFile(tokenizerDataFileName);
            }
        }

        public static IEnumerable<object[]> ModelUrlData()
        {
            yield return new object[] { GPT4, @"https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken" };
            yield return new object[] { GPT2, @"https://pythia.blob.core.windows.net/public/encoding/gpt2.tiktoken" };
            yield return new object[] { P50kBase, @"https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken" };
            yield return new object[] { R50kBase, @"https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken" };
            yield return new object[] { GPT4o, @"https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" };
        }

        [Theory]
        [MemberData(nameof(ModelUrlData))]
        public async void TestTokenizerUsingExternalVocab(Tokenizer tokenizer, string url)
        {
            string tokenizerDataFileName = Utils.CreateTemporaryFile("tiktoken");
            await Utils.DownloadFile(url, tokenizerDataFileName);

            try
            {
                Tiktoken tiktoken = (tokenizer as Tiktoken)!;
                Tokenizer externalTokenizer = new Tiktoken(tokenizerDataFileName, tokenizer.PreTokenizer, tiktoken.SpecialTokens);

                IReadOnlyDictionary<ReadOnlyMemory<byte>, int> encoder = tiktoken.Encoder;
                IReadOnlyDictionary<ReadOnlyMemory<byte>, int> externalEncoder = (externalTokenizer as Tiktoken)!.Encoder;

                Assert.Equal(externalEncoder.Count, encoder.Count);
                foreach (KeyValuePair<ReadOnlyMemory<byte>, int> kvp in encoder)
                {
                    Assert.True(externalEncoder.TryGetValue(kvp.Key, out int value));
                    Assert.Equal(kvp.Value, value);
                }
            }
            finally
            {
                Utils.DeleteFile(tokenizerDataFileName);
            }
        }

        private void TestGPT4TokenizationEncoding(Tokenizer tokenizer)
        {
            string text = "Hello World";
            IReadOnlyList<int> encoded = tokenizer.EncodeToIds(text);
            Assert.Equal(new List<int>() { 9906, 4435 }, encoded);
            Assert.Equal(text, tokenizer.Decode(encoded)!);

            IReadOnlyList<Token> result = tokenizer.Encode(text, out string? normalizedString);
            int idsCount = tokenizer.CountTokens(text);

            int[] ids = result.Select(token => token.Id).ToArray();
            string[] tokens = result.Select(token => token.Value).ToArray();
            (int, int)[] offsets = result.Select(token => token.Offset).ToArray();
            Assert.Equal(encoded, ids);
            Assert.Equal(new string[] { "Hello", " World" }, tokens);
            Assert.Equal(new List<(int, int)> { (0, 5), (5, 6) }, offsets);
            Assert.Equal(encoded.Count, idsCount);
            Assert.Equal(encoded, ids);

            TestGPT4Tokenizer(tokenizer);
        }

        [Fact]
        public void TestEncode1()
        {
            var text = "<|im_start|>Hello World<|im_end|>";
            IReadOnlyList<int> encoded = GPT4.EncodeToIds(text);
            Assert.Equal(new List<int>() { 100264, 9906, 4435, 100265 }, encoded);
            Assert.Equal(text, GPT4.Decode(encoded));

            IReadOnlyList<Token> result = GPT4.Encode(text, out string? normalizedString);
            int idsCount = GPT4.CountTokens(text);

            int[] ids = result.Select(token => token.Id).ToArray();
            string[] tokens = result.Select(token => token.Value).ToArray();
            (int, int)[] offsets = result.Select(token => token.Offset).ToArray();

            Assert.Equal(encoded, ids);
            Assert.Equal(new string[] { "<|im_start|>", "Hello", " World", "<|im_end|>" }, tokens);
            Assert.Equal(new List<(int, int)> { (0, 12), (12, 5), (17, 6), (23, 10) }, offsets);
            Assert.Equal(encoded.Count, idsCount);
            Assert.Equal(encoded, ids);
        }

        private void TestGPT4Tokenizer(Tokenizer gpt4Tokenizer)
        {
            string text = ReadAndSanitizeFile("./Data/lib.rs.txt");
            IReadOnlyList<int> encoded = gpt4Tokenizer.EncodeToIds(text);
            Assert.Equal(5584, encoded.Count);
            int idsCount = gpt4Tokenizer.CountTokens(text);
            Assert.Equal(encoded.Count, idsCount);

            using (Stream stream = File.OpenRead("./Data/tokens.json"))
            {
                int[]? expected = JsonSerializer.Deserialize<int[]>(stream) as int[];
                Assert.Equal(expected!, encoded);
            }

            string? decoded = gpt4Tokenizer.Decode(encoded);
            Assert.Equal(text, decoded!);

            TokenizerTests.TestTokenLimits(gpt4Tokenizer);
        }

        [Fact]
        public void TestEncode3()
        {
            string text = "<|im_start|>Hello<|im_end|> World";
            IReadOnlyList<int> encoded = GPT4.EncodeToIds(text);
            Assert.Equal(new List<int>() { 100264, 9906, 100265, 4435 }, encoded);
            string? decoded = GPT4.Decode(encoded);
            Assert.Equal(text, decoded);

            IReadOnlyList<Token> result = GPT4.Encode(text, out string? normalizedString);
            int[] ids = result.Select(token => token.Id).ToArray();
            string[] tokens = result.Select(token => token.Value).ToArray();
            (int, int)[] offsets = result.Select(token => token.Offset).ToArray();

            int idsCount = GPT4.CountTokens(text);
            Assert.Equal(encoded, ids);
            Assert.Equal(encoded.Count, idsCount);
            Assert.Equal(new string[] { "<|im_start|>", "Hello", "<|im_end|>", " World" }, tokens);
            Assert.Equal(new List<(int, int)> { (0, 12), (12, 5), (17, 10), (27, 6) }, offsets);
        }

        [Fact]
        public void TestEncode4()
        {
            string text = "";
            IReadOnlyList<int> encoded = GPT4.EncodeToIds(text);
            Assert.Empty(encoded);

            IReadOnlyList<Token> result = GPT4.Encode(text, out string? normalizedString);
            int idsCount = GPT4.CountTokens(text);
            Assert.Empty(result);
            Assert.Equal(0, idsCount);
        }

        [Fact]
        public void TestEncode5()
        {
            string text = "<|im_start|>Hello ⭐ World<|im_end|>";
            IReadOnlyList<int> encoded = GPT4.EncodeToIds(text);
            int idsCount = GPT4.CountTokens(text);
            Assert.Equal(new List<int>() { 100264, 9906, 2928, 99834, 4435, 100265 }, encoded);
            Assert.Equal(text, GPT4.Decode(encoded));

            IReadOnlyList<Token> result = GPT4.Encode(text, out string? normalizedString);
            Assert.Equal(encoded, result.Select(token => token.Id).ToArray());
            Assert.Equal(encoded.Count, idsCount);
            Assert.Equal(new string[] { "<|im_start|>", "Hello", " ⭐", "⭐", " World", "<|im_end|>" }, result.Select(token => token.Value).ToArray());
            Assert.Equal(new List<(int, int)> { (0, 12), (12, 5), (17, 2), (18, 1), (19, 6), (25, 10) }, result.Select(token => token.Offset).ToArray());
        }

        [Fact]
        public void TestEncodeGpt4o()
        {
            string text = ReadAndSanitizeFile("./Data/lib.rs.txt");
            IReadOnlyList<int> encoded = GPT4o.EncodeToIds(text);
            int idsCount = GPT4o.CountTokens(text);

            Assert.Equal(5609, encoded.Count);
            Assert.Equal(encoded.Count, idsCount);

            using (Stream stream = File.OpenRead("./Data/tokens_gpt4o.json"))
            {
                int[]? expected = JsonSerializer.Deserialize<int[]>(stream) as int[];
                Assert.Equal(expected!, encoded);
            }

            string? decoded = GPT4o.Decode(encoded);
            Assert.Equal(text, decoded);

            text = "<|endoftext|>Hello ⭐ World<|endofprompt|>";

            encoded = GPT4o.EncodeToIds(text);
            idsCount = GPT4o.CountTokens(text);
            Assert.Equal(new List<int>() { 199999, 13225, 161181, 5922, 200018 }, encoded);
            Assert.Equal(text, GPT4o.Decode(encoded));

            IReadOnlyList<Token> result = GPT4o.Encode(text, out string? normalizedString);

            Assert.Equal(encoded, result.Select(token => token.Id).ToArray());
            Assert.Equal(encoded.Count, idsCount);
            Assert.Equal(new string[] { "<|endoftext|>", "Hello", " ⭐", " World", "<|endofprompt|>" }, result.Select(token => token.Value).ToArray());
            Assert.Equal(new List<(int, int)> { (0, 13), (13, 5), (18, 2), (20, 6), (26, 15) }, result.Select(token => token.Offset).ToArray());

            TokenizerTests.TestTokenLimits(GPT4o);
        }

        [Fact]
        public void TestEncodeGpt2()
        {
            string text = ReadAndSanitizeFile("./Data/lib.rs.txt");
            IReadOnlyList<int> encoded = GPT2.EncodeToIds(text);
            int idsCount = GPT2.CountTokens(text);
            Assert.Equal(11378, encoded.Count);
            Assert.Equal(encoded.Count, idsCount);

            using (Stream stream = File.OpenRead("./Data/tokens_gpt2.json"))
            {
                int[]? expected = JsonSerializer.Deserialize<int[]>(stream) as int[];
                Assert.Equal(expected!, encoded);
            }

            string? decoded = GPT2.Decode(encoded);
            Assert.Equal(text, decoded);
        }

        [Fact]
        public void TestEncodeP50kBase()
        {
            string text = ReadAndSanitizeFile("./Data/lib.rs.txt");
            IReadOnlyList<int> encoded = P50kBase.EncodeToIds(text);
            int idsCount = P50kBase.CountTokens(text);
            Assert.Equal(7230, encoded.Count);
            Assert.Equal(encoded.Count, idsCount);

            using (Stream stream = File.OpenRead("./Data/tokens_p50k_base.json"))
            {
                int[]? expected = JsonSerializer.Deserialize<int[]>(stream) as int[];
                Assert.Equal(expected!, encoded);
            }

            string? decoded = P50kBase.Decode(encoded);
            Assert.Equal(text, decoded);
        }

        [Fact]
        public void TestEncodeP50kEdit()
        {
            string text = ReadAndSanitizeFile("./Data/lib.rs.txt");
            IReadOnlyList<int> encoded = P50kEdit.EncodeToIds(text);
            int idsCount = P50kEdit.CountTokens(text);
            Assert.Equal(7230, encoded.Count);
            Assert.Equal(encoded.Count, idsCount);

            using (Stream stream = File.OpenRead("./Data/tokens_p50k_edit.json"))
            {
                int[]? expected = JsonSerializer.Deserialize<int[]>(stream) as int[];
                Assert.Equal(expected!, encoded);
            }

            string? decoded = P50kEdit.Decode(encoded);
            Assert.Equal(text, decoded);
        }

        [Fact]
        public void TestEncodeR50kBase()
        {
            string text = ReadAndSanitizeFile("./Data/lib.rs.txt");
            IReadOnlyList<int> encoded = R50kBase.EncodeToIds(text);
            int idsCount = R50kBase.CountTokens(text);
            Assert.Equal(11378, encoded.Count);
            Assert.Equal(encoded.Count, idsCount);

            using (Stream stream = File.OpenRead("./Data/tokens_r50k_base.json"))
            {
                int[]? expected = JsonSerializer.Deserialize<int[]>(stream) as int[];
                Assert.Equal(expected!, encoded);
            }

            string? decoded = R50kBase.Decode(encoded);
            Assert.Equal(text, decoded);
        }

        [Theory]
        [InlineData("gpt-4o")]
        [InlineData("gpt-4o-")]
        [InlineData("gpt-4")]
        [InlineData("gpt-4-")]
        [InlineData("gpt-3.5-")]
        [InlineData("gpt-3.5-turbo")]
        [InlineData("gpt-3.5-turbo-")]
        [InlineData("gpt-3.5-turbo-16k")]
        [InlineData("gpt-35")]
        [InlineData("gpt-35-")]
        [InlineData("gpt-35-turbo")]
        [InlineData("gpt-35-turbo-16k")]
        [InlineData("gpt-35-turbo-")]
        [InlineData("text-davinci-003")]
        [InlineData("text-davinci-002")]
        [InlineData("text-davinci-001")]
        [InlineData("text-curie-001")]
        [InlineData("text-babbage-001")]
        [InlineData("text-ada-001")]
        [InlineData("davinci")]
        [InlineData("curie")]
        [InlineData("babbage")]
        [InlineData("ada")]
        [InlineData("code-davinci-002")]
        [InlineData("code-davinci-001")]
        [InlineData("code-cushman-002")]
        [InlineData("code-cushman-001")]
        [InlineData("davinci-codex")]
        [InlineData("cushman-codex")]
        [InlineData("text-davinci-edit-001")]
        [InlineData("code-davinci-edit-001")]
        [InlineData("text-embedding-ada-002")]
        [InlineData("text-embedding-3-small")]
        [InlineData("text-embedding-3-large")]
        [InlineData("text-similarity-davinci-001")]
        [InlineData("text-similarity-curie-001")]
        [InlineData("text-similarity-babbage-001")]
        [InlineData("text-similarity-ada-001")]
        [InlineData("text-search-davinci-doc-001")]
        [InlineData("text-search-curie-doc-001")]
        [InlineData("text-search-babbage-doc-001")]
        [InlineData("text-search-ada-doc-001")]
        [InlineData("code-search-babbage-code-001")]
        [InlineData("code-search-ada-code-001")]
        [InlineData("gpt2")]
        public void TestAllSupportedModelNames(string modelName)
        {
            Tokenizer tokenizer = Tokenizer.CreateTiktokenForModel(modelName);
            Assert.True(tokenizer is Tiktoken);
            Assert.NotNull(tokenizer.PreTokenizer);
        }

        [Theory]
        [InlineData("r50k_base")]
        [InlineData("p50k_base")]
        [InlineData("p50k_edit")]
        [InlineData("cl100k_base")]
        [InlineData("o200k_base")]
        public void TestAllSupportedEncodingNames(string encodingName)
        {
            Tokenizer tokenizer = Tokenizer.CreateTiktokenForEncoding(encodingName);
            Assert.True(tokenizer is Tiktoken);
            Assert.NotNull(tokenizer.PreTokenizer);

            string modelName = encodingName.ToLowerInvariant() switch
            {
                "r50k_base" => "text-davinci-001",
                "p50k_base" => "text-davinci-003",
                "p50k_edit" => "text-davinci-edit-001",
                "cl100k_base" => "gpt-4",
                "o200k_base" => "gpt-4o",
                _ => throw new ArgumentException("Invalid encoding name"),
            };

            Tokenizer tokenizer1 = Tokenizer.CreateTiktokenForModel(modelName);

            Assert.True(tokenizer is Tiktoken);
            Assert.True(tokenizer1 is Tiktoken);

            Tiktoken tiktoken = (tokenizer as Tiktoken)!;
            Tiktoken tiktoken1 = (tokenizer1 as Tiktoken)!;

            Assert.Equal(tiktoken1.Encoder, tiktoken.Encoder);
            Assert.Equal(tiktoken1.Decoder, tiktoken.Decoder);
            Assert.Equal(tiktoken1.SpecialTokens, tiktoken.SpecialTokens);
            Assert.Equal(tiktoken1.Vocab, tiktoken.Vocab);
        }

        [Fact]
        public void TestEncodingNamesNegativeCases()
        {
            Assert.Throws<ArgumentNullException>(() => Tokenizer.CreateTiktokenForEncoding(null!));
            Assert.Throws<ArgumentException>(() => Tokenizer.CreateTiktokenForEncoding("r50k_base_"));
            Assert.Throws<ArgumentException>(() => Tokenizer.CreateTiktokenForEncoding("p50k_base_"));
            Assert.Throws<ArgumentException>(() => Tokenizer.CreateTiktokenForEncoding("p50k_edit_"));
            Assert.Throws<ArgumentException>(() => Tokenizer.CreateTiktokenForEncoding("cl100k_base_"));
            Assert.Throws<ArgumentException>(() => Tokenizer.CreateTiktokenForEncoding("o200k_base_"));
        }

        [InlineData("gpt-4")]
        [InlineData("gpt-4o")]
        [InlineData("text-davinci-003")]
        [InlineData("text-curie-001")]
        [InlineData("text-davinci-edit-001")]
        [ConditionalTheory(typeof(RemoteExecutor), nameof(RemoteExecutor.IsSupported))]
        public void TestCreationUsingModel(string modelName)
        {
            RemoteExecutor.Invoke(static (name) =>
            {
                Tokenizer tokenizer = Tokenizer.CreateTiktokenForModel(name);
                Assert.True(tokenizer is Tiktoken);
                Assert.NotNull(tokenizer.PreTokenizer);
            }, modelName).Dispose();
        }

        public static IEnumerable<object?[]> TokenizerTestData
        {
            get
            {
                // string to tokenize, produced tokens, the token offsets
                yield return new object?[]
                {
                    "the brown fox jumped over the lazy dog!",
                    new string[] { "the", " brown", " fox", " jumped", " over", " the", " lazy", " dog", "!" },
                    new (int Index, int Length)[] { (0, 3), (3, 6), (9, 4), (13, 7), (20, 5), (25, 4), (29, 5), (34, 4), (38, 1) },
                    new int[] { 1820, 14198, 39935, 27096, 927, 279, 16053, 5679, 0 }
                };
                yield return new object?[]
                {
                    "he traveled to Egypt during the summer, the weather was hot and ammunition." ,
                    new string[] { "he", " traveled", " to", " Egypt", " during", " the", " summer", ",", " the", " weather", " was", " hot", " and", " ammunition", "." },
                    new (int Index, int Length)[] { (0, 2), (2, 9), (11, 3), (14, 6), (20, 7), (27, 4), (31, 7), (38, 1), (39, 4), (43, 8), (51, 4), (55, 4), (59, 4), (63, 11), (74, 1) },
                    new int[] { 383, 31796, 311, 15212, 2391, 279, 7474, 11, 279, 9282, 574, 4106, 323, 37768, 13 }
                };
                yield return new object?[]
                {
                    "She played many games and she felt exhausted afterward",
                    new string[] { "She", " played", " many", " games", " and", " she", " felt", " exhausted", " afterward" },
                    new (int Index, int Length)[] { (0, 3), (3, 7), (10, 5), (15, 6), (21, 4), (25, 4), (29, 5), (34, 10), (44, 10) },
                    new int[] { 8100, 6476, 1690, 3953, 323, 1364, 6612, 39019, 49043 }
                };
                yield return new object?[]
                {
                    "Hello, y'all! How are you 😁 ?",
                    new string[] { "Hello", ",", " y", "'all", "!", " How", " are", " you", " 😁", "😁", " ?" },
                    new (int Index, int Length)[] { (0, 5), (5, 1), (6, 2), (8, 4), (12, 1), (13, 4), (17, 4), (21, 4), (25, 3), (26, 2), (28, 2) },
                    new int[] { 9906, 11, 379, 65948, 0, 2650, 527, 499, 27623, 223, 949 }
                };
            }
        }

        [Theory]
        [MemberData(nameof(TokenizerTestData))]
        public void TestTokenizerEncoding(string text, string[] expectedTokens, (int Index, int Length)[] expectedOffsets, int[] expectedIds)
        {
            Tokenizer tokenizer = GPT4;

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

            Assert.Equal(expectedIds.Take(expectedIds.Length - 4), tokenizer.EncodeToIds(text, expectedIds.Length - 4, out normalizedString, out length));
            Assert.Null(normalizedString);
            int expectedLength = expectedOffsets[expectedOffsets.Length - 5].Index + expectedOffsets[expectedOffsets.Length - 5].Length;
            Assert.Equal(expectedLength, length);
            Assert.Equal(expectedIds.Take(expectedIds.Length - 4), tokenizer.EncodeToIds(text.AsSpan(), expectedIds.Length - 4, out normalizedString, out length));
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

        // Test running copy the test data files to the output folder but sometimes the file content is mutated replacing '\n' with '\r\n'.
        // This method reads the file and removes the extra inserted '\r' characters. Having '\r' in the file content will cause the tests to fail.
        private string ReadAndSanitizeFile(string path)
        {
            // Didn't use String.Replace because the version accept stringComparison parameter is not supported on NETFX.
            string text = File.ReadAllText(path);
            StringBuilder sb = new StringBuilder();

            foreach (char c in text)
            {
                if (c != '\r')
                {
                    sb.Append(c);
                }
            }
            return sb.ToString();
        }

        public static IEnumerable<object?[]> TokenizerLimitsTestData
        {
            get
            {
                // string to tokenize, produced tokens, the token offsets, the token ids
                yield return new object?[]
                {
                    "Hello ⭐ World",
                    new string[] { "Hello", " ⭐", "⭐", " World" },
                    new (int Index, int Length)[] { (0, 5), (5, 2), (6, 1), (7, 6) },
                    new int[] { 9906, 2928, 99834, 4435 }
                };

                yield return new object?[]
                {
                    "⭐", // encoded to multiple tokens
                    new string[] { "⭐", "⭐" },
                    new (int Index, int Length)[] { (0, 1), (0, 1) },
                    new int[] { 158, 99834 }
                };

                yield return new object?[]
                {
                    "Hi 😀", // Surrogates
                    new string[] { "Hi", " 😀" },
                    new (int Index, int Length)[] { (0, 2), (2, 3) },
                    new int[] { 13347, 91416 }
                };

                yield return new object?[]
                {
                    "⭐😀", // character encoded to multiple tokens and surrogates
                    new string[] { "⭐", "⭐", "😀", "😀" },
                    new (int Index, int Length)[] { (0, 1), (0, 1), (1, 2), (1, 2) },
                    new int[] { 158, 99834, 76460, 222 }
                };

                yield return new object?[]
                {
                    "From: Adele Vance\nSubject: TestSubject\nTestBodyContent",
                    new string[] { "From", ":", " Ade", "le", " Vance", "\n", "Subject", ":", " Test", "Subject", "\n", "Test", "Body", "Content" },
                    new (int Index, int Length)[] { (0, 4), (4, 1), (5, 4), (9, 2), (11, 6), (17, 1), (18, 7), (25, 1), (26, 5), (31, 7), (38, 1), (39, 4), (43, 4), (47, 7)},
                    new int[] { 3915, 25, 63140, 273, 92368, 198, 13317, 25, 3475, 13317, 198, 2323, 5561, 2831 }
                };
            }
        }

        [Theory]
        [MemberData(nameof(TokenizerLimitsTestData))]
        public void TestPreciseTokenLimits(string text, string[] expectedTokens, (int Index, int Length)[] expectedOffsets, int[] expectedIds)
        {
            IReadOnlyList<Token> result = GPT4.Encode(text, out _);
            int[] ids = result.Select(r => r.Id).ToArray();
            (int Index, int Length)[] offsets = result.Select(r => r.Offset).ToArray();
            Assert.Equal(expectedTokens, result.Select(r => r.Value));
            Assert.Equal(expectedIds, ids);
            Assert.Equal(expectedOffsets, offsets);
            Assert.Equal(expectedIds, GPT4.EncodeToIds(text));
            Assert.Equal(expectedIds.Length, GPT4.CountTokens(text));

            for (int tokenCount = 1; tokenCount <= ids.Length; tokenCount++)
            {
                int length = GPT4.IndexOfTokenCount(text, tokenCount, out _, out int count);
                Assert.True(count <= ids.Length);

                if (count < tokenCount)
                {
                    Assert.True(count < ids.Length - 1);
                    Assert.True(offsets[count + 1].Index < offsets[count].Index + offsets[count].Length);
                }

                if (count > 0)
                {
                    Assert.Equal(offsets[count - 1].Index + offsets[count - 1].Length, length);
                }
                else
                {
                    Assert.Equal(0, length);
                }

                int index = GPT4.LastIndexOfTokenCount(text, tokenCount, out _, out count);
                Assert.True(count <= ids.Length);

                if (count < tokenCount)
                {
                    Assert.True(ids.Length - tokenCount > 0);
                    Assert.True(offsets[offsets.Length - tokenCount].Index < offsets[offsets.Length - tokenCount - 1].Index + offsets[offsets.Length - tokenCount - 1].Length);
                }

                if (count > 0)
                {
                    Assert.Equal(offsets[offsets.Length - count].Index, index);
                }
                else
                {
                    Assert.Equal(text.Length, index);
                }
            }
        }
    }
}

