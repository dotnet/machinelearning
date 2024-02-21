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

        public static Tokenizer GPT4 { get; } = Tokenizer.CreateByModelNameAsync("gpt-4", _specialTokens).GetAwaiter().GetResult();
        public static Tokenizer GPT2 { get; } = Tokenizer.CreateByModelNameAsync("gpt2").GetAwaiter().GetResult();
        public static Tokenizer P50kBase { get; } = Tokenizer.CreateByModelNameAsync("text-davinci-003").GetAwaiter().GetResult();
        public static Tokenizer R50kBase { get; } = Tokenizer.CreateByModelNameAsync("ada").GetAwaiter().GetResult();
        public static Tokenizer P50kEdit { get; } = Tokenizer.CreateByModelNameAsync("text-davinci-edit-001").GetAwaiter().GetResult();

        [Fact]
        public async void TestTokenizerCreation()
        {
            TestGPT4TokenizationEncoding(GPT4);

            Assert.True(GPT4.Model is Tiktoken);
            IReadOnlyDictionary<string, int>? specialTokensEncoder = (GPT4.Model as Tiktoken)!.SpecialTokensEncoder;
            string tokenizerDataFileName = "./Data/cl100k_base.tiktoken";

            Tokenizer tokenizer = new Tokenizer(new Tiktoken(tokenizerDataFileName, specialTokensEncoder), GPT4.PreTokenizer);
            TestGPT4TokenizationEncoding(tokenizer);

            using (Stream stream = File.OpenRead(tokenizerDataFileName))
            {
                tokenizer = new Tokenizer(new Tiktoken(stream, specialTokensEncoder), GPT4.PreTokenizer);
            }
            TestGPT4TokenizationEncoding(tokenizer);

            tokenizer = new Tokenizer(await Tiktoken.CreateAsync(tokenizerDataFileName, specialTokensEncoder), GPT4.PreTokenizer);
            TestGPT4TokenizationEncoding(tokenizer);

            using (Stream stream = File.OpenRead(tokenizerDataFileName))
            {
                tokenizer = new Tokenizer(await Tiktoken.CreateAsync(stream, specialTokensEncoder), GPT4.PreTokenizer);
            }
            TestGPT4TokenizationEncoding(tokenizer);

            Tiktoken? tiktoken = GPT4.Model as Tiktoken;
            tokenizer = new Tokenizer(new Tiktoken(tiktoken!.Encoder, tiktoken!.Decoder, tiktoken!.GetVocab(), specialTokensEncoder), GPT4.PreTokenizer);
            TestGPT4TokenizationEncoding(tokenizer);

            tokenizer = new Tokenizer(new Tiktoken(tiktoken!.Encoder, decoder: null, vocab: null, specialTokensEncoder), GPT4.PreTokenizer);
            TestGPT4TokenizationEncoding(tokenizer);
        }

        private void TestGPT4TokenizationEncoding(Tokenizer tokenizer)
        {
            string text = "Hello World";
            IReadOnlyList<int> encoded = tokenizer.EncodeToIds(text);
            Assert.Equal(new List<int>() { 9906, 4435 }, encoded);
            Assert.Equal(text, tokenizer.Decode(encoded.ToArray())!);

            EncodingResult result = tokenizer.Encode(text);
            int idsCount = tokenizer.CountTokens(text);
            Assert.Equal(encoded, result.Ids);
            Assert.Equal(new string[] { "Hello", " World" }, result.Tokens);
            Assert.Equal(new List<(int, int)> { (0, 5), (5, 11) }, result.Offsets);
            Assert.Equal(encoded.Count, idsCount);
            Assert.Equal(encoded, result.Ids);
        }

        [Fact]
        public void TestEncode1()
        {
            var text = "<|im_start|>Hello World<|im_end|>";
            IReadOnlyList<int> encoded = GPT4.EncodeToIds(text);
            Assert.Equal(new List<int>() { 100264, 9906, 4435, 100265 }, encoded);
            Assert.Equal(text, GPT4.Decode(encoded.ToArray()));

            EncodingResult result = GPT4.Encode(text);
            int idsCount = GPT4.CountTokens(text);
            Assert.Equal(encoded, result.Ids);
            Assert.Equal(new string[] { "<|im_start|>", "Hello", " World", "<|im_end|>" }, result.Tokens);
            Assert.Equal(new List<(int, int)> { (0, 12), (12, 17), (17, 23), (23, 33) }, result.Offsets);
            Assert.Equal(encoded.Count, idsCount);
            Assert.Equal(encoded, result.Ids);
        }

        [Fact]
        public void TestEncode2()
        {
            string text = ReadAndSanitizeFile("./Data/lib.rs.txt");
            IReadOnlyList<int> encoded = GPT4.EncodeToIds(text, considerSpecialTokens: false);
            Assert.Equal(5584, encoded.Count);
            int idsCount = GPT4.CountTokens(text, considerSpecialTokens: false);
            Assert.Equal(encoded.Count, idsCount);

            using (Stream stream = File.OpenRead("./Data/tokens.json"))
            {
                int[]? expected = JsonSerializer.Deserialize<int[]>(stream) as int[];
                Assert.Equal(expected!, encoded.ToArray());
            }

            string? decoded = GPT4.Decode(encoded.ToArray());
            Assert.Equal(text, decoded!);
        }

        [Fact]
        public void TestEncode3()
        {
            string text = "<|im_start|>Hello<|im_end|> World";
            IReadOnlyList<int> encoded = GPT4.EncodeToIds(text);
            Assert.Equal(new List<int>() { 100264, 9906, 100265, 4435 }, encoded);
            string? decoded = GPT4.Decode(encoded.ToArray());
            Assert.Equal(text, decoded);

            EncodingResult result = GPT4.Encode(text);
            int idsCount = GPT4.CountTokens(text);
            Assert.Equal(encoded, result.Ids);
            Assert.Equal(encoded.Count, idsCount);
            Assert.Equal(new string[] { "<|im_start|>", "Hello", "<|im_end|>", " World" }, result.Tokens);
            Assert.Equal(new List<(int, int)> { (0, 12), (12, 17), (17, 27), (27, 33) }, result.Offsets);
        }

        [Fact]
        public void TestEncode4()
        {
            string text = "";
            IReadOnlyList<int> encoded = GPT4.EncodeToIds(text);
            Assert.Empty(encoded);

            EncodingResult result = GPT4.Encode(text);
            int idsCount = GPT4.CountTokens(text);
            Assert.Empty(result.Ids);
            Assert.Empty(result.Tokens);
            Assert.Empty(result.Offsets);
            Assert.Equal(result.Ids.Count, idsCount);
        }

        [Fact]
        public void TestEncode5()
        {
            string text = "<|im_start|>Hello ⭐ World<|im_end|>";
            IReadOnlyList<int> encoded = GPT4.EncodeToIds(text);
            int idsCount = GPT4.CountTokens(text);
            Assert.Equal(new List<int>() { 100264, 9906, 2928, 99834, 4435, 100265 }, encoded);
            Assert.Equal(text, GPT4.Decode(encoded.ToArray()));

            EncodingResult result = GPT4.Encode(text);
            Assert.Equal(encoded, result.Ids);
            Assert.Equal(encoded.Count, idsCount);
            Assert.Equal(new string[] { "<|im_start|>", "Hello", " ⭐", "", " World", "<|im_end|>" }, result.Tokens);
            Assert.Equal(new List<(int, int)> { (0, 12), (12, 17), (17, 19), (19, 19), (19, 25), (25, 35) }, result.Offsets);
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
                Assert.Equal(expected!, encoded.ToArray());
            }

            string? decoded = GPT2.Decode(encoded.ToArray());
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
                Assert.Equal(expected!, encoded.ToArray());
            }

            string? decoded = P50kBase.Decode(encoded.ToArray());
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
                Assert.Equal(expected!, encoded.ToArray());
            }

            string? decoded = P50kEdit.Decode(encoded.ToArray());
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
                Assert.Equal(expected!, encoded.ToArray());
            }

            string? decoded = R50kBase.Decode(encoded.ToArray());
            Assert.Equal(text, decoded);
        }

        [Theory]
        [InlineData("gpt-4")]
        [InlineData("gpt-4-")]
        [InlineData("gpt-3.5-turbo")]
        [InlineData("gpt-3.5-turbo-")]
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
        public async void TestAllSupportedModelNames(string modelName)
        {
            Tokenizer tokenizer = await Tokenizer.CreateByModelNameAsync(modelName);
            Assert.NotNull(tokenizer.Model);
            Assert.NotNull(tokenizer.PreTokenizer);
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
    }
}

