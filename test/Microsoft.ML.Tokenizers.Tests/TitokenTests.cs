// Licensed to the .NET Foundation under one or more agreements.
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

        [Fact]
        public async void TestTokenizerCreation()
        {
            TestGPT4TokenizationEncoding(GPT4);

            Assert.True(GPT4.Model is Tiktoken);
            IReadOnlyDictionary<string, int>? specialTokensEncoder = (GPT4.Model as Tiktoken)!.SpecialTokensEncoder;

            string tokenizerDataFileName = Utils.CreateTemporaryFile("tiktoken");

            using Stream compressedStream = typeof(Tokenizer).Assembly.GetManifestResourceStream("cl100k_base.tiktoken.deflate")!;
            using Stream deflateStream = new DeflateStream(compressedStream, CompressionMode.Decompress);

            using (Stream fileStream = File.OpenWrite(tokenizerDataFileName))
            {
                deflateStream.CopyTo(fileStream);
            }

            try
            {
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
        }

        [Theory]
        [MemberData(nameof(ModelUrlData))]
        public async void TestTokenizerUsingExternalVocab(Tokenizer tokenizer, string url)
        {
            string tokenizerDataFileName = Utils.CreateTemporaryFile("tiktoken");
            await Utils.DownloadFile(url, tokenizerDataFileName);

            try
            {
                Tiktoken tiktoken = (tokenizer.Model as Tiktoken)!;
                Tokenizer externalTokenizer = new Tokenizer(new Tiktoken(tokenizerDataFileName, tiktoken.SpecialTokensEncoder), tokenizer.PreTokenizer);

                IReadOnlyDictionary<ReadOnlyMemory<byte>, int> encoder = tiktoken.Encoder;
                IReadOnlyDictionary<ReadOnlyMemory<byte>, int> externalEncoder = (externalTokenizer.Model as Tiktoken)!.Encoder;

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
            Assert.Equal(text, tokenizer.Decode(encoded.ToArray())!);

            EncodingResult result = tokenizer.Encode(text);
            int idsCount = tokenizer.CountTokens(text);
            Assert.Equal(encoded, result.Ids);
            Assert.Equal(new string[] { "Hello", " World" }, result.Tokens);
            Assert.Equal(new List<(int, int)> { (0, 5), (5, 6) }, result.Offsets);
            Assert.Equal(encoded.Count, idsCount);
            Assert.Equal(encoded, result.Ids);

            TestGPT4Tokenizer(tokenizer);
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
            Assert.Equal(new List<(int, int)> { (0, 12), (12, 5), (17, 6), (23, 10) }, result.Offsets);
            Assert.Equal(encoded.Count, idsCount);
            Assert.Equal(encoded, result.Ids);
        }

        private void TestGPT4Tokenizer(Tokenizer gpt4Tokenizer)
        {
            string text = ReadAndSanitizeFile("./Data/lib.rs.txt");
            IReadOnlyList<int> encoded = gpt4Tokenizer.EncodeToIds(text, considerSpecialTokens: false);
            Assert.Equal(5584, encoded.Count);
            int idsCount = gpt4Tokenizer.CountTokens(text, considerSpecialTokens: false);
            Assert.Equal(encoded.Count, idsCount);

            using (Stream stream = File.OpenRead("./Data/tokens.json"))
            {
                int[]? expected = JsonSerializer.Deserialize<int[]>(stream) as int[];
                Assert.Equal(expected!, encoded.ToArray());
            }

            string? decoded = gpt4Tokenizer.Decode(encoded.ToArray());
            Assert.Equal(text, decoded!);

            TokenizerTests.TestTokenLimits(gpt4Tokenizer);
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
            Assert.Equal(new List<(int, int)> { (0, 12), (12, 5), (17, 10), (27, 6) }, result.Offsets);
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
            Assert.Equal(new List<(int, int)> { (0, 12), (12, 5), (17, 2), (19, 0), (19, 6), (25, 10) }, result.Offsets);
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
        public void TestAllSupportedModelNames(string modelName)
        {
            Tokenizer tokenizer = Tokenizer.CreateTiktokenForModel(modelName);
            Assert.NotNull(tokenizer.Model);
            Assert.NotNull(tokenizer.PreTokenizer);
        }

        [InlineData("gpt-4")]
        [InlineData("text-davinci-003")]
        [InlineData("text-curie-001")]
        [InlineData("text-davinci-edit-001")]
        [ConditionalTheory(typeof(RemoteExecutor), nameof(RemoteExecutor.IsSupported))]
        public void TestCreationUsingModel(string modelName)
        {
            RemoteExecutor.Invoke(static (name) =>
            {
                Tokenizer tokenizer = Tokenizer.CreateTiktokenForModel(name);
                Assert.NotNull(tokenizer.Model);
                Assert.NotNull(tokenizer.PreTokenizer);
            }, modelName).Dispose();
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

