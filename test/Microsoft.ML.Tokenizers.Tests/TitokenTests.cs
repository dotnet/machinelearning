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
        public void TestGPT4TokenizationEncoding()
        {
            string text = "Hello World";
            IReadOnlyList<int> encoded = GPT4.EncodeToIds(text);
            Assert.Equal(new List<int>() { 9906, 4435 }, encoded);
            Assert.Equal(text, GPT4.Decode(encoded.ToArray())!);

            TokenizerResult result = GPT4.Encode(text);
            int idsCount = GPT4.GetEncodedIdsCount(text);
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

            TokenizerResult result = GPT4.Encode(text);
            int idsCount = GPT4.GetEncodedIdsCount(text);
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
            IReadOnlyList<int> encoded = GPT4.EncodeToIds(text, skipSpecialTokens: true);
            Assert.Equal(5584, encoded.Count);
            int idsCount = GPT4.GetEncodedIdsCount(text, skipSpecialTokens: true);
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

            TokenizerResult result = GPT4.Encode(text);
            int idsCount = GPT4.GetEncodedIdsCount(text);
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

            TokenizerResult result = GPT4.Encode(text);
            int idsCount = GPT4.GetEncodedIdsCount(text);
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
            int idsCount = GPT4.GetEncodedIdsCount(text);
            Assert.Equal(new List<int>() { 100264, 9906, 2928, 99834, 4435, 100265 }, encoded);
            Assert.Equal(text, GPT4.Decode(encoded.ToArray()));

            TokenizerResult result = GPT4.Encode(text);
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
            int idsCount = GPT2.GetEncodedIdsCount(text);
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
            int idsCount = P50kBase.GetEncodedIdsCount(text);
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
            int idsCount = P50kEdit.GetEncodedIdsCount(text);
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
            int idsCount = R50kBase.GetEncodedIdsCount(text);
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

