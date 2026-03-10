// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class SentencePieceTests
    {
        [Fact]
        public void CreateWithNullStreamThrows()
        {
            Assert.ThrowsAny<ArgumentException>(() => SentencePieceTokenizer.Create(null!));
        }

        [Fact]
        public void CreateWithEmptyStreamThrows()
        {
            using MemoryStream empty = new MemoryStream(Array.Empty<byte>());
            Assert.ThrowsAny<ArgumentException>(() => SentencePieceTokenizer.Create(empty));
        }

        [Fact]
        public void CreateWithTruncatedStreamThrows()
        {
            // A protobuf tag claiming a length-delimited field longer than remaining bytes.
            byte[] truncated = new byte[] { 0x0A, 0xFF, 0x01 }; // field 1, length 255 – but only 0 data bytes follow
            using MemoryStream ms = new MemoryStream(truncated);
            Assert.ThrowsAny<Exception>(() => SentencePieceTokenizer.Create(ms));
        }

        [Fact]
        public void CreateBpeViaSentencePieceTokenizer()
        {
            // Verify that the generic SentencePieceTokenizer.Create() factory method
            // works for BPE models (not just LlamaTokenizer.Create()).
            using Stream stream = File.OpenRead(Path.Combine(@"Llama", "tokenizer.model"));
            SentencePieceTokenizer tokenizer = SentencePieceTokenizer.Create(stream);

            IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens("Hello", out _);
            Assert.True(tokens.Count > 0);
            Assert.Equal("Hello", tokenizer.Decode(tokens.Select(t => t.Id)));
        }
    }
}
