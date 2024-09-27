// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class TokenizerDataTests
    {
        [Theory]
        [InlineData("gpt-4o", "Microsoft.ML.Tokenizers.Data.O200kBase")]            // O200kBase
        [InlineData("gpt-4", "Microsoft.ML.Tokenizers.Data.Cl100kBase")]            // Cl100kBase
        [InlineData("text-davinci-003", "Microsoft.ML.Tokenizers.Data.P50kBase")]   // P50kBase
        [InlineData("text-davinci-001", "Microsoft.ML.Tokenizers.Data.R50kBase")]   // R50kBase
        [InlineData("gpt2", "Microsoft.ML.Tokenizers.Data.Gpt2")]                   // Gpt2
        public void TestMissingDataPackages(string modelName, string packageName)
        {
            var exception = Record.Exception(() => TiktokenTokenizer.CreateForModel(modelName));
            Assert.NotNull(exception);
            Assert.Contains(packageName, exception.Message);
        }

        public static IEnumerable<object[]> ModelUrlData()
        {
            yield return new object[] { @"https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken" };
            yield return new object[] { @"https://fossies.org/linux/misc/whisper-20231117.tar.gz/whisper-20231117/whisper/assets/gpt2.tiktoken?m=b" };
            yield return new object[] { @"https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken" };
            yield return new object[] { @"https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken" };
            yield return new object[] { @"https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" };
        }

        [Theory]
        [MemberData(nameof(ModelUrlData))]
        public async Task TestTokenizerCreationWithProvidedData(string url)
        {
            string tokenizerDataFileName = Utils.CreateTemporaryFile("tiktoken");
            await Utils.DownloadFile(url, tokenizerDataFileName);

            try
            {
                TiktokenTokenizer externalTokenizer = TiktokenTokenizer.Create(tokenizerDataFileName, preTokenizer: null, normalizer: null);
                Assert.NotNull(externalTokenizer);
            }
            finally
            {
                Utils.DeleteFile(tokenizerDataFileName);
            }
        }
    }
}

