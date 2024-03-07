// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class TokenizerTests
    {
        internal static void TestTokenLimits(Tokenizer tokenizer)
        {
            string input = @"
                OpenAI's large language models (sometimes referred to as GPT's) process text using tokens, which are common sequences of characters found in a set of text.
                The models learn to understand the statistical relationships between these tokens, and excel at producing the next token in a sequence of tokens.
                You can use the tool below to understand how a piece of text might be tokenized by a language model, and the total count of tokens in that piece of text.
                It's important to note that the exact tokenization process varies between models. Newer models like GPT-3.5 and GPT-4 use a different tokenizer than previous models,
                and will produce different tokens for the same input text.
            ";

            IReadOnlyList<int> fullIdsList = tokenizer.EncodeToIds(input);

            for (int i = 1; i <= fullIdsList.Count; i++)
            {
                int index1 = tokenizer.IndexOfTokenCount(input, maxTokenCount: i, out string processedText1, out int tokenCount1);
                int index2 = tokenizer.LastIndexOfTokenCount(input, maxTokenCount: i, out string processedText2, out int tokenCount2);


                IReadOnlyList<int>? prefixIds = null;
                IReadOnlyList<int>? suffixIds = null;

                if (tokenCount1 > 0)
                {
                    string prefixString = processedText1.Substring(0, index1);
                    prefixIds = tokenizer.EncodeToIds(prefixString);
                    Assert.Equal(tokenCount1, prefixIds.Count);
                    Assert.Equal(prefixIds, fullIdsList.Take(prefixIds.Count));
                }

                if (tokenCount2 > 0)
                {
                    string suffixString = processedText2.Substring(index2);
                    suffixIds = tokenizer.EncodeToIds(suffixString);
                    Assert.Equal(tokenCount2, suffixIds.Count);
                    Assert.Equal(suffixIds, fullIdsList.Skip(fullIdsList.Count - suffixIds.Count));
                }

                if (i == fullIdsList.Count)
                {
                    Assert.Equal(processedText1.Length, index1);
                    Assert.Equal(0, index2);
                    Assert.Equal(fullIdsList, prefixIds);
                    Assert.Equal(fullIdsList, suffixIds);
                }
            }

            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.IndexOfTokenCount(input, maxTokenCount: 0, out _, out _));
            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.IndexOfTokenCount(input, maxTokenCount: -1, out _, out _));
            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.LastIndexOfTokenCount(input, maxTokenCount: 0, out _, out _));
            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.LastIndexOfTokenCount(input, maxTokenCount: -1, out _, out _));

            Assert.Throws<ArgumentNullException>(() => tokenizer.IndexOfTokenCount(null!, maxTokenCount: 10, out _, out _));
            Assert.Throws<ArgumentNullException>(() => tokenizer.LastIndexOfTokenCount(null!, maxTokenCount: 10, out _, out _));
        }
    }
}