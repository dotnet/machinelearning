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
                (string Text, int Offset, int Length, int TokenCount) result1 = tokenizer.TrimWithinTokenLimit(input, maxTokenCount: i, fromStart: true);
                (string Text, int Offset, int Length, int TokenCount) result2 = tokenizer.TrimWithinTokenLimit(input, maxTokenCount: i, fromStart: false);

                IReadOnlyList<int>? prefixIds = null;
                IReadOnlyList<int>? suffixIds = null;

                if (result1.TokenCount > 0)
                {
                    Assert.Equal(0, result1.Offset);
                    string prefixString = result1.Text.Substring(result1.Offset, result1.Length);
                    prefixIds = tokenizer.EncodeToIds(prefixString);
                    Assert.Equal(result1.TokenCount, prefixIds.Count);
                    Assert.Equal(prefixIds, fullIdsList.Take(prefixIds.Count));
                }

                if (result2.TokenCount > 0)
                {
                    Assert.Equal(result2.Text.Length, result2.Offset + result2.Length);
                    string suffixString = result2.Text.Substring(result2.Offset, result2.Length);
                    suffixIds = tokenizer.EncodeToIds(suffixString);
                    Assert.Equal(result2.TokenCount, suffixIds.Count);
                    Assert.Equal(suffixIds, fullIdsList.Skip(fullIdsList.Count - suffixIds.Count));
                }

                if (i == fullIdsList.Count)
                {
                    Assert.Equal(result1.Text.Length, result1.Length);
                    Assert.Equal(result2.Text.Length, result2.Length);
                    Assert.Equal(fullIdsList, prefixIds);
                    Assert.Equal(fullIdsList, suffixIds);
                }
            }

            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.TrimWithinTokenLimit(input, maxTokenCount: 0, fromStart: true));
            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.TrimWithinTokenLimit(input, maxTokenCount: -1, fromStart: true));
            Assert.Throws<ArgumentNullException>(() => tokenizer.TrimWithinTokenLimit(null!, maxTokenCount: 0, fromStart: false));
        }
    }
}