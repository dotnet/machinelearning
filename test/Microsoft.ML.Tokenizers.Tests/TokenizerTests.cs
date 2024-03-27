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

                // It is possible with Llama tokenizer to produce start of sentence token <s> token only if we have the maxTokenCount is 1.
                // In this case, we'll get index1 equal to zero and nothing really will need to be tested.
                if (tokenCount1 > 0 && index1 > 0)
                {
                    string prefixString = processedText1.Substring(0, index1);

                    if (tokenizer.Model is SentencePieceBpe)
                    {
                        // SentencePieceBpe model normalize the text and insert more characters.
                        // We call the model directly to bypass the normalization step
                        prefixIds = new List<int>();
                        tokenizer.Model.EncodeToIds(prefixString.AsSpan(), (prefixIds as IList<int>)!, out _);
                    }
                    else
                    {
                        prefixIds = tokenizer.EncodeToIds(prefixString);
                    }
                    Assert.Equal(tokenCount1, prefixIds.Count);
                    Assert.Equal(prefixIds, fullIdsList.Take(prefixIds.Count));
                }

                if (tokenCount2 > 0)
                {
                    string suffixString = processedText2.Substring(index2);

                    if (tokenizer.Model is SentencePieceBpe)
                    {
                        // SentencePieceBpe model normalize the text and insert more characters.
                        // We call the model directly to bypass the normalization step
                        suffixIds = new List<int>();
                        tokenizer.Model.EncodeToIds(suffixString.AsSpan(), (suffixIds as IList<int>)!, out _);
                        if (i < fullIdsList.Count)
                        {
                            suffixIds = suffixIds.Skip(1).ToList(); // Skip the start of sentence token <s>
                        }
                    }
                    else
                    {
                        suffixIds = tokenizer.EncodeToIds(suffixString);
                    }

                    Assert.Equal(tokenCount2, suffixIds.Count);
                    Assert.Equal(suffixIds, fullIdsList.Skip(fullIdsList.Count - suffixIds.Count));
                }

                if (i == fullIdsList.Count)
                {
                    if (index1 != processedText1.Length)
                    {
                        // It's possible that the remaining text on the left doesn't produce any tokens, as in the case of BPE,
                        // where the pre-tokenizer removes spaces and the left text consists entirely of spaces.
                        Assert.True(index1 < processedText1.Length);
                        Assert.Equal(0, tokenizer.CountTokens(processedText1.Substring(index1)));
                    }

                    if (index2 != 0)
                    {
                        // It's possible that the remaining text on the right doesn't produce any tokens, as in the case of BPE,
                        // where the pre-tokenizer removes spaces and the left text consists entirely of spaces.
                        Assert.True(index2 > 0);
                        Assert.Equal(0, tokenizer.CountTokens(processedText1.Substring(0, index2)));
                    }

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