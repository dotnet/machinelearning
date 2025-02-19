// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class TokenizerTests
    {
        [Fact]
        public void Decode_DefaultImplementation()
        {
            var tokenizer = new EnglishAlphabetTokenizer();

            Assert.Equal("", tokenizer.Decode([]));

            Assert.Equal("hello", tokenizer.Decode([7, 4, 11, 11, 14]));

            Assert.Equal(
                string.Concat(Enumerable.Repeat("abcdefghijklmnopqrstuvwxyz", 100)),
                tokenizer.Decode(Enumerable.Repeat("abcdefghijklmnopqrstuvwxyz", 100).SelectMany(s => s.Select(c => c - 'a'))));

            Assert.Throws<InvalidOperationException>(() => tokenizer.Decode([26, 27, 28, 29]));
        }

        [Fact]
        public void EncodeToIds_DefaultImplementation()
        {
            var tokenizer = new EnglishAlphabetTokenizer();

            IReadOnlyList<int> ids = tokenizer.EncodeToIds("hello, world", 5, out string? normalizedText, out int charsConsumed);

            Assert.Equal([7, 4, 11, 11, 14], ids);
            Assert.Null(normalizedText);
            Assert.Equal(5, charsConsumed);
        }

        [Fact]
        public void CountTokens_DefaultImplementation()
        {
            var tokenizer = new EnglishAlphabetTokenizer();

            Assert.Equal(5, tokenizer.CountTokens("hello"));
        }

        [Fact]
        public void CountTokens_WithMaxTokenCount()
        {
            var tokenizer = new EnglishAlphabetTokenizer();

            Assert.Equal(3, tokenizer.CountTokens("hello", maxTokenCount: 3));
        }

        [Fact]
        public void GetIndexByTokenCount_DefaultImplementation()
        {
            var tokenizer = new EnglishAlphabetTokenizer();

            Assert.Equal(2, tokenizer.GetIndexByTokenCount("hello", 2, out string? normalizedText, out int tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(2, tokenCount);

            Assert.Equal(5, tokenizer.GetIndexByTokenCount("hello", 8, out normalizedText, out tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(5, tokenCount);
        }

        [Fact]
        public void GetIndexByTokenCountFromEnd_DefaultImplementation()
        {
            var tokenizer = new EnglishAlphabetTokenizer();

            Assert.Equal(3, tokenizer.GetIndexByTokenCountFromEnd("hello", 2, out string? normalizedText, out int tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(2, tokenCount);

            Assert.Equal(0, tokenizer.GetIndexByTokenCountFromEnd("hello", 8, out normalizedText, out tokenCount));
            Assert.Null(normalizedText);
            Assert.Equal(5, tokenCount);
        }

        private sealed class EnglishAlphabetTokenizer : Tokenizer
        {
            public override OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, out int idsConsumed, out int charsWritten)
            {
                int pos = 0;
                foreach (int i in ids)
                {
                    if (pos >= destination.Length)
                    {
                        charsWritten = idsConsumed = pos;
                        return OperationStatus.DestinationTooSmall;
                    }

                    if (i is < 0 or >= 26)
                    {
                        charsWritten = idsConsumed = pos;
                        return OperationStatus.InvalidData;
                    }

                    destination[pos++] = (char)('a' + i);
                }

                charsWritten = idsConsumed = pos;
                return OperationStatus.Done;
            }

            protected override EncodeResults<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
            {
                var tokens = new List<EncodedToken>();

                int count = 0;
                foreach (char c in textSpan)
                {
                    if (count >= settings.MaxTokenCount)
                        break;

                    tokens.Add(new EncodedToken(c - 'a', c.ToString(), new Range(count, count + 1)));
                    count++;
                }

                return new EncodeResults<EncodedToken> { Tokens = tokens, CharsConsumed = count };
            }
        }

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
                int index1 = tokenizer.GetIndexByTokenCount(input, maxTokenCount: i, out string? processedText1, out int tokenCount1);
                int index2 = tokenizer.GetIndexByTokenCountFromEnd(input, maxTokenCount: i, out string? processedText2, out int tokenCount2);
                IReadOnlyList<int> partialIdsList = tokenizer.EncodeToIds(input, maxTokenCount: i, out string? processedText, out int charsConsumed);

                Assert.True(processedText is null || charsConsumed <= processedText.Length);
                Assert.True(tokenizer.Normalizer is not null || processedText is null);

                Assert.Equal(fullIdsList.Take(partialIdsList.Count), partialIdsList);

                IReadOnlyList<int>? prefixIds = null;
                IReadOnlyList<int>? suffixIds = null;

                // It is possible with Llama tokenizer to produce start of sentence token <s> token only if we have the maxTokenCount is 1.
                // In this case, we'll get index1 equal to zero and nothing really will need to be tested.
                if (tokenCount1 > 0 && index1 > 0)
                {
                    string prefixString = (processedText1 ?? input).Substring(0, index1);

                    if (tokenizer is SentencePieceTokenizer)
                    {
                        // SentencePieceBpe model normalize the text and insert more characters.
                        // We call the model directly to bypass the normalization step
                        prefixIds = tokenizer.EncodeToIds(prefixString.AsSpan(), considerNormalization: false);
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
                    string suffixString = (processedText2 ?? input).Substring(index2);

                    if (tokenizer is SentencePieceTokenizer)
                    {
                        // SentencePieceBpe model normalize the text and insert more characters.
                        // We call the model directly to bypass the normalization step
                        suffixIds = tokenizer.EncodeToIds(suffixString.AsSpan(), considerNormalization: false);
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
                    string s = processedText1 ?? input;
                    if (index1 != s.Length)
                    {
                        // It's possible that the remaining text on the left doesn't produce any tokens, as in the case of BPE,
                        // where the pre-tokenizer removes spaces and the left text consists entirely of spaces.
                        Assert.True(index1 < s.Length);
                        Assert.Equal(0, tokenizer.CountTokens(s.Substring(index1)));
                    }

                    if (index2 != 0)
                    {
                        // It's possible that the remaining text on the right doesn't produce any tokens, as in the case of BPE,
                        // where the pre-tokenizer removes spaces and the left text consists entirely of spaces.
                        Assert.True(index2 > 0);
                        Assert.Equal(0, tokenizer.CountTokens(s.Substring(0, index2)));
                    }

                    Assert.Equal(fullIdsList, prefixIds);
                    Assert.Equal(fullIdsList, suffixIds);
                }
            }

            Assert.Equal(0, tokenizer.GetIndexByTokenCount((string)null!, maxTokenCount: 10, out _, out _));
            Assert.Equal(0, tokenizer.GetIndexByTokenCountFromEnd((string)null!, maxTokenCount: 10, out _, out _));
            Assert.Equal(0, tokenizer.GetIndexByTokenCount(Span<char>.Empty, maxTokenCount: 10, out _, out _));
            Assert.Equal(0, tokenizer.GetIndexByTokenCountFromEnd(Span<char>.Empty, maxTokenCount: 10, out _, out _));

            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.GetIndexByTokenCount(input, maxTokenCount: 0, out _, out _));
            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.GetIndexByTokenCount(input, maxTokenCount: -1, out _, out _));
            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.GetIndexByTokenCountFromEnd(input, maxTokenCount: 0, out _, out _));
            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.GetIndexByTokenCountFromEnd(input, maxTokenCount: -1, out _, out _));
        }
    }
}
