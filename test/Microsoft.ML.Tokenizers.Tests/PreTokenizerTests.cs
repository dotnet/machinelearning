// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Tokenizers;
using System;
using System.Linq;
using System.Collections.Generic;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class PreTokenizersTests
    {
        public static IEnumerable<object[]> PreTokenizerData
        {
            get
            {
                yield return new object[]
                {
                    PreTokenizer.CreateWhiteSpace(),
                    "How are you doing?",
                    new (int Offset, int Length)[] { (0, 3), (4, 3), (8, 3), (12, 5), (17, 1), }
                };

                yield return new object[]
                {
                    PreTokenizer.CreateWhiteSpace(),
                    "I_am_Just_Fine!",
                    new (int Offset, int Length)[] { (0, 14), (14, 1) }
                };

                yield return new object[]
                {
                    new SpacePreTokenizer(),
                    "How are    you doing?!",
                    new (int Offset, int Length)[] { (0, 3), (4, 3), (11, 3), (15, 7) }
                };

                yield return new object[]
                {
                    new SpacePreTokenizer(),
                    new string(' ', 100),
                    new (int Offset, int Length)[] { }
                };
            }
        }

        [Theory]
        [MemberData(nameof(PreTokenizerData))]
        public void TestPreTokenizer(PreTokenizer preTokenizer, string text, (int Offset, int Length)[] splits)
        {
            (int Offset, int Length)[] splitParts = preTokenizer.PreTokenize(text).ToArray<(int Offset, int Length)>();
            Assert.Equal(splits, splitParts);

            // Empty tokenizer which tokenize all parts as unknown tokens.
            Tokenizer tokenizer = BpeTests.CreateEmptyBpe(normalizer: null, preTokenizer: preTokenizer);

            IReadOnlyList<EncodedToken> encoding = tokenizer.EncodeToTokens(text, out _);
            Assert.True(encoding.Count >= splitParts.Length, $"Expected to have {encoding.Count} >= {splitParts.Length}");
        }

        [Fact]
        public void TestWhiteSpacePreTokenizer()
        {
            Assert.Empty(PreTokenizer.CreateWhiteSpace().PreTokenize((string)null!));
        }

        public class SpacePreTokenizer : PreTokenizer
        {
            public override IEnumerable<(int Offset, int Length)> PreTokenize(ReadOnlySpan<char> text)
            {
                if (text.IsEmpty)
                {
                    return [];
                }

                List<(int Offset, int Length)> splits = new();

                int index = 0;
                while (true)
                {
                    while (index < text.Length && char.IsWhiteSpace(text[index]))
                    {
                        index++;
                    }

                    int end = index + 1;
                    while (end < text.Length && !char.IsWhiteSpace(text[end]))
                    {
                        end++;
                    }

                    if (index < text.Length)
                    {
                        splits.Add((index, end - index));
                    }
                    else
                    {
                        break;
                    }

                    index = end + 1;
                }

                return splits;
            }

            public override IEnumerable<(int Offset, int Length)> PreTokenize(string text)
            {
                if (string.IsNullOrEmpty(text))
                {
                    return [];
                }

                return PreTokenize(text.AsSpan());
            }
        }
    }
}
