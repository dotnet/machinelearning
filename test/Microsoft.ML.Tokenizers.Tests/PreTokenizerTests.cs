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
                    WhiteSpace.Instance,
                    "How are you doing?",
                    new Split[] { new Split("How", (0, 3)), new Split("are", (4, 3)), new Split("you", (8, 3)), new Split("doing", (12, 5)), new Split("?", (17, 1)),}
                };

                yield return new object[]
                {
                    WhiteSpace.Instance,
                    "I_am_Just_Fine!",
                    new Split[] { new Split("I_am_Just_Fine", (0, 14)), new Split("!", (14, 1)) }
                };

                yield return new object[]
                {
                    new SpacePreTokenizer(),
                    "How are    you doing?!",
                    new Split[] { new Split("How", (0, 3)), new Split("are", (4, 3)), new Split("you", (11, 3)), new Split("doing?!", (15, 7)) }
                };

                yield return new object[]
                {
                    new SpacePreTokenizer(),
                    new string(' ', 100),
                    new Split[] { }
                };
            }
        }

        [Theory]
        [MemberData(nameof(PreTokenizerData))]
        public void TestPreTokenizer(PreTokenizer preTokenizer, string text, Split[] splits)
        {
            Split[] splitParts = preTokenizer.PreTokenize(text).ToArray<Split>();
            Assert.Equal(splits, splitParts);

            // Empty tokenizer which tokenize all parts as unknown tokens.
            Tokenizer tokenizer = BpeTests.CreateEmptyBpe(normalizer: null, preTokenizer: preTokenizer);

            IReadOnlyList<Token> encoding = tokenizer.Encode(text, out _);
            Assert.True(encoding.Count >= splitParts.Length, $"Expected to have {encoding.Count} >= {splitParts.Length}");
        }

        [Fact]
        public void TestWhiteSpacePreTokenizer()
        {
            Assert.Empty(WhiteSpace.Instance.PreTokenize((string)null!));
        }

        public class SpacePreTokenizer : PreTokenizer
        {
            public override IEnumerable<Split> PreTokenize(string text)
            {
                List<Split> splits = new();
                if (string.IsNullOrEmpty(text))
                {
                    return splits;
                }

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
                        splits.Add(new Split(text.Substring(index, end - index), (index, end - index)));
                    }
                    else
                    {
                        break;
                    }

                    index = end + 1;
                }

                return splits;
            }
        }
    }
}
