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
                    new Split[] { new Split("How", (0, 3)), new Split("are", (4, 7)), new Split("you", (8, 11)), new Split("doing", (12, 17)), new Split("?", (17, 18)),}
                };

                yield return new object[]
                {
                    WhiteSpace.Instance,
                    "I_am_Just_Fine!",
                    new Split[] { new Split("I_am_Just_Fine", (0, 14)), new Split("!", (14, 15)) }
                };

                yield return new object[]
                {
                    new SpacePreTokenizer(),
                    "How are    you doing?!",
                    new Split[] { new Split("How", (0, 3)), new Split("are", (4, 7)), new Split("you", (11, 14)), new Split("doing?!", (15, 22)) }
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
        public void TestPreTokenizer(PreTokenizer preTokenizer, string sentence, Split[] splits)
        {
            Split[] splitParts = preTokenizer.PreTokenize(sentence).ToArray<Split>();
            Assert.Equal(splits, splitParts);

            // Empty tokenizer which tokenize all parts as unknown tokens.
            Bpe bpe = new Bpe();

            Tokenizer tokenizer = new Tokenizer(bpe, preTokenizer);

            TokenizerResult encoding = tokenizer.Encode(sentence);
            Assert.True(encoding.Tokens.Count >= splitParts.Length, $"Expected to have {encoding.Tokens.Count} >= {splitParts.Length}");
        }

        [Fact]
        public void TestWhiteSpacePreTokenizer()
        {
            WhiteSpace.Instance.PreTokenize(null);
        }

        public class SpacePreTokenizer : PreTokenizer
        {
            public override IReadOnlyList<Split> PreTokenize(string sentence)
            {
                List<Split> splits = new();
                if (string.IsNullOrEmpty(sentence))
                {
                    return splits;
                }

                int index = 0;
                while (true)
                {
                    while (index < sentence.Length && char.IsWhiteSpace(sentence[index]))
                    {
                        index++;
                    }

                    int end = index + 1;
                    while (end < sentence.Length && !char.IsWhiteSpace(sentence[end]))
                    {
                        end++;
                    }

                    if (index < sentence.Length)
                    {
                        splits.Add(new Split(sentence.Substring(index, end - index), (index, end)));
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
