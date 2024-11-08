// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    public class NormalizerTests
    {
        public static IEnumerable<object?[]> NormalizerData
        {
            get
            {
                yield return new object?[]
                {
                    LowerCaseNormalizer.Instance,
                    "How Are You Doing?",
                    "how are you doing?",
                };

                yield return new object?[]
                {
                    UpperCaseNormalizer.Instance,
                    "How Are You Doing?",
                    "HOW ARE YOU DOING?",
                };

                yield return new object?[]
                {
                    new RemoveQuotesNormalizer(),
                    "This is already normalized string",
                    "This is already normalized string",
                };

                yield return new object?[]
                {
                    new RemoveQuotesNormalizer(),
                    "String \"to\" normalize",
                    "String to normalize",
                };

                yield return new object?[]
                {
                    new UnicodeNormalizer(NormalizationForm.FormKD),
                    "\uFB01", // Composed form of the character 'fi' one character
                    "fi", // normalized in 2 characters 'f' and 'i'
                };
            }
        }

        [Theory]
        [MemberData(nameof(NormalizerData))]
        public void TestNormalizer(Normalizer normalizer, string text, string normalized)
        {
            string? normalizedText = normalizer.Normalize(text);
            Assert.Equal(normalized, normalizedText);

            Tokenizer tokenizer = BpeTests.CreateEmptyBpe(preTokenizer: null, normalizer);
            IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens(text, out normalizedText);
            Assert.Equal(normalized, normalizedText);
        }

        public class RemoveQuotesNormalizer : Normalizer
        {
            public override string Normalize(string original)
            {
                int index = original.IndexOf('"');
                if (index <= 0)
                {
                    return original;
                }

                return RemoveQuotes(original.AsSpan(), index);
            }

            public override string Normalize(ReadOnlySpan<char> original)
            {
                int index = original.IndexOf('"');
                if (index <= 0)
                {
                    return original.ToString();
                }

                return RemoveQuotes(original, index);
            }

            private string RemoveQuotes(ReadOnlySpan<char> original, int index)
            {
                StringBuilder sb = new StringBuilder(original.Length);
                List<int> mapping = new List<int>();

                int start = 0;

                do
                {
                    for (int i = start; i < index; i++)
                    {
                        sb.Append(original[i]);
                        mapping.Add(i);
                    }

                    start = index + 1;

                    if (start >= original.Length)
                    {
                        break;
                    }

                    index = original.Slice(start).IndexOf('"');
                    if (index <= 0)
                    {
                        for (int i = start; i < original.Length; i++)
                        {
                            sb.Append(original[i]);
                            mapping.Add(i);
                        }
                        break;
                    }

                    index += start;
                } while (true);

                return sb.ToString();
            }
        }

        public class UnicodeNormalizer : Normalizer
        {
            private NormalizationForm _normalizationForm;
            public UnicodeNormalizer(NormalizationForm form)
            {
                _normalizationForm = form;
            }

            public override string Normalize(string original)
            {
                if (string.IsNullOrEmpty(original))
                {
                    return string.Empty;
                }

                return original.Normalize(_normalizationForm);
            }

            public override string Normalize(ReadOnlySpan<char> original)
            {
                if (original.IsEmpty)
                {
                    return string.Empty;
                }

                return original.ToString().Normalize(_normalizationForm);
            }
        }
    }
}
