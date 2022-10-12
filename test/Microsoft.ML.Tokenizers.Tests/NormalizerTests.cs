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
                    new LowerCaseNormalizer(),
                    "How Are You Doing?",
                    "how are you doing?",
                    true,   // IsOneToOneMapping
                    true,   // CanMapToOriginal
                    null,   // NormalizedToOriginalMapping
                };

                yield return new object?[]
                {
                    new UpperCaseNormalizer(),
                    "How Are You Doing?",
                    "HOW ARE YOU DOING?",
                    true,   // IsOneToOneMapping
                    true,   // CanMapToOriginal
                    null,   // NormalizedToOriginalMapping
                };

                yield return new object?[]
                {
                    new RemoveQuotesNormalizer(),
                    "This is already normalized string",
                    "This is already normalized string",
                    true,   // IsOneToOneMapping
                    true,   // CanMapToOriginal
                    null,   // NormalizedToOriginalMapping
                };

                yield return new object?[]
                {
                    new RemoveQuotesNormalizer(),
                    "String \"to\" normalize",
                    "String to normalize",
                    false,   // IsOneToOneMapping
                    true,    // CanMapToOriginal
                    new int[] { 0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 },    // NormalizedToOriginalMapping
                };

                yield return new object?[]
                {
                    new UnicodeNormalizer(NormalizationForm.FormKD),
                    "\uFB01", // Composed form of the character 'fi' one character
                    "fi", // normalized in 2 characters 'f' and 'i'
                    false,   // IsOneToOneMapping
                    false,    // CanMapToOriginal
                    null,    // NormalizedToOriginalMapping
                };
            }
        }

        [Theory]
        [MemberData(nameof(NormalizerData))]
        public void TestNormalizer(Normalizer normalizer, string sentence, string normalized, bool isOneToOneMapping, bool canMapToOriginal, int[] normalizedToOriginalMapping)
        {
            NormalizedString ns = normalizer.Normalize(sentence);
            Assert.Equal(normalized, ns.Normalized);
            Assert.Equal(isOneToOneMapping, ns.IsOneToOneMapping);
            Assert.Equal(canMapToOriginal, ns.CanMapToOriginal);
            Assert.Equal(normalizedToOriginalMapping, ns.NormalizedToOriginalMapping);

            Tokenizer tokenizer = new Tokenizer(new Bpe(), WhiteSpace.Instance, normalizer);
            TokenizerResult encoding = tokenizer.Encode(sentence);
            Assert.Equal(canMapToOriginal, encoding.OffsetsMappedToOriginalString);
            Assert.Equal(sentence, encoding.OriginalString);
            Assert.Equal(normalized, encoding.NormalizedString);
        }

        public class RemoveQuotesNormalizer : Normalizer
        {
            public override NormalizedString Normalize(string original)
            {
                int index = original.IndexOf('"');
                if (index <= 0)
                {
                    return new NormalizedString(original, original, null, true);
                }

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

                    index = original.IndexOf('"', start);
                    if (index <= 0)
                    {
                        for (int i = start; i < original.Length; i++)
                        {
                            sb.Append(original[i]);
                            mapping.Add(i);
                        }
                        break;
                    }
                } while (true);

                return new NormalizedString(original, sb.ToString(), mapping.ToArray(), false);
            }
        }

        public class UnicodeNormalizer : Normalizer
        {
            private NormalizationForm _normalizationForm;
            public UnicodeNormalizer(NormalizationForm form)
            {
                _normalizationForm = form;
            }

            public override NormalizedString Normalize(string original)
            {
                if (string.IsNullOrEmpty(original))
                {
                    return new NormalizedString(original, "", null, true);
                }

                return new NormalizedString(original, original.Normalize(_normalizationForm), null, false);
            }
        }
    }
}
