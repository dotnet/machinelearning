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
                };

                yield return new object?[]
                {
                    new UpperCaseNormalizer(),
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
            string normalizedText = normalizer.Normalize(text);
            Assert.Equal(normalized, normalizedText);

            Tokenizer tokenizer = new Tokenizer(BpeTests.CreateEmptyBpe(), WhiteSpace.Instance, normalizer);
            EncodingResult encoding = tokenizer.Encode(text);
            Assert.Equal(text, encoding.OriginalString);
            Assert.Equal(normalized, encoding.NormalizedString);
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
        }
    }
}
