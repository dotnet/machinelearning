// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Normalizer that performs the Bert model normalization.
    /// </summary>
    internal sealed class BertNormalizer : Normalizer
    {
        private readonly bool _lowerCase;
        private readonly bool _individuallyTokenizeCjk;
        private readonly bool _removeNonSpacingMarks;

        /// <summary>
        /// Normalize the input string.
        /// </summary>
        /// <param name="original">The input string to normalize.</param>
        /// <returns>The normalized string.</returns>
        public override string Normalize(string original)
        {
            if (string.IsNullOrEmpty(original))
            {
                return string.Empty;
            }

            if (_removeNonSpacingMarks)
            {
                original = original.Normalize(NormalizationForm.FormD);
            }

            Span<char> casingBuffer = stackalloc char[10];
            char[] buffer = ArrayPool<char>.Shared.Rent(original.Length);
            int index = 0;

            for (int i = 0; i < original.Length; i++)
            {
                char c = original[i];

                if (c == '\u0000' || c == '\uFFFD')
                {
                    continue;
                }

                int inc = 0;
                int codePoint = (int)c;
                if (char.IsHighSurrogate(c) && i + 1 < original.Length && char.IsLowSurrogate(original[i + 1]))
                {
                    codePoint = char.ConvertToUtf32(c, original[i + 1]);
                    inc = 1;
                }

                UnicodeCategory category = CharUnicodeInfo.GetUnicodeCategory(original, i);

                if (category == UnicodeCategory.Control)
                {
                    i += inc;
                    continue;
                }

                if (category == UnicodeCategory.SpaceSeparator)
                {
                    AddChar(ref buffer, ref index, ' ');
                    i += inc;
                    continue;
                }

                if (_removeNonSpacingMarks && category is UnicodeCategory.NonSpacingMark)
                {
                    i += inc;
                    continue;
                }

                if (_lowerCase && category == UnicodeCategory.UppercaseLetter)
                {
                    int length = original.AsSpan().Slice(i, inc + 1).ToLowerInvariant(casingBuffer);
                    Debug.Assert(length > 0);

                    AddSpan(ref buffer, ref index, casingBuffer.Slice(0, length));

                    i += inc;
                    continue;
                }

                if (_individuallyTokenizeCjk && IsCjkChar(codePoint))
                {
                    AddChar(ref buffer, ref index, ' ');
                    AddChar(ref buffer, ref index, c);
                    if (inc > 0)
                    {
                        AddChar(ref buffer, ref index, original[i + 1]);
                    }
                    AddChar(ref buffer, ref index, ' ');

                    i += inc;
                    continue;
                }

                AddChar(ref buffer, ref index, c);
                if (inc > 0)
                {
                    AddChar(ref buffer, ref index, original[i + 1]);
                }
                i += inc;
            }

            string result = index == 0 ? string.Empty : new string(buffer, 0, index).Normalize(NormalizationForm.FormC);
            ArrayPool<char>.Shared.Return(buffer);
            return result;
        }

        /// <summary>
        /// Normalize the input character span.
        /// </summary>
        /// <param name="original">The input character span to normalize.</param>
        /// <returns>The normalized string.</returns>
        public override string Normalize(ReadOnlySpan<char> original)
        {
            if (original.IsEmpty)
            {
                return string.Empty;
            }

            return Normalize(original.ToString());
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BertNormalizer"/> class.
        /// </summary>
        /// <param name="lowerCase">Whether to lowercase the input.</param>
        /// <param name="individuallyTokenizeCjk">Whether to tokenize CJK characters.</param>
        /// <param name="removeNonSpacingMarks">Whether to strip accents from the input.</param>
        public BertNormalizer(bool lowerCase, bool individuallyTokenizeCjk, bool removeNonSpacingMarks)
        {
            _lowerCase = lowerCase;
            _individuallyTokenizeCjk = individuallyTokenizeCjk;
            _removeNonSpacingMarks = removeNonSpacingMarks;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void AddChar(ref char[] buffer, ref int index, char c)
        {
            if (index >= buffer.Length)
            {
                Helpers.ArrayPoolGrow(ref buffer, index + 40);
            }

            buffer[index++] = c;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void AddSpan(ref char[] buffer, ref int index, Span<char> chars)
        {
            if (index + chars.Length >= buffer.Length)
            {
                Helpers.ArrayPoolGrow(ref buffer, index + buffer.Length + 10);
            }

            chars.CopyTo(buffer.AsSpan(index));
            index += chars.Length;
        }

        /// <summary>
        /// Checks whether CP is the codepoint of a CJK character.
        /// This defines a "chinese character" as anything in the CJK Unicode block:
        ///   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        /// </summary>
        /// <param name="codePoint">The codepoint to check.</param>
        /// <remarks>
        /// The CJK Unicode block is NOT all Japanese and Korean characters,
        /// despite its name. The modern Korean Hangul alphabet is a different block,
        /// as is Japanese Hiragana and Katakana. Those alphabets are used to write
        /// space-separated words, so they are not treated specially and handled
        /// like the all of the other languages.
        /// </remarks>
        /// <returns>True if the codepoint is a CJK character, false otherwise.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsCjkChar(int codePoint)
        {
            return (codePoint > 0x3400) && // Quick check to exit early if the codepoint is outside of the CJK range
               (((uint)(codePoint - 0x3400) <= (uint)(0x4DBF - 0x3400)) ||
                ((uint)(codePoint - 0xF900) <= (uint)(0xFAFF - 0xF900)) ||
                ((uint)(codePoint - 0x4E00) <= (uint)(0x9FFF - 0x4E00)) ||
                ((uint)(codePoint - 0x20000) <= (uint)(0x2A6DF - 0x20000)) ||
                ((uint)(codePoint - 0x2A700) <= (uint)(0x2B73F - 0x2A700)) ||
                ((uint)(codePoint - 0x2B740) <= (uint)(0x2B81F - 0x2B740)) ||
                ((uint)(codePoint - 0x2B820) <= (uint)(0x2CEAF - 0x2B820)) ||
                ((uint)(codePoint - 0x2F800) <= (uint)(0x2FA1F - 0x2F800)));
        }
    }
}