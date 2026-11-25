// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Removes the non-spacing marks (for example, accents) that are part of precomposed characters, without using
    /// <see cref="string.Normalize(NormalizationForm)"/>, which returns its input unchanged when globalization-invariant mode is enabled.
    /// </summary>
    internal static partial class NonSpacingMarkRemover
    {
        // No character below this one is in the table.
        private const char FirstCandidate = '\u00C0';

        /// <summary>
        /// Replaces each precomposed character that contains non-spacing marks with the character without them.
        /// For example, "Café" becomes "Cafe". Like decomposing the text (NFD), it leaves standalone non-spacing marks for the caller to remove.
        /// </summary>
        /// <param name="text">The text to remove the non-spacing marks from.</param>
        /// <returns>The text with those characters replaced, or <paramref name="text"/> itself when it has none.</returns>
        internal static string Remove(string text)
        {
            ValueStringBuilder result = default;
            bool changed = false;
            int pendingStart = 0; // Start of the characters not copied to the result yet.

            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];
                if (c < FirstCandidate)
                {
                    continue;
                }

                int codePoint = c;
                int charCount = 1;
                if (char.IsHighSurrogate(c) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
                {
                    codePoint = char.ConvertToUtf32(c, text[i + 1]);
                    charCount = 2;
                }

                int index = Array.BinarySearch(_codePoints, codePoint);

                if (index >= 0)
                {
                    if (!changed)
                    {
                        result = new ValueStringBuilder(text.Length);
                        changed = true;
                    }

                    result.Append(text.AsSpan(pendingStart, i - pendingStart));
                    pendingStart = i + charCount;

                    // Each replacement takes two UTF-16 code units, the second one being '\0' when unused.
                    result.Append(Replacements[2 * index]);
                    if (Replacements[2 * index + 1] != '\0')
                    {
                        result.Append(Replacements[2 * index + 1]);
                    }
                }

                i += charCount - 1;
            }

            if (!changed)
            {
                return text;
            }

            result.Append(text.AsSpan(pendingStart));
            return result.ToString();
        }
    }
}
