// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Normalize the string to lowercase form before processing it with the tokenizer.
    /// </summary>
    public sealed class SentencePieceNormalizer : Normalizer
    {
        internal const char DummyPrefix = '\u2581'; // '▁' (LOWER ONE EIGHT BLOCK)

        /// <summary>
        /// Creates a LowerCaseNormalizer object.
        /// </summary>
        public SentencePieceNormalizer(bool removeExtraWhiteSpaces, bool addDummyPrefix, bool escapeWhiteSpaces, bool treatWhitespaceAsSuffix)
        {
            RemoveExtraWhiteSpaces = removeExtraWhiteSpaces;
            AddDummyPrefix = addDummyPrefix;
            EscapeWhiteSpaces = escapeWhiteSpaces;
            TreatWhitespaceAsSuffix = treatWhitespaceAsSuffix;
        }

        /// <summary>
        /// Indicate removing extra white spaces from the original string during the normalization.
        /// </summary>
        public bool RemoveExtraWhiteSpaces { get; }

        /// <summary>
        /// Indicate emitting the dummy prefix character U+2581 at the beginning of sentence token during the encoding.
        /// </summary>
        public bool AddDummyPrefix { get; }

        public bool EscapeWhiteSpaces { get; }

        public bool TreatWhitespaceAsSuffix { get; }

        /// <summary>
        /// Normalize the original string according to SentencePiece normalization.
        /// </summary>
        /// <param name="original">The original string to normalize.</param>
        /// <returns>The normalized string.</returns>
        public override string Normalize(string original)
        {
            if (string.IsNullOrEmpty(original))
            {
                return string.Empty;
            }

            return Normalize(original.AsSpan());
        }

        /// <summary>
        /// Normalize the original string according to SentencePiece normalization.
        /// </summary>
        /// <param name="original">The original string to normalize.</param>
        /// <returns>The normalized string.</returns>
        public override string Normalize(ReadOnlySpan<char> original)
        {
            int startIndex = 0;
            int endIndex = original.Length - 1;

            if (RemoveExtraWhiteSpaces)
            {
                while (startIndex <= endIndex && original[startIndex] == ' ')
                {
                    startIndex++;
                }

                while (endIndex >= startIndex && original[endIndex] == ' ')
                {
                    endIndex--;
                }

                if (startIndex == endIndex)
                {
                    return string.Empty;
                }
            }

            int length = endIndex - startIndex + 1;

            Span<char> span = stackalloc char[512];
            char[]? buffer = null;

            if (span.Length < length + 1)
            {
                // Add dummy prefix if needed
                buffer = ArrayPool<char>.Shared.Rent(AddDummyPrefix ? length + 1 : length);
                span = buffer;
            }

            int bufferIndex = 0;
            if (AddDummyPrefix && !TreatWhitespaceAsSuffix)
            {
                span[bufferIndex++] = EscapeWhiteSpaces ? DummyPrefix : ' ';
            }

            while (startIndex <= endIndex)
            {
                char c = original[startIndex++];
                if (c == ' ')
                {
                    span[bufferIndex++] = EscapeWhiteSpaces ? DummyPrefix : c;

                    if (RemoveExtraWhiteSpaces)
                    {
                        while (startIndex <= endIndex && original[startIndex] == ' ')
                        {
                            startIndex++;
                        }
                    }
                }
                else
                {
                    span[bufferIndex++] = c;
                }
            }

            if (AddDummyPrefix && TreatWhitespaceAsSuffix)
            {
                span[bufferIndex++] = EscapeWhiteSpaces ? DummyPrefix : ' ';
            }

            string result = span.Slice(0, bufferIndex).ToString();

            if (buffer is not null)
            {
                ArrayPool<char>.Shared.Return(buffer);
            }
            return result;
        }
    }
}
