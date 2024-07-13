// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;

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
        public SentencePieceNormalizer(bool removeExtraWhiteSpaces, bool addDummyPrefix, bool escapeWhiteSpaces, bool treatWhitespaceAsSuffix, IReadOnlyDictionary<string, int>? specialTokens)
        {
            RemoveExtraWhiteSpaces = removeExtraWhiteSpaces;
            AddDummyPrefix = addDummyPrefix;
            EscapeWhiteSpaces = escapeWhiteSpaces;
            TreatWhitespaceAsSuffix = treatWhitespaceAsSuffix;
            SpecialTokens = specialTokens;
        }

        /// <summary>
        /// Indicate removing extra white spaces from the original string during the normalization.
        /// </summary>
        public bool RemoveExtraWhiteSpaces { get; }

        /// <summary>
        /// Indicate emitting the dummy prefix character U+2581 at the beginning of sentence token during the encoding.
        /// </summary>
        public bool AddDummyPrefix { get; }

        /// <summary>
        /// Indicate escaping white spaces by adding the dummy prefix character U+2581.
        /// </summary>
        public bool EscapeWhiteSpaces { get; }

        /// <summary>
        /// Indicate treating white space as suffix.
        /// </summary>
        public bool TreatWhitespaceAsSuffix { get; private set; }

        /// <summary>
        /// Indicate the added tokens.
        /// </summary>
        public IReadOnlyDictionary<string, int>? SpecialTokens { get; }

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

            int spanLength = AddDummyPrefix ? length + 1 : length;

            if (span.Length < spanLength)
            {
                // Add dummy prefix if needed
                buffer = ArrayPool<char>.Shared.Rent(spanLength);
                span = buffer;
            }

            span = span.Slice(0, spanLength);

            int bufferIndex = 0;
            if (AddDummyPrefix && !TreatWhitespaceAsSuffix)
            {
                if (SpecialTokens is not null)
                {
                    InsertDummyPrefix(original, ref startIndex, endIndex, span, ref bufferIndex);
                }
                else
                {
                    span[bufferIndex++] = EscapeWhiteSpaces ? DummyPrefix : ' ';
                }
            }

            int originalStart = startIndex;

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
                if (SpecialTokens is not null)
                {
                    InsertDummyPrefixAtEnd(span, ref bufferIndex);
                }
                else
                {
                    // Add dummy prefix if needed
                    span[bufferIndex++] = EscapeWhiteSpaces ? DummyPrefix : ' ';
                }
            }

            string result = span.Slice(0, bufferIndex).ToString();

            if (buffer is not null)
            {
                ArrayPool<char>.Shared.Return(buffer);
            }
            return result;
        }

        private void InsertDummyPrefix(ReadOnlySpan<char> original, ref int startIndex, int endIndex, Span<char> span, ref int bufferIndex)
        {
            int currentStartIndex;
            endIndex++;

            do
            {
                currentStartIndex = startIndex;
                foreach (var kvp in SpecialTokens!)
                {
                    var token = kvp.Key;
                    var tokenLength = token.Length;
                    if (startIndex + tokenLength <= endIndex && original.Slice(startIndex, tokenLength).SequenceEqual(token.AsSpan()))
                    {
                        token.AsSpan().CopyTo(span.Slice(bufferIndex));
                        bufferIndex += tokenLength;
                        startIndex += tokenLength;
                        break;
                    }
                }
            } while (currentStartIndex < startIndex);

            if (startIndex < endIndex)
            {
                // prefix should be followed with more characters, otherwise startIndex should be greater endIndex
                Debug.Assert(bufferIndex < span.Length - 1);
                span[bufferIndex++] = EscapeWhiteSpaces ? DummyPrefix : ' ';
            }
        }

        private void InsertDummyPrefixAtEnd(Span<char> span, ref int bufferIndex)
        {
            int currentIndex;
            int currentBufferIndex = bufferIndex - 1;

            if (currentBufferIndex < 0)
            {
                return;
            }

            do
            {
                currentIndex = currentBufferIndex;
                foreach (var kvp in SpecialTokens!)
                {
                    var token = kvp.Key;
                    var tokenLength = token.Length;
                    if (currentIndex >= tokenLength - 1 && span.Slice(currentIndex - tokenLength + 1, tokenLength).SequenceEqual(token.AsSpan()))
                    {
                        currentBufferIndex -= tokenLength;
                        break;
                    }
                }
            } while (currentBufferIndex > 0 && currentBufferIndex < currentIndex);

            if (currentBufferIndex > 0)
            {
                // prefix should be proceeded with more characters, otherwise currentBufferIndex should be 0 or less
                Debug.Assert(bufferIndex < span.Length);
                int i = bufferIndex;
                while (i > currentBufferIndex + 1)
                {
                    span[i] = span[i - 1];
                    i--;
                }
                span[currentBufferIndex + 1] = EscapeWhiteSpaces ? DummyPrefix : ' ';
                bufferIndex++;
            }
        }
    }
}
