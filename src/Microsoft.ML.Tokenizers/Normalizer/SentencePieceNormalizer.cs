// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Normalize the string according to SentencePiece normalization.
    /// </summary>
    public sealed class SentencePieceNormalizer : Normalizer
    {
        // Maximum size of the return value of Trie, which corresponds to the maximum size of shared common prefix in the chars map.
        private const int MaxTrieResultsSize = 32;
        internal const char DummyPrefix = '\u2581'; // '▁' (LOWER ONE EIGHT BLOCK)
        private static readonly byte[] _spaceSymbol = { 0xe2, 0x96, 0x81 }; // Utf8 of DummyPrefix; Null terminated.
        private static readonly byte[] _space = { (byte)' ' };
        private static readonly byte[] _replacementBytes = { 0xEF, 0xBF, 0xBD, 0 }; // Utf8 of 0xFFFD; Null terminated.

        private readonly DoubleArrayTrie? _trie;
        private readonly byte[]? _normalized;

        /// <summary>
        /// Creates a SentencePieceNormalizer object.
        /// </summary>
        public SentencePieceNormalizer(bool removeExtraWhiteSpaces, bool addDummyPrefix, bool escapeWhiteSpaces, bool treatWhitespaceAsSuffix, IReadOnlyDictionary<string, int>? specialTokens)
        {
            RemoveExtraWhiteSpaces = removeExtraWhiteSpaces;
            AddDummyPrefix = addDummyPrefix;
            EscapeWhiteSpaces = escapeWhiteSpaces;
            TreatWhitespaceAsSuffix = treatWhitespaceAsSuffix;
            SpecialTokens = specialTokens;
        }

        internal SentencePieceNormalizer(
                    ReadOnlySpan<byte> precompiledCharsMap,
                    bool removeExtraWhiteSpaces,
                    bool addDummyPrefix,
                    bool escapeWhiteSpaces,
                    bool treatWhitespaceAsSuffix,
                    IReadOnlyDictionary<string, int>? specialTokens) : this(removeExtraWhiteSpaces, addDummyPrefix, escapeWhiteSpaces, treatWhitespaceAsSuffix, specialTokens)
        {
            if (precompiledCharsMap.IsEmpty)
            {
                return;
            }

            DecodePrecompiledCharsMap(precompiledCharsMap, out DoubleArrayUnit[]? trieBlob, out _normalized);

            Debug.Assert(trieBlob is not null);
            _trie = new DoubleArrayTrie(trieBlob!);
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

        // Returns the longest consumed prefix of |input| that can be normalized.
        // if we return normalizedPrefix == default, means no normalization and the original input span should be used.
        private int NormalizePrefix(ReadOnlySpan<byte> input, out Memory<byte> normalizedPrefix)
        {
            Debug.Assert(!input.IsEmpty);

            int longestLength = 0;
            int longestValue = 0;

            if (_trie is not null)
            {
                // Allocates trie_results in stack, which makes the encoding speed 36% faster. (38k sentences/sec => 60k sentences/sec).
                // Builder checks that the result size never exceeds kMaxTrieResultsSize. This array consumes 0.5kByte in stack,
                // which is less than default stack frames (16kByte).
                Span<DoubleArrayResultPair> trieResults = stackalloc DoubleArrayResultPair[MaxTrieResultsSize];

                int numNodes = _trie.CommonPrefixSearch(input, trieResults);

                // Finds the longest rule.
                for (int k = 0; k < numNodes; ++k)
                {
                    if (longestLength == 0 || trieResults[k].Length > longestLength)
                    {
                        longestLength = trieResults[k].Length;  // length of prefix
                        longestValue = trieResults[k].Value;    // pointer to |_normalized|.
                    }
                }
            }

            int result;

            if (longestLength == 0)
            {
                if (!Helpers.IsValidDecodeUtf8(input, out int length))
                {
                    // Found a malformed utf8.
                    // The rune is set to be 0xFFFD (REPLACEMENT CHARACTER), which is a valid Unicode of three bytes in utf8, but here we only consume one byte.
                    result = 1;
                    normalizedPrefix = new Memory<byte>(_replacementBytes, 0, 3);
                }
                else
                {
                    result = length;
                    normalizedPrefix = default;
                }
            }
            else
            {
                Debug.Assert(_normalized is not null);

                result = longestLength;

                // Calculate the length of the normalized prefix.
                int normalizedLength = longestValue;
                while (normalizedLength < _normalized!.Length && _normalized[normalizedLength] != 0)
                {
                    normalizedLength++;
                }
                normalizedPrefix = new Memory<byte>(_normalized, longestValue, normalizedLength - longestValue);
            }

            return result;
        }

        internal int Normalize(ReadOnlySpan<byte> input, ref Span<byte> normalized, ref byte[]? poolArray)
        {
            if (input.IsEmpty)
            {
                return 0;
            }

            int consumed = 0;

            // Ignores heading space.
            if (RemoveExtraWhiteSpaces)
            {
                while (!input.IsEmpty)
                {
                    int p = NormalizePrefix(input, out Memory<byte> normalizedPrefix);

                    Debug.Assert(p > 0);

                    if (p != 1)
                    {
                        break;
                    }

                    ReadOnlySpan<byte> normalizedByte = normalizedPrefix.Equals(default(Memory<byte>)) ? input.Slice(0, p) : normalizedPrefix.Span;
                    if (normalizedByte[0] != (byte)' ')
                    {
                        break;
                    }

                    input = input.Slice(p);
                    consumed += p;
                }
            }

            // all chars are whitespace.
            if (input.IsEmpty)
            {
                return 0;
            }

            int normalizedIndex = 0;

            // Adds a space symbol as a prefix (default is true) With this prefix, "world" and "hello world" are converted into
            // "_world" and "_hello_world", which help the trainer to extract "_world" as one symbol.
            if (!TreatWhitespaceAsSuffix && AddDummyPrefix)
            {
                AddWhiteSpace(this, normalized, ref normalizedIndex, ref poolArray);
            }

            bool isPrevSpace = RemoveExtraWhiteSpaces;

            while (!input.IsEmpty)
            {
                int p = NormalizePrefix(input, out Memory<byte> normalizedPrefix);
                ReadOnlySpan<byte> sp = normalizedPrefix.Equals(default(Memory<byte>)) ? input.Slice(0, p) : normalizedPrefix.Span;

                // Removes heading spaces in sentence piece, if the previous sentence piece ends with whitespace.
                while (isPrevSpace && sp.Length > 0 && sp[0] == (byte)' ')
                {
                    sp = sp.Slice(1);
                }

                if (!sp.IsEmpty)
                {
                    for (int n = 0; n < sp.Length; ++n)
                    {
                        if (EscapeWhiteSpaces && sp[n] == ' ')
                        {
                            if (normalized.Length <= normalizedIndex + _spaceSymbol.Length)
                            {
                                Helpers.ArrayPoolGrow(ref normalized, ref poolArray, (normalizedIndex + _spaceSymbol.Length) << 1);
                            }

                            // replace ' ' with _spaceSymbol.
                            _spaceSymbol.AsSpan().CopyTo(normalized.Slice(normalizedIndex));
                            normalizedIndex += _spaceSymbol.Length;

                        }
                        else
                        {
                            if (normalized.Length <= normalizedIndex + 1)
                            {
                                Helpers.ArrayPoolGrow(ref normalized, ref poolArray, (normalizedIndex + 1) << 1);
                            }

                            normalized[normalizedIndex++] = sp[n];

                        }
                    }

                    // Checks whether the last character of sp is whitespace.
                    isPrevSpace = sp[sp.Length - 1] == (byte)' ';
                }

                input = input.Slice(p);

                if (!RemoveExtraWhiteSpaces)
                {
                    isPrevSpace = false;
                }
            }

            // Ignores trailing space.
            if (RemoveExtraWhiteSpaces)
            {
                Span<byte> space = EscapeWhiteSpaces ? _spaceSymbol : _space;
                while (normalized.Slice(0, normalizedIndex).EndsWith(space))
                {
                    int length = normalizedIndex - space.Length;
                    if (length < 0)
                    {
                        return normalizedIndex;
                    }

                    normalizedIndex = length; // cut spaces

                }
            }

            // Adds a space symbol as a suffix (default is false)
            if (TreatWhitespaceAsSuffix && AddDummyPrefix)
            {
                AddWhiteSpace(this, normalized, ref normalizedIndex, ref poolArray);
            }

            return normalizedIndex;

            // adds _spaceSymbol to the current context.
            static void AddWhiteSpace(SentencePieceNormalizer normalizer, Span<byte> normalized, ref int normalizedIndex, ref byte[]? poolArray)
            {
                if (normalizer.EscapeWhiteSpaces)
                {
                    if (normalized.Length <= normalizedIndex + _spaceSymbol.Length)
                    {
                        Helpers.ArrayPoolGrow(ref normalized, ref poolArray, (normalizedIndex + _spaceSymbol.Length) << 1);
                    }
                    _spaceSymbol.AsSpan().CopyTo(normalized.Slice(normalizedIndex));
                    normalizedIndex += _spaceSymbol.Length;
                }
                else
                {
                    if (normalized.Length <= normalizedIndex + 1)
                    {
                        Helpers.ArrayPoolGrow(ref normalized, ref poolArray, (normalizedIndex + 1) << 1);
                    }
                    normalized[normalizedIndex] = (byte)' ';
                    normalizedIndex++;
                }
            }
        }

        private unsafe void DecodePrecompiledCharsMap(ReadOnlySpan<byte> blob, out DoubleArrayUnit[]? trieBlob, out byte[]? normalized)
        {
            uint trieBlobSize = 0;

            if (blob.Length <= sizeof(uint))
            {
                throw new ArgumentException("Blob for normalization rule is broken.");
            }

            fixed (byte* pBlob = blob)
            {
                trieBlobSize = *(uint*)pBlob;
            }

            if (!BitConverter.IsLittleEndian)
            {
                trieBlobSize = Helpers.Swap32(trieBlobSize);
            }

            if (trieBlobSize >= blob.Length)
            {
                throw new ArgumentException("Trie data size exceeds the input blob size.");
            }

            blob = blob.Slice(sizeof(uint));

            if (!BitConverter.IsLittleEndian)
            {
                fixed (byte* pBlob = blob)
                {
                    uint* data = (uint*)pBlob;

                    // Perform necessary operations for Big Endian
                    for (int i = 0; i < trieBlobSize / 4; ++i)
                    {
                        data[i] = Helpers.Swap32(data[i]);
                    }
                }
            }

            fixed (byte* pBlob = blob.Slice(0, (int)trieBlobSize))
            {

                trieBlob = new Span<DoubleArrayUnit>((DoubleArrayUnit*)pBlob, (int)trieBlobSize / 4).ToArray();
            }

            normalized = blob.Slice((int)trieBlobSize).ToArray();
        }
    }
}
