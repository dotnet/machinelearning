// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Runtime.Data
{
    public static class ReadOnlyMemoryUtils
    {

        /// <summary>
        /// Compare equality with the given system string value.
        /// </summary>
        public static bool EqualsStr(string s, ReadOnlyMemory<char> memory)
        {
            Contracts.CheckValueOrNull(s);

            if (s == null)
                return memory.Length == 0;

            if (s.Length != memory.Length)
                return false;

            return memory.Span.SequenceEqual(s.AsSpan());
        }

        public static IEnumerable<ReadOnlyMemory<char>> Split(ReadOnlyMemory<char> memory, char[] separators)
        {
            Contracts.CheckValueOrNull(separators);

            if (memory.IsEmpty)
                yield break;

            if (separators == null || separators.Length == 0)
            {
                yield return memory;
                yield break;
            }

            if (separators.Length == 1)
            {
                char chSep = separators[0];
                for (int ichCur = 0; ;)
                {
                    int ichMinLocal = ichCur;
                    for (; ; ichCur++)
                    {
                        Contracts.Assert(ichCur <= memory.Length);
                        if (ichCur >= memory.Length)
                        {
                            yield return memory.Slice(ichMinLocal, ichCur - ichMinLocal);
                            yield break;
                        }
                        if (memory.Span[ichCur] == chSep)
                            break;
                    }

                    yield return memory.Slice(ichMinLocal, ichCur - ichMinLocal);

                    // Skip the separator.
                    ichCur++;
                }
            }
            else
            {
                for (int ichCur = 0; ;)
                {
                    int ichMinLocal = ichCur;
                    for (; ; ichCur++)
                    {
                        Contracts.Assert(ichCur <= memory.Length);
                        if (ichCur >= memory.Length)
                        {
                            yield return memory.Slice(ichMinLocal, ichCur - ichMinLocal);
                            yield break;
                        }
                        // REVIEW: Can this be faster?
                        if (ContainsChar(memory.Span[ichCur], separators))
                            break;
                    }

                    yield return memory.Slice(ichMinLocal, ichCur - ichMinLocal);

                    // Skip the separator.
                    ichCur++;
                }
            }
        }

        /// <summary>
        /// Splits <paramref name="memory"/> on the left-most occurrence of separator and produces the left
        /// and right <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/> values. If <paramref name="memory"/> does not contain the separator character,
        /// this returns false and sets <paramref name="left"/> to this instance and <paramref name="right"/>
        /// to the default <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/> value.
        /// </summary>
        public static bool SplitOne(ReadOnlyMemory<char> memory, char separator, out ReadOnlyMemory<char> left, out ReadOnlyMemory<char> right)
        {
            if (memory.IsEmpty)
            {
                left = memory;
                right = default;
                return false;
            }

            int index = memory.Span.IndexOf(separator);
            if (index == -1)
            {
                left = memory;
                right = default;
                return false;
            }

            left = memory.Slice(0, index);
            right = memory.Slice(index + 1, memory.Length - index - 1);
            return true;
        }

        /// <summary>
        /// Splits <paramref name="memory"/> on the left-most occurrence of an element of separators character array and
        /// produces the left and right <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/> values. If <paramref name="memory"/> does not contain any of the
        /// characters in separators, this return false and initializes <paramref name="left"/> to this instance
        /// and <paramref name="right"/> to the default <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/> value.
        /// </summary>
        public static bool SplitOne(ReadOnlyMemory<char> memory, char[] separators, out ReadOnlyMemory<char> left, out ReadOnlyMemory<char> right)
        {
            Contracts.CheckValueOrNull(separators);

            if (memory.IsEmpty || separators == null || separators.Length == 0)
            {
                left = memory;
                right = default;
                return false;
            }

            int index;
            if (separators.Length == 1)
                index = memory.Span.IndexOf(separators[0]);
            else
                index = memory.Span.IndexOfAny(separators);

            if (index == -1)
            {
                left = memory;
                right = default;
                return false;
            }

            left = memory.Slice(0, index);
            right = memory.Slice(index + 1, memory.Length - index - 1);
            return true;
        }

        /// <summary>
        /// Returns a <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/> with leading and trailing spaces trimmed. Note that this
        /// will remove only spaces, not any form of whitespace.
        /// </summary>
        public static ReadOnlyMemory<char> TrimSpaces(ReadOnlyMemory<char> memory)
        {
            if (memory.IsEmpty)
                return memory;

            int ichLim = memory.Length;
            int ichMin = 0;
            var span = memory.Span;
            if (span[ichMin] != ' ' && span[ichLim - 1] != ' ')
                return memory;

            while (ichMin < ichLim && span[ichMin] == ' ')
                ichMin++;
            while (ichMin < ichLim && span[ichLim - 1] == ' ')
                ichLim--;
            return memory.Slice(ichMin, ichLim - ichMin);
        }

        public static ReadOnlySpan<char> TrimSpaces(ReadOnlySpan<char> span)
        {
            if (span.IsEmpty)
                return span;

            int ichLim = span.Length;
            int ichMin = 0;
            if (span[ichMin] != ' ' && span[ichLim - 1] != ' ')
                return span;

            while (ichMin < ichLim && span[ichMin] == ' ')
                ichMin++;
            while (ichMin < ichLim && span[ichLim - 1] == ' ')
                ichLim--;
            return span.Slice(ichMin, ichLim - ichMin);
        }

        /// <summary>
        /// Returns a <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/> with leading and trailing whitespace trimmed.
        /// </summary>
        public static ReadOnlyMemory<char> TrimWhiteSpace(ReadOnlyMemory<char> memory)
        {
            if (memory.IsEmpty)
                return memory;

            int ichMin = 0;
            int ichLim = memory.Length;
            var span = memory.Span;
            if (!char.IsWhiteSpace(span[ichMin]) && !char.IsWhiteSpace(span[ichLim - 1]))
                return memory;

            while (ichMin < ichLim && char.IsWhiteSpace(span[ichMin]))
                ichMin++;
            while (ichMin < ichLim && char.IsWhiteSpace(span[ichLim - 1]))
                ichLim--;

            return memory.Slice(ichMin, ichLim - ichMin);
        }

        /// <summary>
        /// Returns a <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/> with trailing whitespace trimmed.
        /// </summary>
        public static ReadOnlyMemory<char> TrimEndWhiteSpace(ReadOnlyMemory<char> memory)
        {
            if (memory.IsEmpty)
                return memory;

            int ichLim = memory.Length;
            var span = memory.Span;
            if (!char.IsWhiteSpace(span[ichLim - 1]))
                return memory;

            while (0 < ichLim && char.IsWhiteSpace(span[ichLim - 1]))
                ichLim--;

            return memory.Slice(0, ichLim);
        }

        /// <summary>
        /// Returns a <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/> with trailing whitespace trimmed.
        /// </summary>
        public static ReadOnlyMemory<char> TrimEndWhiteSpace(ReadOnlyMemory<char> memory, ReadOnlySpan<char> span)
        {
            if (memory.IsEmpty)
                return memory;

            int ichLim = memory.Length;
            if (!char.IsWhiteSpace(span[ichLim - 1]))
                return memory;

            while (0 < ichLim && char.IsWhiteSpace(span[ichLim - 1]))
                ichLim--;

            return memory.Slice(0, ichLim);
        }

        /// <summary>
        /// This produces zero for an empty string.
        /// </summary>
        public static bool TryParse(ReadOnlySpan<char> span, out Single value)
        {
            var res = DoubleParser.Parse(out value, span);
            Contracts.Assert(res != DoubleParser.Result.Empty || value == 0);
            return res <= DoubleParser.Result.Empty;
        }

        /// <summary>
        /// This produces zero for an empty string.
        /// </summary>
        public static bool TryParse(ReadOnlySpan<char> span, out Double value)
        {
            var res = DoubleParser.Parse(out value, span);
            Contracts.Assert(res != DoubleParser.Result.Empty || value == 0);
            return res <= DoubleParser.Result.Empty;
        }

        public static uint Hash(ReadOnlySpan<char> span, uint seed) => Hashing.MurmurHash(seed, span);

        public static NormStr AddToPool(ReadOnlyMemory<char> memory, NormStr.Pool pool)
        {
            Contracts.CheckValue(pool, nameof(pool));
            return pool.Add(memory);
        }

        public static NormStr FindInPool(ReadOnlyMemory<char> memory, NormStr.Pool pool)
        {
            Contracts.CheckValue(pool, nameof(pool));
            return pool.Get(memory);
        }

        public static void AddToStringBuilder(ReadOnlyMemory<char> memory, StringBuilder sb)
        {
            Contracts.CheckValue(sb, nameof(sb));
            if (!memory.IsEmpty)
                sb.AppendAll(memory);
        }

        public static void AddLowerCaseToStringBuilder(ReadOnlyMemory<char> memory, StringBuilder sb)
        {
            Contracts.CheckValue(sb, nameof(sb));

            if (!memory.IsEmpty)
            {
                int min = 0;
                int j;
                var span = memory.Span;
                for (j = min; j < memory.Length; j++)
                {
                    char ch = CharUtils.ToLowerInvariant(span[j]);
                    if (ch != span[j])
                    {
                        sb.Append(memory, min, j - min).Append(ch);
                        min = j + 1;
                    }
                }

                Contracts.Assert(j == memory.Length);
                if (min != j)
                    sb.Append(memory, min, j - min);
            }
        }

        private static bool ContainsChar(char ch, char[] rgch)
        {
            Contracts.AssertNonEmpty(rgch, nameof(rgch));

            for (int i = 0; i < rgch.Length; i++)
            {
                if (rgch[i] == ch)
                    return true;
            }
            return false;
        }

        public static StringBuilder AppendAll(this StringBuilder sb, ReadOnlyMemory<char> memory) => Append(sb, memory, 0, memory.Length);

        public static StringBuilder Append(this StringBuilder sb, ReadOnlyMemory<char> memory, int startIndex, int length)
        {
            Contracts.Check(startIndex >= 0, nameof(startIndex));
            Contracts.Check(length >= 0, nameof(length));

            int ichLim = startIndex + length;

            Contracts.Check(memory.Length >= ichLim, nameof(memory));

            var span = memory.Span;
            for (int index = startIndex; index < ichLim; index++)
                sb.Append(span[index]);

            return sb;
        }

        public static StringBuilder Append(this StringBuilder sb, ReadOnlySpan<char> span)
        {
            for (int index = 0; index < span.Length; index++)
                sb.Append((char)span[index]);

            return sb;
        }
    }
}
