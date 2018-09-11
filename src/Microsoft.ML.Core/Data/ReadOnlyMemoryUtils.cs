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
        /// Compares two <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/> byte by byte.
        /// </summary>
        public static bool Equals(ReadOnlyMemory<char> b, ReadOnlyMemory<char> a)
        {
            if (a.Length != b.Length)
                return false;

            Contracts.Assert(a.IsEmpty == b.IsEmpty);
            var aSpan = a.Span;
            var bSpan = b.Span;

            for (int i = 0; i < a.Length; i++)
            {
                if (aSpan[i] != bSpan[i])
                    return false;
            }
            return true;
        }

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

            var span = memory.Span;
            for (int i = 0; i < memory.Length; i++)
            {
                if (s[i] != span[i])
                    return false;
            }
            return true;
        }

        /// <summary>
        /// For implementation of <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/>. Uses code point comparison.
        /// Generally, this is not appropriate for sorting for presentation to a user.
        /// </summary>
        public static int CompareTo(ReadOnlyMemory<char> b, ReadOnlyMemory<char> a)
        {
            int len = Math.Min(a.Length, b.Length);
            var aSpan = a.Span;
            var bSpan = b.Span;
            for (int ich = 0; ich < len; ich++)
            {
                char ch1 = aSpan[ich];
                char ch2 = bSpan[ich];
                if (ch1 != ch2)
                    return ch1 < ch2 ? -1 : +1;
            }
            if (len < b.Length)
                return -1;
            if (len < a.Length)
                return +1;
            return 0;
        }

        public static IEnumerable<ReadOnlyMemory<char>> Split(char[] separators, ReadOnlyMemory<char> memory)
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
        public static bool SplitOne(char separator, out ReadOnlyMemory<char> left, out ReadOnlyMemory<char> right, ReadOnlyMemory<char> memory)
        {
            if (memory.IsEmpty)
            {
                left = memory;
                right = default;
                return false;
            }

            var span = memory.Span;
            int ichCur = 0;
            for (; ; ichCur++)
            {
                Contracts.Assert(0 <= ichCur && ichCur <= memory.Length);
                if (ichCur >= memory.Length)
                {
                    left = memory;
                    right = default;
                    return false;
                }
                if (span[ichCur] == separator)
                    break;
            }

            left = memory.Slice(0, ichCur);
            right = memory.Slice(ichCur + 1, memory.Length - ichCur - 1);
            return true;
        }

        /// <summary>
        /// Splits <paramref name="memory"/> on the left-most occurrence of an element of separators character array and
        /// produces the left and right <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/> values. If <paramref name="memory"/> does not contain any of the
        /// characters in separators, this return false and initializes <paramref name="left"/> to this instance
        /// and <paramref name="right"/> to the default <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/> value.
        /// </summary>
        public static bool SplitOne(char[] separators, out ReadOnlyMemory<char> left, out ReadOnlyMemory<char> right, ReadOnlyMemory<char> memory)
        {
            Contracts.CheckValueOrNull(separators);

            if (memory.IsEmpty || separators == null || separators.Length == 0)
            {
                left = memory;
                right = default;
                return false;
            }

            var span = memory.Span;

            int ichCur = 0;
            if (separators.Length == 1)
            {
                // Note: This duplicates code of the other SplitOne, but doing so improves perf because this is
                // used so heavily in instances parsing.
                char chSep = separators[0];
                for (; ; ichCur++)
                {
                    Contracts.Assert(0 <= ichCur && ichCur <= memory.Length);
                    if (ichCur >= memory.Length)
                    {
                        left = memory;
                        right = default;
                        return false;
                    }
                    if (span[ichCur] == chSep)
                        break;
                }
            }
            else
            {
                for (; ; ichCur++)
                {
                    Contracts.Assert(0 <= ichCur && ichCur <= memory.Length);
                    if (ichCur >= memory.Length)
                    {
                        left = memory;
                        right = default;
                        return false;
                    }
                    // REVIEW: Can this be faster?
                    if (ContainsChar(span[ichCur], separators))
                        break;
                }
            }

            left = memory.Slice(0, ichCur);
            right = memory.Slice(ichCur + 1, memory.Length - ichCur - 1);
            return true;
        }

        /// <summary>
        /// Returns a <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/> with leading and trailing spaces trimmed. Note that this
        /// will remove only spaces, not any form of whitespace.
        /// </summary>
        public static ReadOnlyMemory<char> Trim(ReadOnlyMemory<char> memory)
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
        /// This produces zero for an empty string.
        /// </summary>
        public static bool TryParse(out Single value, ReadOnlyMemory<char> memory)
        {
            var res = DoubleParser.Parse(out value, memory.Span);
            Contracts.Assert(res != DoubleParser.Result.Empty || value == 0);
            return res <= DoubleParser.Result.Empty;
        }

        /// <summary>
        /// This produces zero for an empty string.
        /// </summary>
        public static bool TryParse(out Double value, ReadOnlyMemory<char> memory)
        {
            var res = DoubleParser.Parse(out value, memory.Span);
            Contracts.Assert(res != DoubleParser.Result.Empty || value == 0);
            return res <= DoubleParser.Result.Empty;
        }

        public static uint Hash(uint seed, ReadOnlyMemory<char> memory) => Hashing.MurmurHash(seed, memory);

        public static NormStr AddToPool(NormStr.Pool pool, ReadOnlyMemory<char> memory)
        {
            Contracts.CheckValue(pool, nameof(pool));
            return pool.Add(memory);
        }

        public static NormStr FindInPool(NormStr.Pool pool, ReadOnlyMemory<char> memory)
        {
            Contracts.CheckValue(pool, nameof(pool));
            return pool.Get(memory);
        }

        public static void AddToStringBuilder(StringBuilder sb, ReadOnlyMemory<char> memory)
        {
            Contracts.CheckValue(sb, nameof(sb));
            if (!memory.IsEmpty)
                sb.AppendAll(memory);
        }

        public static void AddLowerCaseToStringBuilder(StringBuilder sb, ReadOnlyMemory<char> memory)
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
    }
}
