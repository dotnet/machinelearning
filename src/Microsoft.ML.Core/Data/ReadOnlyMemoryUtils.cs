// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
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

            var span = memory.Span;
            if (separators.Length == 1)
            {
                char chSep = separators[0];
                for (int ichCur = 0; ;)
                {
                    int nextSep = span.IndexOf(chSep);
                    if (nextSep == -1)
                    {
                        yield return memory.Slice(ichCur);
                        yield break;
                    }

                    yield return memory.Slice(ichCur, nextSep);

                    // Skip the separator.
                    ichCur += nextSep + 1;
                    span = memory.Slice(ichCur).Span;
                }
            }
            else
            {
                for (int ichCur = 0; ;)
                {
                    int nextSep = span.IndexOfAny(separators);
                    if (nextSep == -1)
                    {
                        yield return memory.Slice(ichCur);
                        yield break;
                    }

                    yield return memory.Slice(ichCur, nextSep);

                    // Skip the separator.
                    ichCur += nextSep + 1;
                    span = memory.Slice(ichCur).Span;
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

        public static void AddLowerCaseToStringBuilder(ReadOnlySpan<char> span, StringBuilder sb)
        {
            Contracts.CheckValue(sb, nameof(sb));

            if (!span.IsEmpty)
            {
                int min = 0;
                int j;
                for (j = min; j < span.Length; j++)
                {
                    char ch = CharUtils.ToLowerInvariant(span[j]);
                    if (ch != span[j])
                    {
                        sb.AppendSpan(span.Slice(min, j - min)).Append(ch);
                        min = j + 1;
                    }
                }

                Contracts.Assert(j == span.Length);
                if (min != j)
                    sb.AppendSpan(span.Slice(min, j - min));
            }
        }

        public static StringBuilder AppendMemory(this StringBuilder sb, ReadOnlyMemory<char> memory)
        {
            Contracts.CheckValue(sb, nameof(sb));
            if (!memory.IsEmpty)
                sb.AppendSpan(memory.Span);

            return sb;
        }

        public static StringBuilder AppendSpan(this StringBuilder sb, ReadOnlySpan<char> span)
        {
            unsafe
            {
                fixed (char* valueChars = &MemoryMarshal.GetReference(span))
                {
                    sb.Append(valueChars, span.Length);
                }
            }

            return sb;
        }
    }
}
