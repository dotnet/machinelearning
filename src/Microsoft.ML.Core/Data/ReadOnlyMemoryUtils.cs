using Microsoft.ML.Runtime;
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
        /// This method retrieves the raw buffer information. The only characters that should be
        /// referenced in the returned string are those between the returned min and lim indices.
        /// If this is an NA value, the min will be zero and the lim will be -1. For either an
        /// empty or NA value, the returned string may be null.
        /// </summary>
        public static string GetRawUnderlyingBufferInfo(out int ichMin, out int ichLim, ReadOnlyMemory<char> memory)
        {
            MemoryMarshal.TryGetString(memory, out string outerBuffer, out ichMin, out int length);
            ichLim = ichMin + length;
            return outerBuffer;
        }

        public static int GetHashCode(this ReadOnlyMemory<char> memory) => (int)Hash(42, memory);

        public static bool Equals(this ReadOnlyMemory<char> memory, object obj)
        {
            if (obj is ReadOnlyMemory<char>)
                return Equals((ReadOnlyMemory<char>)obj, memory);
            return false;
        }

        /// <summary>
        /// This implements IEquatable's Equals method. Returns true if both are NA.
        /// For NA propagating equality comparison, use the == operator.
        /// </summary>
        public static bool Equals(ReadOnlyMemory<char> b, ReadOnlyMemory<char> memory)
        {
            if (memory.Length != b.Length)
                return false;
            Contracts.Assert(memory.IsEmpty == b.IsEmpty);

            MemoryMarshal.TryGetString(memory, out string outerBuffer, out int ichMin, out int length);
            int ichLim = ichMin + length;

            MemoryMarshal.TryGetString(b, out string bOuterBuffer, out int bIchMin, out int bLength);
            int bIchLim = bIchMin + bLength;
            for (int i = 0; i < memory.Length; i++)
            {
                if (outerBuffer[ichMin + i] != bOuterBuffer[bIchMin + i])
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Does not propagate NA values. Returns true if both are NA (same as a.Equals(b)).
        /// For NA propagating equality comparison, use the == operator.
        /// </summary>
        public static bool Identical(ReadOnlyMemory<char> a, ReadOnlyMemory<char> b)
        {
            if (a.Length != b.Length)
                return false;
            if (!a.IsEmpty)
            {
                Contracts.Assert(!b.IsEmpty);
                MemoryMarshal.TryGetString(a, out string aOuterBuffer, out int aIchMin, out int aLength);
                int aIchLim = aIchMin + aLength;

                MemoryMarshal.TryGetString(b, out string bOuterBuffer, out int bIchMin, out int bLength);
                int bIchLim = bIchMin + bLength;

                for (int i = 0; i < a.Length; i++)
                {
                    if (aOuterBuffer[aIchMin + i] != bOuterBuffer[bIchMin + i])
                        return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Compare equality with the given system string value. Returns false if "this" is NA.
        /// </summary>
        public static bool EqualsStr(string s, ReadOnlyMemory<char> memory)
        {
            Contracts.CheckValueOrNull(s);

            // Note that "NA" doesn't match any string.
            if (s == null)
                return memory.Length == 0;

            if (s.Length != memory.Length)
                return false;

            MemoryMarshal.TryGetString(memory, out string outerBuffer, out int ichMin, out int length);
            int ichLim = ichMin + length;
            for (int i = 0; i < memory.Length; i++)
            {
                if (s[i] != outerBuffer[ichMin + i])
                    return false;
            }
            return true;
        }

        /// <summary>
        /// For implementation of ReadOnlyMemory. Uses code point comparison.
        /// Generally, this is not appropriate for sorting for presentation to a user.
        /// Sorts NA before everything else.
        /// </summary>
        public static int CompareTo(ReadOnlyMemory<char> other, ReadOnlyMemory<char> memory)
        {
            int len = Math.Min(memory.Length, other.Length);
            MemoryMarshal.TryGetString(memory, out string outerBuffer, out int ichMin, out int length);
            int ichLim = ichMin + length;

            MemoryMarshal.TryGetString(other, out string otherOuterBuffer, out int otherIchMin, out int otherLength);
            int otherIchLim = otherIchMin + otherLength;

            for (int ich = 0; ich < len; ich++)
            {
                char ch1 = outerBuffer[ichMin + ich];
                char ch2 = otherOuterBuffer[otherIchMin + ich];
                if (ch1 != ch2)
                    return ch1 < ch2 ? -1 : +1;
            }
            if (len < other.Length)
                return -1;
            if (len < memory.Length)
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

            MemoryMarshal.TryGetString(memory, out string outerBuffer, out int ichMin, out int length);
            int ichLim = ichMin + length;
            string text = outerBuffer;
            if (separators.Length == 1)
            {
                char chSep = separators[0];
                for (int ichCur = ichMin; ;)
                {
                    int ichMinLocal = ichCur;
                    for (; ; ichCur++)
                    {
                        Contracts.Assert(ichCur <= ichLim);
                        if (ichCur >= ichLim)
                        {
                            yield return outerBuffer.AsMemory().Slice(ichMinLocal, ichCur - ichMinLocal);
                            yield break;
                        }
                        if (text[ichCur] == chSep)
                            break;
                    }

                    yield return outerBuffer.AsMemory().Slice(ichMinLocal, ichCur - ichMinLocal);

                    // Skip the separator.
                    ichCur++;
                }
            }
            else
            {
                for (int ichCur = ichMin; ;)
                {
                    int ichMinLocal = ichCur;
                    for (; ; ichCur++)
                    {
                        Contracts.Assert(ichCur <= ichLim);
                        if (ichCur >= ichLim)
                        {
                            yield return outerBuffer.AsMemory().Slice(ichMinLocal, ichCur - ichMinLocal);
                            yield break;
                        }
                        // REVIEW: Can this be faster?
                        if (ContainsChar(text[ichCur], separators))
                            break;
                    }

                    yield return outerBuffer.AsMemory().Slice(ichMinLocal, ichCur - ichMinLocal);

                    // Skip the separator.
                    ichCur++;
                }
            }
        }

        /// <summary>
        /// Splits this instance on the left-most occurrence of separator and produces the left
        /// and right ReadOnlyMemory values. If this instance does not contain the separator character,
        /// this returns false and sets <paramref name="left"/> to this instance and <paramref name="right"/>
        /// to the default ReadOnlyMemory value.
        /// </summary>
        public static bool SplitOne(char separator, out ReadOnlyMemory<char> left, out ReadOnlyMemory<char> right, ReadOnlyMemory<char> memory)
        {
            if (memory.IsEmpty)
            {
                left = memory;
                right = default;
                return false;
            }

            MemoryMarshal.TryGetString(memory, out string outerBuffer, out int ichMin, out int length);
            int ichLim = ichMin + length;
            string text = outerBuffer;
            int ichCur = ichMin;
            for (; ; ichCur++)
            {
                Contracts.Assert(ichMin <= ichCur && ichCur <= ichLim);
                if (ichCur >= ichLim)
                {
                    left = memory;
                    right = default;
                    return false;
                }
                if (text[ichCur] == separator)
                    break;
            }

            // Note that we don't use any fields of "this" here in case one
            // of the out parameters is the same as "this".
            left = outerBuffer.AsMemory().Slice(ichMin, ichCur - ichMin);
            right = outerBuffer.AsMemory().Slice(ichCur + 1, ichLim - ichCur - 1);
            return true;
        }

        /// <summary>
        /// Splits this instance on the left-most occurrence of an element of separators character array and
        /// produces the left and right ReadOnlyMemory values. If this instance does not contain any of the
        /// characters in separators, thiss return false and initializes <paramref name="left"/> to this instance
        /// and <paramref name="right"/> to the default ReadOnlyMemory value.
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

            MemoryMarshal.TryGetString(memory, out string outerBuffer, out int ichMin, out int length);
            int ichLim = ichMin + length;
            string text = outerBuffer;

            int ichCur = ichMin;
            if (separators.Length == 1)
            {
                // Note: This duplicates code of the other SplitOne, but doing so improves perf because this is
                // used so heavily in instances parsing.
                char chSep = separators[0];
                for (; ; ichCur++)
                {
                    Contracts.Assert(ichMin <= ichCur && ichCur <= ichLim);
                    if (ichCur >= ichLim)
                    {
                        left = memory;
                        right = default;
                        return false;
                    }
                    if (text[ichCur] == chSep)
                        break;
                }
            }
            else
            {
                for (; ; ichCur++)
                {
                    Contracts.Assert(ichMin <= ichCur && ichCur <= ichLim);
                    if (ichCur >= ichLim)
                    {
                        left = memory;
                        right = default;
                        return false;
                    }
                    // REVIEW: Can this be faster?
                    if (ContainsChar(text[ichCur], separators))
                        break;
                }
            }

            // Note that we don't use any fields of "this" here in case one
            // of the out parameters is the same as "this".
            left = outerBuffer.AsMemory().Slice(ichMin, ichCur - ichMin);
            right = outerBuffer.AsMemory().Slice(ichCur + 1, ichLim - ichCur - 1);
            return true;
        }

        /// <summary>
        /// Returns a text span with leading and trailing spaces trimmed. Note that this
        /// will remove only spaces, not any form of whitespace.
        /// </summary>
        public static ReadOnlyMemory<char> Trim(ReadOnlyMemory<char> memory)
        {
            if (memory.IsEmpty)
                return memory;

            MemoryMarshal.TryGetString(memory, out string outerBuffer, out int ichMin, out int length);
            int ichLim = ichMin + length;
            if (outerBuffer[ichMin] != ' ' && outerBuffer[ichLim - 1] != ' ')
                return memory;

            while (ichMin < ichLim && outerBuffer[ichMin] == ' ')
                ichMin++;
            while (ichMin < ichLim && outerBuffer[ichLim - 1] == ' ')
                ichLim--;
            return outerBuffer.AsMemory().Slice(ichMin, ichLim - ichMin);
        }

        /// <summary>
        /// Returns a text span with leading and trailing whitespace trimmed.
        /// </summary>
        public static ReadOnlyMemory<char> TrimWhiteSpace(ReadOnlyMemory<char> memory)
        {
            if (memory.IsEmpty)
                return memory;

            MemoryMarshal.TryGetString(memory, out string outerBuffer, out int ichMin, out int length);
            int ichLim = ichMin + length;

            if (!char.IsWhiteSpace(outerBuffer[ichMin]) && !char.IsWhiteSpace(outerBuffer[ichLim - 1]))
                return memory;

            while (ichMin < ichLim && char.IsWhiteSpace(outerBuffer[ichMin]))
                ichMin++;
            while (ichMin < ichLim && char.IsWhiteSpace(outerBuffer[ichLim - 1]))
                ichLim--;

            return outerBuffer.AsMemory().Slice(ichMin, ichLim - ichMin);
        }

        /// <summary>
        /// Returns a text span with trailing whitespace trimmed.
        /// </summary>
        public static ReadOnlyMemory<char> TrimEndWhiteSpace(ReadOnlyMemory<char> memory)
        {
            if (memory.IsEmpty)
                return memory;

            MemoryMarshal.TryGetString(memory, out string outerBuffer, out int ichMin, out int length);
            int ichLim = ichMin + length;
            if (!char.IsWhiteSpace(outerBuffer[ichLim - 1]))
                return memory;

            while (ichMin < ichLim && char.IsWhiteSpace(outerBuffer[ichLim - 1]))
                ichLim--;

            return outerBuffer.AsMemory().Slice(ichMin, ichLim - ichMin);
        }

        /// <summary>
        /// This produces zero for an empty string.
        /// </summary>
        public static bool TryParse(out Single value, ReadOnlyMemory<char> memory)
        {
            MemoryMarshal.TryGetString(memory, out string outerBuffer, out int ichMin, out int length);
            int ichLim = ichMin + length;
            var res = DoubleParser.Parse(out value, outerBuffer, ichMin, ichLim);
            Contracts.Assert(res != DoubleParser.Result.Empty || value == 0);
            return res <= DoubleParser.Result.Empty;
        }

        /// <summary>
        /// This produces zero for an empty string.
        /// </summary>
        public static bool TryParse(out Double value, ReadOnlyMemory<char> memory)
        {
            MemoryMarshal.TryGetString(memory, out string outerBuffer, out int ichMin, out int length);
            int ichLim = ichMin + length;
            var res = DoubleParser.Parse(out value, outerBuffer, ichMin, ichLim);
            Contracts.Assert(res != DoubleParser.Result.Empty || value == 0);
            return res <= DoubleParser.Result.Empty;
        }

        public static uint Hash(uint seed, ReadOnlyMemory<char> memory)
        {
            MemoryMarshal.TryGetString(memory, out string outerBuffer, out int ichMin, out int length);
            int ichLim = ichMin + length;
            return Hashing.MurmurHash(seed, outerBuffer, ichMin, ichLim);
        }

        // REVIEW: Add method to NormStr.Pool that deal with ReadOnlyMemory instead of the other way around.
        public static NormStr AddToPool(NormStr.Pool pool, ReadOnlyMemory<char> memory)
        {
            Contracts.CheckValue(pool, nameof(pool));
            MemoryMarshal.TryGetString(memory, out string outerBuffer, out int ichMin, out int length);
            int ichLim = ichMin + length;
            return pool.Add(outerBuffer, ichMin, ichLim);
        }

        public static NormStr FindInPool(NormStr.Pool pool, ReadOnlyMemory<char> memory)
        {
            Contracts.CheckValue(pool, nameof(pool));
            MemoryMarshal.TryGetString(memory, out string outerBuffer, out int ichMin, out int length);
            int ichLim = ichMin + length;
            return pool.Get(outerBuffer, ichMin, ichLim);
        }

        public static void AddToStringBuilder(StringBuilder sb, ReadOnlyMemory<char> memory)
        {
            Contracts.CheckValue(sb, nameof(sb));
            if (!memory.IsEmpty)
            {
                MemoryMarshal.TryGetString(memory, out string outerBuffer, out int ichMin, out int length);
                sb.Append(outerBuffer, ichMin, length);
            }
        }

        public static void AddLowerCaseToStringBuilder(StringBuilder sb, ReadOnlyMemory<char> memory)
        {
            Contracts.CheckValue(sb, nameof(sb));

            if (!memory.IsEmpty)
            {
                MemoryMarshal.TryGetString(memory, out string outerBuffer, out int ichMin, out int length);
                int ichLim = ichMin + length;
                int min = ichMin;
                int j;
                for (j = min; j < ichLim; j++)
                {
                    char ch = CharUtils.ToLowerInvariant(outerBuffer[j]);
                    if (ch != outerBuffer[j])
                    {
                        sb.Append(outerBuffer, min, j - min).Append(ch);
                        min = j + 1;
                    }
                }

                Contracts.Assert(j == ichLim);
                if (min != j)
                    sb.Append(outerBuffer, min, j - min);
            }
        }

        // REVIEW: Can this be faster?
        private static bool ContainsChar(char ch, char[] rgch)
        {
            Contracts.CheckNonEmpty(rgch, nameof(rgch));

            for (int i = 0; i < rgch.Length; i++)
            {
                if (rgch[i] == ch)
                    return true;
            }
            return false;
        }
    }
}
