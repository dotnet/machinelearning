// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// A text value. This essentially wraps a portion of a string. This can distinguish between a length zero
    /// span of characters and "NA", the latter having a Length of -1.
    /// </summary>
    public struct DvText : IEquatable<DvText>, IComparable<DvText>
    {
        /// <summary>
        /// The fields/properties <see cref="_outerBuffer"/>, <see cref="_ichMin"/>, and <see cref="IchLim"/> are
        /// private so client code can't easily "cheat" and look outside the <see cref="DvText"/> characters. Client
        /// code that absolutely needs access to this information can call <see cref="GetRawUnderlyingBufferInfo"/>.
        /// </summary>
        private readonly string _outerBuffer;
        private readonly int _ichMin;

        /// <summary>
        /// For the "NA" value, this is -1; otherwise, it is the number of characters in the text.
        /// </summary>
        public readonly int Length;

        private int IchLim => _ichMin + Length;

        /// <summary>
        /// Gets a DvText that represents "NA", aka "Missing".
        /// </summary>
        public static DvText NA => new DvText(missing: true);

        /// <summary>
        /// Gets an empty (zero character) DvText.
        /// </summary>
        public static DvText Empty => default(DvText);

        /// <summary>
        /// Gets whether this DvText contains any characters. Equivalent to Length > 0.
        /// </summary>
        public bool HasChars => Length > 0;

        /// <summary>
        /// Gets whether this DvText is empty (distinct from NA). Equivalent to Length == 0.
        /// </summary>
        public bool IsEmpty
        {
            get
            {
                Contracts.Assert(Length >= -1);
                return Length == 0;
            }
        }

        /// <summary>
        /// Gets whether this DvText represents "NA". Equivalent to Length == -1.
        /// </summary>
        public bool IsNA
        {
            get
            {
                Contracts.Assert(Length >= -1);
                return Length < 0;
            }
        }

        /// <summary>
        /// Gets the indicated character in the text.
        /// </summary>
        public char this[int ich]
        {
            get
            {
                Contracts.CheckParam(0 <= ich & ich < Length, nameof(ich));
                return _outerBuffer[ich + _ichMin];
            }
        }

        private DvText(bool missing)
        {
            _outerBuffer = null;
            _ichMin = 0;
            Length = missing ? -1 : 0;
        }

        /// <summary>
        /// Constructor using the indicated range of characters in the given string.
        /// </summary>
        public DvText(string text, int ichMin, int ichLim)
        {
            Contracts.CheckValueOrNull(text);
            Contracts.CheckParam(0 <= ichMin & ichMin <= Utils.Size(text), nameof(ichMin));
            Contracts.CheckParam(ichMin <= ichLim & ichLim <= Utils.Size(text), nameof(ichLim));
            Length = ichLim - ichMin;
            if (Length == 0)
            {
                _outerBuffer = null;
                _ichMin = 0;
            }
            else
            {
                _outerBuffer = text;
                _ichMin = ichMin;
            }
        }

        /// <summary>
        /// Constructor using the indicated string.
        /// </summary>
        public DvText(string text)
        {
            Contracts.CheckValueOrNull(text);
            Length = Utils.Size(text);
            if (Length == 0)
                _outerBuffer = null;
            else
                _outerBuffer = text;
            _ichMin = 0;
        }

        /// <summary>
        /// This method retrieves the raw buffer information. The only characters that should be
        /// referenced in the returned string are those between the returned min and lim indices.
        /// If this is an NA value, the min will be zero and the lim will be -1. For either an
        /// empty or NA value, the returned string may be null.
        /// </summary>
        public string GetRawUnderlyingBufferInfo(out int ichMin, out int ichLim)
        {
            ichMin = _ichMin;
            ichLim = ichMin + Length;
            return _outerBuffer;
        }

        /// <summary>
        /// This compares the two text values with NA propagation semantics.
        /// </summary>
        public static DvBool operator ==(DvText a, DvText b)
        {
            if (a.IsNA || b.IsNA)
                return DvBool.NA;

            if (a.Length != b.Length)
                return DvBool.False;
            for (int i = 0; i < a.Length; i++)
            {
                if (a._outerBuffer[a._ichMin + i] != b._outerBuffer[b._ichMin + i])
                    return DvBool.False;
            }
            return DvBool.True;
        }

        /// <summary>
        /// This compares the two text values with NA propagation semantics.
        /// </summary>
        public static DvBool operator !=(DvText a, DvText b)
        {
            if (a.IsNA || b.IsNA)
                return DvBool.NA;

            if (a.Length != b.Length)
                return DvBool.True;
            for (int i = 0; i < a.Length; i++)
            {
                if (a._outerBuffer[a._ichMin + i] != b._outerBuffer[b._ichMin + i])
                    return DvBool.True;
            }
            return DvBool.False;
        }

        public override int GetHashCode()
        {
            if (IsNA)
                return 0;
            return (int)Hash(42);
        }

        public override bool Equals(object obj)
        {
            if (obj is DvText)
                return Equals((DvText)obj);
            return false;
        }

        /// <summary>
        /// This implements IEquatable's Equals method. Returns true if both are NA.
        /// For NA propagating equality comparison, use the == operator.
        /// </summary>
        public bool Equals(DvText b)
        {
            if (Length != b.Length)
                return false;
            Contracts.Assert(HasChars == b.HasChars);
            for (int i = 0; i < Length; i++)
            {
                if (_outerBuffer[_ichMin + i] != b._outerBuffer[b._ichMin + i])
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Does not propagate NA values. Returns true if both are NA (same as a.Equals(b)).
        /// For NA propagating equality comparison, use the == operator.
        /// </summary>
        public static bool Identical(DvText a, DvText b)
        {
            if (a.Length != b.Length)
                return false;
            if (a.HasChars)
            {
                Contracts.Assert(b.HasChars);
                for (int i = 0; i < a.Length; i++)
                {
                    if (a._outerBuffer[a._ichMin + i] != b._outerBuffer[b._ichMin + i])
                        return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Compare equality with the given system string value. Returns false if "this" is NA.
        /// </summary>
        public bool EqualsStr(string s)
        {
            Contracts.CheckValueOrNull(s);

            // Note that "NA" doesn't match any string.
            if (s == null)
                return Length == 0;

            if (s.Length != Length)
                return false;
            for (int i = 0; i < Length; i++)
            {
                if (s[i] != _outerBuffer[_ichMin + i])
                    return false;
            }
            return true;
        }

        /// <summary>
        /// For implementation of <see cref="IComparable{DvText}"/>. Uses code point comparison.
        /// Generally, this is not appropriate for sorting for presentation to a user.
        /// Sorts NA before everything else.
        /// </summary>
        public int CompareTo(DvText other)
        {
            if (IsNA)
                return other.IsNA ? 0 : -1;
            if (other.IsNA)
                return +1;

            int len = Math.Min(Length, other.Length);
            for (int ich = 0; ich < len; ich++)
            {
                char ch1 = _outerBuffer[_ichMin + ich];
                char ch2 = other._outerBuffer[other._ichMin + ich];
                if (ch1 != ch2)
                    return ch1 < ch2 ? -1 : +1;
            }
            if (len < other.Length)
                return -1;
            if (len < Length)
                return +1;
            return 0;
        }

        /// <summary>
        /// Return a DvText consisting of characters from ich to the end of this DvText.
        /// </summary>
        public DvText SubSpan(int ich)
        {
            Contracts.CheckParam(0 <= ich & ich <= Length, nameof(ich));
            return new DvText(_outerBuffer, ich + _ichMin, IchLim);
        }

        /// <summary>
        /// Return a DvText consisting of the indicated range of characters.
        /// </summary>
        public DvText SubSpan(int ichMin, int ichLim)
        {
            Contracts.CheckParam(0 <= ichMin & ichMin <= Length, nameof(ichMin));
            Contracts.CheckParam(ichMin <= ichLim & ichLim <= Length, nameof(ichLim));
            return new DvText(_outerBuffer, ichMin + _ichMin, ichLim + _ichMin);
        }

        /// <summary>
        /// Return a non-null string corresponding to the characters in this DvText.
        /// Note that an empty string is returned for both Empty and NA.
        /// </summary>
        public override string ToString()
        {
            if (!HasChars)
                return "";
            Contracts.AssertNonEmpty(_outerBuffer);
            if (_ichMin == 0 && Length == _outerBuffer.Length)
                return _outerBuffer;
            return _outerBuffer.Substring(_ichMin, Length);
        }

        public string ToString(int ichMin)
        {
            Contracts.CheckParam(0 <= ichMin & ichMin <= Length, nameof(ichMin));
            if (ichMin == Length)
                return "";
            ichMin += _ichMin;
            if (ichMin == 0 && Length == _outerBuffer.Length)
                return _outerBuffer;
            return _outerBuffer.Substring(ichMin, IchLim - ichMin);
        }

        public IEnumerable<DvText> Split(char[] separators)
        {
            Contracts.CheckValueOrNull(separators);

            if (!HasChars)
                yield break;

            if (separators == null || separators.Length == 0)
            {
                yield return this;
                yield break;
            }

            string text = _outerBuffer;
            int ichLim = IchLim;
            if (separators.Length == 1)
            {
                char chSep = separators[0];
                for (int ichCur = _ichMin; ; )
                {
                    int ichMin = ichCur;
                    for (; ; ichCur++)
                    {
                        Contracts.Assert(ichCur <= ichLim);
                        if (ichCur >= ichLim)
                        {
                            yield return new DvText(text, ichMin, ichCur);
                            yield break;
                        }
                        if (text[ichCur] == chSep)
                            break;
                    }

                    yield return new DvText(text, ichMin, ichCur);

                    // Skip the separator.
                    ichCur++;
                }
            }
            else
            {
                for (int ichCur = _ichMin; ; )
                {
                    int ichMin = ichCur;
                    for (; ; ichCur++)
                    {
                        Contracts.Assert(ichCur <= ichLim);
                        if (ichCur >= ichLim)
                        {
                            yield return new DvText(text, ichMin, ichCur);
                            yield break;
                        }
                        // REVIEW: Can this be faster?
                        if (ContainsChar(text[ichCur], separators))
                            break;
                    }

                    yield return new DvText(text, ichMin, ichCur);

                    // Skip the separator.
                    ichCur++;
                }
            }
        }

        /// <summary>
        /// Splits this instance on the left-most occurrence of separator and produces the left
        /// and right <see cref="DvText"/> values. If this instance does not contain the separator character,
        /// this returns false and sets <paramref name="left"/> to this instance and <paramref name="right"/>
        /// to the default <see cref="DvText"/> value.
        /// </summary>
        public bool SplitOne(char separator, out DvText left, out DvText right)
        {
            if (!HasChars)
            {
                left = this;
                right = default(DvText);
                return false;
            }

            string text = _outerBuffer;
            int ichMin = _ichMin;
            int ichLim = IchLim;

            int ichCur = ichMin;
            for (; ; ichCur++)
            {
                Contracts.Assert(ichMin <= ichCur && ichCur <= ichLim);
                if (ichCur >= ichLim)
                {
                    left = this;
                    right = default(DvText);
                    return false;
                }
                if (text[ichCur] == separator)
                    break;
            }

            // Note that we don't use any fields of "this" here in case one
            // of the out parameters is the same as "this".
            left = new DvText(text, ichMin, ichCur);
            right = new DvText(text, ichCur + 1, ichLim);
            return true;
        }

        /// <summary>
        /// Splits this instance on the left-most occurrence of an element of separators character array and
        /// produces the left and right <see cref="DvText"/> values. If this instance does not contain any of the
        /// characters in separators, thiss return false and initializes <paramref name="left"/> to this instance
        /// and <paramref name="right"/> to the default <see cref="DvText"/> value.
        /// </summary>
        public bool SplitOne(char[] separators, out DvText left, out DvText right)
        {
            Contracts.CheckValueOrNull(separators);

            if (!HasChars || separators == null || separators.Length == 0)
            {
                left = this;
                right = default(DvText);
                return false;
            }

            string text = _outerBuffer;
            int ichMin = _ichMin;
            int ichLim = IchLim;

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
                        left = this;
                        right = default(DvText);
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
                        left = this;
                        right = default(DvText);
                        return false;
                    }
                    // REVIEW: Can this be faster?
                    if (ContainsChar(text[ichCur], separators))
                        break;
                }
            }

            // Note that we don't use any fields of "this" here in case one
            // of the out parameters is the same as "this".
            left = new DvText(text, _ichMin, ichCur);
            right = new DvText(text, ichCur + 1, ichLim);
            return true;
        }

        /// <summary>
        /// Splits this instance on the right-most occurrence of separator and produces the left
        /// and right <see cref="DvText"/> values. If this instance does not contain the separator character,
        /// this returns false and sets <paramref name="left"/> to this instance and <paramref name="right"/>
        /// to the default <see cref="DvText"/> value.
        /// </summary>
        public bool SplitOneRight(char separator, out DvText left, out DvText right)
        {
            if (!HasChars)
            {
                left = this;
                right = default(DvText);
                return false;
            }

            string text = _outerBuffer;
            int ichMin = _ichMin;
            int ichLim = IchLim;

            int ichCur = ichLim;
            for (; ; )
            {
                Contracts.Assert(ichMin <= ichCur && ichCur <= ichLim);
                if (--ichCur < ichMin)
                {
                    left = this;
                    right = default(DvText);
                    return false;
                }
                if (text[ichCur] == separator)
                    break;
            }

            // Note that we don't use any fields of "this" here in case one
            // of the out parameters is the same as "this".
            left = new DvText(text, ichMin, ichCur);
            right = new DvText(text, ichCur + 1, ichLim);
            return true;
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

        /// <summary>
        /// Returns a text span with leading and trailing spaces trimmed. Note that this
        /// will remove only spaces, not any form of whitespace.
        /// </summary>
        public DvText Trim()
        {
            if (!HasChars)
                return this;
            int ichMin = _ichMin;
            int ichLim = IchLim;
            if (_outerBuffer[ichMin] != ' ' && _outerBuffer[ichLim - 1] != ' ')
                return this;

            while (ichMin < ichLim && _outerBuffer[ichMin] == ' ')
                ichMin++;
            while (ichMin < ichLim && _outerBuffer[ichLim - 1] == ' ')
                ichLim--;
            return new DvText(_outerBuffer, ichMin, ichLim);
        }

        /// <summary>
        /// Returns a text span with leading and trailing whitespace trimmed.
        /// </summary>
        public DvText TrimWhiteSpace()
        {
            if (!HasChars)
                return this;
            int ichMin = _ichMin;
            int ichLim = IchLim;
            if (!char.IsWhiteSpace(_outerBuffer[ichMin]) && !char.IsWhiteSpace(_outerBuffer[ichLim - 1]))
                return this;

            while (ichMin < ichLim && char.IsWhiteSpace(_outerBuffer[ichMin]))
                ichMin++;
            while (ichMin < ichLim && char.IsWhiteSpace(_outerBuffer[ichLim - 1]))
                ichLim--;
            return new DvText(_outerBuffer, ichMin, ichLim);
        }

        /// <summary>
        /// Returns a text span with trailing whitespace trimmed.
        /// </summary>
        public DvText TrimEndWhiteSpace()
        {
            if (!HasChars)
                return this;

            int ichLim = IchLim;
            if (!char.IsWhiteSpace(_outerBuffer[ichLim - 1]))
                return this;

            int ichMin = _ichMin;
            while (ichMin < ichLim && char.IsWhiteSpace(_outerBuffer[ichLim - 1]))
                ichLim--;

            return new DvText(_outerBuffer, ichMin, ichLim);
        }

        /// <summary>
        /// This produces zero for an empty string.
        /// </summary>
        public bool TryParse(out Single value)
        {
            if (IsNA)
            {
                value = Single.NaN;
                return true;
            }
            var res = DoubleParser.Parse(out value, _outerBuffer, _ichMin, IchLim);
            Contracts.Assert(res != DoubleParser.Result.Empty || value == 0);
            return res <= DoubleParser.Result.Empty;
        }

        /// <summary>
        /// This produces zero for an empty string.
        /// </summary>
        public bool TryParse(out Double value)
        {
            if (IsNA)
            {
                value = Double.NaN;
                return true;
            }
            var res = DoubleParser.Parse(out value, _outerBuffer, _ichMin, IchLim);
            Contracts.Assert(res != DoubleParser.Result.Empty || value == 0);
            return res <= DoubleParser.Result.Empty;
        }

        public uint Hash(uint seed)
        {
            Contracts.Check(!IsNA);
            return Hashing.MurmurHash(seed, _outerBuffer, _ichMin, IchLim);
        }

        // REVIEW: Add method to NormStr.Pool that deal with DvText instead of the other way around.
        public NormStr AddToPool(NormStr.Pool pool)
        {
            Contracts.Check(!IsNA);
            Contracts.CheckValue(pool, nameof(pool));
            return pool.Add(_outerBuffer, _ichMin, IchLim);
        }

        public NormStr FindInPool(NormStr.Pool pool)
        {
            Contracts.CheckValue(pool, nameof(pool));
            if (IsNA)
                return null;
            return pool.Get(_outerBuffer, _ichMin, IchLim);
        }

        public void AddToStringBuilder(StringBuilder sb)
        {
            Contracts.CheckValue(sb, nameof(sb));
            if (HasChars)
                sb.Append(_outerBuffer, _ichMin, Length);
        }

        public void AddLowerCaseToStringBuilder(StringBuilder sb)
        {
            Contracts.CheckValue(sb, nameof(sb));
            if (HasChars)
            {
                int min = _ichMin;
                int j;
                for (j = min; j < IchLim; j++)
                {
                    char ch = CharUtils.ToLowerInvariant(_outerBuffer[j]);
                    if (ch != _outerBuffer[j])
                    {
                        sb.Append(_outerBuffer, min, j - min).Append(ch);
                        min = j + 1;
                    }
                }

                Contracts.Assert(j == IchLim);
                if (min != j)
                    sb.Append(_outerBuffer, min, j - min);
            }
        }
    }

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
        /// For implementation of <see cref="IComparable{DvText}"/>. Uses code point comparison.
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
                            yield return memory.Slice(ichMinLocal, ichCur - ichMinLocal);
                            yield break;
                        }
                        if (text[ichCur] == chSep)
                            break;
                    }

                    yield return memory.Slice(ichMinLocal, ichCur - ichMinLocal);

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
                            yield return memory.Slice(ichMinLocal, ichCur - ichMinLocal);
                            yield break;
                        }
                        // REVIEW: Can this be faster?
                        if (ContainsChar(text[ichCur], separators))
                            break;
                    }

                    yield return memory.Slice(ichMinLocal, ichCur - ichMinLocal);

                    // Skip the separator.
                    ichCur++;
                }
            }
        }

        /// <summary>
        /// Splits this instance on the left-most occurrence of separator and produces the left
        /// and right <see cref="DvText"/> values. If this instance does not contain the separator character,
        /// this returns false and sets <paramref name="left"/> to this instance and <paramref name="right"/>
        /// to the default <see cref="DvText"/> value.
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
            left = memory.Slice(ichMin, ichCur - ichMin);
            right = memory.Slice(ichCur + 1, ichLim - ichCur - 1);
            return true;
        }

        /// <summary>
        /// Splits this instance on the left-most occurrence of an element of separators character array and
        /// produces the left and right <see cref="DvText"/> values. If this instance does not contain any of the
        /// characters in separators, thiss return false and initializes <paramref name="left"/> to this instance
        /// and <paramref name="right"/> to the default <see cref="DvText"/> value.
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
            left = memory.Slice(ichMin, ichCur - ichMin);
            right = memory.Slice(ichCur + 1, ichLim - ichCur - 1);
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
            return memory.Slice(ichMin, ichLim - ichMin);
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

            return memory.Slice(ichMin, ichLim - ichMin);
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

            return memory.Slice(ichMin, ichLim - ichMin);
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

        // REVIEW: Add method to NormStr.Pool that deal with DvText instead of the other way around.
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