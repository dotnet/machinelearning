﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// This Split contains the underlying split token as well as its offsets
    /// in the original string. These offsets are in the `original` referential.
    /// It also contains any `Token` associated to the current split.
    /// </summary>
    public struct Split : IEquatable<Split>
    {
        private readonly string? _originalString;
        private string? _tokenString;

        /// <summary>
        /// Gets the underlying split token. Each SubString is represented by a token
        /// and in the end we might be carrying a lot of SubString representing various parts of the
        /// original input string.
        /// </summary>
        public string TokenString => _tokenString ??= _originalString!.Substring(Offset.Index, Offset.End - Offset.Index);

        /// <summary>
        /// Gets the underlying split token as a span.
        /// </summary>
        public ReadOnlySpan<char> TokenSpan => _tokenString is string s ? s.AsSpan() : _originalString.AsSpan(Offset.Index, Offset.End - Offset.Index);

        /// <summary>
        /// Returns the offset mapping to the original string
        /// </summary>
        public (int Index, int End) Offset { get; }

        /// <summary>
        /// create a Split object using the token and the offset
        /// </summary>
        /// <param name="token">The token string</param>
        /// <param name="offset">The offset mapping to the original string</param>
        /// <param name="isSpecialToken">Indicates whether the token is a special token</param>
        public Split(string token, (int Index, int End) offset, bool isSpecialToken = false)
        {
            _tokenString = token;
            Offset = offset;
            IsSpecialToken = isSpecialToken;
        }

        internal Split(string originalString, string? token, (int Index, int End) offset, bool isSpecialToken = false)
        {
            _originalString = originalString;
            _tokenString = token;
            Offset = offset;
            IsSpecialToken = isSpecialToken;
        }

        /// <summary>
        /// Gets if the current Split is a special token.
        /// </summary>
        public bool IsSpecialToken { get; }

        /// <summary>
        /// Indicates whether the current Split object is equal to another Split object.
        /// </summary>
        /// <param name="other">The Split object to compare with the current object.</param>
        public bool Equals(Split other) =>
            (_originalString == other._originalString || TokenString == other.TokenString) &&
            IsSpecialToken == other.IsSpecialToken &&
            Offset.Index == other.Offset.Index &&
            Offset.End == other.Offset.End;
    }

    /// <summary>
    /// Base class for all pre-tokenizers classes.
    /// The PreTokenizer is in charge of doing the pre-segmentation step.
    /// </summary>
    public abstract class PreTokenizer
    {
        /// <summary>
        /// Splits the given string in multiple substrings at the word boundary, keeping track of the offsets of said substrings from the original string.
        /// </summary>
        /// <param name="sentence">The string to split into tokens.</param>
        /// <param name="skipSpecialTokens">Indicates whether to skip the special tokens.</param>
        /// <returns>The list of the splits containing the tokens and the token's offsets to the original string.</returns>
        public abstract IEnumerable<Split> PreTokenize(string sentence, bool skipSpecialTokens = false);

        internal static IEnumerable<Split> SplitSentence(string sentence, Regex regex)
        {
            (int Offset, int Length) match;
            int beginning = 0;
            while (TryGetMatch(regex, sentence, beginning, sentence.Length - beginning, out match))
            {
                yield return new Split(sentence, null, (match.Offset, match.Offset + match.Length));
                beginning = match.Offset + match.Length;
            }
        }

        internal static bool TryGetMatch(Regex regex, string sentence, int beginning, int length, out (int offset, int length) match)
        {
#if NET7_0_OR_GREATER
            foreach (ValueMatch m in regex.EnumerateMatches(sentence.AsSpan(beginning, length)))
            {
                match = (beginning + m.Index, m.Length);
                return true;
            }
#else
            Match m = regex.Match(sentence, beginning, length);
            if (m.Success)
            {
                match = (m.Index, m.Length);
                return true;
            }
#endif
            match = default;
            return false;
        }
    }
}
