// Licensed to the .NET Foundation under one or more agreements.
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
        public string TokenString => _tokenString ??= _originalString!.Substring(Offset.Index, Offset.Length);

        /// <summary>
        /// Gets the underlying split token as a span.
        /// </summary>
        public ReadOnlySpan<char> TokenSpan => _tokenString is string s ? s.AsSpan() : _originalString.AsSpan(Offset.Index, Offset.Length);

        /// <summary>
        /// Returns the offset mapping to the original string
        /// </summary>
        public (int Index, int Length) Offset { get; }

        /// <summary>
        /// create a Split object using the token and the offset
        /// </summary>
        /// <param name="token">The token string</param>
        /// <param name="offset">The offset mapping to the original string</param>
        public Split(string token, (int Index, int Length) offset)
        {
            _tokenString = token;
            Offset = offset;
        }

        internal Split(string originalString, string? token, (int Index, int Length) offset)
        {
            _originalString = originalString;
            _tokenString = token;
            Offset = offset;
        }

        /// <summary>
        /// Indicates whether the current Split object is equal to another Split object.
        /// </summary>
        /// <param name="other">The Split object to compare with the current object.</param>
        public bool Equals(Split other) =>
            (_originalString == other._originalString || TokenString == other.TokenString) &&
            Offset.Index == other.Offset.Index &&
            Offset.Length == other.Offset.Length;
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
        /// <param name="text">The string to split into tokens.</param>
        /// <returns>The list of the splits containing the tokens and the token's offsets to the original string.</returns>
        public abstract IEnumerable<Split> PreTokenize(string text);

        /// <summary>
        /// Splits the given character span in multiple spans at the word boundary, keeping track of the offsets of said substrings from the original span.
        /// </summary>
        /// <param name="text">The character span to split into tokens.</param>
        /// <returns>The list of the splits containing the tokens and the token's offsets to the original character span.</returns>
        public virtual IEnumerable<Split> PreTokenize(ReadOnlySpan<char> text) => PreTokenize(text.ToString());

        internal static IEnumerable<Split> SplitText(string text, Regex regex)
        {
            (int Offset, int Length) match;
            int beginning = 0;
            while (TryGetMatch(regex, text, beginning, text.Length - beginning, out match))
            {
                yield return new Split(text, null, (match.Offset, match.Length));
                beginning = match.Offset + match.Length;
            }
        }

        internal static bool TryGetMatch(Regex regex, string text, int beginning, int length, out (int offset, int length) match)
        {
#if NET7_0_OR_GREATER
            foreach (ValueMatch m in regex.EnumerateMatches(text.AsSpan(beginning, length)))
            {
                match = (beginning + m.Index, m.Length);
                return true;
            }
#else
            Match m = regex.Match(text, beginning, length);
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
