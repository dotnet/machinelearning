// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// This Split contains the underlying split token as well as its offsets
    /// in the original string. These offsets are in the `original` referential.
    /// It also contains any `Token` associated to the current split.
    /// </summary>
    public readonly struct Split : IEquatable<Split>
    {
        /// <summary>
        /// Gets the underlying split token. Each SubString is represented by a token
        /// and in the end we might be carrying a lot of SubString representing various parts of the
        /// original input string.
        /// </summary>
        public string TokenString { get; }

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
            TokenString = token;
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
            TokenString == other.TokenString &&
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
        internal static readonly IReadOnlyList<Split> EmptyList = new List<Split>();

        /// <summary>
        /// Splits the given string in multiple substrings at the word boundary, keeping track of the offsets of said substrings from the original string.
        /// </summary>
        /// <param name="sentence">The string to split into tokens.</param>
        /// <param name="skipSpecialTokens">Indicates whether to skip the special tokens.</param>
        /// <returns>The list of the splits containing the tokens and the token's offsets to the original string.</returns>
        public abstract IEnumerable<Split> PreTokenize(string sentence, bool skipSpecialTokens = false);
    }

    internal sealed class RegexSplitEnumerable : IEnumerable<Split>
    {
        private readonly static Dictionary<string, Regex> _regexCache = new(StringComparer.Ordinal);
        private readonly Regex _regex;
        private readonly string _sentence;

        public RegexSplitEnumerable(string sentence, string pattern)
        {
            Debug.Assert(sentence is not null);
            Debug.Assert(pattern is not null);

            Regex? regex;
            lock (_regexCache)
            {
                if (!_regexCache.TryGetValue(pattern!, out regex))
                {
                    regex = new Regex(pattern, RegexOptions.Compiled);
                    _regexCache[pattern!] = regex;
                }
            }

            _regex = regex;
            _sentence = sentence!;
        }

        public IEnumerator<Split> GetEnumerator() => new RegexSplitEnumerator(_regex, _sentence);

        IEnumerator IEnumerable.GetEnumerator() => new RegexSplitEnumerator(_regex, _sentence);

        private sealed class RegexSplitEnumerator : IEnumerator<Split>
        {
            private Split _current = default;
            private readonly Regex _regex;
            private Match? _tokenMatch;
            private readonly string _sentence;

            public RegexSplitEnumerator(Regex regex, string sentence)
            {
                Debug.Assert(sentence is not null);
                Debug.Assert(regex is not null);

                _regex = regex!;
                _sentence = sentence!;
            }

            public Split Current => _current;

            object IEnumerator.Current => _current;

            public bool MoveNext()
            {
                if (_tokenMatch is null)
                {
                    _tokenMatch = _regex.Match(_sentence);
                }
                else if (!_tokenMatch.Success)
                {
                    return false;
                }
                else
                {
                    _tokenMatch = _tokenMatch.NextMatch();
                }

                if (!_tokenMatch.Success)
                {
                    return false;
                }

                _current = new Split(_tokenMatch.Value, (_tokenMatch.Index, _tokenMatch.Index + _tokenMatch.Length));
                return true;
            }

            public void Reset()
            {
                _tokenMatch = null;
            }

            public void Dispose()
            {
            }
        }
    }
}
