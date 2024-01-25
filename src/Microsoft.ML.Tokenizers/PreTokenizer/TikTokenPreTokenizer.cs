// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// The pre-tokenizer for Tiktoken tokenizer.
    /// </summary>
    public sealed class TikTokenPreTokenizer : PreTokenizer
    {
        private readonly Regex? _specialTokensRegex;
        private readonly Regex _regex;

        public TikTokenPreTokenizer(string regexPattern, IReadOnlyDictionary<string, int>? specialTokensEncoder)
        {
            if (regexPattern is null)
            {
                throw new ArgumentNullException(nameof(regexPattern));
            }

            _regex = new Regex(regexPattern, RegexOptions.Compiled);

            if (specialTokensEncoder is not null && specialTokensEncoder.Count > 0)
            {
                _specialTokensRegex = new Regex(string.Join("|", specialTokensEncoder.Keys.Select(s => Regex.Escape(s))), RegexOptions.Compiled);
            }
        }

        /// <summary>
        /// Splits the given string in multiple substrings at the word boundary, keeping track of the offsets of said substrings from the original string.
        /// </summary>
        /// <param name="sentence">The string to split into tokens.</param>
        /// <param name="skipSpecialTokens">Indicates whether to skip the special tokens.</param>
        /// <returns>The list of the splits containing the tokens and the token's offsets to the original string.</returns>
        public override IEnumerable<Split> PreTokenize(string sentence, bool skipSpecialTokens = false)
        {
            if (string.IsNullOrEmpty(sentence))
            {
                return EmptyList;
            }

            return new TokenizationEnumerable(sentence, _regex, skipSpecialTokens ? null : _specialTokensRegex);
        }

        private readonly struct TokenizationEnumerable : IEnumerable<Split>
        {
            private readonly string _sentence;
            private readonly Regex _regex;
            private readonly Regex? _specialTokensRegex;

            public TokenizationEnumerable(string sentence, Regex regex, Regex? specialTokensRegex)
            {
                if (sentence is null)
                {
                    throw new ArgumentNullException(nameof(sentence));
                }

                if (regex is null)
                {
                    throw new ArgumentNullException(nameof(regex));
                }

                _sentence = sentence;
                _regex = regex;
                _specialTokensRegex = specialTokensRegex;
            }

            public readonly IEnumerator<Split> GetEnumerator() => new TokenizationEnumerator(_sentence, _regex, _specialTokensRegex);
            IEnumerator IEnumerable.GetEnumerator() => new TokenizationEnumerator(_sentence, _regex, _specialTokensRegex);

            private struct TokenizationEnumerator : IEnumerator<Split>
            {
                private Split _current = default;
                private int _startIndex;
                private int _offset;
                private MatchCollection? _matches;
                private int _matchIndex;
                private Match? _specialTokenMatch;
                private readonly Regex _regex;
                private readonly string _sentence;
                private readonly Regex? _specialTokensRegex;

                public TokenizationEnumerator(string sentence, Regex regex, Regex? specialTokensRegex)
                {
                    Debug.Assert(sentence is not null);
                    Debug.Assert(regex is not null);

                    _sentence = sentence!;
                    _regex = regex!;
                    _specialTokensRegex = specialTokensRegex;
                    _startIndex = 0;
                    _offset = 0;
                }

                readonly object IEnumerator.Current => _current;

                readonly Split IEnumerator<Split>.Current => _current;

                public bool MoveNext()
                {
                    if (_matches is not null && _matchIndex < _matches.Count)
                    {
                        Match match = _matches[_matchIndex];
                        _current = new Split(match.Value, (match.Index + _offset, match.Index + _offset + match.Length), false);
                        _startIndex += match.Length;
                        _matchIndex++;
                        return true;
                    }

                    if (_specialTokenMatch is not null && _specialTokenMatch.Success)
                    {
                        _current = new Split(_specialTokenMatch.Value, (_specialTokenMatch.Index, _specialTokenMatch.Index + _specialTokenMatch.Length), true);
                        _startIndex += _specialTokenMatch.Length;
                        _specialTokenMatch = null;
                        return true;
                    }

                    if (_startIndex >= _sentence.Length)
                    {
                        return false;
                    }

                    if (_specialTokensRegex is not null)
                    {
                        _specialTokenMatch = _specialTokensRegex.Match(_sentence, _startIndex);
                        _offset = _startIndex;
                        _matches = _regex.Matches(_sentence.Substring(_startIndex, _specialTokenMatch.Success ? _specialTokenMatch.Index - _startIndex : _sentence.Length - _startIndex));
                    }
                    else
                    {
                        _matches = _regex.Matches(_sentence);
                    }

                    if (_matches.Count > 0)
                    {
                        Match match = _matches[0];
                        _current = new Split(match.Value, (match.Index + _startIndex, match.Index + _startIndex + match.Length), false);
                        _startIndex += match.Length;
                        _matchIndex = 1;
                        return true;
                    }
                    else if (_specialTokenMatch is not null && _specialTokenMatch.Success)
                    {
                        _current = new Split(_specialTokenMatch.Value, (_specialTokenMatch.Index, _specialTokenMatch.Index + _specialTokenMatch.Length), true);
                        _startIndex += _specialTokenMatch.Length;
                        _specialTokenMatch = null;
                        return true;
                    }

                    return false;
                }

                public void Reset()
                {
                    _current = default;
                    _startIndex = 0;
                    _matches = null;
                    _matchIndex = -1;
                    _specialTokenMatch = null;
                }

                public void Dispose()
                {
                }
            }
        }
    }
}
