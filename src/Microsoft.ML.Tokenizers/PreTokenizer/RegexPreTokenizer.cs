// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// The pre-tokenizer for Tiktoken tokenizer.
    /// </summary>
    public sealed partial class RegexPreTokenizer : PreTokenizer
    {
        private readonly Regex? _specialTokensRegex;
        private readonly Regex _regex;

        /// <summary>
        /// Initializes a new instance of the <see cref="RegexPreTokenizer"/> class.
        /// </summary>
        /// <param name="regex">The regex to use for splitting the text into smaller tokens in the pre-tokenization process.</param>
        /// <param name="specialTokensEncoder">The dictionary containing the special tokens and their corresponding ids.</param>
        /// <exception cref="ArgumentNullException">When regex is null</exception>
        public RegexPreTokenizer(Regex regex, IReadOnlyDictionary<string, int>? specialTokensEncoder)
        {
            if (regex is null)
            {
                throw new ArgumentNullException(nameof(regex));
            }

            _regex = regex;

            if (specialTokensEncoder is { Count: > 0 })
            {
                // We create this Regex object without a timeout, as we expect the match operation to complete in \(O(N)\) time complexity. Note that `specialTokensEncoder` is treated as constants after the pre-tokenizer is created.
                _specialTokensRegex = new Regex(string.Join("|", specialTokensEncoder.Keys.Select(s => Regex.Escape(s))), RegexOptions.Compiled);
            }
        }

        /// <summary>
        /// Get the offsets and lengths of the tokens relative to the <paramref name="text"/>.
        /// </summary>
        /// <param name="text">The string to split into tokens.</param>
        /// <returns>The offsets and lengths of the tokens, expressed as pairs, are relative to the original string.</returns>
        public override IEnumerable<(int Offset, int Length)> PreTokenize(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                return [];
            }

            return SplitText(text, _regex, _specialTokensRegex);

            static IEnumerable<(int Offset, int Length)> SplitText(string text, Regex regex, Regex? specialTokensRegex)
            {
                (int Offset, int Length) match;
                int beginning = 0;

                if (specialTokensRegex is not null)
                {
                    while (true)
                    {
                        (int Offset, int Length) specialMatch;
                        if (!TryGetMatch(specialTokensRegex, text, beginning, text.Length - beginning, out specialMatch))
                        {
                            break;
                        }

                        while (TryGetMatch(regex, text, beginning, specialMatch.Offset - beginning, out match))
                        {
                            yield return (match.Offset, match.Length);
                            beginning = match.Offset + match.Length;
                        }

                        yield return (specialMatch.Offset, specialMatch.Length);
                        beginning = specialMatch.Offset + specialMatch.Length;
                    }
                }

                while (TryGetMatch(regex, text, beginning, text.Length - beginning, out match))
                {
                    yield return (match.Offset, match.Length);
                    beginning = match.Length + match.Offset;
                }
            }
        }

        /// <summary>
        /// Get the offsets and lengths of the tokens relative to the <paramref name="text"/>.
        /// </summary>
        /// <param name="text">The string to split into tokens.</param>
        /// <returns>The offsets and lengths of the tokens, expressed as pairs, are relative to the original string.</returns>
        public override IEnumerable<(int Offset, int Length)> PreTokenize(ReadOnlySpan<char> text)
        {
            if (text.IsEmpty)
            {
                return [];
            }

#if NET7_0_OR_GREATER
            char[] buffer = ArrayPool<char>.Shared.Rent(text.Length);
            text.CopyTo(buffer);
            return SplitText(buffer, _regex, _specialTokensRegex, text.Length);

            static IEnumerable<(int Offset, int Length)> SplitText(char[] text, Regex regex, Regex? specialTokensRegex, int textLength)
            {
                (int Offset, int Length) match;
                int beginning = 0;

                if (specialTokensRegex is not null)
                {
                    while (true)
                    {
                        (int Offset, int Length) specialMatch;
                        if (!TryGetMatch(specialTokensRegex, text.AsSpan(), beginning, textLength - beginning, out specialMatch))
                        {
                            break;
                        }

                        while (TryGetMatch(regex, text.AsSpan(), beginning, specialMatch.Offset - beginning, out match))
                        {
                            yield return (match.Offset, match.Length);
                            beginning = match.Offset + match.Length;
                        }

                        yield return (specialMatch.Offset, specialMatch.Length);
                        beginning = specialMatch.Offset + specialMatch.Length;
                    }
                }

                while (TryGetMatch(regex, text.AsSpan(), beginning, textLength - beginning, out match))
                {
                    yield return (match.Offset, match.Length);
                    beginning = match.Length + match.Offset;
                }

                ArrayPool<char>.Shared.Return(text);
            }
#else
            return PreTokenize(text.ToString());
#endif // NET7_0_OR_GREATER
        }
    }
}
