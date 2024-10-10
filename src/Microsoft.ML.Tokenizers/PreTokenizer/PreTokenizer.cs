// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Base class for all pre-tokenizers classes.
    /// The PreTokenizer is in charge of doing the pre-segmentation step.
    /// </summary>
    public abstract partial class PreTokenizer
    {
        /// <summary>
        /// Get the offsets and lengths of the tokens relative to the <paramref name="text"/>.
        /// </summary>
        /// <param name="text">The string to split into tokens.</param>
        /// <returns>The offsets and lengths of the tokens, expressed as pairs, are relative to the original string.</returns>
        public abstract IEnumerable<(int Offset, int Length)> PreTokenize(string text);

        /// <summary>
        /// Get the offsets and lengths of the tokens relative to the original string.
        /// </summary>
        /// <param name="text">The character span to split into tokens.</param>
        /// <returns>The offsets and lengths of the tokens, expressed as pairs, are relative to the original string.</returns>
        public abstract IEnumerable<(int Offset, int Length)> PreTokenize(ReadOnlySpan<char> text);

        internal static IEnumerable<(int Offset, int Length)> SplitText(string text, Regex regex)
        {
            (int Offset, int Length) match;
            int beginning = 0;
            while (TryGetMatch(regex, text, beginning, text.Length - beginning, out match))
            {
                yield return (match.Offset, match.Length);
                beginning = match.Offset + match.Length;
            }
        }

        private const string WhiteSpacePattern = /*lang=regex*/ @"\w+|[^\w\s]+";
        private static PreTokenizer? _whiteSpacePreTokenizer;
#if NET7_0_OR_GREATER
        [GeneratedRegex(WhiteSpacePattern)]
        private static partial Regex WhiteSpaceRegex();
#else
        private static Regex WhiteSpaceRegex() => new Regex(WhiteSpacePattern, RegexOptions.Compiled);
#endif

        /// <summary>
        /// Create a new instance of the <see cref="PreTokenizer"/> class which split the text at the word boundary.
        /// The word is a set of alphabet, numeric, and underscore characters.
        /// </summary>
        /// <param name="specialTokensEncoder">The dictionary containing the special tokens and their corresponding ids.</param>
        /// <returns>The pre-tokenizer that splits the text at the word boundary.</returns>
        public static PreTokenizer CreateWhiteSpace(IReadOnlyDictionary<string, int>? specialTokensEncoder = null)
        {
            if (specialTokensEncoder is null)
            {
                // return a singleton instance of the WhiteSpace pre-tokenizer
                return _whiteSpacePreTokenizer ??= new RegexPreTokenizer(WhiteSpaceRegex(), null);
            }

            return new RegexPreTokenizer(WhiteSpaceRegex(), specialTokensEncoder);
        }

        internal static IEnumerable<(int Offset, int Length)> SplitText(ReadOnlySpan<char> text, Regex regex)
        {
#if NET7_0_OR_GREATER
            char[] buffer = ArrayPool<char>.Shared.Rent(text.Length);
            text.CopyTo(buffer);
            return SplitText(buffer, regex, text.Length);

            static IEnumerable<(int Offset, int Length)> SplitText(char[] text, Regex regex, int textLength)
            {
                (int Offset, int Length) match;
                int beginning = 0;
                while (TryGetMatch(regex, text, beginning, textLength - beginning, out match))
                {
                    yield return (match.Offset, match.Length);
                    beginning = match.Offset + match.Length;
                }

                ArrayPool<char>.Shared.Return(text);
            }
#else
            return SplitText(text.ToString(), regex);
#endif // NET7_0_OR_GREATER
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

#if NET7_0_OR_GREATER
        internal static bool TryGetMatch(Regex regex, scoped ReadOnlySpan<char> text, int beginning, int length, out (int offset, int length) match)
        {
            foreach (ValueMatch m in regex.EnumerateMatches(text.Slice(beginning, length)))
            {
                match = (beginning + m.Index, m.Length);
                return true;
            }
            match = default;
            return false;
        }
#endif // NET7_0_OR_GREATER
    }
}
