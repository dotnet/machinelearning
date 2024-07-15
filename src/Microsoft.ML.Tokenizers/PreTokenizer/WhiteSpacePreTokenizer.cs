// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// The pre-tokenizer which split the text at the word boundary.
    /// The word is a set of alphabet, numeric, and underscore characters.
    /// </summary>
    public sealed partial class WhiteSpacePreTokenizer : PreTokenizer
    {
        /// <summary>
        /// Gets a singleton instance of the WhiteSpace pre-tokenizer..
        /// </summary>
        public static WhiteSpacePreTokenizer Instance { get; } = new WhiteSpacePreTokenizer();

        private const string PretokenizePattern = /*lang=regex*/ @"\w+|[^\w\s]+";
#if NET7_0_OR_GREATER
        [GeneratedRegex(PretokenizePattern)]
        private static partial Regex PretokenizeRegex();
#else
        private static readonly Regex _regex = new Regex(PretokenizePattern, RegexOptions.Compiled);
        private static Regex PretokenizeRegex() => _regex;
#endif

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

            return SplitText(text, PretokenizeRegex());
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

            return SplitText(text, PretokenizeRegex());
        }
    }
}
