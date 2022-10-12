// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// The pre-tokenizer which split the text at the word boundary.
    /// The word is a set of alphabet, numeric, and underscore characters.
    /// </summary>
    public sealed class WhiteSpace : PreTokenizer
    {
        /// <summary>
        /// Gets a singleton instance of the WhiteSpace pre-tokenizer..
        /// </summary>
        public static readonly WhiteSpace Instance = new WhiteSpace();

        private const string Pattern = @"\w+|[^\w\s]+";

        private static readonly IReadOnlyList<Split> _emptyList = new List<Split>();

        /// <summary>
        /// Splits the given string in multiple substrings at the word boundary, keeping track of the offsets of said substrings from the original string.
        /// </summary>
        /// <param name="sentence">The string to split into tokens.</param>
        /// <returns>The list of the splits containing the tokens and the token's offsets to the original string.</returns>
        public override IReadOnlyList<Split> PreTokenize(string? sentence)
        {
            if (sentence is null)
            {
                return _emptyList;
            }

            List<Split> parts = new List<Split>();

            foreach (Match match in Regex.Matches(sentence, Pattern))
            {
                parts.Add(new Split(match.Value, (match.Index, match.Index + match.Length)));
            }

            return parts;
        }
    }
}
