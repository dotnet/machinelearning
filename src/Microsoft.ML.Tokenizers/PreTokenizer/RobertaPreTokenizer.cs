// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// The pre-tokenizer for Roberta English tokenizer.
    /// </summary>
    public sealed partial class RobertaPreTokenizer : PreTokenizer
    {
        /// <summary>
        /// Gets a singleton instance of the Roberta pre-tokenizer..
        /// </summary>
        public static RobertaPreTokenizer Instance { get; } = new RobertaPreTokenizer();

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

            return SplitText(text, TiktokenTokenizer.P50kBaseRegex());
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

            return SplitText(text, TiktokenTokenizer.P50kBaseRegex());
        }
    }
}
