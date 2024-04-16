// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// The pre-tokenizer for SentencePiece tokenizers.
    /// </summary>
    internal sealed partial class SentencePiecePreTokenizer : PreTokenizer
    {
        /// <summary>
        /// Gets a singleton instance of the Roberta pre-tokenizer..
        /// </summary>
        public static SentencePiecePreTokenizer Instance { get; } = new SentencePiecePreTokenizer();

        /// <summary>
        /// Return the whole text as one chunk.
        /// </summary>
        /// <param name="text">The string to split into tokens.</param>
        /// <returns>The original string as one chunk.</returns>
        public override IEnumerable<Split> PreTokenize(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                yield break;
            }

            yield return new Split(text, (0, text.Length));
        }
    }
}
