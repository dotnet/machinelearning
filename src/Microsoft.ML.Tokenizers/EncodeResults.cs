// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// The result of encoding a text.
    /// </summary>
    /// <typeparam name="T">The type of the tokens.</typeparam>
    public struct EncodeResults<T>
    {
        /// <summary>
        /// Gets or sets the list of tokens generated from the encoded text.
        /// </summary>
        public IReadOnlyList<T> Tokens { get; set; }

        /// <summary>
        /// Gets or sets the normalized text generated during the encoding process. This can be <see langword="null"/> if the encoding process does not normalize the input text.
        /// </summary>
        public string? NormalizedText { get; set; }

        /// <summary>
        /// Gets or sets the count of characters consumed from the input text.
        /// </summary>
        public int CharsConsumed { get; set; }
    }
}
