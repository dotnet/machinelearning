// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// The settings used to encode a text.
    /// </summary>
    public struct EncodeSettings
    {
        public EncodeSettings() { MaxTokenCount = int.MaxValue; }

        /// <summary>
        /// Gets or sets a value indicating whether to consider the input normalization during encoding.
        /// </summary>
        public bool ConsiderNormalization { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to consider the pre-tokenization during encoding.
        /// </summary>
        public bool ConsiderPreTokenization { get; set; }

        /// <summary>
        /// Gets or sets the maximum number of tokens to generate.
        /// </summary>
        public int MaxTokenCount { get; set; }
    }
}

