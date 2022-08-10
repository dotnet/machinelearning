// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Normalize the string to uppercase form before processing it with the tokenizer.
    /// </summary>
    public sealed class UpperCaseNormalizer : Normalizer
    {
        /// <summary>
        /// Creates a UpperCaseNormalizer object.
        /// </summary>
        public UpperCaseNormalizer() { }

        /// <summary>
        /// Uppercase the original string.
        /// </summary>
        /// <param name="original">The original string to normalize to uppercase form.</param>
        /// <returns>The upper-cased normalized string.</returns>
        public override NormalizedString Normalize(string original) => new NormalizedString(original, original.ToUpperInvariant(), mapping: null, isOneToOneMapping: true);
    }
}
