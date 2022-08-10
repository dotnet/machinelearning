// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Normalize the string to lowercase form before processing it with the tokenizer.
    /// </summary>
    public sealed class LowerCaseNormalizer : Normalizer
    {
        /// <summary>
        /// Creates a LowerCaseNormalizer object.
        /// </summary>
        public LowerCaseNormalizer() { }

        /// <summary>
        /// Lowercase the original string.
        /// </summary>
        /// <param name="original">The original string to normalize to lowercase form.</param>
        /// <returns>The lower-cased normalized string.</returns>
        public override NormalizedString Normalize(string original) => new NormalizedString(original, original.ToLowerInvariant(), mapping: null, isOneToOneMapping: true);
    }
}
