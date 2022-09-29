// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Normalize the string before processing it with the tokenizer.
    /// </summary>
    public abstract class Normalizer
    {
        /// <summary>
        /// Process the original string to modify it and obtain a normalized string.
        /// </summary>
        /// <param name="original">The original string to normalize.</param>
        /// <returns>The normalized string along with the mapping to the original string.</returns>
        public abstract NormalizedString Normalize(string original);
    }
}
