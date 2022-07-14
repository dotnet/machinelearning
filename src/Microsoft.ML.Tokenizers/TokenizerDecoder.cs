// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// A Decoder has the responsibility to merge the given list of tokens in a string.
    /// </summary>
    public abstract class TokenizerDecoder
    {
        /// <summary>
        /// Decode by joining all the tokens to a string.
        /// </summary>
        /// <param name="tokens">The list of tokens to merge.</param>
        /// <returns>The string containing all merged tokens.</returns>
        public abstract string Decode(IEnumerable<string> tokens);
    }
}
