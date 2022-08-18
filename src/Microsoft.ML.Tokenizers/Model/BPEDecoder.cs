// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Allows decoding Original BPE by joining all the tokens and then replacing
    /// the suffix used to identify end-of-words by white spaces
    /// </summary>
    public sealed class BpeDecoder : TokenizerDecoder
    {
        private readonly string? _suffix;

        /// <summary>
        /// Construct a new Bpe decoder object.
        /// </summary>
        public BpeDecoder() => _suffix = null;

        /// <summary>
        /// Construct a new Bpe decoder object.
        /// </summary>
        /// <param name="suffix">The suffix that was used to characterize an end-of-word. This suffix will be replaced by white spaces during the decoding.</param>
        public BpeDecoder(string? suffix) => _suffix = suffix;

        /// <summary>
        /// Decode the original BPE by joining all the tokens and then replacing the suffix used to identify end-of-words by white spaces.
        /// </summary>
        /// <param name="tokens">The list of tokens to merge.</param>
        /// <returns>The string containing all merged tokens.</returns>
        public override string Decode(IEnumerable<string> tokens)
        {
            if (tokens is null)
            {
                throw new ArgumentNullException(nameof(tokens));
            }

            if (_suffix == null)
            {
                return string.Join("", tokens);
            }

            return string.Join(" ", tokens).Replace(_suffix, " ");
        }
    }
}
