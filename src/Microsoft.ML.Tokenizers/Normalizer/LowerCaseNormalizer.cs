// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Diagnostics;

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
        /// Gets a singleton instance of the <see cref="LowerCaseNormalizer"/>.
        /// </summary>
        public static LowerCaseNormalizer Instance { get; } = new LowerCaseNormalizer();

        /// <summary>
        /// Lowercase the original string.
        /// </summary>
        /// <param name="original">The original string to normalize to lowercase form.</param>
        /// <returns>The lower-cased normalized string.</returns>
        public override string Normalize(string original) => original.ToLowerInvariant();

        /// <summary>
        /// Lowercase the original string.
        /// </summary>
        /// <param name="original">The original string to normalize to lowercase form.</param>
        /// <returns>The lower-cased normalized string.</returns>
        public override string Normalize(ReadOnlySpan<char> original)
        {
            if (original.IsEmpty)
            {
                return string.Empty;
            }

            char[] arrayPoolArray = ArrayPool<char>.Shared.Rent(original.Length);

            int length = original.ToLowerInvariant(arrayPoolArray);
            Debug.Assert(length == original.Length);

            string result = new string(arrayPoolArray, 0, length);
            ArrayPool<char>.Shared.Return(arrayPoolArray);
            return result;
        }
    }
}
