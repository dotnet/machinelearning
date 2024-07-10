// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Diagnostics;

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
        /// Gets a singleton instance of the <see cref="UpperCaseNormalizer"/>.
        /// </summary>
        public static UpperCaseNormalizer Instance { get; } = new UpperCaseNormalizer();

        /// <summary>
        /// Uppercase the original string.
        /// </summary>
        /// <param name="original">The original string to normalize to uppercase form.</param>
        /// <returns>The upper-cased normalized string.</returns>
        public override string Normalize(string original) => original.ToUpperInvariant();

        /// <summary>
        /// Uppercase the original string.
        /// </summary>
        /// <param name="original">The original string to normalize to uppercase form.</param>
        /// <returns>The upper-cased normalized string.</returns>
        public override string Normalize(ReadOnlySpan<char> original)
        {
            if (original.IsEmpty)
            {
                return string.Empty;
            }

            char[] arrayPoolArray = ArrayPool<char>.Shared.Rent(original.Length);

            int length = original.ToUpperInvariant(arrayPoolArray);
            Debug.Assert(length == original.Length);

            string result = new string(arrayPoolArray, 0, length);
            ArrayPool<char>.Shared.Return(arrayPoolArray);
            return result;
        }
    }
}
