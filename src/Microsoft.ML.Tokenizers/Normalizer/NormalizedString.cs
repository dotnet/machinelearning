// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Contains the normalized string and the mapping to the original string.
    /// </summary>
    public readonly struct NormalizedString
    {
        /// <summary>
        /// Create NormalizedString object containing the normalization of the original string and the mapping
        /// between the original and the normalized string.
        /// </summary>
        /// <param name="original">The original string before normalization.</param>
        /// <param name="normalizedString">The normalized string.</param>
        /// <param name="mapping">The mapping between the normalized string and the original string.</param>
        /// <param name="isOneToOneMapping">Indicate whether the mapping is one-to-one.</param>
        public NormalizedString(string original, string normalizedString, int[]? mapping, bool isOneToOneMapping)
        {
            Original = original;
            Normalized = normalizedString;
            NormalizedToOriginalMapping = mapping;

            if (mapping is not null && mapping.Length < normalizedString.Length)
            {
                throw new ArgumentException($"Mapping array has to cover the whole normalized string length mapping", nameof(mapping));
            }

            IsOneToOneMapping = isOneToOneMapping;
        }

        /// <summary>
        /// Gets the original string before the normalization.
        /// </summary>
        public string Original { get; }

        /// <summary>
        /// Gets the normalized string.
        /// </summary>
        public string Normalized { get; }

        /// <summary>
        /// Gets the mapping between the normalized string and the original string.
        /// </summary>
        /// <remarks>
        /// The mapping can be null if IsOneToOneMapping is true or if the normalization doesn't support the mapping.
        /// </remarks>
        public int[]? NormalizedToOriginalMapping { get; }

        /// <summary>
        /// Gets whether the normalization between the normalized string and the original string is one-to-one.
        /// </summary>
        public bool IsOneToOneMapping { get; }

        /// <summary>
        /// Gets whether can map the normalized string the original string.
        /// </summary>
        public bool CanMapToOriginal => IsOneToOneMapping || NormalizedToOriginalMapping is not null;
    }
}
