// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Checks that <see cref="string.Normalize(NormalizationForm)"/> works, so that the features relying on it fail loudly
    /// rather than silently producing different tokens.
    /// </summary>
    internal static class UnicodeNormalization
    {
        // string.Normalize returns its input unchanged in globalization-invariant mode. There's no public API
        // to detect that mode yet (https://github.com/dotnet/runtime/issues/81429), so check Normalize directly.
        private static readonly bool _isSupported = "\u00E9".Normalize(NormalizationForm.FormD).Length == 2;

        /// <summary>
        /// Throws <see cref="PlatformNotSupportedException"/> when <see cref="string.Normalize(NormalizationForm)"/> returns its input
        /// unchanged, as it does when globalization-invariant mode is enabled.
        /// </summary>
        /// <param name="feature">The feature that needs Unicode normalization, which starts the exception message.</param>
        /// <param name="alternative">An optional sentence that ends the exception message, giving another way to fix the problem.</param>
        internal static void ThrowIfNotSupported(string feature, string? alternative = null)
        {
            if (!_isSupported)
            {
                throw new PlatformNotSupportedException(
                    $"{feature} needs Unicode normalization, which isn't available when globalization-invariant mode is enabled. " +
                    "To use it, set <InvariantGlobalization>false</InvariantGlobalization> in the project file and make sure ICU is available, " +
                    "for example by using a container image that includes it, such as a chiseled '-extra' image." +
                    (alternative is null ? string.Empty : " " + alternative));
            }
        }
    }
}
