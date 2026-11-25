// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Detects whether <see cref="string.Normalize(NormalizationForm)"/> works, so that the features relying on it can fail loudly
    /// rather than silently producing different tokens.
    /// </summary>
    internal static class UnicodeNormalization
    {
        // string.Normalize returns its input unchanged in globalization-invariant mode. There's no public API
        // to detect that mode yet (https://github.com/dotnet/runtime/issues/81429), so check Normalize directly.
        internal static bool IsSupported { get; } = "\u00E9".Normalize(NormalizationForm.FormD).Length == 2;
    }
}
