// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Internal.Utilities
{
    /// <summary>
    /// Optional extension for RNG engines that can produce bulk sequences efficiently.
    /// </summary>
    internal interface IRandomBulkSource
    {
        /// <summary>Fills <paramref name="destination"/> with independent U[0,1) doubles.</summary>
        void NextDoubles(Span<double> destination);

        /// <summary>Fills <paramref name="destination"/> with independent uint values covering the full 32-bit range.</summary>
        void NextUInt32(Span<uint> destination);
    }
}

