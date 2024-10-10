// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using static TorchSharp.torch;

namespace Microsoft.ML.TorchSharp
{
    internal static class RangeUtil
    {
        public static TensorIndex ToTensorIndex(this Range range)
        {
            long? start = !range.Start.IsFromEnd ? range.Start.Value : -1 * range.Start.Value;
            var stop = !range.End.IsFromEnd ? new long?(range.End.Value) : range.End.Value == 0 ? null : new long?(-1 * range.End.Value);
            return TensorIndex.Slice(start, stop);
        }
    }
}
