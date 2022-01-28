// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.SearchSpace
{
    internal static class Contract
    {
        public static void Requires(bool condition, string msg)
        {
            if (!condition)
            {
                throw new Exception(msg);
            }
        }
    }
}
