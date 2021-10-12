// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Data
{
    // taken from https://github.com/dotnet/machinelearning/blob/08318656ed0f0649aa75370b019cf4bcbda5a6a5/src/Microsoft.ML.Core/Utilities/Hashing.cs#L17-L25
    internal static class Hashing
    {
        public static uint CombineHash(uint u1, uint u2)
        {
            return ((u1 << 7) | (u1 >> 25)) ^ u2;
        }

        public static int CombineHash(int n1, int n2)
        {
            return (int)CombineHash((uint)n1, (uint)n2);
        }
    }
}
