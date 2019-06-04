// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading;

namespace Microsoft.ML.AutoML
{
    internal static class AutoMlUtils
    {
        public static readonly ThreadLocal<Random> Random = new ThreadLocal<Random>(() => new Random());

        public static void Assert(bool boolVal, string message = null)
        {
            if (!boolVal)
            {
                message = message ?? "Assertion failed";
                throw new InvalidOperationException(message);
            }
        }
    }
}
