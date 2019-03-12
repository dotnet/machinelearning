// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Experimental
{
    public static class Experimental
    {
        /// <summary>
        /// Stop the exeuction of pipeline in <see cref="MLContext"/>
        /// </summary>
        /// <param name="ctx"></param>
        public static void StopExecution(this MLContext ctx) => ctx.StopExecution();
    }
}
