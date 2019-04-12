// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Auto.Test
{
    internal static class MLNetUtils
    {
        public static bool[] BuildArray(int length, IEnumerable<DataViewSchema.Column> columnsNeeded)
        {
            Contracts.CheckParam(length >= 0, nameof(length));

            var result = new bool[length];
            foreach (var col in columnsNeeded)
            {
                if (col.Index < result.Length)
                    result[col.Index] = true;
            }

            return result;
        }
    }
}
