// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.AutoML.Utils
{
    internal static class StringEditDistance
    {
        public static int GetLevenshteinDistance(string first, string second)
        {
            if (first is null)
            {
                throw new ArgumentNullException(nameof(first));
            }

            if (second is null)
            {
                throw new ArgumentNullException(nameof(second));
            }

            if (first.Length == 0 || second.Length == 0)
            {
                return first.Length + second.Length;
            }

            var currentRow = 0;
            var nextRow = 1;
            var rows = new int[second.Length + 1, second.Length + 1];

            for (var j = 0; j <= second.Length; ++j)
            {
                rows[currentRow, j] = j;
            }

            for (var i = 1; i <= first.Length; ++i)
            {
                rows[nextRow, 0] = i;
                for (var j = 1; j <= second.Length; ++j)
                {
                    var deletion = rows[currentRow, j] + 1;
                    var insertion = rows[nextRow, j - 1] + 1;
                    var substitution = rows[currentRow, j - 1] + (first[i - 1].Equals(second[j - 1]) ? 0 : 1);

                    rows[nextRow, j] = Math.Min(deletion, Math.Min(insertion, substitution));
                }

                if (currentRow == 0)
                {
                    currentRow = 1;
                    nextRow = 0;
                }
                else
                {
                    currentRow = 0;
                    nextRow = 1;
                }
            }

            return rows[currentRow, second.Length];
        }
    }
}
