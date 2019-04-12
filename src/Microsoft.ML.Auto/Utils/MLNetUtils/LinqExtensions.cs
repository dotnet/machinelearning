// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Auto
{
    internal static class LinqExtensions
    {
        public static int ArgMax<T>(this IEnumerable<T> e) where T : IComparable<T>
        {
            T max = e.First();
            int argMax = 0;
            int i = 1;
            foreach (T d in e.Skip(1))
            {
                if (d.CompareTo(max) > 0)
                {
                    argMax = i;
                    max = d;
                }
                ++i;
            }
            return argMax;
        }
    }
}
