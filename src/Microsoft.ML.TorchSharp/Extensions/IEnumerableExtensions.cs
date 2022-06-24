// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.TorchSharp.Extensions
{
    internal static class IEnumerableExtensions
    {
        public static string ToString<T>(this IEnumerable<T> enumerable, bool verbose = true)
        {
            return verbose
                ? $"[{string.Join(", ", enumerable.Select(e => e.ToString()))}]"
                : enumerable?.ToString() ?? string.Empty;
        }

        public static T ArgMin<T>(this IEnumerable<T> source, Func<T, int> getValue)
        {
            var keys = source.ToList();     // avoid enumerate twice
            var values = keys.Select(getValue);
            var (minSource, minValue) = keys.Zip(values, (first, second) => (first, second)).Aggregate((min, x) => min.second <= x.second ? min : x);
            return minValue < int.MaxValue ? minSource : default;
        }
    }
}
