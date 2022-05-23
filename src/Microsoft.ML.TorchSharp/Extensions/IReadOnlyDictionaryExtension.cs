// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.TorchSharp.Extensions
{
    internal static class IReadOnlyDictionaryExtension
    {
        public static IReadOnlyDictionary<TValue, TKey> Reverse<TKey, TValue>(this IReadOnlyDictionary<TKey, TValue> source)
        {
            var dictionary = new Dictionary<TValue, TKey>();
            if (source != null)
            {
                foreach (var pair in source)
                {
                    dictionary[pair.Value] = pair.Key;
                }
            }
            return dictionary;
        }
    }
}
