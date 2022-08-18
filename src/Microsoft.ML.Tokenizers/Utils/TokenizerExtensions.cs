// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Microsoft.ML.Tokenizers
{
    internal static class TokenizerExtensions
    {
        public static T? ArgMin<T>(this IEnumerable<T> source, Func<T, int> getValue)
        {
            var keys = source.ToList();     // avoid enumerate twice
            var values = keys.Select(getValue);
            var (minSource, minValue) = keys.Zip(values, (first, second) => (first, second)).Aggregate((min, x) => min.second <= x.second ? min : x);
            return minValue < int.MaxValue ? minSource : default;
        }

        public static TValue GetOrAdd<TKey, TValue>(this Dictionary<TKey, TValue> dic, TKey key, TValue setValue)
        {
            if (dic.TryGetValue(key, out var value))
            {
                return value;
            }

            dic[key] = setValue;
            return setValue;
        }

        public static IReadOnlyDictionary<TValue, TKey> Reverse<TKey, TValue>(this IReadOnlyDictionary<TKey, TValue> source)
        {
            Dictionary<TValue, TKey> dictionary = new Dictionary<TValue, TKey>();
            if (source != null)
            {
                foreach (KeyValuePair<TKey, TValue> pair in source)
                {
                    dictionary[pair.Value] = pair.Key;
                }
                return dictionary;
            }
            return dictionary;
        }

        public static SortedDictionary<TValue, TKey> ReverseSorted<TKey, TValue>(this IReadOnlyDictionary<TKey, TValue> source)
        {
            SortedDictionary<TValue, TKey> dictionary = new SortedDictionary<TValue, TKey>();
            if (source != null)
            {
                foreach (KeyValuePair<TKey, TValue> pair in source)
                {
                    dictionary[pair.Value] = pair.Key;
                }
                return dictionary;
            }
            return dictionary;
        }
    }

    internal class DictReversingConverter : JsonConverter<SortedDictionary<int, string>>
    {
        public override SortedDictionary<int, string>? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) => null;

        public override void Write(Utf8JsonWriter writer, SortedDictionary<int, string> value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();

            foreach (KeyValuePair<int, string> pair in value)
            {
                if (pair.Key >= 0)
                {
                    writer.WriteNumber(pair.Value, pair.Key);
                }
            }

            writer.WriteEndObject();
        }
    }
}
