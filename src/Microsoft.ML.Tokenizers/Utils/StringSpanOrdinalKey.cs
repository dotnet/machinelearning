// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>Used as a key in a dictionary to enable querying with either a string or a span.</summary>
    /// <remarks>
    /// This should only be used with a Ptr/Length for querying. For storing in a dictionary, this should
    /// always be used with a string.
    /// </remarks>
    internal unsafe readonly struct StringSpanOrdinalKey : IEquatable<StringSpanOrdinalKey>
    {
        public readonly char* Ptr;
        public readonly int Length;
        public readonly string? Data;

        public StringSpanOrdinalKey(char* ptr, int length)
        {
            Ptr = ptr;
            Length = length;
        }

        public StringSpanOrdinalKey(string data) =>
            Data = data;

        private ReadOnlySpan<char> Span => Ptr is not null ?
            new ReadOnlySpan<char>(Ptr, Length) :
            Data.AsSpan();

        public override bool Equals(object? obj) =>
            obj is StringSpanOrdinalKey wrapper && Equals(wrapper);

        public bool Equals(StringSpanOrdinalKey other) =>
            Span.SequenceEqual(other.Span);

        public override int GetHashCode() => Helpers.GetHashCode(Span);
    }

    internal sealed class StringSpanOrdinalKeyCache<TValue>
    {
        private readonly int _capacity;
        private readonly Dictionary<StringSpanOrdinalKey, TValue> _map;

        private object SyncObj => _map;

        internal StringSpanOrdinalKeyCache() : this(Bpe.DefaultCacheCapacity) { }

        internal StringSpanOrdinalKeyCache(int capacity)
        {
            _capacity = capacity;
            _map = new Dictionary<StringSpanOrdinalKey, TValue>(capacity);
        }

        internal bool TryGetValue(string key, out TValue value)
        {
            lock (SyncObj)
            {
                return _map.TryGetValue(new StringSpanOrdinalKey(key), out value!);
            }
        }

        internal unsafe bool TryGetValue(ReadOnlySpan<char> key, out TValue value)
        {
            lock (SyncObj)
            {
                fixed (char* ptr = key)
                {
                    return _map.TryGetValue(new StringSpanOrdinalKey(ptr, key.Length), out value!);
                }
            }
        }

        internal void Remove(string key)
        {
            lock (SyncObj)
            {
                _map.Remove(new StringSpanOrdinalKey(key));
            }
        }

        internal void Set(string k, TValue v)
        {
            lock (SyncObj)
            {
                if (_map.Count < _capacity)
                {
                    _map[new StringSpanOrdinalKey(k)] = v;
                }
            }
        }
    }

    /// <summary>
    /// Custom JSON converter for <see cref="StringSpanOrdinalKey"/>.
    /// </summary>
    internal sealed class StringSpanOrdinalKeyConverter : JsonConverter<StringSpanOrdinalKey>
    {
        public override StringSpanOrdinalKey ReadAsPropertyName(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) =>
            new StringSpanOrdinalKey(reader.GetString()!);

        public override void WriteAsPropertyName(Utf8JsonWriter writer, StringSpanOrdinalKey value, JsonSerializerOptions options) =>
            writer.WriteStringValue(value.Data!);

        public override StringSpanOrdinalKey Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) => new StringSpanOrdinalKey(reader.GetString()!);
        public override void Write(Utf8JsonWriter writer, StringSpanOrdinalKey value, JsonSerializerOptions options) => writer.WriteStringValue(value.Data!);
    }

    /// <summary>
    /// Extension methods for <see cref="StringSpanOrdinalKey"/>.
    /// </summary>
    internal static class StringSpanOrdinalKeyExtensions
    {
        public unsafe static bool TryGetValueUnsafe<TValue>(this IReadOnlyDictionary<StringSpanOrdinalKey, TValue> map, ReadOnlySpan<char> key, out TValue value)
        {
            fixed (char* ptr = key)
            {
                return map.TryGetValue(new StringSpanOrdinalKey(ptr, key.Length), out value!);
            }
        }

        public static bool TryGetValueUnsafe<TValue>(this IReadOnlyDictionary<StringSpanOrdinalKey, TValue> map, string key, out TValue value) =>
            map.TryGetValue(new StringSpanOrdinalKey(key), out value!);
    }
}
