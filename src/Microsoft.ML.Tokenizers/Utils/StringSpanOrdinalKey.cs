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
    internal readonly unsafe struct StringSpanOrdinalKey : IEquatable<StringSpanOrdinalKey>
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

        public override string ToString() => Data ?? Span.ToString();

        public override bool Equals(object? obj) =>
            obj is StringSpanOrdinalKey wrapper && Equals(wrapper);

        public bool Equals(StringSpanOrdinalKey other) =>
            Span.SequenceEqual(other.Span);

        public override int GetHashCode() => Helpers.GetHashCode(Span);
    }

    internal readonly unsafe struct StringSpanOrdinalKeyPair : IEquatable<StringSpanOrdinalKeyPair>
    {
        private readonly StringSpanOrdinalKey _left;
        private readonly StringSpanOrdinalKey _right;

        public StringSpanOrdinalKeyPair(char* ptr1, int length1, char* ptr2, int length2)
        {
            _left = new StringSpanOrdinalKey(ptr1, length1);
            _right = new StringSpanOrdinalKey(ptr2, length2);
        }

        public StringSpanOrdinalKeyPair(string data1, string data2)
        {
            _left = new StringSpanOrdinalKey(data1);
            _right = new StringSpanOrdinalKey(data2);
        }
        public override bool Equals(object? obj) =>
            obj is StringSpanOrdinalKeyPair wrapper && wrapper._left.Equals(_left) && wrapper._right.Equals(_right);

        public bool Equals(StringSpanOrdinalKeyPair other) => other._left.Equals(_left) && other._right.Equals(_right);

        public override int GetHashCode() => HashCode.Combine(_left.GetHashCode(), _right.GetHashCode());
    }


    internal sealed class StringSpanOrdinalKeyCache<TValue>
    {
        private readonly int _capacity;
        private readonly Dictionary<StringSpanOrdinalKey, TValue> _map;

        private object SyncObj => _map;

        internal StringSpanOrdinalKeyCache() : this(BpeTokenizer.DefaultCacheCapacity) { }

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
        public static StringSpanOrdinalKeyConverter Instance { get; } = new StringSpanOrdinalKeyConverter();
        public override StringSpanOrdinalKey ReadAsPropertyName(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) =>
            new StringSpanOrdinalKey(reader.GetString()!);

        public override void WriteAsPropertyName(Utf8JsonWriter writer, StringSpanOrdinalKey value, JsonSerializerOptions options) =>
            writer.WriteStringValue(value.Data!);

        public override StringSpanOrdinalKey Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) => new StringSpanOrdinalKey(reader.GetString()!);
        public override void Write(Utf8JsonWriter writer, StringSpanOrdinalKey value, JsonSerializerOptions options) => writer.WriteStringValue(value.Data!);
    }

    internal class StringSpanOrdinalKeyCustomConverter : JsonConverter<Dictionary<StringSpanOrdinalKey, (int, string)>>
    {
        public static StringSpanOrdinalKeyCustomConverter Instance { get; } = new StringSpanOrdinalKeyCustomConverter();

        public override Dictionary<StringSpanOrdinalKey, (int, string)> Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            var dictionary = new Dictionary<StringSpanOrdinalKey, (int, string)>();
            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                {
                    return dictionary;
                }

                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    var key = reader.GetString();
                    reader.Read();
                    var value = reader.GetInt32();
                    dictionary.Add(new StringSpanOrdinalKey(key!), (value, key!));
                }
            }
            throw new JsonException("Invalid JSON.");
        }

        public override void Write(Utf8JsonWriter writer, Dictionary<StringSpanOrdinalKey, (int, string)> value, JsonSerializerOptions options) => throw new NotImplementedException();
    }

    /// <summary>
    /// Extension methods for <see cref="StringSpanOrdinalKey"/>.
    /// </summary>
    internal static class StringSpanOrdinalKeyExtensions
    {
        public static unsafe bool TryGetValue<TValue>(this Dictionary<StringSpanOrdinalKey, TValue> map, ReadOnlySpan<char> key, out TValue value)
        {
            fixed (char* ptr = key)
            {
                return map.TryGetValue(new StringSpanOrdinalKey(ptr, key.Length), out value!);
            }
        }

        public static bool TryGetValue<TValue>(this Dictionary<StringSpanOrdinalKey, TValue> map, string key, out TValue value) =>
            map.TryGetValue(new StringSpanOrdinalKey(key), out value!);

        public static unsafe bool TryGetValue<TValue>(this Dictionary<StringSpanOrdinalKeyPair, TValue> map, ReadOnlySpan<char> key1, ReadOnlySpan<char> key2, out TValue value)
        {
            fixed (char* ptr1 = key1)
            fixed (char* ptr2 = key2)
            {
                return map.TryGetValue(new StringSpanOrdinalKeyPair(ptr1, key1.Length, ptr2, key2.Length), out value!);
            }
        }

        public static bool TryGetValue<TValue>(this Dictionary<StringSpanOrdinalKeyPair, TValue> map, string key1, string key2, out TValue value) =>
            map.TryGetValue(new StringSpanOrdinalKeyPair(key1, key2), out value!);
    }
}
