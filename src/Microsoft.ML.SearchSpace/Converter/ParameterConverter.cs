// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Microsoft.ML.SearchSpace.Converter
{
    internal class ParameterConverter<TParameter> : JsonConverter<TParameter>
        where TParameter : IParameter
    {
        public override TParameter Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            switch (reader.TokenType)
            {
                case JsonTokenType.StartObject:
                    var array = JsonSerializer.Deserialize<Dictionary<string, Parameter>>(ref reader, options).ToArray();
                    return (TParameter)Parameter.CreateNestedParameter(array);
                case JsonTokenType.String:
                    return (TParameter)Parameter.FromString(JsonSerializer.Deserialize<string>(ref reader, options));
                case JsonTokenType.Number:
                    if (reader.TryGetInt64(out var _long))
                    {
                        return (TParameter)Parameter.FromLong(_long);
                    }
                    else if (reader.TryGetInt32(out var _int))
                    {
                        return (TParameter)Parameter.FromInt(_int);
                    }

                    return (TParameter)Parameter.FromDouble(JsonSerializer.Deserialize<double>(ref reader, options));
                case JsonTokenType.True:
                    return (TParameter)Parameter.FromBool(true);
                case JsonTokenType.False:
                    return (TParameter)Parameter.FromBool(false);
                case JsonTokenType.Null:
                    return default(TParameter);
                case JsonTokenType.StartArray:
                    var list = new List<object>();
                    while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
                    {
                        list.Add(this.Read(ref reader, null, options));
                    }

                    return (TParameter)Parameter.FromIEnumerable(list);
                default:
                    throw new ArgumentException($"Unsupported reader type {reader.TokenType}");
            }
        }

        public override void Write(Utf8JsonWriter writer, TParameter value, JsonSerializerOptions options)
        {
            JsonSerializer.Serialize(writer, value.Value, options);
        }
    }
}
