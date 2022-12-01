// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
#if false
using System.Text.RegularExpressions;

using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Utilities;
//using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;

using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;
#endif

namespace Microsoft.ML.Trainers.XGBoost
{
    public static class Utils
    {
        public class DictionaryStringObjectConverter : JsonConverter<Dictionary<string, object>>
        {
            public override void Write(Utf8JsonWriter writer, Dictionary<string, object> value, JsonSerializerOptions options)
            {
                JsonSerializer.Serialize(writer, value, options);
            }

#nullable enable
            public override Dictionary<string, object>? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
            {
                if (reader.TokenType != JsonTokenType.StartObject)
                {
                    throw new JsonException($"JsonTokenType is not StartObject. Token type is {reader.TokenType}");
                }
                var dict = new Dictionary<string, object>();
                while (reader.Read())
                {
                    if (reader.TokenType == JsonTokenType.EndObject)
                    {
                        return dict;
                    }

                    if (reader.TokenType != JsonTokenType.PropertyName)
                    {
                        throw new JsonException("JsonTokenType is not PropertyName");
                    }

                    var propertyName = reader.GetString();

                    if (string.IsNullOrWhiteSpace(propertyName))
                    {
                        throw new JsonException("Property name is null or empty");
                    }
                    reader.Read();
                    dict.Add(propertyName, ExtractValue(ref reader, options));
                }

                return dict;
            }
#nullable disable

            private object ExtractValue(ref Utf8JsonReader reader, JsonSerializerOptions options)
            {
                switch (reader.TokenType)
                {
                    case JsonTokenType.String:
                        if (reader.TryGetDateTime(out DateTime dateTime))
                        {
                            return dateTime;
                        }
                        return reader.GetString();
                    case JsonTokenType.False:
                        return false;
                    case JsonTokenType.True:
                        return true;
                    case JsonTokenType.Null:
                        return null;
                    case JsonTokenType.Number:
                        if (reader.TryGetInt64(out long int64))
                        {
                            return int64;
                        }
                        return reader.GetDecimal();
                    case JsonTokenType.StartObject:
                        return Read(ref reader, null, options);
                    case JsonTokenType.StartArray:
                        var list = new List<object>();
                        while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
                        {
                            list.Add(ExtractValue(ref reader, options));
                        }
                        return list;
                    default:
                        throw new JsonException($"Unexpected token type {reader.TokenType}");
                }
            }
        }

        public static Dictionary<string, object> ParseBoosterConfig(string boosterConfig)
        {
            var options = new JsonSerializerOptions
            {
                Converters = { new Utils.DictionaryStringObjectConverter() }
            };
            return JsonSerializer.Deserialize<Dictionary<string, object>>(boosterConfig, options);
        }
    }
}
