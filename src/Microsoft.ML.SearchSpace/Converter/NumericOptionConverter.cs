// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.SearchSpace.Option;

namespace Microsoft.ML.SearchSpace.Converter
{
    internal class NumericOptionConverter : JsonConverter<UniformNumericOption>
    {
        class Schema
        {
            /// <summary>
            /// must be one of "int" | "float" | "double"
            /// </summary>
            [JsonPropertyName("type")]
            public string Type { get; set; }

            [JsonPropertyName("default")]
            public object Default { get; set; }

            [JsonPropertyName("min")]
            public object Min { get; set; }

            [JsonPropertyName("max")]
            public object Max { get; set; }

            [JsonPropertyName("log_base")]
            public bool LogBase { get; set; }
        }

        public override UniformNumericOption Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            var schema = JsonSerializer.Deserialize<Schema>(ref reader, options);

            return schema.Type switch
            {
                "int" => new UniformIntOption(Convert.ToInt32(schema.Min), Convert.ToInt32(schema.Max), schema.LogBase, Convert.ToInt32(schema.Default)),
                "float" => new UniformSingleOption(Convert.ToSingle(schema.Min), Convert.ToSingle(schema.Max), schema.LogBase, Convert.ToSingle(schema.Default)),
                "double" => new UniformDoubleOption(Convert.ToDouble(schema.Min), Convert.ToDouble(schema.Max), schema.LogBase, Convert.ToDouble(schema.Default)),
                _ => throw new ArgumentException($"unknown schema type: {schema.Type}"),
            };
        }

        public override void Write(Utf8JsonWriter writer, UniformNumericOption value, JsonSerializerOptions options)
        {
            var schema = value switch
            {
                UniformIntOption intOption => new Schema
                {
                    Type = "int",
                    Default = intOption.SampleFromFeatureSpace(intOption.Default).AsType<int>(),
                    Min = Convert.ToInt32(intOption.Min),
                    Max = Convert.ToInt32(intOption.Max),
                    LogBase = intOption.LogBase,
                },
                UniformDoubleOption doubleOption => new Schema
                {
                    Type = "double",
                    Default = doubleOption.SampleFromFeatureSpace(doubleOption.Default).AsType<double>(),
                    Min = doubleOption.Min,
                    Max = doubleOption.Max,
                    LogBase = doubleOption.LogBase,
                },
                UniformSingleOption singleOption => new Schema
                {
                    Type = "float",
                    Default = singleOption.SampleFromFeatureSpace(singleOption.Default).AsType<Single>(),
                    Min = Convert.ToSingle(singleOption.Min),
                    Max = Convert.ToSingle(singleOption.Max),
                    LogBase = singleOption.LogBase,
                },
                _ => throw new ArgumentException("unknown type"),
            };

            JsonSerializer.Serialize(writer, schema, options);
        }
    }
}
