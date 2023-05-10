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
    internal class ChoiceOptionConverter : JsonConverter<ChoiceOption>
    {
        class Schema
        {
            /// <summary>
            /// must be one of "int" | "float" | "double"
            /// </summary>
            [JsonPropertyName("default")]
            public object Default { get; set; }

            [JsonPropertyName("choices")]
            public object[] Choices { get; set; }
        }

        public override ChoiceOption Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            var schema = JsonSerializer.Deserialize<Schema>(ref reader, options);

            return new ChoiceOption(schema.Choices, schema.Default);
        }

        public override void Write(Utf8JsonWriter writer, ChoiceOption value, JsonSerializerOptions options)
        {
            var schema = new Schema
            {
                Choices = value.Choices,
                Default = value.SampleFromFeatureSpace(value.Default),
            };

            JsonSerializer.Serialize(writer, schema, options);
        }
    }
}
