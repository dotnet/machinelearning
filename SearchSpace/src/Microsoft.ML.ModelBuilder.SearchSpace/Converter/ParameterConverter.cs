// <copyright file="ParameterConverter.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.ModelBuilder.SearchSpace.Converter
{
    internal class ParameterConverter : JsonConverter
    {
        internal static JsonSerializerSettings settings = new JsonSerializerSettings()
        {
            Formatting = Formatting.Indented,
            Culture = System.Globalization.CultureInfo.InvariantCulture,
            NullValueHandling = NullValueHandling.Ignore,
            Converters = new JsonConverter[]
            {
                new StringEnumConverter(),
            },
        };

        public override bool CanRead => true;

        public override bool CanWrite => true;

        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(Parameter);
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {

            var jtoken = JToken.ReadFrom(reader);
            return new Parameter(jtoken);
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            var param = (Parameter)value;
            var json = JsonConvert.SerializeObject(param.Value, settings);
            writer.WriteRawValue(json);
        }
    }
}
