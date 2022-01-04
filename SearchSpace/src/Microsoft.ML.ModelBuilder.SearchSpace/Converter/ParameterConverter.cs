// <copyright file="ParameterConverter.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Linq;
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
            return objectType == typeof(IParameter);
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {

            var jtoken = JToken.ReadFrom(reader);
            return jtoken.Type switch
            {
                JTokenType.Object => Parameter.CreateNestedParameter(jtoken.ToObject<Dictionary<string, IParameter>>().ToArray()),
                JTokenType.String => Parameter.FromString(jtoken.ToObject<string>()),
                JTokenType.Float => Parameter.FromDouble(jtoken.ToObject<double>()),
                JTokenType.Boolean => Parameter.FromBool(jtoken.ToObject<bool>()),
                JTokenType.Integer => Parameter.FromInt(jtoken.ToObject<int>()),
                JTokenType.Array => Parameter.FromIEnumerable((JArray)jtoken),
                JTokenType.Null => null,
                _ => throw new ArgumentException($"Unsupported jtoken type {jtoken.Type}"),
            };
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            var param = (IParameter)value;
            var s = JsonSerializer.Create(settings);
            s.Serialize(writer, param.Value);
        }
    }
}
