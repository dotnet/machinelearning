// <copyright file="ParameterConverter.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.ModelBuilder.SearchSpace.Converter
{
    internal class ParameterConverter : JsonConverter
    {
        public override bool CanRead => false;

        public override bool CanWrite => true;

        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(Parameter);
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            throw new NotImplementedException();
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            var param = (Parameter)value;
            if (param.Value != null)
            {
                serializer.Serialize(writer, param.Value);
            }
            else
            {
                var jobject = new JObject();
                foreach (var kv in param)
                {
                    jobject[kv.Key] = JToken.FromObject(kv.Value);
                }

                serializer.Serialize(writer, jobject);
            }
        }
    }
}
