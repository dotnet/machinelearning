// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    internal class SweepableEstimatorConverter : JsonConverter<SweepableEstimator>
    {
        public override SweepableEstimator Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            var jsonObject = JsonValue.Parse(ref reader);
            var estimatorType = jsonObject["estimatorType"].GetValue<EstimatorType>();
            var parameter = jsonObject["parameter"].GetValue<Parameter>();
            var estimator = new SweepableEstimator(estimatorType);
            estimator.Parameter = parameter;

            return estimator;
        }

        public override void Write(Utf8JsonWriter writer, SweepableEstimator value, JsonSerializerOptions options)
        {
            var jObject = JsonObject.Parse("{}");
            jObject["estimatorType"] = JsonValue.Create(value.EstimatorType);
            jObject["parameter"] = JsonValue.Create(value.Parameter);
            jObject.WriteTo(writer, options);
        }
    }
}
