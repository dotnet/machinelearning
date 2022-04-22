// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    internal class SweepableEstimatorPipelineConverter : JsonConverter<SweepableEstimatorPipeline>
    {
        public override SweepableEstimatorPipeline Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            var jNode = JsonNode.Parse(ref reader);
            var parameter = jNode["parameter"].GetValue<Parameter>();
            var estimators = jNode["estimators"].GetValue<SweepableEstimator[]>();
            var pipeline = new SweepableEstimatorPipeline(estimators, parameter);

            return pipeline;
        }

        public override void Write(Utf8JsonWriter writer, SweepableEstimatorPipeline value, JsonSerializerOptions options)
        {
            var parameter = value.Parameter;
            var estimators = value.Estimators;
            var jNode = JsonNode.Parse("{}");
            jNode["parameter"] = JsonValue.Create(parameter);
            jNode["estimators"] = JsonValue.Create(estimators);

            jNode.WriteTo(writer, options);
        }
    }
}
