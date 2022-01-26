// <copyright file="UniformDoubleOptionSchemaV0.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using Newtonsoft.Json;

namespace Microsoft.ML.ModelBuilder.SearchSpace.Schema
{
    public class UniformDoubleOptionSchemaV0 : SchemaBase
    {
        public override SchemaType SchemaType => SchemaType.UniformDoubleOption;

        public override int Version => 0;

        [JsonProperty(PropertyName = "min", Required = Required.Always)]
        public double Min { get; set; }

        [JsonProperty(PropertyName = "max", Required = Required.Always)]
        public double Max { get; set; }

        [JsonProperty(PropertyName = "log_base", Required = Required.AllowNull)]
        public bool? LogBase { get; set; }
    }
}
