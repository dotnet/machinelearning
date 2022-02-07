// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text.Json.Serialization;

namespace Microsoft.ML.SearchSpace.Schema
{
    internal class UniformDoubleOptionSchemaV0 : SchemaBase
    {
        public override SchemaType SchemaType => SchemaType.UniformDoubleOption;

        public override int Version => 0;

        [JsonPropertyName("min")]
        public double Min { get; set; }

        [JsonPropertyName("max")]
        public double Max { get; set; }

        [JsonPropertyName("log_base")]
        public bool? LogBase { get; set; }
    }
}
