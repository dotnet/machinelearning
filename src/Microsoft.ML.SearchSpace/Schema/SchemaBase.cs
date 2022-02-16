// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text.Json.Serialization;

namespace Microsoft.ML.SearchSpace.Schema
{
    [JsonConverter(typeof(JsonStringEnumConverter))]
    internal enum SchemaType
    {
        UniformDoubleOption = 0,
        IntegerOption = 1,
        ChoiceOption = 2,
        NestOption = 3,
    }

    internal abstract class SchemaBase
    {
        [JsonPropertyName("schema_type")]
        public abstract SchemaType SchemaType { get; }

        [JsonPropertyName("schema_type")]
        public abstract int Version { get; }
    }
}
