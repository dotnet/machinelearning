// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Newtonsoft.Json;

namespace Microsoft.ML.ModelBuilder.SearchSpace.Schema
{
    public abstract class SchemaBase
    {
        [JsonProperty(PropertyName = "schema_type", Required = Required.Always)]
        public abstract SchemaType SchemaType { get; }

        [JsonProperty(PropertyName = "version", Required = Required.Always)]
        public abstract int Version { get; }
    }
}
