// <copyright file="SchemaBase.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;
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
