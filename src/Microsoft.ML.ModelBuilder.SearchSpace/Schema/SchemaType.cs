// <copyright file="SchemaType.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace Microsoft.ML.ModelBuilder.SearchSpace.Schema
{
    [JsonConverter(typeof(StringEnumConverter))]
    public enum SchemaType
    {
        UniformDoubleOption = 0,
        IntegerOption = 1,
        ChoiceOption = 2,
        NestOption = 3,
    }
}
