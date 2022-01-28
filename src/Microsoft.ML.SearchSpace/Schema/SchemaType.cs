// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

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
