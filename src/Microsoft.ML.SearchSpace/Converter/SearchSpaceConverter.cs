// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.SearchSpace.Option;

namespace Microsoft.ML.SearchSpace.Converter
{
    internal class SearchSpaceConverter : JsonConverter<SearchSpace>
    {
        public override SearchSpace Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            var optionKVPairs = JsonSerializer.Deserialize<Dictionary<string, OptionBase>>(ref reader, options);

            return new SearchSpace(optionKVPairs);
        }

        public override void Write(Utf8JsonWriter writer, SearchSpace value, JsonSerializerOptions options)
        {
            JsonSerializer.Serialize<IDictionary<string, OptionBase>>(value, options);
        }
    }
}
