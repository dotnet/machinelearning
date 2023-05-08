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
    internal class OptionConverter : JsonConverter<OptionBase>
    {
        public override OptionBase Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            try
            {
                return JsonSerializer.Deserialize<SearchSpace>(ref reader, options);
            }
            catch (Exception)
            {
                // try choice option
            }

            try
            {
                return JsonSerializer.Deserialize<ChoiceOption>(ref reader, options);
            }
            catch (Exception)
            {
                // try numeric option
            }

            try
            {
                return JsonSerializer.Deserialize<UniformNumericOption>(ref reader, options);
            }
            catch (Exception)
            {
                throw new ArgumentException("unknown option type");
            }
        }

        public override void Write(Utf8JsonWriter writer, OptionBase value, JsonSerializerOptions options)
        {
            if (value is SearchSpace ss)
            {
                JsonSerializer.Serialize(writer, ss, options);
            }
            else if (value is ChoiceOption choiceOption)
            {
                JsonSerializer.Serialize(writer, choiceOption, options);
            }
            else if (value is UniformNumericOption uniformNumericOption)
            {
                JsonSerializer.Serialize(writer, uniformNumericOption, options);
            }
            else
            {
                throw new ArgumentException("unknown option type");
            }
        }
    }
}
