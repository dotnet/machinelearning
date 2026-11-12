// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Microsoft.ML.AutoML.Test
{
    internal class DoubleToDecimalConverter : JsonConverter<double>
    {
        public override double Read(ref Utf8JsonReader reader, Type type, JsonSerializerOptions options)
        {
            return Convert.ToDouble(reader.GetDecimal());
        }

        public override void Write(Utf8JsonWriter writer, double value, JsonSerializerOptions options)
        {
            // Preserve the snapshot's 15 significant digits across runtime versions.
            var normalized = decimal.Parse(value.ToString("G15", CultureInfo.InvariantCulture), NumberStyles.Float, CultureInfo.InvariantCulture);
            writer.WriteNumberValue(normalized);
        }
    }
}
