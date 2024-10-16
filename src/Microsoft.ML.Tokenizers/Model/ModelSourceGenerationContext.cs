// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Microsoft.ML.Tokenizers;

[JsonSerializable(typeof(Dictionary<StringSpanOrdinalKey, int>))]
[JsonSerializable(typeof(Vocabulary))]
internal partial class ModelSourceGenerationContext : JsonSerializerContext
{
}
