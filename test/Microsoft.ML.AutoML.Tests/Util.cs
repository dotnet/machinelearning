// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Xunit;

namespace Microsoft.ML.AutoML.Test
{
    internal static class Util
    {
        public static void AssertObjectMatchesJson<T>(string expectedJson, T obj)
        {
            var actualJson = JsonConvert.SerializeObject(obj,
                Formatting.Indented, new JsonConverter[] { new StringEnumConverter() });
            Assert.Equal(expectedJson, actualJson);
        }

        public static ValueGetter<VBuffer<ReadOnlyMemory<char>>> GetKeyValueGetter(IEnumerable<string> colNames)
        {
            return (ref VBuffer<ReadOnlyMemory<char>> dst) =>
            {
                var editor = VBufferEditor.Create(ref dst, colNames.Count());
                for (int i = 0; i < colNames.Count(); i++)
                {
                    editor.Values[i] = colNames.ElementAt(i).AsMemory();
                }
                dst = editor.Commit();
            };
        }
    }
}
