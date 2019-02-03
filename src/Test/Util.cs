// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace Microsoft.ML.Auto.Test
{
    internal static class Util
    {
        public static void AssertObjectMatchesJson<T>(string expectedJson, T obj)
        {
            var actualJson = JsonConvert.SerializeObject(obj, 
                Formatting.Indented, new JsonConverter[] { new StringEnumConverter() });
            Assert.AreEqual(expectedJson, actualJson);
        }
    }
}
