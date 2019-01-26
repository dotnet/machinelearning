using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json;

namespace Microsoft.ML.Auto.Test
{
    internal static class Util
    {
        public static void AssertObjectMatchesJson<T>(string expectedJson, T obj)
        {
            var actualJson = JsonConvert.SerializeObject(obj);
            var expectedObj = JsonConvert.DeserializeObject<T>(expectedJson);
            expectedJson = JsonConvert.SerializeObject(expectedObj);
            Assert.AreEqual(expectedJson, actualJson);
        }
    }
}
