using System.Collections.Generic;
using Microsoft.ML.LightGBM;
using Xunit;

namespace Microsoft.ML.RunTests
{
    public class SmokeTests
    {
        [Fact]
        public void LightGbmArguments_TreeBoosterArgumentsTest()
        {
            var sut = new LightGbmArguments.TreeBooster(new LightGbmArguments.TreeBooster.Arguments());

            var dictionary = new Dictionary<string, object>();

            sut.UpdateParameters(dictionary);

            Assert.Equal(11, dictionary.Count);
        }

        [Fact]
        public void LightGbmArguments_GossBoosterArgumentsTest()
        {
            var sut = new LightGbmArguments.GossBooster(new LightGbmArguments.GossBooster.Arguments());

            var dictionary = new Dictionary<string, object>();

            sut.UpdateParameters(dictionary);

            Assert.Equal(13, dictionary.Count);
        }

        [Fact]
        public void LightGbmArguments_DartBoosterArgumentsTest()
        {
            var sut = new LightGbmArguments.DartBooster(new LightGbmArguments.DartBooster.Arguments());

            var dictionary = new Dictionary<string, object>();

            sut.UpdateParameters(dictionary);

            Assert.Equal(16, dictionary.Count);
        }
    }
}