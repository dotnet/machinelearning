// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class SuggestedPipelineBuilderTests
    {
        private static MLContext _context = new MLContext();

        [TestMethod]
        public void TrainerWantsCaching()
        {
            TestPipelineBuilderCaching(BuildAveragedPerceptronTrainer(),
                new bool?[] { true, false, null },
                new[] { true, false, true });
        }

        [TestMethod]
        public void TrainerDoesntWantCaching()
        {
            TestPipelineBuilderCaching(BuildLightGbmTrainer(),
                new bool?[] { true, false, null },
                new[] { true, false, false });
        }

        [TestMethod]
        public void TrainerNeedsNormalization()
        {
            var pipeline = BuildSuggestedPipeline(BuildAveragedPerceptronTrainer());
            Assert.AreEqual(EstimatorName.Normalizing.ToString(),
                pipeline.Transforms[0].PipelineNode.Name);
        }

        [TestMethod]
        public void TrainerNotNeedNormalization()
        {
            var pipeline = BuildSuggestedPipeline(BuildLightGbmTrainer());
            Assert.AreEqual(0, pipeline.Transforms.Count);
        }

        private static void TestPipelineBuilderCaching(
            SuggestedTrainer trainer,
            bool?[] enableCachingOptions,
            bool[] resultShouldHaveCaching)
        {
            for (var i = 0; i < enableCachingOptions.Length; i++)
            {
                var suggestedPipeline = BuildSuggestedPipeline(trainer,
                    enableCachingOptions[i]);
                Assert.AreEqual(resultShouldHaveCaching[i],
                    suggestedPipeline.ToPipeline().CacheBeforeTrainer);
            }
        }

        private static SuggestedTrainer BuildAveragedPerceptronTrainer()
        {
            return new SuggestedTrainer(_context,
                new AveragedPerceptronBinaryExtension(),
                new ColumnInformation());
        }

        private static SuggestedTrainer BuildLightGbmTrainer()
        {
            return new SuggestedTrainer(_context,
                new LightGbmBinaryExtension(),
                new ColumnInformation());
        }

        private static SuggestedPipeline BuildSuggestedPipeline(SuggestedTrainer trainer,
            bool? enableCaching = null)
        {
            return SuggestedPipelineBuilder.Build(_context,
                    new List<SuggestedTransform>(),
                    new List<SuggestedTransform>(),
                    trainer, enableCaching);
        }
    }
}
