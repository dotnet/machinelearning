// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.AutoML.Test
{
    [TestClass]
    public class SuggestedPipelineBuilderTests
    {
        private static MLContext _context = new MLContext();

        [TestMethod]
        public void TrainerWantsCaching()
        {
            TestPipelineBuilderCaching(BuildAveragedPerceptronTrainer(),
                new CacheBeforeTrainer[] { CacheBeforeTrainer.On, CacheBeforeTrainer.Off, CacheBeforeTrainer.Auto },
                new[] { true, false, true });
        }

        [TestMethod]
        public void TrainerDoesntWantCaching()
        {
            TestPipelineBuilderCaching(BuildLightGbmTrainer(),
                new CacheBeforeTrainer[] { CacheBeforeTrainer.On, CacheBeforeTrainer.Off, CacheBeforeTrainer.Auto },
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
            CacheBeforeTrainer[] cacheBeforeTrainerSettings,
            bool[] resultShouldHaveCaching)
        {
            for (var i = 0; i < cacheBeforeTrainerSettings.Length; i++)
            {
                var suggestedPipeline = BuildSuggestedPipeline(trainer,
                    cacheBeforeTrainerSettings[i]);
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
            CacheBeforeTrainer cacheBeforeTrainer = CacheBeforeTrainer.Auto)
        {
            return SuggestedPipelineBuilder.Build(_context,
                    new List<SuggestedTransform>(),
                    new List<SuggestedTransform>(),
                    trainer, cacheBeforeTrainer);
        }
    }
}
