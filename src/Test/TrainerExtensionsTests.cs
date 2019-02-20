// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class TrainerExtensionsTests
    {
        [TestMethod]
        public void TrainerExtensionInstanceTests()
        {
            var context = new MLContext();
            var trainerNames = Enum.GetValues(typeof(TrainerName)).Cast<TrainerName>();
            foreach (var trainerName in trainerNames)
            {
                var extension = TrainerExtensionCatalog.GetTrainerExtension(trainerName);
                var instance = extension.CreateInstance(context, null);
                Assert.IsNotNull(instance);
                var sweepParams = extension.GetHyperparamSweepRanges();
                Assert.IsNotNull(sweepParams);
            }
        }

        [TestMethod]
        public void BuildPipelineNodePropsLightGbm()
        {
            var sweepParams = SweepableParams.BuildLightGbmParams();
            foreach (var sweepParam in sweepParams)
            {
                sweepParam.RawValue = 1;
            }

            var lightGbmBinaryProps = TrainerExtensionUtil.BuildPipelineNodeProps(TrainerName.LightGbmBinary, sweepParams);
            var lightGbmMultiProps = TrainerExtensionUtil.BuildPipelineNodeProps(TrainerName.LightGbmMulti, sweepParams);
            var lightGbmRegressionProps = TrainerExtensionUtil.BuildPipelineNodeProps(TrainerName.LightGbmRegression, sweepParams);

            var expectedJson = @"{
  ""NumBoostRound"": 20,
  ""LearningRate"": 1,
  ""NumLeaves"": 1,
  ""MinDataPerLeaf"": 10,
  ""UseSoftmax"": false,
  ""UseCat"": false,
  ""UseMissing"": false,
  ""MinDataPerGroup"": 50,
  ""MaxCatThreshold"": 16,
  ""CatSmooth"": 10,
  ""CatL2"": 0.5,
  ""Booster"": {
    ""Name"": ""Options.TreeBooster.Arguments"",
    ""Properties"": {
      ""RegLambda"": 0.5,
      ""RegAlpha"": 0.5
    }
  }
}";
            Util.AssertObjectMatchesJson(expectedJson, lightGbmBinaryProps);
            Util.AssertObjectMatchesJson(expectedJson, lightGbmMultiProps);
            Util.AssertObjectMatchesJson(expectedJson, lightGbmRegressionProps);
        }

        [TestMethod]
        public void BuildPipelineNodePropsSdca()
        {
            var sweepParams = SweepableParams.BuildSdcaParams();
            foreach (var sweepParam in sweepParams)
            {
                sweepParam.RawValue = 1;
            }

            var sdcaBinaryProps = TrainerExtensionUtil.BuildPipelineNodeProps(TrainerName.SdcaBinary, sweepParams);
            var expectedJson = @"{
  ""L2Const"": 1E-07,
  ""L1Threshold"": 0.0,
  ""ConvergenceTolerance"": 0.01,
  ""MaxIterations"": 10,
  ""Shuffle"": true,
  ""BiasLearningRate"": 0.01
}";
            Util.AssertObjectMatchesJson(expectedJson, sdcaBinaryProps);
        }

        [TestMethod]
        public void BuildParameterSetLightGbm()
        {
            var props = new Dictionary<string, object>()
            {
                {"NumBoostRound", 1 },
                {"LearningRate", 1 },
                {"Booster", new CustomProperty() {
                    Name = "Options.TreeBooster.Arguments",
                    Properties = new Dictionary<string, object>()
                    {
                        {"RegLambda", 1 },
                        {"RegAlpha", 1 },
                    }
                } },
            };
            var binaryParams = TrainerExtensionUtil.BuildParameterSet(TrainerName.LightGbmBinary, props);
            var multiParams = TrainerExtensionUtil.BuildParameterSet(TrainerName.LightGbmMulti, props);
            var regressionParams = TrainerExtensionUtil.BuildParameterSet(TrainerName.LightGbmRegression, props);

            foreach (var paramSet in new ParameterSet[] { binaryParams, multiParams, regressionParams })
            {
                Assert.AreEqual(4, paramSet.Count);
                Assert.AreEqual("1", paramSet["NumBoostRound"].ValueText);
                Assert.AreEqual("1", paramSet["LearningRate"].ValueText);
                Assert.AreEqual("1", paramSet["RegLambda"].ValueText);
                Assert.AreEqual("1", paramSet["RegAlpha"].ValueText);
            }
        }

        [TestMethod]
        public void BuildParameterSetSdca()
        {
            var props = new Dictionary<string, object>()
            {
                {"LearningRate", 1 },
            };

            var sdcaParams = TrainerExtensionUtil.BuildParameterSet(TrainerName.SdcaBinary, props);

            Assert.AreEqual(1, sdcaParams.Count);
            Assert.AreEqual("1", sdcaParams["LearningRate"].ValueText);
        }
    }
}
