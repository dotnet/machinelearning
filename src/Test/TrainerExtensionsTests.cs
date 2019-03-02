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
            var columnInfo = new ColumnInformation();
            var trainerNames = Enum.GetValues(typeof(TrainerName)).Cast<TrainerName>();
            foreach (var trainerName in trainerNames)
            {
                var extension = TrainerExtensionCatalog.GetTrainerExtension(trainerName);
                var sweepParams = extension.GetHyperparamSweepRanges();
                Assert.IsNotNull(sweepParams);
                foreach (var sweepParam in sweepParams)
                {
                    sweepParam.RawValue = 1;
                }
                var instance = extension.CreateInstance(context, sweepParams, columnInfo);
                Assert.IsNotNull(instance);
                var pipelineNode = extension.CreatePipelineNode(null, columnInfo);
                Assert.IsNotNull(pipelineNode);
            }
        }

        [TestMethod]
        public void BuildLightGbmPipelineNode()
        {
            var sweepParams = SweepableParams.BuildLightGbmParams();
            foreach (var sweepParam in sweepParams)
            {
                sweepParam.RawValue = 1;
            }

            var pipelineNode = new LightGbmBinaryExtension().CreatePipelineNode(sweepParams, new ColumnInformation());

            var expectedJson = @"{
  ""Name"": ""LightGbmBinary"",
  ""NodeType"": ""Trainer"",
  ""InColumns"": [
    ""Features""
  ],
  ""OutColumns"": [
    ""Score""
  ],
  ""Properties"": {
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
      ""Name"": ""Options.TreeBooster.Options"",
      ""Properties"": {
        ""RegLambda"": 0.5,
        ""RegAlpha"": 0.5
      }
    },
    ""LabelColumn"": ""Label""
  }
}";
            Util.AssertObjectMatchesJson(expectedJson, pipelineNode);
        }

        [TestMethod]
        public void BuildSdcaPipelineNode()
        {
            var sweepParams = SweepableParams.BuildSdcaParams();
            foreach (var sweepParam in sweepParams)
            {
                sweepParam.RawValue = 1;
            }

            var pipelineNode = new SdcaBinaryExtension().CreatePipelineNode(sweepParams, new ColumnInformation());
            var expectedJson = @"{
  ""Name"": ""SdcaBinary"",
  ""NodeType"": ""Trainer"",
  ""InColumns"": [
    ""Features""
  ],
  ""OutColumns"": [
    ""Score""
  ],
  ""Properties"": {
    ""L2Const"": 1E-07,
    ""L1Threshold"": 0.0,
    ""ConvergenceTolerance"": 0.01,
    ""MaxIterations"": 10,
    ""Shuffle"": true,
    ""BiasLearningRate"": 0.01,
    ""LabelColumn"": ""Label""
  }
}";
            Util.AssertObjectMatchesJson(expectedJson, pipelineNode);
        }

        [TestMethod]
        public void BuildPipelineNodeWithCustomColumns()
        {
            var columnInfo = new ColumnInformation()
            {
                LabelColumn = "L",
                WeightColumn = "W"
            };
            var sweepParams = SweepableParams.BuildFastForestParams();
            foreach (var sweepParam in sweepParams)
            {
                sweepParam.RawValue = 1;
            }

            var pipelineNode = new FastForestBinaryExtension().CreatePipelineNode(sweepParams, columnInfo);
            var expectedJson = @"{
  ""Name"": ""FastForestBinary"",
  ""NodeType"": ""Trainer"",
  ""InColumns"": [
    ""Features""
  ],
  ""OutColumns"": [
    ""Score""
  ],
  ""Properties"": {
    ""NumLeaves"": 1,
    ""MinDocumentsInLeafs"": 10,
    ""NumTrees"": 100,
    ""LabelColumn"": ""L"",
    ""WeightColumn"": ""W""
  }
}";
            Util.AssertObjectMatchesJson(expectedJson, pipelineNode);
        }

        [TestMethod]
        public void BuildDefaultAveragedPerceptronPipelineNode()
        {
            var pipelineNode = new AveragedPerceptronBinaryExtension().CreatePipelineNode(null, new ColumnInformation() { LabelColumn = "L" });
            var expectedJson = @"{
  ""Name"": ""AveragedPerceptronBinary"",
  ""NodeType"": ""Trainer"",
  ""InColumns"": [
    ""Features""
  ],
  ""OutColumns"": [
    ""Score""
  ],
  ""Properties"": {
    ""LabelColumn"": ""L"",
    ""NumberOfIterations"": ""10""
  }
}";
            Util.AssertObjectMatchesJson(expectedJson, pipelineNode);
        }

        [TestMethod]
        public void BuildOvaPipelineNode()
        {
            var pipelineNode = new FastForestOvaExtension().CreatePipelineNode(null, new ColumnInformation());
            var expectedJson = @"{
  ""Name"": ""FastForestOva"",
  ""NodeType"": ""Trainer"",
  ""InColumns"": [
    ""Features""
  ],
  ""OutColumns"": [
    ""Score""
  ],
  ""Properties"": {
    ""LabelColumn"": ""Label""
  }
}";
            Util.AssertObjectMatchesJson(expectedJson, pipelineNode);
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

        [TestMethod]
        public void PublicToPrivateTrainerNamesBinaryTest()
        {
            var publicNames = Enum.GetValues(typeof(BinaryClassificationTrainer)).Cast<BinaryClassificationTrainer>();
            var internalNames = TrainerExtensionUtil.GetTrainerNames(publicNames);
            Assert.AreEqual(publicNames.Distinct().Count(), internalNames.Distinct().Count());
        }

        [TestMethod]
        public void PublicToPrivateTrainerNamesMultiTest()
        {
            var publicNames = Enum.GetValues(typeof(MulticlassClassificationTrainer)).Cast<MulticlassClassificationTrainer>();
            var internalNames = TrainerExtensionUtil.GetTrainerNames(publicNames);
            Assert.AreEqual(publicNames.Distinct().Count(), internalNames.Distinct().Count());
        }

        [TestMethod]
        public void PublicToPrivateTrainerNamesRegressionTest()
        {
            var publicNames = Enum.GetValues(typeof(RegressionTrainer)).Cast<RegressionTrainer>();
            var internalNames = TrainerExtensionUtil.GetTrainerNames(publicNames);
            Assert.AreEqual(publicNames.Distinct().Count(), internalNames.Distinct().Count());
        }

        [TestMethod]
        public void PublicToPrivateTrainerNamesNullTest()
        {
            var internalNames = TrainerExtensionUtil.GetTrainerNames(null as IEnumerable<BinaryClassificationTrainer>);
            Assert.AreEqual(null, internalNames);
        }

        [TestMethod]
        public void AllowedTrainersWhitelistNullTest()
        {
            var trainers = RecipeInference.AllowedTrainers(new MLContext(), TaskKind.BinaryClassification, new ColumnInformation(), null);
            Assert.IsTrue(trainers.Any());
        }

        [TestMethod]
        public void AllowedTrainersWhitelistTest()
        {
            var whitelist = new[] { TrainerName.AveragedPerceptronBinary, TrainerName.FastForestBinary };
            var trainers = RecipeInference.AllowedTrainers(new MLContext(), TaskKind.BinaryClassification, new ColumnInformation(), whitelist);
            Assert.AreEqual(whitelist.Count(), trainers.Count());
        }
    }
}
