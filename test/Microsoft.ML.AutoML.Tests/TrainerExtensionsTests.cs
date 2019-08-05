// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace Microsoft.ML.AutoML.Test
{
    
    public class TrainerExtensionsTests
    {
        [Fact]
        public void TrainerExtensionInstanceTests()
        {
            var context = new MLContext();
            var columnInfo = new ColumnInformation();
            var trainerNames = Enum.GetValues(typeof(TrainerName)).Cast<TrainerName>()
                .Except(new[] { TrainerName.Ova });
            foreach (var trainerName in trainerNames)
            {
                var extension = TrainerExtensionCatalog.GetTrainerExtension(trainerName);
                var sweepParams = extension.GetHyperparamSweepRanges();
                Assert.NotNull(sweepParams);
                foreach (var sweepParam in sweepParams)
                {
                    sweepParam.RawValue = 1;
                }
                var instance = extension.CreateInstance(context, sweepParams, columnInfo);
                Assert.NotNull(instance);
                var pipelineNode = extension.CreatePipelineNode(null, columnInfo);
                Assert.NotNull(pipelineNode);
            }
        }

        [Fact]
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
    ""NumberOfIterations"": 20,
    ""LearningRate"": 1,
    ""NumberOfLeaves"": 1,
    ""MinimumExampleCountPerLeaf"": 10,
    ""UseCategoricalSplit"": false,
    ""HandleMissingValue"": false,
    ""MinimumExampleCountPerGroup"": 50,
    ""MaximumCategoricalSplitPointCount"": 16,
    ""CategoricalSmoothing"": 10,
    ""L2CategoricalRegularization"": 0.5,
    ""Booster"": {
      ""Name"": ""GradientBooster.Options"",
      ""Properties"": {
        ""L2Regularization"": 0.5,
        ""L1Regularization"": 0.5
      }
    },
    ""LabelColumnName"": ""Label""
  }
}";
            Util.AssertObjectMatchesJson(expectedJson, pipelineNode);
        }

        [Fact]
        public void BuildSdcaPipelineNode()
        {
            var sweepParams = SweepableParams.BuildSdcaParams();
            foreach (var sweepParam in sweepParams)
            {
                sweepParam.RawValue = 1;
            }

            var pipelineNode = new SdcaLogisticRegressionBinaryExtension().CreatePipelineNode(sweepParams, new ColumnInformation());
            var expectedJson = @"{
  ""Name"": ""SdcaLogisticRegressionBinary"",
  ""NodeType"": ""Trainer"",
  ""InColumns"": [
    ""Features""
  ],
  ""OutColumns"": [
    ""Score""
  ],
  ""Properties"": {
    ""L2Regularization"": 1E-07,
    ""L1Regularization"": 0.0,
    ""ConvergenceTolerance"": 0.01,
    ""MaximumNumberOfIterations"": 10,
    ""Shuffle"": true,
    ""BiasLearningRate"": 0.01,
    ""LabelColumnName"": ""Label""
  }
}";
            Util.AssertObjectMatchesJson(expectedJson, pipelineNode);
        }

        [Fact]
        public void BuildLightGbmPipelineNodeDefaultParams()
        {
            var pipelineNode = new LightGbmBinaryExtension().CreatePipelineNode(
                new List<SweepableParam>(), 
                new ColumnInformation());
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
    ""LabelColumnName"": ""Label""
  }
}";
            Util.AssertObjectMatchesJson(expectedJson, pipelineNode);
        }

        [Fact]
        public void BuildPipelineNodeWithCustomColumns()
        {
            var columnInfo = new ColumnInformation()
            {
                LabelColumnName = "L",
                ExampleWeightColumnName = "W"
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
    ""NumberOfLeaves"": 1,
    ""MinimumExampleCountPerLeaf"": 10,
    ""NumberOfTrees"": 100,
    ""LabelColumnName"": ""L"",
    ""ExampleWeightColumnName"": ""W""
  }
}";
            Util.AssertObjectMatchesJson(expectedJson, pipelineNode);
        }

        [Fact]
        public void BuildDefaultAveragedPerceptronPipelineNode()
        {
            var pipelineNode = new AveragedPerceptronBinaryExtension().CreatePipelineNode(null, new ColumnInformation() { LabelColumnName = "L" });
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
    ""LabelColumnName"": ""L"",
    ""NumberOfIterations"": 10
  }
}";
            Util.AssertObjectMatchesJson(expectedJson, pipelineNode);
        }

        [Fact]
        public void BuildOvaPipelineNode()
        {
            var pipelineNode = new FastForestOvaExtension().CreatePipelineNode(null, new ColumnInformation());
            var expectedJson = @"{
  ""Name"": ""Ova"",
  ""NodeType"": ""Trainer"",
  ""InColumns"": null,
  ""OutColumns"": null,
  ""Properties"": {
    ""LabelColumnName"": ""Label"",
    ""BinaryTrainer"": {
      ""Name"": ""FastForestBinary"",
      ""NodeType"": ""Trainer"",
      ""InColumns"": [
        ""Features""
      ],
      ""OutColumns"": [
        ""Score""
      ],
      ""Properties"": {
        ""LabelColumnName"": ""Label""
      }
    }
  }
}";
            Util.AssertObjectMatchesJson(expectedJson, pipelineNode);
        }

        [Fact]
        public void BuildParameterSetLightGbm()
        {
            var props = new Dictionary<string, object>()
            {
                {"NumberOfIterations", 1 },
                {"LearningRate", 1 },
                {"Booster", new CustomProperty() {
                    Name = "GradientBooster.Options",
                    Properties = new Dictionary<string, object>()
                    {
                        {"L2Regularization", 1 },
                        {"L1Regularization", 1 },
                    }
                } },
            };
            var binaryParams = TrainerExtensionUtil.BuildParameterSet(TrainerName.LightGbmBinary, props);
            var multiParams = TrainerExtensionUtil.BuildParameterSet(TrainerName.LightGbmMulti, props);
            var regressionParams = TrainerExtensionUtil.BuildParameterSet(TrainerName.LightGbmRegression, props);

            foreach (var paramSet in new ParameterSet[] { binaryParams, multiParams, regressionParams })
            {
                Assert.Equal(4, paramSet.Count);
                Assert.Equal("1", paramSet["NumberOfIterations"].ValueText);
                Assert.Equal("1", paramSet["LearningRate"].ValueText);
                Assert.Equal("1", paramSet["L2Regularization"].ValueText);
                Assert.Equal("1", paramSet["L1Regularization"].ValueText);
            }
        }

        [Fact]
        public void BuildParameterSetSdca()
        {
            var props = new Dictionary<string, object>()
            {
                {"LearningRate", 1 },
            };

            var sdcaParams = TrainerExtensionUtil.BuildParameterSet(TrainerName.SdcaLogisticRegressionBinary, props);

            Assert.Equal(1, sdcaParams.Count);
            Assert.Equal("1", sdcaParams["LearningRate"].ValueText);
        }

        [Fact]
        public void PublicToPrivateTrainerNamesBinaryTest()
        {
            var publicNames = Enum.GetValues(typeof(BinaryClassificationTrainer)).Cast<BinaryClassificationTrainer>();
            var internalNames = TrainerExtensionUtil.GetTrainerNames(publicNames);
            Assert.Equal(publicNames.Distinct().Count(), internalNames.Distinct().Count());
        }

        [Fact]
        public void PublicToPrivateTrainerNamesMultiTest()
        {
            var publicNames = Enum.GetValues(typeof(MulticlassClassificationTrainer)).Cast<MulticlassClassificationTrainer>();
            var internalNames = TrainerExtensionUtil.GetTrainerNames(publicNames);
            Assert.Equal(publicNames.Distinct().Count(), internalNames.Distinct().Count());
        }

        [Fact]
        public void PublicToPrivateTrainerNamesRegressionTest()
        {
            var publicNames = Enum.GetValues(typeof(RegressionTrainer)).Cast<RegressionTrainer>();
            var internalNames = TrainerExtensionUtil.GetTrainerNames(publicNames);
            Assert.Equal(publicNames.Distinct().Count(), internalNames.Distinct().Count());
        }

        [Fact]
        public void PublicToPrivateTrainerNamesNullTest()
        {
            var internalNames = TrainerExtensionUtil.GetTrainerNames(null as IEnumerable<BinaryClassificationTrainer>);
            Assert.Null(internalNames);
        }

        [Fact]
        public void AllowedTrainersWhitelistNullTest()
        {
            var trainers = RecipeInference.AllowedTrainers(new MLContext(), TaskKind.BinaryClassification, new ColumnInformation(), null);
            Assert.True(trainers.Any());
        }

        [Fact]
        public void AllowedTrainersWhitelistTest()
        {
            var whitelist = new[] { TrainerName.AveragedPerceptronBinary, TrainerName.FastForestBinary };
            var trainers = RecipeInference.AllowedTrainers(new MLContext(), TaskKind.BinaryClassification, new ColumnInformation(), whitelist);
            Assert.Equal(whitelist.Count(), trainers.Count());
        }
    }
}
