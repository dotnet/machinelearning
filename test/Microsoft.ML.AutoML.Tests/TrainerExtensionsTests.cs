// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Newtonsoft.Json;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    
    public class TrainerExtensionsTests : BaseTestClass
    {
        public TrainerExtensionsTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TrainerExtensionInstanceTests()
        {
            var context = new MLContext(1);
            var columnInfo = new ColumnInformation();
            var trainerNames = Enum.GetValues(typeof(TrainerName)).Cast<TrainerName>()
                .Except(new[] { TrainerName.Ova });
            foreach (var trainerName in trainerNames)
            {
                var extension = TrainerExtensionCatalog.GetTrainerExtension(trainerName);

                IEnumerable<SweepableParam> sweepParams = null;
                if (trainerName != TrainerName.ImageClassification)
                {
                    sweepParams = extension.GetHyperparamSweepRanges();
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
        }

        [TensorFlowFact]
        public void TrainerExtensionTensorFlowInstanceTests()
        {
            var context = new MLContext(1);
            var columnInfo = new ColumnInformation();
            var extension = TrainerExtensionCatalog.GetTrainerExtension(TrainerName.ImageClassification);
            var instance = extension.CreateInstance(context, null, columnInfo);
            Assert.NotNull(instance);
            var pipelineNode = extension.CreatePipelineNode(null, columnInfo);
            Assert.NotNull(pipelineNode);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void BuildMatrixFactorizationPipelineNode()
        {
            var sweepParams = SweepableParams.BuildMatrixFactorizationParams();
            foreach (var sweepParam in sweepParams)
            {
                sweepParam.RawValue = 1;
            }

            var pipelineNode = new MatrixFactorizationExtension().CreatePipelineNode(sweepParams, new ColumnInformation());
            Approvals.Verify(JsonConvert.SerializeObject(pipelineNode, Formatting.Indented));
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void BuildLightGbmPipelineNode()
        {
            var sweepParams = SweepableParams.BuildLightGbmParams();
            foreach (var sweepParam in sweepParams)
            {
                sweepParam.RawValue = 1;
            }

            var pipelineNode = new LightGbmBinaryExtension().CreatePipelineNode(sweepParams, new ColumnInformation());
            Approvals.Verify(JsonConvert.SerializeObject(pipelineNode, Formatting.Indented));
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void BuildSdcaPipelineNode()
        {
            var sweepParams = SweepableParams.BuildSdcaParams();
            foreach (var sweepParam in sweepParams)
            {
                sweepParam.RawValue = 1;
            }

            var pipelineNode = new SdcaLogisticRegressionBinaryExtension().CreatePipelineNode(sweepParams, new ColumnInformation());
            Approvals.Verify(JsonConvert.SerializeObject(pipelineNode, Formatting.Indented));
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void BuildLightGbmPipelineNodeDefaultParams()
        {
            var pipelineNode = new LightGbmBinaryExtension().CreatePipelineNode(
                new List<SweepableParam>(), 
                new ColumnInformation());
            Approvals.Verify(JsonConvert.SerializeObject(pipelineNode, Formatting.Indented));
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
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
            Approvals.Verify(JsonConvert.SerializeObject(pipelineNode, Formatting.Indented));
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void BuildDefaultAveragedPerceptronPipelineNode()
        {
            var pipelineNode = new AveragedPerceptronBinaryExtension().CreatePipelineNode(null, new ColumnInformation() { LabelColumnName = "ticky\"label\"" });
            Approvals.Verify(JsonConvert.SerializeObject(pipelineNode, Formatting.Indented));
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void BuildOvaPipelineNode()
        {
            var pipelineNode = new FastForestOvaExtension().CreatePipelineNode(null, new ColumnInformation());
            Approvals.Verify(JsonConvert.SerializeObject(pipelineNode, Formatting.Indented));
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [UseApprovalSubdirectory("ApprovalTests")]
        public void BuildFastTreeRankingPipelineNode()
        {
            var columnInfo = new ColumnInformation()
            {
                LabelColumnName = "L",
                GroupIdColumnName = "GId"
            };
            var pipelineNode = new FastTreeRankingExtension().CreatePipelineNode(null, columnInfo);
            Approvals.Verify(JsonConvert.SerializeObject(pipelineNode, Formatting.Indented));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
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
            var rankingParams = TrainerExtensionUtil.BuildParameterSet(TrainerName.LightGbmRanking, props);

            foreach (var paramSet in new ParameterSet[] { binaryParams, multiParams, regressionParams, rankingParams })
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
        public void PublicToPrivateTrainerNamesRecommendationTest()
        {
            var publicNames = Enum.GetValues(typeof(RecommendationTrainer)).Cast<RecommendationTrainer>();
            var internalNames = TrainerExtensionUtil.GetTrainerNames(publicNames);
            Assert.Equal(publicNames.Distinct().Count(), internalNames.Distinct().Count());
        }

        [Fact]
        public void PublicToPrivateTrainerNamesRankingTest()
        {
            var publicNames = Enum.GetValues(typeof(RankingTrainer)).Cast<RankingTrainer>();
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
        public void AllowedTrainersAllowListNullTest()
        {
            var trainers = RecipeInference.AllowedTrainers(new MLContext(1), TaskKind.BinaryClassification, new ColumnInformation(), null);
            Assert.True(trainers.Any());
        }

        [Fact]
        public void AllowedTrainersAllowListTest()
        {
            var allowList = new[] { TrainerName.AveragedPerceptronBinary, TrainerName.FastForestBinary };
            var trainers = RecipeInference.AllowedTrainers(new MLContext(1), TaskKind.BinaryClassification, new ColumnInformation(), allowList);
            Assert.Equal(allowList.Count(), trainers.Count());
        }
    }
}
