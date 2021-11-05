// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using BenchmarkDotNet.Attributes;
using Microsoft.ML.Data;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.PerformanceTests
{
    [Config(typeof(TrainConfig))]
    public class RankingTrain : BenchmarkBase
    {
        private string _mslrWeb10kValidate;
        private string _mslrWeb10kTrain;

        [GlobalSetup]
        public void SetupTrainingSpeedTests()
        {
            _mslrWeb10kValidate = GetBenchmarkDataPathAndEnsureData(TestDatasets.MSLRWeb.validFilename, TestDatasets.MSLRWeb.path);
            _mslrWeb10kTrain = GetBenchmarkDataPathAndEnsureData(TestDatasets.MSLRWeb.trainFilename, TestDatasets.MSLRWeb.path);

            if (!File.Exists(_mslrWeb10kValidate))
                throw new FileNotFoundException(string.Format(Errors.DatasetNotFound, _mslrWeb10kValidate));

            if (!File.Exists(_mslrWeb10kTrain))
                throw new FileNotFoundException(string.Format(Errors.DatasetNotFound, _mslrWeb10kTrain));
        }

        [Benchmark]
        public void TrainTest_Ranking_MSLRWeb10K_RawNumericFeatures_FastTreeRanking()
        {
            string cmd = @"TrainTest test=" + _mslrWeb10kValidate +
                " eval=RankingEvaluator{t=10}" +
                " data=" + _mslrWeb10kTrain +
                " loader=TextLoader{col=Label:R4:0 col=GroupId:TX:1 col=Features:R4:2-138}" +
                " xf=HashTransform{col=GroupId} xf=NAHandleTransform{col=Features}" +
                " tr=FastTreeRanking{}";

            var environment = EnvironmentFactory.CreateRankingEnvironment<RankingEvaluator, TextLoader, HashingTransformer, FastTreeRankingTrainer, FastTreeRankingModelParameters>();
            cmd.ExecuteMamlCommand(environment);
        }

        [Benchmark]
        public void TrainTest_Ranking_MSLRWeb10K_RawNumericFeatures_LightGBMRanking()
        {
            string cmd = @"TrainTest test=" + _mslrWeb10kValidate +
                " eval=RankingEvaluator{t=10}" +
                " data=" + _mslrWeb10kTrain +
                " loader=TextLoader{col=Label:R4:0 col=GroupId:TX:1 col=Features:R4:2-138}" +
                " xf=HashTransform{col=GroupId}" +
                " xf=NAHandleTransform{col=Features}" +
                " tr=LightGBMRanking{}";

            var environment = EnvironmentFactory.CreateRankingEnvironment<RankingEvaluator, TextLoader, HashingTransformer, LightGbmMulticlassTrainer, OneVersusAllModelParameters>();
            cmd.ExecuteMamlCommand(environment);
        }
    }

    public class RankingTest : BenchmarkBase
    {
        private string _mslrWeb10kValidate;
        private string _mslrWeb10kTrain;
        private string _mslrWeb10kTest;
        private string _modelPathMslr;

        [GlobalSetup]
        public void SetupScoringSpeedTests()
        {
            _mslrWeb10kTest = GetBenchmarkDataPathAndEnsureData(TestDatasets.MSLRWeb.testFilename, TestDatasets.MSLRWeb.path);
            _mslrWeb10kValidate = GetBenchmarkDataPathAndEnsureData(TestDatasets.MSLRWeb.validFilename, TestDatasets.MSLRWeb.path);
            _mslrWeb10kTrain = GetBenchmarkDataPathAndEnsureData(TestDatasets.MSLRWeb.trainFilename, TestDatasets.MSLRWeb.path);

            if (!File.Exists(_mslrWeb10kTest))
                throw new FileNotFoundException(string.Format(Errors.DatasetNotFound, _mslrWeb10kTest));

            if (!File.Exists(_mslrWeb10kValidate))
                throw new FileNotFoundException(string.Format(Errors.DatasetNotFound, _mslrWeb10kValidate));

            if (!File.Exists(_mslrWeb10kTrain))
                throw new FileNotFoundException(string.Format(Errors.DatasetNotFound, _mslrWeb10kTrain));

            _modelPathMslr = Path.Combine(Path.GetDirectoryName(typeof(RankingTest).Assembly.Location), "FastTreeRankingModel.zip");

            string cmd = @"TrainTest test=" + _mslrWeb10kValidate +
                " eval=RankingEvaluator{t=10}" +
                " data=" + _mslrWeb10kTrain +
                " loader=TextLoader{col=Label:R4:0 col=GroupId:TX:1 col=Features:R4:2-138}" +
                " xf=HashTransform{col=GroupId}" +
                " xf=NAHandleTransform{col=Features}" +
                " tr=FastTreeRanking{}" +
                " out={" + _modelPathMslr + "}";

            var environment = EnvironmentFactory.CreateRankingEnvironment<RankingEvaluator, TextLoader, HashingTransformer, FastTreeRankingTrainer, FastTreeRankingModelParameters>();
            cmd.ExecuteMamlCommand(environment);
        }

        [Benchmark]
        public void Test_Ranking_MSLRWeb10K_RawNumericFeatures_FastTreeRanking()
        {
            // This benchmark is profiling bulk scoring speed and not training speed. 
            string cmd = @"Test data=" + _mslrWeb10kTest + " in=" + _modelPathMslr;

            var environment = EnvironmentFactory.CreateRankingEnvironment<RankingEvaluator, TextLoader, HashingTransformer, FastTreeRankingTrainer, FastTreeRankingModelParameters>();
            cmd.ExecuteMamlCommand(environment);
        }
    }
}
