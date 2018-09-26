// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using System.IO;

namespace Microsoft.ML.Benchmarks
{
    [Config(typeof(TrainConfig))]
    public class RankingTrain
    {
        private string _mslrWeb10k_Validate;
        private string _mslrWeb10k_Train;

        [GlobalSetup]
        public void SetupTrainingSpeedTests()
        {
            _mslrWeb10k_Validate = Path.GetFullPath(TestDatasets.MSLRWeb.validFilename);
            _mslrWeb10k_Train = Path.GetFullPath(TestDatasets.MSLRWeb.trainFilename);
            
            if (!File.Exists(_mslrWeb10k_Validate))
                throw new FileNotFoundException(string.Format(Helpers.DatasetNotFound, _mslrWeb10k_Validate));

            if (!File.Exists(_mslrWeb10k_Train))
                throw new FileNotFoundException(string.Format(Helpers.DatasetNotFound, _mslrWeb10k_Train));
        }

        [Benchmark]
        public void TrainTest_Ranking_MSLRWeb10K_RawNumericFeatures_FastTreeRanking()
        {
            string cmd = @"TrainTest test=" + _mslrWeb10k_Validate +
                " eval=RankingEvaluator{t=10}" +
                " data=" + _mslrWeb10k_Train +
                " loader=TextLoader{col=Label:R4:0 col=GroupId:TX:1 col=Features:R4:2-138}" +
                " xf=HashTransform{col=GroupId} xf=NAHandleTransform{col=Features}" +
                " tr=FastTreeRanking{}";

            using (var environment = new ConsoleEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                Maml.MainCore(environment, cmd, alwaysPrintStacktrace: false);
            }
        }

        [Benchmark]
        public void TrainTest_Ranking_MSLRWeb10K_RawNumericFeatures_LightGBMRanking()
        {
            string cmd = @"TrainTest test=" + _mslrWeb10k_Validate +
                " eval=RankingEvaluator{t=10}" +
                " data=" + _mslrWeb10k_Train +
                " loader=TextLoader{col=Label:R4:0 col=GroupId:TX:1 col=Features:R4:2-138}" +
                " xf=HashTransform{col=GroupId}" +
                " xf=NAHandleTransform{col=Features}" +
                " tr=LightGBMRanking{}";

            using (var environment = new ConsoleEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                Maml.MainCore(environment, cmd, alwaysPrintStacktrace: false);
            }
        }
    }

    [Config(typeof(PredictConfig))]
    public class RankingTest
    {
        private string _mslrWeb10k_Validate;
        private string _mslrWeb10k_Train;
        private string _mslrWeb10k_Test;
        private string _modelPath_MSLR;

        [GlobalSetup]
        public void SetupScoringSpeedTests()
        {
            _mslrWeb10k_Test = Path.GetFullPath(TestDatasets.MSLRWeb.testFilename);
            _mslrWeb10k_Validate = Path.GetFullPath(TestDatasets.MSLRWeb.validFilename);
            _mslrWeb10k_Train = Path.GetFullPath(TestDatasets.MSLRWeb.trainFilename);

            if (!File.Exists(_mslrWeb10k_Test))
                throw new FileNotFoundException(string.Format(Helpers.DatasetNotFound, _mslrWeb10k_Test));

            if (!File.Exists(_mslrWeb10k_Validate))
                throw new FileNotFoundException(string.Format(Helpers.DatasetNotFound, _mslrWeb10k_Validate));

            if (!File.Exists(_mslrWeb10k_Train))
                throw new FileNotFoundException(string.Format(Helpers.DatasetNotFound, _mslrWeb10k_Train));

            _modelPath_MSLR = Path.Combine(Directory.GetCurrentDirectory(), @"FastTreeRankingModel.zip");

            string cmd = @"TrainTest test=" + _mslrWeb10k_Validate +
                " eval=RankingEvaluator{t=10}" +
                " data=" + _mslrWeb10k_Train +
                " loader=TextLoader{col=Label:R4:0 col=GroupId:TX:1 col=Features:R4:2-138}" +
                " xf=HashTransform{col=GroupId}" +
                " xf=NAHandleTransform{col=Features}" +
                " tr=FastTreeRanking{}" +
                " out={" + _modelPath_MSLR + "}";

            using (var environment = new ConsoleEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                Maml.MainCore(environment, cmd, alwaysPrintStacktrace: false);
            }
        }

        [Benchmark]
        public void Test_Ranking_MSLRWeb10K_RawNumericFeatures_FastTreeRanking()
        {
            // This benchmark is profiling bulk scoring speed and not training speed. 
            string cmd = @"Test data=" + _mslrWeb10k_Test + " in=" + _modelPath_MSLR;
            using (var environment = new ConsoleEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                Maml.MainCore(environment, cmd, alwaysPrintStacktrace: false);
            }
        }
    }
}
