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
    public class Ranking
    {
        public string _mslrWeb10k_Validate;
        public string _mslrWeb10k_Train;
        public string _mslrWeb10k_Test;
        private string _modelPath_MSLR;

        [GlobalSetup(Targets = new string[] {
            nameof(TrainTest_Multiclass_MSLRWeb10K_Ranking_FastTree),
            nameof(TrainTest_Multiclass_MSLRWeb10K_Ranking_LightGBM) })]
        public void SetupTrainingSpeedTests()
        {
            _mslrWeb10k_Validate = Path.GetFullPath(TestDatasets.MSLRWeb.validFilename);
            _mslrWeb10k_Train = Path.GetFullPath(TestDatasets.MSLRWeb.trainFilename);
            
            if (!File.Exists(_mslrWeb10k_Validate))
            {
                throw new FileNotFoundException($"Could not find {_mslrWeb10k_Validate} Please ensure you have run 'build.cmd -- /t:DownloadExternalTestFiles /p:IncludeBenchmarkData=true' from the root");
            }

            if (!File.Exists(_mslrWeb10k_Train))
            {
                throw new FileNotFoundException($"Could not find {_mslrWeb10k_Train} Please ensure you have run 'build.cmd -- /t:DownloadExternalTestFiles /p:IncludeBenchmarkData=true' from the root");
            }
        }

        [GlobalSetup(Target = nameof(Test_Multiclass_MSLRWeb10K_Ranking_FastTree))]
        public void SetupScoringSpeedTests()
        {
            _mslrWeb10k_Test = Path.GetFullPath(TestDatasets.MSLRWeb.testFilename);
            if (!File.Exists(_mslrWeb10k_Test))
            {
                throw new FileNotFoundException($"Could not find {_mslrWeb10k_Test} Please ensure you have run 'build.cmd -- /t:DownloadExternalTestFiles /p:IncludeBenchmarkData=true' from the root");
            
            }
            
            SetupTrainingSpeedTests();
            _modelPath_MSLR = Path.Combine(Directory.GetCurrentDirectory(), @"FastTreeRankingModel.zip");
            string cmd = @"TrainTest test=" + _mslrWeb10k_Validate + " eval=RankingEvaluator{t=10} data=" + _mslrWeb10k_Train+ "  loader=TextLoader{col=Label:R4:0 col=GroupId:TX:1 col=Features:R4:2-138} xf=HashTransform{col=GroupId} xf=NAHandleTransform{col=Features} tr=FastTreeRanking{} out={" + _modelPath_MSLR + "}";
            using (var tlc = new TlcEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                Maml.MainCore(tlc, cmd, alwaysPrintStacktrace: false);
            }
        }

        [Benchmark]
        public void TrainTest_Multiclass_MSLRWeb10K_Ranking_FastTree()
        {
            string cmd = @"TrainTest test=" + _mslrWeb10k_Validate + " eval=RankingEvaluator{t=10} data=" + _mslrWeb10k_Train + " loader=TextLoader{col=Label:R4:0 col=GroupId:TX:1 col=Features:R4:2-138} xf=HashTransform{col=GroupId} xf=NAHandleTransform{col=Features} tr=FastTreeRanking{}";
            using (var tlc = new TlcEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                Maml.MainCore(tlc, cmd, alwaysPrintStacktrace: false);
            }
        }

        [Benchmark]
        public void TrainTest_Multiclass_MSLRWeb10K_Ranking_LightGBM()
        {
            string cmd = @"TrainTest test=" + _mslrWeb10k_Validate + " eval=RankingEvaluator{t=10} data=" + _mslrWeb10k_Train + " loader=TextLoader{col=Label:R4:0 col=GroupId:TX:1 col=Features:R4:2-138} xf=HashTransform{col=GroupId} xf=NAHandleTransform{col=Features} tr=LightGBMRanking{}";
            using (var tlc = new TlcEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                Maml.MainCore(tlc, cmd, alwaysPrintStacktrace: false);
            }
        }

        [Benchmark]
        public void Test_Multiclass_MSLRWeb10K_Ranking_FastTree()
        {
            // This benchmark is profiling bulk scoring speed and not training speed. 
            string modelpath = Path.Combine(Directory.GetCurrentDirectory(), @"FastTreeRankingModel.zip");
            string cmd = @"Test data=" + _mslrWeb10k_Test + " in="+ modelpath;
            using (var tlc = new TlcEnvironment(verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                Maml.MainCore(tlc, cmd, alwaysPrintStacktrace: false);
            }
        }
    }
}
