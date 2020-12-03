// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using System.Threading;
using BenchmarkDotNet.Running;

namespace Microsoft.ML.PerformanceTests
{
    class Program
    {
        /// <summary>
        /// execute dotnet run -c Release and choose the benchmarks you want to run
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            // enforce Neutral Language as "en-us" because the input data files use dot as decimal separator (and it fails for cultures with ",")
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;

            /*
            Available Benchmarks:
                #0  TextPredictionEngineCreationBenchmark
                #1  CacheDataViewBench
                #2  FeaturizeTextBench
                #3  HashBench
                #4  ImageClassificationBench
                #5  KMeansAndLogisticRegressionBench
                #6  MulticlassClassificationTest
                #7  MulticlassClassificationTrain
                #8  PredictionEngineBench
                #9  RankingTest
                #10 RankingTrain
                #11 RffTransformTrain
                #12 ShuffleRowsBench
                #13 StochasticDualCoordinateAscentClassifierBench
                #14 TextLoaderBench
            */

            // TO-DO: Further investigate how to run each benchmark separately.
            // Arcade's `-performanceTest` command results in a predefined dotnet call that does not allow for additional arguments
            // to be passed (say, for selecting individual benchmarks to run).
            // Link to code:
            //     https://github.com/dotnet/arcade/blob/4873d157a8f34f8cc7e28b3f9938b32c642ef542/src/Microsoft.DotNet.Arcade.Sdk/tools/Performance.targets#L16-L19
            BenchmarkSwitcher
               .FromAssembly(typeof(Program).Assembly)
               .RunAll(new RecommendedConfig());

        }
    }
}
