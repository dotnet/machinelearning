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

            Console.WriteLine("args");

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

            You should select the target benchmark(s). Please, print a number of a benchmark(e.g. `0`) or a contained benchmark caption(e.g. `TextPredictionEngineCreationBenchmark`).
            If you want to select few, please separate them with space ` ` (e.g. `1 2 3`).
            */

            BenchmarkSwitcher
               .FromAssembly(typeof(Program).Assembly)
               .Run(args, new RecommendedConfig());

        }
    }
}
