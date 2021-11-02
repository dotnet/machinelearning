// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BenchmarkDotNet.Attributes;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using Xunit;

namespace Microsoft.ML.PerformanceTests
{
    [Config(typeof(TrainConfig))]
    public class FeaturizeTextBench : BenchmarkBase
    {
        private MLContext _mlContext;
        private IDataView _dataset;
        private static int _numColumns = 1000;
        private static int _numRows = 300;
        private static int _maxWordLength = 15;

        [GlobalSetup]
        public void SetupData()
        {
            _mlContext = new MLContext(seed: 1);
            var path = Path.GetTempFileName();
            Console.WriteLine($"Created dataset in temporary file:\n{path}\n");
            path = RandomFile.CreateRandomFile(path, _numRows, _numColumns, _maxWordLength);

            var columns = new List<TextLoader.Column>();
            for (int i = 0; i < _numColumns; i++)
            {
                columns.Add(new TextLoader.Column($"Column{i}", DataKind.String, i));
            }

            var textLoader = _mlContext.Data.CreateTextLoader(new TextLoader.Options()
            {
                Columns = columns.ToArray(),
                HasHeader = false,
                Separators = new char[] { ',' },
                AllowQuoting = true
            });

            _dataset = textLoader.Load(path);
        }

        [Benchmark]
        public ITransformer TrainFeaturizeText()
        {
            var textColumns = new List<string>();
            for (int i = 0; i < 20; i++) // Only load first 20 columns
            {
                textColumns.Add($"Column{i}");
            }

            var featurizers = new List<TextFeaturizingEstimator>();
            foreach (var textColumn in textColumns)
            {
                var featurizer = _mlContext.Transforms.Text.FeaturizeText(textColumn, new TextFeaturizingEstimator.Options()
                {
                    CharFeatureExtractor = null,
                    WordFeatureExtractor = new WordBagEstimator.Options()
                    {
                        NgramLength = 2,
                        MaximumNgramsCount = new int[] { 200000 }
                    }
                });
                featurizers.Add(featurizer);
            }

            IEstimator<ITransformer> pipeline = featurizers.First();
            foreach (var featurizer in featurizers.Skip(1))
            {
                pipeline = pipeline.Append(featurizer);
            }

            var model = pipeline.Fit(_dataset);

            // BENCHMARK OUTPUT
            // * Summary *

            //BenchmarkDotNet = v0.11.3, OS = Windows 10.0.18363
            //Intel Xeon W - 2133 CPU 3.60GHz, 1 CPU, 12 logical and 6 physical cores
            //.NET Core SDK = 3.0.100
            //[Host]     : .NET Core 2.1.13(CoreCLR 4.6.28008.01, CoreFX 4.6.28008.01), 64bit RyuJIT
            //Job - KDKCUJ : .NET Core 2.1.13(CoreCLR 4.6.28008.01, CoreFX 4.6.28008.01), 64bit RyuJIT

            //Arguments =/ p:Configuration = Release  Toolchain = netcoreapp2.1  IterationCount = 1
            //LaunchCount = 3  MaxIterationCount = 20  RunStrategy = ColdStart
            //UnrollFactor = 1  WarmupCount = 1

            //             Method | Mean     | Error    | StdDev    | Extra Metric  | Gen 0 / 1k Op | Gen 1 / 1k Op | Gen 2 / 1k Op | Allocated Memory / Op |
            //------------------- | --------:| --------:| ---------:| -------------:| -------------:| ------------: | ------------: | --------------------: |
            // TrainFeaturizeText | 17.00 s  | 6.337 s  | 0.3474 s  | -             | 1949000.0000  | 721000.0000   | 36000.0000    | 315.48 MB             |

            //// * Legends *
            //  Mean                : Arithmetic mean of all measurements
            //  Error               : Half of 99.9 % confidence interval
            //  StdDev              : Standard deviation of all measurements
            //  Extra Metric: Value of the provided extra metric
            //  Gen 0 / 1k Op         : GC Generation 0 collects per 1k Operations
            //  Gen 1 / 1k Op         : GC Generation 1 collects per 1k Operations
            //  Gen 2 / 1k Op         : GC Generation 2 collects per 1k Operations
            //  Allocated Memory/ Op : Allocated memory per single operation(managed only, inclusive, 1KB = 1024B)
            //  1 s: 1 Second(1 sec)

            //// * Diagnostic Output - MemoryDiagnoser *
            //// ***** BenchmarkRunner: End *****
            //  Run time: 00:01:52(112.92 sec), executed benchmarks: 1

            //// * Artifacts cleanup *
            //  Global total time: 00:01:59(119.89 sec), executed benchmarks: 1

            return model;
        }
    }
}
