// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using BenchmarkDotNet.Attributes;
using Microsoft.ML.Transforms.Text;
using Xunit;

namespace Microsoft.ML.Benchmarks
{
    [Config(typeof(TrainConfig))]
    public class FeaturizeTextBench
    {
        private MLContext mlContext;
        private IDataView dataset;
        private static int numColumns = 1000;
        private static int numRows = 300;
        private static int maxWordLength = 15;

        [GlobalSetup]
        public void SetupData()
        {
            Path.GetTempFileName();
            mlContext = new MLContext(seed: 1);
            var path = Path.GetTempFileName();
            Console.WriteLine($"Created dataset in temporary file:\n{path}\n");
            path = CreateRandomFile(path);

            var columns = new List<TextLoader.Column>();
            for(int i = 0; i < numColumns; i++)
            {
                columns.Add(new TextLoader.Column($"Column{i}", DataKind.String, i));
            }

            var textLoader = mlContext.Data.CreateTextLoader(new TextLoader.Options()
            {
                Columns = columns.ToArray(),
                HasHeader = false,
                Separators = new char[] { ',' }
            });

            dataset = textLoader.Load(path);
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
                var featurizer = mlContext.Transforms.Text.FeaturizeText(textColumn, new TextFeaturizingEstimator.Options()
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

            IEstimator<ITransformer>  pipeline = featurizers.First();
            foreach (var featurizer in featurizers.Skip(1))
            {
                pipeline = pipeline.Append(featurizer);
            }

            var model = pipeline.Fit(dataset);

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

        public static string CreateRandomFile(string path)
        {
            // Create file with random strings
            // to use as dataset of the benchmark

            Random random = new Random(1);

            using (StreamWriter file = new StreamWriter(path))
            {
                for(int i = 0; i < numRows; i++)
                    file.WriteLine(CreateRandomLine(numColumns, random));
            }
            return path;
        }

        public static string CreateRandomLine(int columns, Random random)
        {
            var lineSB = new System.Text.StringBuilder();
            for(int i = 0; i < columns; i++)
            {
                lineSB.Append(CreateRandomColumn(random, random.Next(100)));
                lineSB.Append(",");
            }
            return lineSB.ToString();
        }

        public static string CreateRandomColumn(Random random, int numwords)
        {
            const string characters =
                "01234567890" +
                "abcdefghijklmnopqrstuvwxyz" +
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

            var columnSB = new System.Text.StringBuilder();
            int wordLength;

            for(int i = 0; i < numwords; i++)
            {
                wordLength = random.Next(1, maxWordLength);
                for(int j = 0; j < wordLength; j++)
                    columnSB.Append(characters[random.Next(characters.Length)]);
                
                columnSB.Append(" ");
            }

            if (random.Next(2) == 0) // sometimes return the column as lowercase
                return columnSB.ToString().ToLower();

            return columnSB.ToString();
        }
    }
}
