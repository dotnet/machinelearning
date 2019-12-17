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
            var memoryUsage = GC.GetTotalMemory(true);
            Console.WriteLine($"Memory Used: {memoryUsage/1000000:0,0.00}MB");
            Assert.True(memoryUsage < 400000000, $"This benchmark should use less than 400MB of memory, but it's using {memoryUsage/1000000:0,0.00}MB"); // Memory usage should be less than 1GB after PR https://github.com/dotnet/machinelearning/pull/4576

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
