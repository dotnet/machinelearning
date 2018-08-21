// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Benchmarks
{
    public class StochasticDualCoordinateAscentClassifierBench
    {
        internal static ClassificationMetrics s_metrics;
        private static PredictionModel<IrisData, IrisPrediction> s_trainedModel;
        private static string s_dataPath;
        private static IrisData[][] s_batches;
        private static readonly int[] s_batchSizes = new int[] { 1, 2, 5 };
        private readonly Random r = new Random(0);
        private static readonly IrisData s_example = new IrisData()
        {
            SepalLength = 3.3f,
            SepalWidth = 1.6f,
            PetalLength = 0.2f,
            PetalWidth = 5.1f,
        };

        [Benchmark]
        public PredictionModel<IrisData, IrisPrediction> TrainIris() => TrainCore();

        [Benchmark]
        public float[] PredictIris() => s_trainedModel.Predict(s_example).PredictedLabels;

        [Benchmark]
        public IEnumerable<IrisPrediction> PredictIrisBatchOf1() => s_trainedModel.Predict(s_batches[0]);
        [Benchmark]
        public IEnumerable<IrisPrediction> PredictIrisBatchOf2() => s_trainedModel.Predict(s_batches[1]);
        [Benchmark]
        public IEnumerable<IrisPrediction> PredictIrisBatchOf5() => s_trainedModel.Predict(s_batches[2]);

        [GlobalSetup]
        public void Setup()
        {
            s_dataPath = Program.GetDataPath("iris.txt");
            s_trainedModel = TrainCore();
            IrisPrediction prediction = s_trainedModel.Predict(s_example);

            var testData = new TextLoader(s_dataPath).CreateFrom<IrisData>(useHeader: true);
            var evaluator = new ClassificationEvaluator();
            s_metrics = evaluator.Evaluate(s_trainedModel, testData);

            s_batches = new IrisData[s_batchSizes.Length][];
            for (int i = 0; i < s_batches.Length; i++)
            {
                var batch = new IrisData[s_batchSizes[i]];
                s_batches[i] = batch;
                for (int bi = 0; bi < batch.Length; bi++)
                {
                    batch[bi] = s_example;
                }
            }
        }

        private static PredictionModel<IrisData, IrisPrediction> TrainCore()
        {
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader(s_dataPath).CreateFrom<IrisData>(useHeader: true));
            pipeline.Add(new ColumnConcatenator(outputColumn: "Features",
                "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            PredictionModel<IrisData, IrisPrediction> model = pipeline.Train<IrisData, IrisPrediction>();
            return model;
        }

        public class IrisData
        {
            [Column("0")]
            public float Label;

            [Column("1")]
            public float SepalLength;

            [Column("2")]
            public float SepalWidth;

            [Column("3")]
            public float PetalLength;

            [Column("4")]
            public float PetalWidth;
        }

        public class IrisPrediction
        {
            [ColumnName("Score")]
            public float[] PredictedLabels;
        }
    }
}
