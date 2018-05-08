// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;


namespace Microsoft.ML.Benchmarks
{
    [KeepBenchmarkFiles]
    public class TrainPredictionBench
    {
        [Benchmark]
        public void Iris()
        {
            var dataPath = Program.GetDataPath("iris.txt");
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader<IrisData>(dataPath, useHeader: true, separator: "tab"));
            pipeline.Add(new ColumnConcatenator(outputColumn: "Features",
                "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            PredictionModel<IrisData, IrisPrediction> model = pipeline.Train<IrisData, IrisPrediction>();

            IrisPrediction prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            });

            prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.1f,
                SepalWidth = 5.5f,
                PetalLength = 2.2f,
                PetalWidth = 6.4f,
            });

            prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.1f,
                SepalWidth = 2.5f,
                PetalLength = 1.2f,
                PetalWidth = 4.4f,
            });
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
