// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Globalization;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using Microsoft.ML.Benchmarks.Harness;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;

namespace Microsoft.ML.Benchmarks
{
    [CIBenchmark]
    public class StochasticDualCoordinateAscentClassifierBench : WithExtraMetrics
    {
        private readonly string _dataPath = BaseTestClass.GetDataPath("iris.txt");
        private readonly string _sentimentDataPath = BaseTestClass.GetDataPath("wikipedia-detox-250-line-data.tsv");
        private readonly Consumer _consumer = new Consumer(); // BenchmarkDotNet utility type used to prevent dead code elimination

        private readonly MLContext mlContext = new MLContext(seed: 1);

        private readonly int[] _batchSizes = new int[] { 1, 2, 5 };

        private readonly IrisData _example = new IrisData()
        {
            SepalLength = 3.3f,
            SepalWidth = 1.6f,
            PetalLength = 0.2f,
            PetalWidth = 5.1f,
        };

        private TransformerChain<MulticlassPredictionTransformer<MaximumEntropyModelParameters>> _trainedModel;
        private PredictionEngine<IrisData, IrisPrediction> _predictionEngine;
        private IrisData[][] _batches;
        private MulticlassClassificationMetrics _metrics;

        protected override IEnumerable<Metric> GetMetrics()
        {
            if (_metrics != null)
            {
                yield return new Metric(
                    nameof(MulticlassClassificationMetrics.MicroAccuracy),
                    _metrics.MicroAccuracy.ToString("0.##", CultureInfo.InvariantCulture));
                yield return new Metric(
                    nameof(MulticlassClassificationMetrics.MacroAccuracy),
                    _metrics.MacroAccuracy.ToString("0.##", CultureInfo.InvariantCulture));
            }
        }

        [Benchmark]
        public TransformerChain<MulticlassPredictionTransformer<MaximumEntropyModelParameters>> TrainIris() => Train(_dataPath);

        private TransformerChain<MulticlassPredictionTransformer<MaximumEntropyModelParameters>> Train(string dataPath)
        {
            // Create text loader.
            var options = new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("Label", DataKind.Single, 0),
                    new TextLoader.Column("SepalLength", DataKind.Single, 1),
                    new TextLoader.Column("SepalWidth", DataKind.Single, 2),
                    new TextLoader.Column("PetalLength", DataKind.Single, 3),
                    new TextLoader.Column("PetalWidth", DataKind.Single, 4),
                },
                HasHeader = true,
            };
            var loader = new TextLoader(mlContext, options: options);

            IDataView data = loader.Load(dataPath);

            var pipeline = new ColumnConcatenatingEstimator(mlContext, "Features", new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" })
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaCalibrated());

            return pipeline.Fit(data);
        }

        [Benchmark]
        public void TrainSentiment()
        {
            // Pipeline
            var arguments = new TextLoader.Options()
            {
                Columns = new TextLoader.Column[]
                {
                    new TextLoader.Column("Label", DataKind.Single, new[] { new TextLoader.Range() { Min = 0, Max = 0 } }),
                    new TextLoader.Column("SentimentText", DataKind.String, new[] { new TextLoader.Range() { Min = 1, Max = 1 } })
                },
                HasHeader = true,
                AllowQuoting = false,
                AllowSparse = false
            };

            var loader = mlContext.Data.LoadFromTextFile(_sentimentDataPath, arguments);
            var text = mlContext.Transforms.Text.FeaturizeText("WordEmbeddings", new TextFeaturizingEstimator.Options
            {
                OutputTokensColumnName = "WordEmbeddings_TransformedText",
                KeepPunctuations = false,
                StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options(),
                Norm = TextFeaturizingEstimator.NormFunction.None,
                CharFeatureExtractor = null,
                WordFeatureExtractor = null,
            }, "SentimentText").Fit(loader).Transform(loader);

            var trans = mlContext.Transforms.Text.ApplyWordEmbedding("Features", "WordEmbeddings_TransformedText",
                WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding)
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .Fit(text).Transform(text);

            // Train
            var trainer = mlContext.MulticlassClassification.Trainers.SdcaCalibrated();
            var predicted = trainer.Fit(trans);
            _consumer.Consume(predicted);
        }

        [GlobalSetup(Targets = new string[] { nameof(PredictIris), nameof(PredictIrisBatchOf1), nameof(PredictIrisBatchOf2), nameof(PredictIrisBatchOf5) })]
        public void SetupPredictBenchmarks()
        {
            _trainedModel = Train(_dataPath);
            _predictionEngine = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(_trainedModel);
            _consumer.Consume(_predictionEngine.Predict(_example));

            // Create text loader.
            var options = new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("Label", DataKind.Single, 0),
                    new TextLoader.Column("SepalLength", DataKind.Single, 1),
                    new TextLoader.Column("SepalWidth", DataKind.Single, 2),
                    new TextLoader.Column("PetalLength", DataKind.Single, 3),
                    new TextLoader.Column("PetalWidth", DataKind.Single, 4),
                },
                HasHeader = true,
            };
            var loader = new TextLoader(mlContext, options: options);

            IDataView testData = loader.Load(_dataPath);
            IDataView scoredTestData = _trainedModel.Transform(testData);
            var evaluator = new MulticlassClassificationEvaluator(mlContext, new MulticlassClassificationEvaluator.Arguments());
            _metrics = evaluator.Evaluate(scoredTestData, DefaultColumnNames.Label, DefaultColumnNames.Score, DefaultColumnNames.PredictedLabel);

            _batches = new IrisData[_batchSizes.Length][];
            for (int i = 0; i < _batches.Length; i++)
            {
                var batch = new IrisData[_batchSizes[i]];
                for (int bi = 0; bi < batch.Length; bi++)
                {
                    batch[bi] = _example;
                }
                _batches[i] = batch;
            }
        }

        [Benchmark]
        public float[] PredictIris() => _predictionEngine.Predict(_example).PredictedLabels;

        [Benchmark]
        public void PredictIrisBatchOf1() => _trainedModel.Transform(mlContext.Data.LoadFromEnumerable(_batches[0]));

        [Benchmark]
        public void PredictIrisBatchOf2() => _trainedModel.Transform(mlContext.Data.LoadFromEnumerable(_batches[1]));

        [Benchmark]
        public void PredictIrisBatchOf5() => _trainedModel.Transform(mlContext.Data.LoadFromEnumerable(_batches[2]));
    }

    public class IrisData
    {
        [LoadColumn(0)]
        public float Label;

        [LoadColumn(1)]
        public float SepalLength;

        [LoadColumn(2)]
        public float SepalWidth;

        [LoadColumn(3)]
        public float PetalLength;

        [LoadColumn(4)]
        public float PetalWidth;
    }

    public class IrisPrediction
    {
        [ColumnName("Score")]
        public float[] PredictedLabels;
    }
}
