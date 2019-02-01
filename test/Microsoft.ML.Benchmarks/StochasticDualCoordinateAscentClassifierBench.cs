// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Globalization;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using Microsoft.Data.DataView;
using Microsoft.ML.Benchmarks.Harness;
using Microsoft.ML.Data;
using Microsoft.ML.Learners;
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

        private readonly MLContext _env = new MLContext(seed: 1);

        private readonly int[] _batchSizes = new int[] { 1, 2, 5 };

        private readonly IrisData _example = new IrisData()
        {
            SepalLength = 3.3f,
            SepalWidth = 1.6f,
            PetalLength = 0.2f,
            PetalWidth = 5.1f,
        };

        private TransformerChain<MulticlassPredictionTransformer<MulticlassLogisticRegressionModelParameters>> _trainedModel;
        private PredictionEngine<IrisData, IrisPrediction> _predictionEngine;
        private IrisData[][] _batches;
        private MultiClassClassifierMetrics _metrics;

        protected override IEnumerable<Metric> GetMetrics()
        {
            if (_metrics != null)
                yield return new Metric(
                    nameof(MultiClassClassifierMetrics.AccuracyMacro),
                    _metrics.AccuracyMacro.ToString("0.##", CultureInfo.InvariantCulture));
        }

        [Benchmark]
        public TransformerChain<MulticlassPredictionTransformer<MulticlassLogisticRegressionModelParameters>> TrainIris() => Train(_dataPath);

        private TransformerChain<MulticlassPredictionTransformer<MulticlassLogisticRegressionModelParameters>> Train(string dataPath)
        {
            var reader = new TextLoader(_env,
                    columns: new[]
                    {
                            new TextLoader.Column("Label", DataKind.R4, 0),
                            new TextLoader.Column("SepalLength", DataKind.R4, 1),
                            new TextLoader.Column("SepalWidth", DataKind.R4, 2),
                            new TextLoader.Column("PetalLength", DataKind.R4, 3),
                            new TextLoader.Column("PetalWidth", DataKind.R4, 4),
                    },
                    hasHeader: true
                );

            IDataView data = reader.Read(dataPath);

            var pipeline = new ColumnConcatenatingEstimator(_env, "Features", new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" })
                .Append(_env.MulticlassClassification.Trainers.StochasticDualCoordinateAscent());

            return pipeline.Fit(data);
        }

        [Benchmark]
        public void TrainSentiment()
        {
            // Pipeline
            var arguments = new TextLoader.Arguments()
            {
                Columns = new TextLoader.Column[]
                {
                    new TextLoader.Column()
                    {
                        Name = "Label",
                        Source = new[] { new TextLoader.Range() { Min = 0, Max = 0 } },
                        Type = DataKind.Num
                    },

                    new TextLoader.Column()
                    {
                        Name = "SentimentText",
                        Source = new[] { new TextLoader.Range() { Min = 1, Max = 1 } },
                        Type = DataKind.Text
                    }
                },
                HasHeader = true,
                AllowQuoting = false,
                AllowSparse = false
            };
            var loader = _env.Data.ReadFromTextFile(_sentimentDataPath, arguments);
            var text = new TextFeaturizingEstimator(_env, "WordEmbeddings", "SentimentText", args =>
            {
                args.OutputTokens = true;
                args.KeepPunctuations = false;
                args.UseStopRemover = true;
                args.VectorNormalizer = TextFeaturizingEstimator.TextNormKind.None;
                args.UseCharExtractor = false;
                args.UseWordExtractor = false;
            }).Fit(loader).Transform(loader);
            var trans = _env.Transforms.Text.ExtractWordEmbeddings("Features", "WordEmbeddings_TransformedText", 
                WordEmbeddingsExtractingTransformer.PretrainedModelKind.Sswe).Fit(text).Transform(text);
            // Train
            var trainer = _env.MulticlassClassification.Trainers.StochasticDualCoordinateAscent();
            var predicted = trainer.Fit(trans);
            _consumer.Consume(predicted);
        }

        [GlobalSetup(Targets = new string[] { nameof(PredictIris), nameof(PredictIrisBatchOf1), nameof(PredictIrisBatchOf2), nameof(PredictIrisBatchOf5) })]
        public void SetupPredictBenchmarks()
        {
            _trainedModel = Train(_dataPath);
            _predictionEngine = _trainedModel.CreatePredictionEngine<IrisData, IrisPrediction>(_env);
            _consumer.Consume(_predictionEngine.Predict(_example));

            var reader = new TextLoader(_env,
                    columns: new[]
                    {
                            new TextLoader.Column("Label", DataKind.R4, 0),
                            new TextLoader.Column("SepalLength", DataKind.R4, 1),
                            new TextLoader.Column("SepalWidth", DataKind.R4, 2),
                            new TextLoader.Column("PetalLength", DataKind.R4, 3),
                            new TextLoader.Column("PetalWidth", DataKind.R4, 4),
                    },
                    hasHeader: true
                );

            IDataView testData = reader.Read(_dataPath);
            IDataView scoredTestData = _trainedModel.Transform(testData);
            var evaluator = new MultiClassClassifierEvaluator(_env, new MultiClassClassifierEvaluator.Arguments());
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
        public void PredictIrisBatchOf1() => _trainedModel.Transform(_env.Data.ReadFromEnumerable(_batches[0]));

        [Benchmark]
        public void PredictIrisBatchOf2() => _trainedModel.Transform(_env.Data.ReadFromEnumerable(_batches[1]));

        [Benchmark]
        public void PredictIrisBatchOf5() => _trainedModel.Transform(_env.Data.ReadFromEnumerable(_batches[2]));
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
