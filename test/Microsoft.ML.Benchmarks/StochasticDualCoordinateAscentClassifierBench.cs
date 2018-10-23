// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers;
using System.Collections.Generic;
using System.Globalization;

namespace Microsoft.ML.Benchmarks
{
    public class StochasticDualCoordinateAscentClassifierBench : WithExtraMetrics
    {
        private readonly string _dataPath = Program.GetInvariantCultureDataPath("iris.txt");
        private readonly string _sentimentDataPath = Program.GetInvariantCultureDataPath("wikipedia-detox-250-line-data.tsv");
        private readonly Consumer _consumer = new Consumer(); // BenchmarkDotNet utility type used to prevent dead code elimination

        private readonly int[] _batchSizes = new int[] { 1, 2, 5 };
        private readonly IrisData _example = new IrisData()
        {
            SepalLength = 3.3f,
            SepalWidth = 1.6f,
            PetalLength = 0.2f,
            PetalWidth = 5.1f,
        };

        private Legacy.PredictionModel<IrisData, IrisPrediction> _trainedModel;
        private IrisData[][] _batches;
        private ClassificationMetrics _metrics;

        protected override IEnumerable<Metric> GetMetrics()
        {
            if (_metrics != null)
                yield return new Metric(
                    nameof(ClassificationMetrics.AccuracyMacro),
                    _metrics.AccuracyMacro.ToString("0.##", CultureInfo.InvariantCulture));
        }

        [Benchmark]
        public Legacy.PredictionModel<IrisData, IrisPrediction> TrainIris() => Train(_dataPath);

        private Legacy.PredictionModel<IrisData, IrisPrediction> Train(string dataPath)
        {
            var pipeline = new Legacy.LearningPipeline();

            pipeline.Add(new Legacy.Data.TextLoader(dataPath).CreateFrom<IrisData>(useHeader: true));
            pipeline.Add(new ColumnConcatenator(outputColumn: "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            return pipeline.Train<IrisData, IrisPrediction>();
        }

        [Benchmark]
        public void TrainSentiment()
        {
            using (var env = new ConsoleEnvironment(seed: 1))
            {
                // Pipeline
                var loader = TextLoader.ReadFile(env,
                    new TextLoader.Arguments()
                    {
                        AllowQuoting = false,
                        AllowSparse = false,
                        Separator = "tab",
                        HasHeader = true,
                        Column = new[]
                        {
                            new TextLoader.Column()
                            {
                                Name = "Label",
                                Source = new [] { new TextLoader.Range() { Min=0, Max=0} },
                                Type = DataKind.Num
                            },

                            new TextLoader.Column()
                            {
                                Name = "SentimentText",
                                Source = new [] { new TextLoader.Range() { Min=1, Max=1} },
                                Type = DataKind.Text
                            }
                        }
                    }, new MultiFileSource(_sentimentDataPath));

                var text = TextTransform.Create(env,
                    new TextTransform.Arguments()
                    {
                        Column = new TextTransform.Column
                        {
                            Name = "WordEmbeddings",
                            Source = new[] { "SentimentText" }
                        },
                        OutputTokens = true,
                        KeepPunctuations=false,
                        StopWordsRemover = new Runtime.TextAnalytics.PredefinedStopWordsRemoverFactory(),
                        VectorNormalizer = TextTransform.TextNormKind.None,
                        CharFeatureExtractor = null,
                        WordFeatureExtractor = null,
                    }, loader);

                var trans = WordEmbeddingsTransform.Create(env,
                    new WordEmbeddingsTransform.Arguments()
                    {
                        Column = new WordEmbeddingsTransform.Column[1]
                        {
                            new WordEmbeddingsTransform.Column
                            {
                                Name = "Features",
                                Source = "WordEmbeddings_TransformedText"
                            }
                        },
                        ModelKind = WordEmbeddingsTransform.PretrainedModelKind.Sswe,
                    }, text);

                // Train
                var trainer = new SdcaMultiClassTrainer(env, "Features", "Label", maxIterations: 20);
                var trainRoles = new RoleMappedData(trans, label: "Label", feature: "Features");

                var predicted = trainer.Train(trainRoles);
                _consumer.Consume(predicted);
            }
        }

        [GlobalSetup(Targets = new string[] { nameof(PredictIris), nameof(PredictIrisBatchOf1), nameof(PredictIrisBatchOf2), nameof(PredictIrisBatchOf5) })]
        public void SetupPredictBenchmarks()
        {
            _trainedModel = Train(_dataPath);
            _consumer.Consume(_trainedModel.Predict(_example));

            var testData = new Legacy.Data.TextLoader(_dataPath).CreateFrom<IrisData>(useHeader: true);
            var evaluator = new ClassificationEvaluator();
            _metrics = evaluator.Evaluate(_trainedModel, testData);

            _batches = new IrisData[_batchSizes.Length][];
            for (int i = 0; i < _batches.Length; i++)
            {
                var batch = new IrisData[_batchSizes[i]];
                _batches[i] = batch;
                for (int bi = 0; bi < batch.Length; bi++)
                {
                    batch[bi] = _example;
                }
            }
        }

        [Benchmark]
        public float[] PredictIris() => _trainedModel.Predict(_example).PredictedLabels;

        [Benchmark]
        public void PredictIrisBatchOf1() => Consume(_trainedModel.Predict(_batches[0]));

        [Benchmark]
        public void PredictIrisBatchOf2() => Consume(_trainedModel.Predict(_batches[1]));

        [Benchmark]
        public void PredictIrisBatchOf5() => Consume(_trainedModel.Predict(_batches[2]));

        private void Consume(IEnumerable<IrisPrediction> predictions)
        {
            foreach (var prediction in predictions)
                _consumer.Consume(prediction);
        }
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
