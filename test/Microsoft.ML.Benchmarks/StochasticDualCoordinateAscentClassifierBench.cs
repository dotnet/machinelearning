// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
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
        private static string s_sentimentDataPath;
        private static IrisData[][] s_batches;
        private static readonly int[] s_batchSizes = new int[] { 1, 2, 5 };
        private readonly Random r = new Random(0);
        private readonly Consumer _consumer = new Consumer();
        private static readonly IrisData s_example = new IrisData()
        {
            SepalLength = 3.3f,
            SepalWidth = 1.6f,
            PetalLength = 0.2f,
            PetalWidth = 5.1f,
        };

        [GlobalSetup]
        public void Setup()
        {
            s_dataPath = Program.GetDataPath("iris.txt");
            s_sentimentDataPath = Program.GetDataPath("wikipedia-detox-250-line-data.tsv");
            s_trainedModel = TrainCore();
            IrisPrediction prediction = s_trainedModel.Predict(s_example);

            var testData = new Data.TextLoader(s_dataPath).CreateFrom<IrisData>(useHeader: true);
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

        [Benchmark]
        public PredictionModel<IrisData, IrisPrediction> TrainIris() => TrainCore();

        [Benchmark]
        public float[] PredictIris() => s_trainedModel.Predict(s_example).PredictedLabels;

        [Benchmark]
        public void PredictIrisBatchOf1() => Consume(s_trainedModel.Predict(s_batches[0]));

        [Benchmark]
        public void PredictIrisBatchOf2() => Consume(s_trainedModel.Predict(s_batches[1]));

        [Benchmark]
        public void PredictIrisBatchOf5() => Consume(s_trainedModel.Predict(s_batches[2]));

        [Benchmark]
        public IPredictor TrainSentiment() => TrainSentimentCore();

        private void Consume(IEnumerable<IrisPrediction> predictions)
        {
            foreach (var prediction in predictions)
                _consumer.Consume(prediction);
        }

        private static PredictionModel<IrisData, IrisPrediction> TrainCore()
        {
            var pipeline = new LearningPipeline();

            pipeline.Add(new Data.TextLoader(s_dataPath).CreateFrom<IrisData>(useHeader: true));
            pipeline.Add(new ColumnConcatenator(outputColumn: "Features",
                "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            PredictionModel<IrisData, IrisPrediction> model = pipeline.Train<IrisData, IrisPrediction>();
            return model;
        }

        private static IPredictor TrainSentimentCore()
        {
            var dataPath = s_sentimentDataPath;
            using (var env = new TlcEnvironment(seed: 1))
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
                    }, new MultiFileSource(dataPath));

                var text = TextTransform.Create(env,
                    new TextTransform.Arguments()
                    {
                        Column = new TextTransform.Column
                        {
                            Name = "WordEmbeddings",
                            Source = new[] { "SentimentText" }
                        },
                        KeepDiacritics = false,
                        KeepPunctuations = false,
                        TextCase = Runtime.TextAnalytics.TextNormalizerTransform.CaseNormalizationMode.Lower,
                        OutputTokens = true,
                        StopWordsRemover = new Runtime.TextAnalytics.PredefinedStopWordsRemoverFactory(),
                        VectorNormalizer = TextTransform.TextNormKind.None,
                        CharFeatureExtractor = null,
                        WordFeatureExtractor = null,
                    }, loader);

                var trans = new WordEmbeddingsTransform(env, 
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
                var trainer = new SdcaMultiClassTrainer(env, new SdcaMultiClassTrainer.Arguments() { MaxIterations = 20 });
                var trainRoles = new RoleMappedData(trans, label: "Label", feature: "Features");
                return trainer.Train(trainRoles);
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
}
