// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Benchmarks.Harness;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Benchmarks
{
    [CIBenchmark]
    public class PredictionEngineBench
    {
        private IrisData _irisExample;
        private PredictionEngine<IrisData, IrisPrediction> _irisModel;

        private SentimentData _sentimentExample;
        private PredictionEngine<SentimentData, SentimentPrediction> _sentimentModel;

        private BreastCancerData _breastCancerExample;
        private PredictionEngine<BreastCancerData, BreastCancerPrediction> _breastCancerModel;

        [GlobalSetup(Target = nameof(MakeIrisPredictions))]
        public void SetupIrisPipeline()
        {
            _irisExample = new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            };

            string _irisDataPath = BaseTestClass.GetDataPath("iris.txt");

            var env = new MLContext(seed: 1);

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
            var loader = new TextLoader(env, options: options);

            IDataView data = loader.Load(_irisDataPath);

            var pipeline = new ColumnConcatenatingEstimator(env, "Features", new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" })
                .Append(env.Transforms.Conversion.MapValueToKey("Label"))
                .Append(env.MulticlassClassification.Trainers.SdcaCalibrated(
                    new SdcaCalibratedMulticlassTrainer.Options { NumberOfThreads = 1, ConvergenceTolerance = 1e-2f, }));

            var model = pipeline.Fit(data);

            _irisModel = env.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model);
        }

        [GlobalSetup(Target = nameof(MakeSentimentPredictions))]
        public void SetupSentimentPipeline()
        {
            _sentimentExample = new SentimentData()
            {
                SentimentText = "Not a big fan of this."
            };

            string _sentimentDataPath = BaseTestClass.GetDataPath("wikipedia-detox-250-line-data.tsv");

            var mlContext = new MLContext(seed: 1);

            // Create text loader.
            var options = new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("Label", DataKind.Boolean, 0),
                    new TextLoader.Column("SentimentText", DataKind.String, 1)
                },
                HasHeader = true,
            };
            var loader = new TextLoader(mlContext, options: options);

            IDataView data = loader.Load(_sentimentDataPath);

            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .Append(mlContext.BinaryClassification.Trainers.SdcaNonCalibrated(
                    new SdcaNonCalibratedBinaryTrainer.Options { NumberOfThreads = 1, ConvergenceTolerance = 1e-2f, }));

            var model = pipeline.Fit(data);

            _sentimentModel = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
        }

        [GlobalSetup(Target = nameof(MakeBreastCancerPredictions))]
        public void SetupBreastCancerPipeline()
        {
            _breastCancerExample = new BreastCancerData()
            {
                Features = new[] { 5f, 1f, 1f, 1f, 2f, 1f, 3f, 1f, 1f }
            };

            string _breastCancerDataPath = BaseTestClass.GetDataPath("breast-cancer.txt");

            var env = new MLContext(seed: 1);

            // Create text loader.
            var options = new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("Label", DataKind.Boolean, 0),
                    new TextLoader.Column("Features", DataKind.Single, new[] { new TextLoader.Range(1, 9) })
                },
                HasHeader = false,
            };
            var loader = new TextLoader(env, options: options);

            IDataView data = loader.Load(_breastCancerDataPath);

            var pipeline = env.BinaryClassification.Trainers.SdcaNonCalibrated(
                new SdcaNonCalibratedBinaryTrainer.Options { NumberOfThreads = 1, ConvergenceTolerance = 1e-2f, });

            var model = pipeline.Fit(data);

            _breastCancerModel = env.Model.CreatePredictionEngine<BreastCancerData, BreastCancerPrediction>(model);
        }

        [Benchmark]
        public void MakeIrisPredictions()
        {
            for (int i = 0; i < 10000; i++)
            {
                _irisModel.Predict(_irisExample);
            }
        }

        [Benchmark]
        public void MakeSentimentPredictions()
        {
            for (int i = 0; i < 10000; i++)
            {
                _sentimentModel.Predict(_sentimentExample);
            }
        }

        [Benchmark]
        public void MakeBreastCancerPredictions()
        {
            for (int i = 0; i < 10000; i++)
            {
                _breastCancerModel.Predict(_breastCancerExample);
            }
        }
    }

    public class SentimentData
    {
        [ColumnName("Label"), LoadColumn(0)]
        public bool Sentiment;

        [LoadColumn(1)]
        public string SentimentText;
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Sentiment;

        public float Score;
    }

    public class BreastCancerData
    {
        [ColumnName("Label")]
        public bool Label;

        [ColumnName("Features"), VectorType(9)]
        public float[] Features;
    }

    public class BreastCancerPrediction
    {
        [ColumnName("Score")]
        public float Score;
    }
}
