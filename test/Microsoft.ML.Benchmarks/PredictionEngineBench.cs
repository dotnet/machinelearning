// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;

namespace Microsoft.ML.Benchmarks
{
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

            string _irisDataPath = Program.GetInvariantCultureDataPath("iris.txt");

            var env = new MLContext(seed: 1, conc: 1);
            var reader = new TextLoader(env,
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

            IDataView data = reader.Read(_irisDataPath);

            var pipeline = new ColumnConcatenatingEstimator(env, "Features", new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" })
                .Append(new SdcaMultiClassTrainer(env, "Label", "Features", advancedSettings: (s) => { s.NumThreads = 1; s.ConvergenceTolerance = 1e-2f; }));

            var model = pipeline.Fit(data);

            _irisModel = model.CreatePredictionEngine<IrisData, IrisPrediction>(env);
        }

        [GlobalSetup(Target = nameof(MakeSentimentPredictions))]
        public void SetupSentimentPipeline()
        {
            _sentimentExample = new SentimentData()
            {
                SentimentText = "Not a big fan of this."
            };

            string _sentimentDataPath = Program.GetInvariantCultureDataPath("wikipedia-detox-250-line-data.tsv");

            var env = new MLContext(seed: 1, conc: 1);
            var reader = new TextLoader(env, columns: new[]
                        {
                            new TextLoader.Column("Label", DataKind.BL, 0),
                            new TextLoader.Column("SentimentText", DataKind.Text, 1)
                        },
                        hasHeader: true                        
                    );

            IDataView data = reader.Read(_sentimentDataPath);

            var pipeline = new TextFeaturizingEstimator(env, "SentimentText", "Features")
                .Append(new SdcaBinaryTrainer(env, "Label", "Features", advancedSettings: (s) => { s.NumThreads = 1; s.ConvergenceTolerance = 1e-2f; }));

            var model = pipeline.Fit(data);

            _sentimentModel = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(env);
        }

        [GlobalSetup(Target = nameof(MakeBreastCancerPredictions))]
        public void SetupBreastCancerPipeline()
        {
            _breastCancerExample = new BreastCancerData()
            {
                Features = new[] { 5f, 1f, 1f, 1f, 2f, 1f, 3f, 1f, 1f }
            };

            string _breastCancerDataPath = Program.GetInvariantCultureDataPath("breast-cancer.txt");

            var env = new MLContext(seed: 1, conc: 1);
            var reader = new TextLoader(env, columns: new[]
                        {
                            new TextLoader.Column("Label", DataKind.BL, 0),
                            new TextLoader.Column("Features", DataKind.R4, new[] { new TextLoader.Range(1, 9) })
                        }, 
                        hasHeader: false
                    );

            IDataView data = reader.Read(_breastCancerDataPath);

            var pipeline = new SdcaBinaryTrainer(env, "Label", "Features", advancedSettings: (s) => { s.NumThreads = 1; s.ConvergenceTolerance = 1e-2f; });

            var model = pipeline.Fit(data);

            _breastCancerModel = model.CreatePredictionEngine<BreastCancerData, BreastCancerPrediction>(env);
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
        [ColumnName("Label"), Column("0")]
        public bool Label;

        [ColumnName("Features"), Column("1-9"), VectorType(9)]
        public float[] Features;
    }

    public class BreastCancerPrediction
    {
        [ColumnName("Score")]
        public float Score;
    }
}
