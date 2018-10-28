// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.Benchmarks
{
    public class PredictionEngineBench
    {
        private IrisData _irisExample;
        private PredictionFunction<IrisData, IrisPrediction> _irisModel;

        private SentimentData _sentimentExample;
        private PredictionFunction<SentimentData, SentimentPrediction> _sentimentModel;

        private BreastCancerData _breastCancerExample;
        private PredictionFunction<BreastCancerData, BreastCancerPrediction> _breastCancerModel;

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

            using (var env = new ConsoleEnvironment(seed: 1, conc: 1, verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                var reader = new TextLoader(env,
                    new TextLoader.Arguments()
                    {
                        Separator = "\t",
                        HasHeader = true,
                        Column = new[]
                        {
                            new TextLoader.Column("Label", DataKind.R4, 0),
                            new TextLoader.Column("SepalLength", DataKind.R4, 1),
                            new TextLoader.Column("SepalWidth", DataKind.R4, 2),
                            new TextLoader.Column("PetalLength", DataKind.R4, 3),
                            new TextLoader.Column("PetalWidth", DataKind.R4, 4),
                        }
                    });

                IDataView data = reader.Read(_irisDataPath);

                var pipeline = new ConcatEstimator(env, "Features", new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" })
                    .Append(new SdcaMultiClassTrainer(env, "Features", "Label", advancedSettings: (s) => { s.NumThreads = 1; s.ConvergenceTolerance = 1e-2f; }));

                var model = pipeline.Fit(data);

                _irisModel = model.MakePredictionFunction<IrisData, IrisPrediction>(env);
            }
        }

        [GlobalSetup(Target = nameof(MakeSentimentPredictions))]
        public void SetupSentimentPipeline()
        {
            _sentimentExample = new SentimentData()
            {
                SentimentText = "Not a big fan of this."
            };

            string _sentimentDataPath = Program.GetInvariantCultureDataPath("wikipedia-detox-250-line-data.tsv");

            using (var env = new ConsoleEnvironment(seed: 1, conc: 1, verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                var reader = new TextLoader(env,
                    new TextLoader.Arguments()
                    {
                        Separator = "\t",
                        HasHeader = true,
                        Column = new[]
                        {
                            new TextLoader.Column("Label", DataKind.BL, 0),
                            new TextLoader.Column("SentimentText", DataKind.Text, 1)
                        }
                    });

                IDataView data = reader.Read(_sentimentDataPath);

                var pipeline = new TextTransform(env, "SentimentText", "Features")
                    .Append(new SdcaBinaryTrainer(env, "Features", "Label", advancedSettings: (s) => { s.NumThreads = 1; s.ConvergenceTolerance = 1e-2f; }));

                var model = pipeline.Fit(data);

                _sentimentModel = model.MakePredictionFunction<SentimentData, SentimentPrediction>(env);
            }
        }

        [GlobalSetup(Target = nameof(MakeBreastCancerPredictions))]
        public void SetupBreastCancerPipeline()
        {
            _breastCancerExample = new BreastCancerData()
            {
                Features = new[] { 5f, 1f, 1f, 1f, 2f, 1f, 3f, 1f, 1f }
            };

            string _breastCancerDataPath = Program.GetInvariantCultureDataPath("breast-cancer.txt");

            using (var env = new ConsoleEnvironment(seed: 1, conc: 1, verbose: false, sensitivity: MessageSensitivity.None, outWriter: EmptyWriter.Instance))
            {
                var reader = new TextLoader(env,
                    new TextLoader.Arguments()
                    {
                        Separator = "\t",
                        HasHeader = false,
                        Column = new[]
                        {
                            new TextLoader.Column("Label", DataKind.BL, 0),
                            new TextLoader.Column("Features", DataKind.R4, new[] { new TextLoader.Range(1, 9) })
                        }
                    });

                IDataView data = reader.Read(_breastCancerDataPath);

                var pipeline = new SdcaBinaryTrainer(env, "Features", "Label", advancedSettings: (s) => { s.NumThreads = 1; s.ConvergenceTolerance = 1e-2f; });

                var model = pipeline.Fit(data);

                _breastCancerModel = model.MakePredictionFunction<BreastCancerData, BreastCancerPrediction>(env);
            }
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
        [ColumnName("Label"), Column("0")]
        public bool Sentiment;

        [Column("1")]
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
