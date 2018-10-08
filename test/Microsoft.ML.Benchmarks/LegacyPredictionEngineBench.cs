// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Legacy.Trainers;

namespace Microsoft.ML.Benchmarks
{
    public class LegacyPredictionEngineBench
    {
        private IrisData _irisExample;
        private PredictionModel<IrisData, IrisPrediction> _irisModel;

        private SentimentData _sentimentExample;
        private PredictionModel<SentimentData, SentimentPrediction> _sentimentModel;

        private BreastCancerData _breastCancerExample;
        private PredictionModel<BreastCancerData, BreastCancerPrediction> _breastCancerModel;

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

            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(_irisDataPath).CreateFrom<IrisData>(useHeader: true, separator: '\t'));
            pipeline.Add(new ColumnConcatenator("Features", new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" }));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier() { NumThreads = 1, ConvergenceTolerance = 1e-2f });

            _irisModel = pipeline.Train<IrisData, IrisPrediction>();
        }

        [GlobalSetup(Target = nameof(MakeSentimentPredictions))]
        public void SetupSentimentPipeline()
        {
            _sentimentExample = new SentimentData()
            {
                SentimentText = "Not a big fan of this."
            };

            string _sentimentDataPath = Program.GetInvariantCultureDataPath("wikipedia-detox-250-line-data.tsv");

            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(_sentimentDataPath).CreateFrom<SentimentData>(useHeader: true, separator: '\t'));
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
            pipeline.Add(new StochasticDualCoordinateAscentBinaryClassifier() { NumThreads = 1, ConvergenceTolerance = 1e-2f });

            _sentimentModel = pipeline.Train<SentimentData, SentimentPrediction>();
        }

        [GlobalSetup(Target = nameof(MakeBreastCancerPredictions))]
        public void SetupBreastCancerPipeline()
        {
            _breastCancerExample = new BreastCancerData()
            {
                Features = new[] { 5f, 1f, 1f, 1f, 2f, 1f, 3f, 1f, 1f }
            };

            string _breastCancerDataPath = Program.GetInvariantCultureDataPath("breast-cancer.txt");

            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(_breastCancerDataPath).CreateFrom<BreastCancerData>(useHeader: false, separator: '\t'));
            pipeline.Add(new StochasticDualCoordinateAscentBinaryClassifier() { NumThreads = 1, ConvergenceTolerance = 1e-2f });

            _breastCancerModel = pipeline.Train<BreastCancerData, BreastCancerPrediction>();
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
}
