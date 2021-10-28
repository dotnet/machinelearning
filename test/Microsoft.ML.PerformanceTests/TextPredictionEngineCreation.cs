// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using BenchmarkDotNet.Attributes;
using Microsoft.ML;
using Microsoft.ML.PerformanceTests;
using Microsoft.ML.Trainers;

namespace micro
{
    [SimpleJob]
    public class TextPredictionEngineCreationBenchmark : BenchmarkBase
    {
        private MLContext _context;
        private ITransformer _trainedModel;
        private ITransformer _trainedModelOldFormat;

        [GlobalSetup]
        public void Setup()
        {
            _context = new MLContext(1);
            var data = _context.Data.LoadFromTextFile<SentimentData>(
                GetBenchmarkDataPathAndEnsureData("wikipedia-detox-250-line-data.tsv"), hasHeader: true);

            // Pipeline.
            var pipeline = _context.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(_context)
                .Append(_context.BinaryClassification.Trainers.SdcaNonCalibrated(
                    new SdcaNonCalibratedBinaryTrainer.Options { NumberOfThreads = 1 }));

            // Train.
            var model = pipeline.Fit(data);
            var modelPath = "temp.zip";

            // Save model. 
            _context.Model.Save(model, data.Schema, modelPath);

            // Load model.
            _trainedModel = _context.Model.Load(modelPath, out var inputSchema);

            _trainedModelOldFormat = _context.Model.Load(Path.Combine("TestModels", "SentimentModel.zip"), out inputSchema);
        }

        [Benchmark]
        public PredictionEngine<SentimentData, SentimentPrediction> CreatePredictionEngine()
        {
            return _context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(_trainedModel);
        }

        [Benchmark]
        public PredictionEngine<SentimentData, SentimentPrediction> CreatePredictionEngineFromOldFormat()
        {
            return _context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(_trainedModelOldFormat);
        }
    }
}
