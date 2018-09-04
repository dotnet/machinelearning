// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// This file contains code examples that currently do not even compile. 
// They serve as the reference point of the 'desired user-facing API' for the future work.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public class AspirationalExamples
    {
        public class IrisPrediction
        {
            public string PredictedLabel;
        }

        public class IrisExample
        {
            public float SepalWidth { get; set; }
            public float SepalLength { get; set; }
            public float PetalWidth { get; set; }
            public float PetalLength { get; set; }
        }

        public void FirstExperienceWithML()
        {
            // Load the data into the system.
            string dataPath = "iris-data.txt";
            var data = TextReader.FitAndRead(env, dataPath, row => (
                Label: row.ReadString(0),
                SepalWidth: row.ReadFloat(1),
                SepalLength: row.ReadFloat(2),
                PetalWidth: row.ReadFloat(3),
                PetalLength: row.ReadFloat(4)));


            var preprocess = data.Schema.MakeEstimator(row => (
                // Convert string label to key.
                Label: row.Label.DictionarizeLabel(),
                // Concatenate all features into a vector.
                Features: row.SepalWidth.ConcatWith(row.SepalLength, row.PetalWidth, row.PetalLength)));

            var pipeline = preprocess
                // Append the trainer to the training pipeline.
                .AppendEstimator(row => row.Label.PredictWithSdca(row.Features))
                .AppendEstimator(row => row.PredictedLabel.KeyToValue());

            // Train the model and make some predictions.
            var model = pipeline.Fit<IrisExample, IrisPrediction>(data);

            IrisPrediction prediction = model.Predict(new IrisExample
            {
                SepalWidth = 3.3f,
                SepalLength = 1.6f,
                PetalWidth = 0.2f,
                PetalLength = 5.1f
            });
        }


        [Fact]
        public void SimpleIrisDescisionTrees()
        {
            var env = new TlcEnvironment(new SysRandom(0), verbose: true);
            string dataPath = "iris-data.txt";
            // Create reader with specific schema.
            var dataReader = TextLoader.CreateReader(env, ctx => (
               Label: ctx.LoadText(0),
               SepalWidth: ctx.LoadFloat(1),
               SepalLength: ctx.LoadFloat(2),
               PetalWidth: ctx.LoadFloat(3),
               PetalLength: ctx.LoadFloat(4)),
               dataPath);

            // Load the data into the system.
            var data = dataReader.Read(dataPath);

            var preprocess = data.Schema.MakeEstimator(row => (
                // Convert string label to key.
                Label: row.Label.Dictionarize(),
                // Concatenate all features into a vector.
                Features: row.SepalWidth.ConcatWith(row.SepalLength, row.PetalWidth, row.PetalLength)));


            var pipeline = preprocess
                .Append(row => row.Label.PredictWithDecisionTrees(row.Features))
                // shoul it be BagVectorize instead of KeyToValue?
                .Append(row => row.PredictedLabel.KeyToValue());

            var model = pipeline.Fit(data);

            var predictions = model.Transform(dataReader.Read(testDataPath));
            var evaluator = new MultiClassEvaluator(env);
            var metrics = evaluator.Evaluate(predictions);
        }

        [Fact]
        public void TwitterSentimentAnalysis()
        {
            var env = new TlcEnvironment(new SysRandom(0), verbose: true);
            var dataPath = "wikipedia-detox-250-line-data.tsv";
            // Load the data into the system.
            var data = TextLoader.CreateReader(env, ctx => (
                   Label: ctx.LoadFloat(0),
                   Text: ctx.LoadText(1)),
                   dataPath, hasHeader: true).Read(dataPath);

            var preprocess = data.Schema.MakeEstimator(row => (
                Label: row.Label,
                // Concatenate all features into a vector.
                Features: row.Text.TextFeaturizer()));

            var pipeline = preprocess.
                Append(row => row.Label.TrainLinearClassification(row.Features));

            var (trainData, testData) = CrossValidator.TrainTestSplit(env, data: data, trainFraction: 0.7);
            var model = pipeline.Fit(trainData);
            var predictions = model.Transform(testData);
            var evaluator = new BinaryClassifierEvaluator(env);
            var metrics = evaluator.Evaluate(predictions);
        }

        [Fact]
        public void TwentyNewsGroups()
        {
            var env = new TlcEnvironment(new SysRandom(0), verbose: true);
            var dataPath = "20newsGroups.txt";
            // Load the data into the system.
            var data = TextLoader.CreateReader(env, ctx => (
                   Label: ctx.LoadText(1),
                   Subject: ctx.LoadText(1),
                   Content: ctx.LoadText(2)),
                   dataPath, hasHeader: true).Read(dataPath);

            var preprocess = data.Schema.MakeEstimator(row => (
                // Convert string label to key.
                Label: row.Label.Dictionarize(),
                // Concatenate all features into a vector.
                Features: row.Subject.Concat(row.Content).TextFeaturizer()));

            var pipeline = preprocess.
                Append(row => row.Label.TrainSDCAClassifier(row.Features)).
                Append(row => row.PredictedLabel.KeyToValue());

            var (trainData, testData) = CrossValidator.TrainTestSplit(env, data: data, trainFraction: 0.8);
            var model = pipeline.Fit(trainData);

            var predictions = model.Transform(testData);
            var evaluator = new MultiClassEvaluator(env);
            var metrics = evaluator.Evaluate(predictions);
        }
    }
}
