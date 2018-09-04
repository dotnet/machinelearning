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
            var env = new ConsoleEnvironment();
            string dataPath = "iris.txt";
            // Create reader with specific schema.
            var dataReader = TextLoader.CreateReader(env, ctx => (
               label: ctx.LoadText(0),
               sepalWidth: ctx.LoadFloat(1),
               sepalLength: ctx.LoadFloat(2),
               petalWidth: ctx.LoadFloat(3),
               petalLength: ctx.LoadFloat(4)),
               dataPath);


            var pipeline = data.MakeEstimator()
                .Append(row => (
                    // Convert string label to key.
                    label: row.label.Dictionarize(),
                    // Concatenate all features into a vector.
                    features: row.sepalWidth.ConcatWith(row.sepalLength, row.petalWidth, row.petalLength)))
                .Append(row => (
                    label: row.label,
                    prediction: row.label.PredictWithMultiClassFastTree(row.features)));

            // Load the data into the system.
            var data = dataReader.Read(dataPath);
            var model = pipeline.Fit(data);

            var predictions = model.Transform(dataReader.Read(testDataPath));
            var evaluator = new MultiClassEvaluator(env);
            var metrics = MultiClassEvaluator.Evaluate(predictions, row => row.label, row => row.prediction);
        }

        [Fact]
        public void TwitterSentimentAnalysis()
        {
            var env = new ConsoleEnvironment();
            var dataPath = "wikipedia-detox-250-line-data.tsv";
            // Load the data into the system.
            var dataReader = TextLoader.CreateReader(env, ctx => (
                   label: ctx.LoadFloat(0),
                   text: ctx.LoadText(1)),
                   dataPath, hasHeader: true);

            var pipeline = dataMakeEstimator()
                .Append(row => (
                    label: row.label,
                    // Concatenate all features into a vector.
                    Features: row.text.TextFeaturizer()))
                 .Append(row => (
                    label: row.label,
                    row.label.TrainLinearClassification(row.features)));


            var (trainData, testData) = CrossValidator.TrainTestSplit(env, data: dataReader.Read(dataPath), trainFraction: 0.7);
            var model = pipeline.Fit(trainData);

            var predictions = model.Transform(testData);
            var metrics = BinaryClassifierEvaluator.Evaluate(predictions, row => row.label, row => row.prediction);
        }

        [Fact]
        public void TwentyNewsGroups()
        {
            var env = new ConsoleEnvironment();
            var dataPath = "20newsGroups.txt";
            // Load the data into the system.
            var dataReader = TextLoader.CreateReader(env, ctx => (
                   label: ctx.LoadText(1),
                   subject: ctx.LoadText(1),
                   content: ctx.LoadText(2)),
                   dataPath, hasHeader: true);

            var preprocess = data.MakeEstimator().
                Append(row => (
                    // Convert string label to key.
                    label: row.label.Dictionarize(),
                // Concatenate all features into a vector.
                features: row.subject.Concat(row.content).TextFeaturizer()))
                 .Append(row => (
                    label: row.label,
                    prediction: row.label.TrainSDCAClassifier(row.features)));

            var (trainData, testData) = CrossValidator.TrainTestSplit(env, data: dataReader.Read(dataPath), trainFraction: 0.8);
            var model = pipeline.Fit(trainData);

            var predictions = model.Transform(testData);
            var metrics = MultiClassEvaluator.Evaluate(predictions, row => row.label, row => row.prediction);
        }
    }
}
