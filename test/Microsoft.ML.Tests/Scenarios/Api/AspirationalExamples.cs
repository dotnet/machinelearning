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

        public class IssueInput
        {
            public string Description { get; set; }
            public string Title { get; set; }
        }

        public void FirstExperienceWithML()
        {
            // Load the data into the system.
            string dataPath = "iris.txt";
            var dataReader = TextReader.CreateReader(env, dataPath, row => (
                label: row.LoadText(0),
                sepalWidth: row.LoadFloat(1),
                sepalLength: row.LoadFloat(2),
                petalWidth: row.LoadFloat(3),
                petalLength: row.LoadFloat(4)));


            var pipeline = dataReader.MakeEstimator()
                .Append(row => (
                    // Convert string label to key.
                    label: row.Label.DictionarizeLabel(),
                    // Concatenate all features into a vector.
                    features: row.SepalWidth.ConcatWith(row.sepalLength, row.petalWidth, row.petalLength)))
                .Append(row => row.label.PredictSdcaMultiClass(row.features))
                .Append(row => row.predictedLabel.KeyToValue());

            // Read the data
            var data = reader.Read(dataPath);

            // Fit the data
            var model = estimator.Fit(data);

            var predictor = model.MakePredictionFunction<IrisExample, IrisPrediction>();

            // (Scalar<float> score, Scalar<float> probability, Scalar<string> predictedLabel) 
            var prediction = predictor.PredictSdcaMultiClass(new IrisExample
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
            var env = new LocalEnvironment();
            string dataPath = "iris.txt";
            // Create reader with specific schema.
            var dataReader = TextLoader.CreateReader(env, dataPath,
                ctx => (
                    label: ctx.LoadText(0),
                    sepalWidth: ctx.LoadFloat(1),
                    sepalLength: ctx.LoadFloat(2),
                    petalWidth: ctx.LoadFloat(3),
                    petalLength: ctx.LoadFloat(4)
                ));

            var pipeline = dataReader.MakeEstimator()
                .Append(row => (
                    // Convert string label to key.
                    label: row.label.Dictionarize(),
                    // Concatenate all features into a vector.
                    features: row.sepalWidth.ConcatWith(row.sepalLength, row.petalWidth, row.petalLength)))
                .Append(row => (
                    label: row.label,
                    prediction: row.label.PredictFastTreeMultiClass(row.features)));

            // Load the data into the system.
            var data = dataReader.Read(dataPath);
            var model = pipeline.Fit(data);

            var predictions = model.Transform(dataReader.Read(testDataPath));
            var metrics = MultiClassEvaluator.Evaluate(predictions, row => row.label, row => row.prediction);
        }

        [Fact]
        public void TwitterSentimentAnalysis()
        {
            var env = new LocalEnvironment();
            var dataPath = "wikipedia-detox-250-line-data.tsv";
            // Load the data into the system.
            var dataReader = TextLoader.CreateReader(env, dataPath,
                    ctx => (
                        label: ctx.LoadFloat(0),
                        text: ctx.LoadText(1)),
                    hasHeader: true);

            var pipeline = dataReader.MakeEstimator()
                .Append(row => (
                    label: row.label,
                    // Concatenate all features into a vector.
                    features: row.text.TextFeaturizer()))
                 .Append(row => (
                    label: row.label,
                    row.label.TrainLinearClassification(row.features)));


            var (trainData, testData) = dataReader.Read(dataPath).TrainTestSplit(trainFraction: 0.7);
            var model = pipeline.Fit(trainData);
            var predictions = model.Transform(testData);
            var metrics = BinaryClassifierEvaluator.Evaluate(predictions, row => row.label, row => row.prediction);
        }

        [Fact]
        public void TwentyNewsGroups()
        {
            var env = new LocalEnvironment();
            var dataPath = "20newsGroups.txt";
            // Load the data into the system.
            var dataReader = TextLoader.CreateReader(env, dataPath,
                   ctx => (
                        label: ctx.LoadText(1),
                        subject: ctx.LoadText(1),
                        content: ctx.LoadText(2)),
                   hasHeader: true);

            var pipeline = dataReader.MakeEstimator().
                Append(row => (
                    // Convert string label to key.
                    label: row.label.Dictionarize(),
                    // Concatenate all features into a vector.
                    features: row.subject.ConcatWith(row.content).TextFeaturizer()))
                 .Append(row => (
                    label: row.label,
                    prediction: row.label.PredictSdcaMultiClass(row.features)));

            var (trainData, testData) = dataReader.Read(dataPath).TrainTestSplit(trainFraction: 0.8);
            var model = pipeline.Fit(trainData);

            var predictions = model.Transform(testData);
            var metrics = MultiClassEvaluator.Evaluate(predictions, row => row.label, row => row.prediction);
        }
    }

    public class GithubClassification
    {
        public void ClassifyGithubIssues()
        {
            var env = new LocalEnvironment(new SysRandom(0), verbose: true);

            string dataPath = "corefx-issues-train.tsv";

            // Create reader with specific schema. 
            // string :ID, string: Area, string:Title, string:Description
            var reader = TextLoader.CreateReader(env, ctx =>
                                          (area: ctx.LoadText(1),
                                            title: ctx.LoadText(2),
                                            description: ctx.LoadText(3),
                                            dataPath,
                                            useHeader: true));

            var estimator = reader.MakeEstimator()
                .Append(row => (
                    // Convert string label to key. 
                    label: row.area.Dictionarize(),
                    // Featurizes 'description'
                    description: row.description.FeaturizeText(),
                    // Featurizes 'title'
                    title: row.title.FeaturizeText()))
                .Append(row => (
                    // Concatenate the two features into a vector.
                    features: row.description.ConcatWith(r.title),
                    // Preserve the label
                    label: row.label))
                .Append(row => r.label.PredictSdcaMultiClass(row.features));

            // Read the data
            var data = reader.Read(dataPath);

            // Fit the data
            var model = estimator.Fit(data);

            string modelPath = "github-Model.zip";

            // We don't currently have the WriteAsync
            await model.WriteAsync(modelPath);
        }

        public void PredictGithubIssues()
        {
            ClassifyGithubIssues();

            // We don't currently have the ReadAsync.
            var model = await PredictionModel.ReadAsync(ModelPath);
            var predictor = model.MakePredictionFunction<IssueInput, IssuePrediction>();

            var prediction = predictor.PredictSdcaMultiClass(new IssueInput
                {
                    Title = "Include fragment and query in Uri.LocalPath on Unix\t", 
                    Description = "While testing XmlUriResolver, @pjanotti discovered that any segments of a file path following a '#' symbol will be cut out of Uri.LocalPath on Unix. Based on additional tests, this also occurs for the '?' symbol. This is happening because the Unix specific case for local path only uses the path component of the URI:  https://github.com/dotnet/corefx/blob/9e8d443ff78c4f0a9a6bedf7f95961cf96ceff0a/src/System.Private.Uri/src/System/Uri.cs#L1032-L1037    The fix here is to include the fragment and query in LocalPath in the Unix path specific case. This PR enables the test case in XmlUriResolver that uncovered this issues, and adds some additional cases to our URI tests.    Fixes: #28486 "
                });
        }
    }

    public class SimpleTransform
    {
        public void ScaleData()
        {
            var env = new LocalEnvironment(new SysRandom(0), verbose: true);

            string dataPath = "iris.txt";

            // Create reader with specific schema.
            var dataReader = TextLoader.CreateReader(env, ctx => (
               label: ctx.LoadText(0),
               sepalWidth: ctx.LoadFloat(1),
               sepalLength: ctx.LoadFloat(2),
               petalWidth: ctx.LoadFloat(3),
               petalLength: ctx.LoadFloat(4)),
               dataPath);

            // Define estimator.
            var estimator = dataReader.MakeEstimator()
                 .Append(row => (
                    // Convert string label to key. 
                    sepalWidthNorm: row.SepalWidth.Normalize(Normalizer.NormalizerMode.MinMax),
                    // featurizes 'description'
                    sepalLengthNorm: row.sepalLength.Normalize(Normalizer.NormalizerMode.MeanVariance),
                    //preserve the columns
                    petalWidth: row.petalWidth,
                    petalLength: row.petalLength
                    ));

            // Read the data.
            var data = dataReader.Read(dataPath);

            // Fit/train the model.
            // var is ITransformer id data is IDataView.
            var model = estimator.Fit(data);

            // Let's assume this iris.txt is a file with the same schema as the one previously used
            // but with different values.
            // Should use the filters to split the file in two parts, to make it less confusing. 
            string anotherDataFile = "iris.txt";

            // Apply the model/transformation to the new data.
            var transformedData = model.Transform(dataReader.Read(anotherDataFile));

        }
    }
}
