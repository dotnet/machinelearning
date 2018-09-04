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
            string dataPath = "iris.txt";
            var data = TextReader.CreateReader(env, dataPath, row => (
                label: row.LoadText(0),
                sepalWidth: row.LoadFloat(1),
                sepalLength: row.LoadFloat(2),
                petalWidth: row.LoadFloat(3),
                petalLength: row.LoadFloat(4)));


            var preprocess = data.MakeEstimator()
                .Append(row => (
                // Convert string label to key.
                label: row.Label.DictionarizeLabel(),
                // Concatenate all features into a vector.
                features: row.SepalWidth.ConcatWith(row.sepalLength, row.petalWidth, row.petalLength)))
                .Append(row => row.label.PredictWithSdca(row.features))
                .Append(row => row.predictedLabel.KeyToValue());

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
    }

    public class GithubClassification
    {
        public void ClassifyGithubIssues()
        {
            var env = new TlcEnvironment(new SysRandom(0), verbose: true);

            string dataPath = "corefx-issues-train.tsv";

            // Create reader with specific schema. 
            // string :ID, string: Area, string:Title, string:Description
            var reader = TextLoader.CreateReader(env, ctx =>
                                          (area: ctx.LoadText(1),
                                            title: ctx.LoadText(2),
                                            description: ctx.LoadText(3),
                                            dataSource,
                                            useHeader: true));

            var data = reader.Read(dataPath);

            var estimator = data.MakeEstimator()
                .Append(row => (
                    // Convert string label to key. 
                    label: row.area.Dictionarize(),
                    // featurizes 'description'
                    description: row.description.FeaturizeText(),
                    // featurizes 'title'
                    title: row.title.FeaturizeText()))
                .Append(row => (
                    // Concatenate the two features into a vector.
                    features: row.description.ConcatWith(r.title),
                    // preserve the label
                    label: row.label))
                // how do we specify arguments for the trainer?
                .Append(row => r.label.TrainLinearClassification(row.Features));

            var model = estimator.Fit(data);

            string modelPath = "github-Model.zip";

            // we don't currently have the WriteAsync
            await model.WriteAsync(modelPath);
        }

        public void PredictGithubIssues()
        {
            ClassifyGithubIssues();

            // we don't currently have the ReadAsync
            var model = await PredictionModel.ReadAsync(ModelPath);
            var predictor = model.MakePredictionFunction<IssueInput, IssuePrediction>();

            // (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) 
            var prediction = predictor.Predict(new IssueInput
                {
                    ID = "29338\t",
                    Area = "area-System.Net\t", 
                    Title = "Include fragment and query in Uri.LocalPath on Unix\t", 
                    Description = "While testing XmlUriResolver, @pjanotti discovered that any segments of a file path following a '#' symbol will be cut out of Uri.LocalPath on Unix. Based on additional tests, this also occurs for the '?' symbol. This is happening because the Unix specific case for local path only uses the path component of the URI:  https://github.com/dotnet/corefx/blob/9e8d443ff78c4f0a9a6bedf7f95961cf96ceff0a/src/System.Private.Uri/src/System/Uri.cs#L1032-L1037    The fix here is to include the fragment and query in LocalPath in the Unix path specific case. This PR enables the test case in XmlUriResolver that uncovered this issues, and adds some additional cases to our URI tests.    Fixes: #28486 "
                });
        }
    }

    public class SimpleTransform
    {
        public void ScaleData()
        {
            var env = new TlcEnvironment(new SysRandom(0), verbose: true);

            string dataPath = "iris.txt";

            // Create reader with specific schema.
            var dataReader = TextLoader.CreateReader(env, ctx => (
               label: ctx.LoadText(0),
               sepalWidth: ctx.LoadFloat(1),
               sepalLength: ctx.LoadFloat(2),
               petalWidth: ctx.LoadFloat(3),
               petalLength: ctx.LoadFloat(4)),
               dataPath);

            // Define estimator
            var estimator = data.MakeEstimator()
                 .Append(row => (
                    // Convert string label to key. 
                    sepalWidthNorm: row.SepalWidth.Normalize(Normalizer.NormalizerMode.MinMax),
                    // featurizes 'description'
                    sepalLengthNorm: row.sepalLength.Normalize(Normalizer.NormalizerMode.MeanVariance),
                    //preserve the columns
                    petalWidth: row.petalWidth,
                    petalLength: row.petalLength
                    ));

            // read the data
            var data = dataReader.Read(dataPath);

            // fit/train the model
            //var is ITransformer id data is IDataView
            var model = estimator.Fit(data);

            // let's assume this iris-data.txt is a file with the same schema as the one previously used
            // but with different values.
            // should use the filters to split the file in two parts, to make it less confusing. 
            string anotherDataFile = "iris.txt";

            // apply the model/transformation to the new data
            var transformedData = model.Transform(dataReader.Read(anotherDataFile));

        }
    }
}
