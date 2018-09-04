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

            var data = reader.Read(dataPath).AsDynamic;

            var schema = textData.Schema;

            // Verify that the columns are there. 
            // There ought to be at least one column corresponding to the identifiers in the tuple.
            CheckSchemaHasColumn(schema, "area", out int areaIdx);
            CheckSchemaHasColumn(schema, "title", out int titleIdx);
            CheckSchemaHasColumn(schema, "description", out int descriptionIdx);

            // Next verify they have the expected types.
            Assert.Equal(TextType.Instance, schema.GetColumnType(areaIdx));
            Assert.Equal(TextType.Instance, schema.GetColumnType(titleIdx));
            Assert.Equal(TextType.Instance, schema.GetColumnType(descriptionIdx));

            // sefilipi: Do we need to inspect the data, as in Tom's example?	

            var estimator = Estimator.MakeNew(data)
                .Append(row => (
                    // Convert string label to key. 
                    label: row.area.Dictionarize(),
                    // featurizes 'description'
                    description1: row.description.FeaturizeText(),
                    // featurizes 'title'
                    title1: row.title.FeaturizeText()))
                .Append(row => (
                    // Concatenate the two features into a vector.
                    // Do we need to make other use of the TextTransform output columns?
                    features: row.description1.ConcatWith(r.title1),
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

            //create an example issue:
            const string data = "29338	" +
                                "area-System.Net 	" +
                                "Include fragment and query in Uri.LocalPath on Unix	" +
                                "While testing XmlUriResolver, @pjanotti discovered that any segments of a file path following a '#' symbol will be cut out of Uri.LocalPath on Unix. Based on additional tests, this also occurs for the '?' symbol. This is happening because the Unix specific case for local path only uses the path component of the URI:  https://github.com/dotnet/corefx/blob/9e8d443ff78c4f0a9a6bedf7f95961cf96ceff0a/src/System.Private.Uri/src/System/Uri.cs#L1032-L1037    The fix here is to include the fragment and query in LocalPath in the Unix path specific case. This PR enables the test case in XmlUriResolver that uncovered this issues, and adds some additional cases to our URI tests.    Fixes: #28486 ";
            var dataSource = new BytesStreamSource(data);

            var text = TextLoader.CreateReader(env, ctx =>
                                          (area: ctx.LoadText(1),
                                            title: ctx.LoadText(2),
                                            description: ctx.LoadText(3),
                                            dataSource,
                                            useHeader: true));

            // (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) 
            var prediction = model.Predict();
        }
    }
}
