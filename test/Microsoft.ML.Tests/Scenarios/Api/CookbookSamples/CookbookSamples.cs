// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.TestFramework;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;
using Xunit.Abstractions;
using System.Linq;
using Microsoft.ML.Runtime.Training;
using System.IO;
using Microsoft.ML.Core.Data;

namespace Microsoft.ML.Tests.Scenarios.Api.CookbookSamples
{
    /// <summary>
    /// Samples that are written as part of 'ML.NET Cookbook' are also added here as tests.
    /// These tests don't actually test anything, other than the fact that the code compiles and
    /// doesn't throw when it is executed.
    /// </summary>
    public sealed class CookbookSamples : BaseTestClass
    {
        public CookbookSamples(ITestOutputHelper output) : base(output)
        {
        }

        private void IntermediateData(string dataPath)
        {
            // Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var env = new LocalEnvironment();

            // Create the reader: define the data columns and where to find them in the text file.
            var reader = TextLoader.CreateReader(env, ctx => (
                    // A boolean column depicting the 'target label'.
                    IsOver50K: ctx.LoadBool(0),
                    // Three text columns.
                    Workclass: ctx.LoadText(1),
                    Education: ctx.LoadText(2),
                    MaritalStatus: ctx.LoadText(3)),
                hasHeader: true);

            // Start creating our processing pipeline. For now, let's just concatenate all the text columns
            // together into one.
            var dataPipeline = reader.MakeNewEstimator()
                .Append(row =>
                    (
                        row.IsOver50K,
                        AllFeatures: row.Workclass.ConcatWith(row.Education, row.MaritalStatus)
                    ));

            // Let's verify that the data has been read correctly. 
            // First, we read the data file.
            var data = reader.Read(new MultiFileSource(dataPath));

            // Fit our data pipeline and transform data with it.
            var transformedData = dataPipeline.Fit(data).Transform(data);

            // Extract the 'AllFeatures' column.
            // This will give the entire dataset: make sure to only take several row
            // in case the dataset is huge.
            var featureColumns = transformedData.GetColumn(r => r.AllFeatures)
                .Take(20).ToArray();

            // The same extension method also applies to the dynamic-typed data, except you have to
            // specify the column name and type:
            var dynamicData = transformedData.AsDynamic;
            var sameFeatureColumns = dynamicData.GetColumn<string[]>(env, "AllFeatures")
                .Take(20).ToArray();
        }

        [Fact]
        public void InspectIntermediateDataGetColumn()
            => IntermediateData(GetDataPath("adult.tiny.with-schema.txt"));


        private void TrainRegression(string trainDataPath, string testDataPath, string modelPath)
        {
            // Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var env = new LocalEnvironment();

            // Step one: read the data as an IDataView.
            // First, we define the reader: specify the data columns and where to find them in the text file.
            var reader = TextLoader.CreateReader(env, ctx => (
                    // We read the first 11 values as a single float vector.
                    FeatureVector: ctx.LoadFloat(0, 10),
                    // Separately, read the target variable.
                    Target: ctx.LoadFloat(11)
                ),
                // The data file has header.
                hasHeader: true,
                // Default separator is tab, but we need a semicolon.
                separator: ';');


            // Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
            var trainData = reader.Read(new MultiFileSource(trainDataPath));

            // Step two: define the learning pipeline. 
            // We know that this is a regression task, so we create a regression context: it will give us the algorithms
            // we need, as well as the evaluation procedure.
            var regression = new RegressionContext(env);

            // We 'start' the pipeline with the output of the reader.
            var learningPipeline = reader.MakeNewEstimator()
                // Now we can add any 'training steps' to it. In our case we want to 'normalize' the data (rescale to be
                // between -1 and 1 for all examples), and then train the model.
                .Append(r => (
                    // Retain the 'Target' column for evaluation purposes.
                    r.Target,
                    // We choose the SDCA regression trainer. Note that we normalize the 'FeatureVector' right here in
                    // the the same call.
                    Prediction: regression.Trainers.Sdca(label: r.Target, features: r.FeatureVector.Normalize())));

            var fx = trainData.GetColumn(x => x.FeatureVector);

            // Step three. Train the pipeline.
            var model = learningPipeline.Fit(trainData);

            // Read the test dataset.
            var testData = reader.Read(new MultiFileSource(testDataPath));
            // Calculate metrics of the model on the test data.
            // We are using the 'regression' context object here to perform evaluation.
            var metrics = regression.Evaluate(model.Transform(testData), label: r => r.Target, score: r => r.Prediction);

            using (var stream = File.Create(modelPath))
            {
                // Saving and loading happens to 'dynamic' models, so the static typing is lost in the process.
                model.AsDynamic.SaveTo(env, stream);
            }

            // Potentially, the lines below can be in a different process altogether.

            // When you load the model, it's a 'dynamic' transformer. 
            ITransformer loadedModel;
            using (var stream = File.OpenRead(modelPath))
                loadedModel = TransformerChain.LoadFrom(env, stream);
        }

        [Fact]
        public void TrainRegressionModel()
            => TrainRegression(GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename), GetDataPath(TestDatasets.generatedRegressionDataset.testFilename),
                DeleteOutputPath("cook_model.zip"));
    }
}
