// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Normalizers;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Scenarios.Api.CookbookSamples
{
    /// <summary>
    /// Samples that are written as part of 'ML.NET Cookbook' are also added here as tests.
    /// These tests don't actually test anything, other than the fact that the code compiles and
    /// doesn't throw when it is executed.
    /// </summary>
    public sealed class CookbookSamplesDynamicApi : BaseTestClass
    {
        public CookbookSamplesDynamicApi(ITestOutputHelper output) : base(output)
        {
        }

        private void IntermediateData(string dataPath)
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Create the reader: define the data columns and where to find them in the text file.
            var reader = mlContext.Data.TextReader(new TextLoader.Arguments
            {
                Column = new[] {
                    // A boolean column depicting the 'label'.
                    new TextLoader.Column("IsOver50K", DataKind.BL, 0),
                    // Three text columns.
                    new TextLoader.Column("Workclass", DataKind.TX, 1),
                    new TextLoader.Column("Education", DataKind.TX, 2),
                    new TextLoader.Column("MaritalStatus", DataKind.TX, 3)
                },
                // First line of the file is a header, not a data row.
                HasHeader = true
            });

            // Start creating our processing pipeline. For now, let's just concatenate all the text columns
            // together into one.
            var dynamicPipeline = mlContext.Transforms.Concatenate("AllFeatures", "Education", "MaritalStatus");

            // Let's verify that the data has been read correctly. 
            // First, we read the data file.
            var data = reader.Read(dataPath);

            // Fit our data pipeline and transform data with it.
            var transformedData = dynamicPipeline.Fit(data).Transform(data);

            // 'transformedData' is a 'promise' of data. Let's actually read it.
            var someRows = transformedData
                // Convert to an enumerable of user-defined type. 
                .AsEnumerable<InspectedRow>(mlContext, reuseRowObject: false)
                // Take a couple values as an array.
                .Take(4).ToArray();

            // Extract the 'AllFeatures' column.
            // This will give the entire dataset: make sure to only take several row
            // in case the dataset is huge. The is similar to the static API, except
            // you have to specify the column name and type.
            var featureColumns = transformedData.GetColumn<string[]>(mlContext, "AllFeatures")
                .Take(20).ToArray();
        }

        [Fact]
        public void InspectIntermediateDataGetColumn()
            => IntermediateData(GetDataPath("adult.tiny.with-schema.txt"));

        private void TrainRegression(string trainDataPath, string testDataPath, string modelPath)
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step one: read the data as an IDataView.
            // First, we define the reader: specify the data columns and where to find them in the text file.
            var reader = mlContext.Data.TextReader(new TextLoader.Arguments
            {
                Column = new[] {
                    // We read the first 11 values as a single float vector.
                    new TextLoader.Column("FeatureVector", DataKind.R4, 0, 10),

                    // Separately, read the target variable.
                    new TextLoader.Column("Target", DataKind.R4, 11),
                },
                // First line of the file is a header, not a data row.
                HasHeader = true,
                // Default separator is tab, but we need a semicolon.
                Separator = ";"
            });

            // Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
            var trainData = reader.Read(trainDataPath);

            // Step two: define the learning pipeline. 

            // We 'start' the pipeline with the output of the reader.
            var dynamicPipeline =
                // First 'normalize' the data (rescale to be
                // between -1 and 1 for all examples), and then train the model.
                mlContext.Transforms.Normalize("FeatureVector")
                // Add the SDCA regression trainer.
                .Append(mlContext.Regression.Trainers.StochasticDualCoordinateAscent(label: "Target", features: "FeatureVector"));

            // Step three. Fit the pipeline to the training data.
            var model = dynamicPipeline.Fit(trainData);

            // Read the test dataset.
            var testData = reader.Read(testDataPath);
            // Calculate metrics of the model on the test data.
            var metrics = mlContext.Regression.Evaluate(model.Transform(testData), label: "Target");

            using (var stream = File.Create(modelPath))
            {
                // Saving and loading happens to 'dynamic' models.
                mlContext.Model.Save(model, stream);
            }

            // Potentially, the lines below can be in a different process altogether.

            // When you load the model, it's a 'dynamic' transformer. 
            ITransformer loadedModel;
            using (var stream = File.OpenRead(modelPath))
                loadedModel = mlContext.Model.Load(stream);
        }

        [Fact]
        public void TrainRegressionModel()
            => TrainRegression(GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename), GetDataPath(TestDatasets.generatedRegressionDataset.testFilename),
                DeleteOutputPath("cook_model.zip"));

        private ITransformer TrainOnIris(string irisDataPath)
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step one: read the data as an IDataView.
            // First, we define the reader: specify the data columns and where to find them in the text file.
            var reader = mlContext.Data.TextReader(new TextLoader.Arguments
            {
                Column = new[] {
                    new TextLoader.Column("SepalLength", DataKind.R4, 0),
                    new TextLoader.Column("SepalWidth", DataKind.R4, 1),
                    new TextLoader.Column("PetalLength", DataKind.R4, 2),
                    new TextLoader.Column("PetalWidth", DataKind.R4, 3),
                    // Label: kind of iris.
                    new TextLoader.Column("Label", DataKind.TX, 4),
                },
                // Default separator is tab, but the dataset has comma.
                Separator = ","
            });

            // Retrieve the training data.
            var trainData = reader.Read(irisDataPath);

            // Build the training pipeline.
            var dynamicPipeline =
                // Concatenate all the features together into one column 'Features'.
                mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                // Note that the label is text, so it needs to be converted to key.
                .Append(mlContext.Transforms.Categorical.MapValueToKey("Label"), TransformerScope.TrainTest)
                // Use the multi-class SDCA model to predict the label using features.
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent())
                // Apply the inverse conversion from 'PredictedLabel' column back to string value.
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(("PredictedLabel", "Data")));

            // Train the model.
            var model = dynamicPipeline.Fit(trainData);
            return model;
        }

        private void PredictOnIris(ITransformer model)
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Use the model for one-time prediction.
            // Make the prediction function object. Note that, on average, this call takes around 200x longer
            // than one prediction, so you might want to cache and reuse the prediction function, instead of
            // creating one per prediction.
            var predictionFunc = model.MakePredictionFunction<IrisInput, IrisPrediction>(mlContext);

            // Obtain the prediction. Remember that 'Predict' is not reentrant. If you want to use multiple threads
            // for simultaneous prediction, make sure each thread is using its own PredictionFunction.
            var prediction = predictionFunc.Predict(new IrisInput
            {
                SepalLength = 4.1f,
                SepalWidth = 0.1f,
                PetalLength = 3.2f,
                PetalWidth = 1.4f
            });
        }

        [Fact]
        public void TrainAndPredictOnIris()
            => PredictOnIris(TrainOnIris(GetDataPath("iris.data")));

        private void NormalizationWorkout(string dataPath)
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Define the reader: specify the data columns and where to find them in the text file.
            var reader = mlContext.Data.TextReader(new TextLoader.Arguments
            {
                Column = new[] {
                    // The four features of the Iris dataset will be grouped together as one Features column.
                    new TextLoader.Column("Features", DataKind.R4, 0, 3),
                    // Label: kind of iris.
                    new TextLoader.Column("Label", DataKind.TX, 4),
                },
                // Default separator is tab, but the dataset has comma.
                Separator = ","
            });

            // Read the training data.
            var trainData = reader.Read(dataPath);

            // Apply all kinds of standard ML.NET normalization to the raw features.
            var pipeline =
                mlContext.Transforms.Normalize(
                    new NormalizingEstimator.MinMaxColumn("Features", "MinMaxNormalized", fixZero: true),
                    new NormalizingEstimator.MeanVarColumn("Features", "MeanVarNormalized", fixZero: true),
                    new NormalizingEstimator.BinningColumn("Features", "BinNormalized", numBins: 256));

            // Let's train our pipeline of normalizers, and then apply it to the same data.
            var normalizedData = pipeline.Fit(trainData).Transform(trainData);

            // Inspect one column of the resulting dataset.
            var meanVarValues = normalizedData.GetColumn<float[]>(mlContext, "MeanVarNormalized").ToArray();
        }

        [Fact]
        public void Normalization()
            => NormalizationWorkout(GetDataPath("iris.data"));

        private class IrisInput
        {
            // Unfortunately, we still need the dummy 'Label' column to be present.
            [ColumnName("Label")]
            public string IgnoredLabel { get; set; }
            public float SepalLength { get; set; }
            public float SepalWidth { get; set; }
            public float PetalLength { get; set; }
            public float PetalWidth { get; set; }
        }

        private IEnumerable<CustomerChurnInfo> GetChurnInfo()
        {
            var r = new Random(454);
            return Enumerable.Range(0, 500)
                .Select(x => new CustomerChurnInfo
                {
                    HasChurned = x % 2 == 0 || (r.NextDouble() < 0.05),
                    DemographicCategory = (x % 10).ToString(),
                    LastVisits = new float[] { x, x * 2, x * 3, x * 4, x * 5 }
                });
        }

        private void TextFeaturizationOn(string dataPath)
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Define the reader: specify the data columns and where to find them in the text file.
            var reader = mlContext.Data.TextReader(new TextLoader.Arguments
            {
                Column = new[] {
                    new TextLoader.Column("IsToxic", DataKind.BL, 0),
                    new TextLoader.Column("Message", DataKind.TX, 1),
                },
                HasHeader = true
            });

            // Read the data.
            var data = reader.Read(dataPath);

            // Inspect the message texts that are read from the file.
            var messageTexts = data.GetColumn<string>(mlContext, "Message").Take(20).ToArray();

            // Apply various kinds of text operations supported by ML.NET.
            var dynamicPipeline =
                // One-stop shop to run the full text featurization.
                mlContext.Transforms.Text.FeaturizeText("Message", "TextFeatures")

                // Normalize the message for later transforms
                .Append(mlContext.Transforms.Text.NormalizeText("Message", "NormalizedMessage"))

                // NLP pipeline 1: bag of words.
                .Append(new WordBagEstimator(mlContext, "NormalizedMessage", "BagOfWords"))

                // NLP pipeline 2: bag of bigrams, using hashes instead of dictionary indices.
                .Append(new WordHashBagEstimator(mlContext, "NormalizedMessage", "BagOfBigrams",
                            ngramLength: 2, allLengths: false))

                // NLP pipeline 3: bag of tri-character sequences with TF-IDF weighting.
                .Append(mlContext.Transforms.Text.TokenizeCharacters("Message", "MessageChars"))
                .Append(new NgramEstimator(mlContext, "MessageChars", "BagOfTrichar",
                            ngramLength: 3, weighting: NgramTransform.WeightingCriteria.TfIdf))

                // NLP pipeline 4: word embeddings.
                .Append(mlContext.Transforms.Text.TokenizeWords("NormalizedMessage", "TokenizedMessage"))
                .Append(mlContext.Transforms.Text.ExtractWordEmbeedings("TokenizedMessage", "Embeddings",
                            WordEmbeddingsTransform.PretrainedModelKind.GloVeTwitter25D));

            // Let's train our pipeline, and then apply it to the same data.
            // Note that even on a small dataset of 70KB the pipeline above can take up to a minute to completely train.
            var transformedData = dynamicPipeline.Fit(data).Transform(data);

            // Inspect some columns of the resulting dataset.
            var embeddings = transformedData.GetColumn<float[]>(mlContext, "Embeddings").Take(10).ToArray();
            var unigrams = transformedData.GetColumn<float[]>(mlContext, "BagOfWords").Take(10).ToArray();
        }

        [Fact (Skip = "This test is running for one minute")]
        public void TextFeaturization()
            => TextFeaturizationOn(GetDataPath("wikipedia-detox-250-line-data.tsv"));

        [Fact]
        public void CategoricalFeaturization()
            => CategoricalFeaturizationOn(GetDataPath("adult.tiny.with-schema.txt"));

        [Fact]
        public void ReadMultipleFiles()
            => CategoricalFeaturizationOn(GetDataPath("adult.tiny.with-schema.txt"), GetDataPath("adult.tiny.with-schema.txt"));

        private void CategoricalFeaturizationOn(params string[] dataPath)
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Define the reader: specify the data columns and where to find them in the text file.
            var reader = mlContext.Data.TextReader(new TextLoader.Arguments
            {
                Column = new[] {
                    new TextLoader.Column("Label", DataKind.BL, 0),
                    // We will load all the categorical features into one vector column of size 8.
                    new TextLoader.Column("CategoricalFeatures", DataKind.TX, 1, 8),
                    // Similarly, load all numerical features into one vector of size 6.
                    new TextLoader.Column("NumericalFeatures", DataKind.R4, 9, 14),
                    // Let's also separately load the 'Workclass' column.
                    new TextLoader.Column("Workclass", DataKind.TX, 1),
                },
                HasHeader = true
            });

            // Read the data.
            var data = reader.Read(dataPath);

            // Inspect the first 10 records of the categorical columns to check that they are correctly read.
            var catColumns = data.GetColumn<string[]>(mlContext, "CategoricalFeatures").Take(10).ToArray();

            // Build several alternative featurization pipelines.
            var dynamicPipeline =
                // Convert each categorical feature into one-hot encoding independently.
                mlContext.Transforms.Categorical.OneHotEncoding("CategoricalFeatures", "CategoricalOneHot")
                // Convert all categorical features into indices, and build a 'word bag' of these.
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("CategoricalFeatures", "CategoricalBag", CategoricalTransform.OutputKind.Bag))
                // One-hot encode the workclass column, then drop all the categories that have fewer than 10 instances in the train set.
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Workclass", "WorkclassOneHot"))
                .Append(new CountFeatureSelector(mlContext, "WorkclassOneHot", "WorkclassOneHotTrimmed", count: 10));

            // Let's train our pipeline, and then apply it to the same data.
            var transformedData = dynamicPipeline.Fit(data).Transform(data);

            // Inspect some columns of the resulting dataset.
            var categoricalBags = transformedData.GetColumn<float[]>(mlContext, "CategoricalBag").Take(10).ToArray();
            var workclasses = transformedData.GetColumn<float[]>(mlContext, "WorkclassOneHotTrimmed").Take(10).ToArray();

            // Of course, if we want to train the model, we will need to compose a single float vector of all the features.
            // Here's how we could do this:

            var fullLearningPipeline = dynamicPipeline
                // Concatenate two of the 3 categorical pipelines, and the numeric features.
                .Append(mlContext.Transforms.Concatenate("Features", "NumericalFeatures", "CategoricalBag", "WorkclassOneHotTrimmed"))
                // Now we're ready to train. We chose our FastTree trainer for this classification task.
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numTrees: 50));

            // Train the model.
            var model = fullLearningPipeline.Fit(data);
        }

        [Fact]
        public void CrossValidationIris()
            => CrossValidationOn(GetDataPath("iris.data"));

        private void CrossValidationOn(string dataPath)
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step one: read the data as an IDataView.
            // First, we define the reader: specify the data columns and where to find them in the text file.
            var reader = mlContext.Data.TextReader(new TextLoader.Arguments
            {
                Column = new[] {
                    // We read the first 11 values as a single float vector.
                    new TextLoader.Column("SepalLength", DataKind.R4, 0),
                    new TextLoader.Column("SepalWidth", DataKind.R4, 1),
                    new TextLoader.Column("PetalLength", DataKind.R4, 2),
                    new TextLoader.Column("PetalWidth", DataKind.R4, 3),
                    // Label: kind of iris.
                    new TextLoader.Column("Label", DataKind.TX, 4),
                },
                // Default separator is tab, but the dataset has comma.
                Separator = ","
            });

            // Read the data.
            var data = reader.Read(dataPath);

            // Build the training pipeline.
            var dynamicPipeline =
                // Concatenate all the features together into one column 'Features'.
                mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                // Note that the label is text, so it needs to be converted to key.
                .Append(mlContext.Transforms.Categorical.MapValueToKey("Label"), TransformerScope.TrainTest)
                // Use the multi-class SDCA model to predict the label using features.
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent());

            // Split the data 90:10 into train and test sets, train and evaluate.
            var (trainData, testData) = mlContext.MulticlassClassification.TrainTestSplit(data, testFraction: 0.1);

            // Train the model.
            var model = dynamicPipeline.Fit(trainData);
            // Compute quality metrics on the test set.
            var metrics = mlContext.MulticlassClassification.Evaluate(model.Transform(testData));
            Console.WriteLine(metrics.AccuracyMicro);

            // Now run the 5-fold cross-validation experiment, using the same pipeline.
            var cvResults = mlContext.MulticlassClassification.CrossValidate(data, dynamicPipeline, numFolds: 5);

            // The results object is an array of 5 elements. For each of the 5 folds, we have metrics, model and scored test data.
            // Let's compute the average micro-accuracy.
            var microAccuracies = cvResults.Select(r => r.metrics.AccuracyMicro);
            Console.WriteLine(microAccuracies.Average());
        }
 
        [Fact]
        public void ReadData()
        {
            ReadDataDynamic(GetDataPath("generated_regression_dataset.csv"));
        }

        private void ReadDataDynamic(string dataPath)
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Create the reader: define the data columns and where to find them in the text file.
            var reader = mlContext.Data.TextReader(new[] {
	                // We read the first 10 values as a single float vector.
                    new TextLoader.Column("FeatureVector", DataKind.R4, new[] {new TextLoader.Range(0, 9)}),
                    // Separately, read the target variable.
                    new TextLoader.Column("Target", DataKind.R4, 10)
                },
                // Default separator is tab, but we need a comma.
                s => s.Separator = ",");

            // Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
            var data = reader.Read(dataPath);
        }

        private class CustomerChurnInfo
        {
            public string CustomerID { get; set; }
            public bool HasChurned { get; set; }
            public string DemographicCategory { get; set; }
            // Visits during last 5 days, latest to newest.
            [VectorType(5)]
            public float[] LastVisits { get; set; }
        }

        private class IrisPrediction
        {
            [ColumnName("Data")]
            public string PredictedClass { get; set; }
        }

        private class InspectedRow
        {
            public bool IsOver50K { get; set; }
            public string Workclass { get; set; }
            public string Education { get; set; }
            public string MaritalStatus { get; set; }
            public string[] AllFeatures { get; set; }
        }
    }
}
