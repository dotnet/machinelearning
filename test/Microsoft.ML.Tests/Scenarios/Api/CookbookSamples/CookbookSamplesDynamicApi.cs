﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Normalizers;
using Microsoft.ML.Transforms.Text;
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

            // Read the data into a data view.
            var data = mlContext.Data.ReadFromTextFile<InspectedRow>(dataPath,
                // First line of the file is a header, not a data row.
                hasHeader: true
            );

            // Start creating our processing pipeline. For now, let's just concatenate all the text columns
            // together into one.
            var pipeline = mlContext.Transforms.Concatenate("AllFeatures", "Education", "MaritalStatus");

            // Fit our data pipeline and transform data with it.
            var transformedData = pipeline.Fit(data).Transform(data);

            // 'transformedData' is a 'promise' of data. Let's actually read it.
            var someRows = mlContext
                // Convert to an enumerable of user-defined type. 
                .CreateEnumerable<InspectedRowWithAllFeatures>(transformedData, reuseRowObject: false)
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
            // Read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
            var trainData = mlContext.Data.ReadFromTextFile<AdultData>(trainDataPath,
                // Default separator is tab, but we need a semicolon.
                separatorChar: ';'
,
                // First line of the file is a header, not a data row.
                hasHeader: true);

            // Sometime, caching data in-memory after its first access can save some loading time when the data is going to be used
            // several times somewhere. The caching mechanism is also lazy; it only caches things after being used.
            // User can replace all the subsequently uses of "trainData" with "cachedTrainData". We still use "trainData" because
            // a caching step, which provides the same caching function, will be inserted in the considered "pipeline."
            var cachedTrainData = mlContext.Data.Cache(trainData);

            // Step two: define the learning pipeline. 

            // We 'start' the pipeline with the output of the reader.
            var pipeline =
                // First 'normalize' the data (rescale to be
                // between -1 and 1 for all examples), and then train the model.
                mlContext.Transforms.Normalize("FeatureVector")
                // We add a step for caching data in memory so that the downstream iterative training
                // algorithm can efficiently scan through the data multiple times. Otherwise, the following
                // trainer will read data from disk multiple times. The caching mechanism uses an on-demand strategy.
                // The data accessed in any downstream step will be cached since its first use. In general, you only
                // need to add a caching step before trainable step, because caching is not helpful if the data is
                // only scanned once. This step can be removed if user doesn't have enough memory to store the whole
                // data set. Notice that in the upstream Transforms.Normalize step, we only scan through the data
                // once so adding a caching step before it is not helpful.
                .AppendCacheCheckpoint(mlContext)
                // Add the SDCA regression trainer.
                .Append(mlContext.Regression.Trainers.StochasticDualCoordinateAscent(labelColumnName: "Target", featureColumnName: "FeatureVector"));

            // Step three. Fit the pipeline to the training data.
            var model = pipeline.Fit(trainData);

            // Read the test dataset.
            var testData = mlContext.Data.ReadFromTextFile<AdultData>(testDataPath,
                // Default separator is tab, but we need a semicolon.
                separatorChar: ';'
,
                // First line of the file is a header, not a data row.
                hasHeader: true);

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
            //  Retrieve the training data.
            var trainData = mlContext.Data.ReadFromTextFile<IrisInput>(irisDataPath,
                // Default separator is tab, but the dataset has comma.
                separatorChar: ','
            );

            //Preview the data
            var dataPreview = trainData.Preview();

            // Build the training pipeline.
            var pipeline =
                // Concatenate all the features together into one column 'Features'.
                mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                // Note that the label is text, so it needs to be converted to key.
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"), TransformerScope.TrainTest)
                // Cache data in memory for steps after the cache check point stage.
                .AppendCacheCheckpoint(mlContext)
                // Use the multi-class SDCA model to predict the label using features.
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent());

            // Train the model.
            var trainedModel = pipeline.Fit(trainData);

            // Inspect the model parameters. 
            var modelParameters = trainedModel.LastTransformer.Model as MulticlassLogisticRegressionModelParameters;

            // Get the weights and the numbers of classes
            VBuffer<float>[] weights = default;
            modelParameters.GetWeights(ref weights, out int numClasses);

            // numClasses
            // 3
            // weights
            // {float[4]}       { float[4]}         { float[4]}
            // 2.45233274       0.181766108         -3.05772042
            // 4.61404276       0.0578986146        -4.85828352
            // - 6.934741       -0.0424297452       6.63682
            // - 3.64960361     -4.072106           7.55050659

            // Get the biases
            var biases = modelParameters.GetBiases();
            // 		[0]	1.151999	float
            //      [1]	8.337694	float
            // 		[2]	-9.709775	float

            // Apply the inverse conversion from 'PredictedLabel' column back to string value.
            var finalPipeline = pipeline.Append(mlContext.Transforms.Conversion.MapKeyToValue(("Data", "PredictedLabel")));
            dataPreview = finalPipeline.Preview(trainData);

            return finalPipeline.Fit(trainData);
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
            var predictionFunc = model.CreatePredictionEngine<IrisInput, IrisPrediction>(mlContext);

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

            // Read the training data.
            var trainData = mlContext.Data.ReadFromTextFile<IrisInputAllFeatures>(dataPath,
                // Default separator is tab, but the dataset has comma.
                separatorChar: ','
            );

            // Apply all kinds of standard ML.NET normalization to the raw features.
            var pipeline =
                mlContext.Transforms.Normalize(
                    new NormalizingEstimator.MinMaxColumn("MinMaxNormalized", "Features", fixZero: true),
                    new NormalizingEstimator.MeanVarColumn("MeanVarNormalized", "Features", fixZero: true),
                    new NormalizingEstimator.BinningColumn("BinNormalized", "Features", numBins: 256));

            // Let's train our pipeline of normalizers, and then apply it to the same data.
            var normalizedData = pipeline.Fit(trainData).Transform(trainData);

            // Inspect one column of the resulting dataset.
            var meanVarValues = normalizedData.GetColumn<float[]>(mlContext, "MeanVarNormalized").ToArray();
        }

        [Fact]
        public void Normalization()
            => NormalizationWorkout(GetDataPath("iris.data"));

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
            var reader = mlContext.Data.CreateTextLoader(new[]
                {
                    new TextLoader.Column("IsToxic", DataKind.BL, 0),
                    new TextLoader.Column("Message", DataKind.TX, 1),
                },
                hasHeader: true
            );

            // Read the data.
            var data = reader.Read(dataPath);

            // Inspect the message texts that are read from the file.
            var messageTexts = data.GetColumn<string>(mlContext, "Message").Take(20).ToArray();

            // Apply various kinds of text operations supported by ML.NET.
            var pipeline =
                // One-stop shop to run the full text featurization.
                mlContext.Transforms.Text.FeaturizeText("TextFeatures", "Message")

                // Normalize the message for later transforms
                .Append(mlContext.Transforms.Text.NormalizeText("NormalizedMessage", "Message"))

                // NLP pipeline 1: bag of words.
                .Append(new WordBagEstimator(mlContext, "BagOfWords", "NormalizedMessage"))

                // NLP pipeline 2: bag of bigrams, using hashes instead of dictionary indices.
                .Append(new WordHashBagEstimator(mlContext, "BagOfBigrams","NormalizedMessage", 
                            ngramLength: 2, allLengths: false))

                // NLP pipeline 3: bag of tri-character sequences with TF-IDF weighting.
                .Append(mlContext.Transforms.Text.TokenizeCharacters("MessageChars", "Message"))
                .Append(new NgramExtractingEstimator(mlContext, "BagOfTrichar", "MessageChars", 
                            ngramLength: 3, weighting: NgramExtractingEstimator.WeightingCriteria.TfIdf))

                // NLP pipeline 4: word embeddings.
                .Append(mlContext.Transforms.Text.TokenizeWords("TokenizedMessage", "NormalizedMessage"))
                .Append(mlContext.Transforms.Text.ExtractWordEmbeddings("Embeddings", "TokenizedMessage",
                            WordEmbeddingsExtractingEstimator.PretrainedModelKind.GloVeTwitter25D));

            // Let's train our pipeline, and then apply it to the same data.
            // Note that even on a small dataset of 70KB the pipeline above can take up to a minute to completely train.
            var transformedData = pipeline.Fit(data).Transform(data);

            // Inspect some columns of the resulting dataset.
            var embeddings = transformedData.GetColumn<float[]>(mlContext, "Embeddings").Take(10).ToArray();
            var unigrams = transformedData.GetColumn<float[]>(mlContext, "BagOfWords").Take(10).ToArray();
        }

        [Fact(Skip = "This test is running for one minute")]
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
            var reader = mlContext.Data.CreateTextLoader(new[]
                {
                    new TextLoader.Column("Label", DataKind.BL, 0),
                    // We will load all the categorical features into one vector column of size 8.
                    new TextLoader.Column("CategoricalFeatures", DataKind.TX, 1, 8),
                    // Similarly, load all numerical features into one vector of size 6.
                    new TextLoader.Column("NumericalFeatures", DataKind.R4, 9, 14),
                    // Let's also separately load the 'Workclass' column.
                    new TextLoader.Column("Workclass", DataKind.TX, 1),
                },
                hasHeader: true
            );

            // Read the data.
            var data = reader.Read(dataPath);

            // Inspect the first 10 records of the categorical columns to check that they are correctly read.
            var catColumns = data.GetColumn<string[]>(mlContext, "CategoricalFeatures").Take(10).ToArray();

            // Build several alternative featurization pipelines.
            var pipeline =
                // Convert each categorical feature into one-hot encoding independently.
                mlContext.Transforms.Categorical.OneHotEncoding("CategoricalOneHot", "CategoricalFeatures")
                // Convert all categorical features into indices, and build a 'word bag' of these.
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("CategoricalBag", "CategoricalFeatures", OneHotEncodingTransformer.OutputKind.Bag))
                // One-hot encode the workclass column, then drop all the categories that have fewer than 10 instances in the train set.
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("WorkclassOneHot", "Workclass"))
                .Append(mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnCount("WorkclassOneHotTrimmed", "WorkclassOneHot", count: 10));

            // Let's train our pipeline, and then apply it to the same data.
            var transformedData = pipeline.Fit(data).Transform(data);

            // Inspect some columns of the resulting dataset.
            var categoricalBags = transformedData.GetColumn<float[]>(mlContext, "CategoricalBag").Take(10).ToArray();
            var workclasses = transformedData.GetColumn<float[]>(mlContext, "WorkclassOneHotTrimmed").Take(10).ToArray();

            // Of course, if we want to train the model, we will need to compose a single float vector of all the features.
            // Here's how we could do this:

            var fullLearningPipeline = pipeline
                // Concatenate two of the 3 categorical pipelines, and the numeric features.
                .Append(mlContext.Transforms.Concatenate("Features", "NumericalFeatures", "CategoricalBag", "WorkclassOneHotTrimmed"))
                // Cache data in memory so that the following trainer will be able to access training examples without
                // reading them from disk multiple times.
                .AppendCacheCheckpoint(mlContext)
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
            var data = mlContext.Data.ReadFromTextFile<IrisInput>(dataPath,
                // Default separator is tab, but the dataset has comma.
                separatorChar: ','
            );

            // Build the training pipeline.
            var pipeline =
                // Concatenate all the features together into one column 'Features'.
                mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                // Note that the label is text, so it needs to be converted to key.
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"), TransformerScope.TrainTest)
                // Cache data in memory so that SDCA trainer will be able to randomly access training examples without
                // reading data from disk multiple times. Data will be cached at its first use in any downstream step.
                // Notice that unused part in the data may not be cached.
                .AppendCacheCheckpoint(mlContext)
                // Use the multi-class SDCA model to predict the label using features.
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent());

            // Split the data 90:10 into train and test sets, train and evaluate.
            var split = mlContext.MulticlassClassification.TrainTestSplit(data, testFraction: 0.1);

            // Train the model.
            var model = pipeline.Fit(split.TrainSet);
            // Compute quality metrics on the test set.
            var metrics = mlContext.MulticlassClassification.Evaluate(model.Transform(split.TestSet));
            Console.WriteLine(metrics.AccuracyMicro);

            // Now run the 5-fold cross-validation experiment, using the same pipeline.
            var cvResults = mlContext.MulticlassClassification.CrossValidate(data, pipeline, numFolds: 5);

            // The results object is an array of 5 elements. For each of the 5 folds, we have metrics, model and scored test data.
            // Let's compute the average micro-accuracy.
            var microAccuracies = cvResults.Select(r => r.Metrics.AccuracyMicro);
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

            // Now read the file (remember though, readers are lazy, so the actual reading will happen when the data is accessed).
            var reader = mlContext.Data.ReadFromTextFile<AdultData>(dataPath,
                // Default separator is tab, but we need a comma.
                separatorChar: ',');
        }

        // Define a class for all the input columns that we intend to consume.
        public class InputRow
        {
            public float Income { get; set; }
        }

        // Define a class for all output columns that we intend to produce.
        public class OutputRow
        {
            public bool Label { get; set; }
        }

        [Fact]
        public void CustomTransformer()
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.ReadFromTextFile(GetDataPath("adult.tiny.with-schema.txt"), new[]
            {
                new TextLoader.Column("Income", DataKind.R4, 10),
                new TextLoader.Column("Features", DataKind.R4, 12, 14)
            }, hasHeader: true);

            PrepareData(mlContext, data);
            TrainModel(mlContext, data);

            RunEndToEnd(mlContext, data, DeleteOutputPath("custom-model.zip"));
        }

        /// <summary>
        /// One class that contains the custom mapping functionality that we need for our model.
        /// 
        /// It has a <see cref="CustomMappingFactoryAttributeAttribute"/> on it and
        /// derives from <see cref="CustomMappingFactory{TSrc, TDst}"/>.
        /// </summary>
        [CustomMappingFactoryAttribute(nameof(CustomMappings.IncomeMapping))]
        public class CustomMappings : CustomMappingFactory<InputRow, OutputRow>
        {
            // This is the custom mapping. We now separate it into a method, so that we can use it both in training and in loading.
            public static void IncomeMapping(InputRow input, OutputRow output) => output.Label = input.Income > 50000;

            // This factory method will be called when loading the model to get the mapping operation.
            public override Action<InputRow, OutputRow> GetMapping()
            {
                return IncomeMapping;
            }
        }

        private static void RunEndToEnd(MLContext mlContext, IDataView trainData, string modelPath)
        {
            // Construct the learning pipeline. Note that we are now providing a contract name for the custom mapping:
            // otherwise we will not be able to save the model.
            var estimator = mlContext.Transforms.CustomMapping<InputRow, OutputRow>(CustomMappings.IncomeMapping, nameof(CustomMappings.IncomeMapping))
                .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumn: "Label"));

            // If memory is enough, we can cache the data in-memory to avoid reading them from file
            // when it will be accessed multiple times. 
            var cachedTrainData = mlContext.Data.Cache(trainData);

            // Train the model.
            var model = estimator.Fit(cachedTrainData);

            // Save the model.
            using (var fs = File.Create(modelPath))
                mlContext.Model.Save(model, fs);

            // Now pretend we are in a different process.
            var newContext = new MLContext();

            // Register the assembly that contains 'CustomMappings' with the ComponentCatalog
            // so it can be found when loading the model.
            newContext.ComponentCatalog.RegisterAssembly(typeof(CustomMappings).Assembly);

            // Now we can load the model.
            ITransformer loadedModel;
            using (var fs = File.OpenRead(modelPath))
                loadedModel = newContext.Model.Load(fs);
        }

        public static IDataView PrepareData(MLContext mlContext, IDataView data)
        {
            // Define the operation code.
            Action<InputRow, OutputRow> mapping = (input, output) => output.Label = input.Income > 50000;
            // Make a custom transformer and transform the data.
            var transformer = mlContext.Transforms.CustomMappingTransformer(mapping, null);
            return transformer.Transform(data);
        }

        public static ITransformer TrainModel(MLContext mlContext, IDataView trainData)
        {
            // Define the custom operation.
            Action<InputRow, OutputRow> mapping = (input, output) => output.Label = input.Income > 50000;
            // Construct the learning pipeline.
            var estimator = mlContext.Transforms.CustomMapping(mapping, null)
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumn: "Label"));

            return estimator.Fit(trainData);
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
            [LoadColumn(0)]
            public bool IsOver50K { get; set; }

            [LoadColumn(1)]
            public string Workclass { get; set; }

            [LoadColumn(2)]
            public string Education { get; set; }

            [LoadColumn(3)]
            public string MaritalStatus { get; set; }

        }

        private class InspectedRowWithAllFeatures : InspectedRow
        {
            public string[] AllFeatures { get; set; }
        }

        private class IrisInput
        {
            // Unfortunately, we still need the dummy 'Label' column to be present.
            [ColumnName("Label"), LoadColumn(4)]
            public string IgnoredLabel { get; set; }

            [LoadColumn(0)]
            public float SepalLength { get; set; }

            [LoadColumn(1)]
            public float SepalWidth { get; set; }

            [LoadColumn(2)]
            public float PetalLength { get; set; }

            [LoadColumn(3)]
            public float PetalWidth { get; set; }
        }

        private class IrisInputAllFeatures
        {
            // Unfortunately, we still need the dummy 'Label' column to be present.
            [ColumnName("Label"), LoadColumn(4)]
            public string IgnoredLabel { get; set; }

            [LoadColumn(0, 3)]
            public float Features { get; set; }
        }

        private class AdultData
        {
            [LoadColumn(0, 10), ColumnName("FeatureVector")]
            public float Features { get; set; }

            [LoadColumn(11)]
            public float Target { get; set; }
        }

    }
}
