// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Conversions;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
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
    public sealed class CookbookSamples : BaseTestClass
    {
        public CookbookSamples(ITestOutputHelper output) : base(output)
        {
        }

        private void IntermediateData(string dataPath)
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Create the reader: define the data columns and where to find them in the text file.
            var reader = mlContext.Data.TextReader(ctx => (
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
                .Append(row => (
                        row.IsOver50K,
                        AllFeatures: row.Workclass.ConcatWith(row.Education, row.MaritalStatus)
                    ));

            // Let's verify that the data has been read correctly. 
            // First, we read the data file.
            var data = reader.Read(dataPath);

            // Fit our data pipeline and transform data with it.
            var transformedData = dataPipeline.Fit(data).Transform(data);

            // 'transformedData' is a 'promise' of data. Let's actually read it.
            var someRows = transformedData.AsDynamic
                // Convert to an enumerable of user-defined type. 
                .AsEnumerable<InspectedRow>(mlContext, reuseRowObject: false)
                // Take a couple values as an array.
                .Take(4).ToArray();

            // Extract the 'AllFeatures' column.
            // This will give the entire dataset: make sure to only take several row
            // in case the dataset is huge.
            var featureColumns = transformedData.GetColumn(r => r.AllFeatures)
                .Take(20).ToArray();

            // The same extension method also applies to the dynamic-typed data, except you have to
            // specify the column name and type:
            var dynamicData = transformedData.AsDynamic;
            var sameFeatureColumns = dynamicData.GetColumn<string[]>(mlContext, "AllFeatures")
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
            var reader = mlContext.Data.TextReader(ctx => (
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
            var trainData = reader.Read(trainDataPath);

            // Step two: define the learning pipeline. 

            // We 'start' the pipeline with the output of the reader.
            var learningPipeline = reader.MakeNewEstimator()
                // Now we can add any 'training steps' to it. In our case we want to 'normalize' the data (rescale to be
                // between -1 and 1 for all examples), and then train the model.
                .Append(r => (
                    // Retain the 'Target' column for evaluation purposes.
                    r.Target,
                    // We choose the SDCA regression trainer. Note that we normalize the 'FeatureVector' right here in
                    // the the same call.
                    Prediction: mlContext.Regression.Trainers.Sdca(label: r.Target, features: r.FeatureVector.Normalize())));

            var fx = trainData.GetColumn(x => x.FeatureVector);

            // Step three. Train the pipeline.
            var model = learningPipeline.Fit(trainData);

            // Read the test dataset.
            var testData = reader.Read(testDataPath);
            // Calculate metrics of the model on the test data.
            var metrics = mlContext.Regression.Evaluate(model.Transform(testData), label: r => r.Target, score: r => r.Prediction);

            using (var stream = File.Create(modelPath))
            {
                // Saving and loading happens to 'dynamic' models, so the static typing is lost in the process.
                mlContext.Model.Save(model.AsDynamic, stream);
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
            var reader = mlContext.Data.TextReader(ctx => (
                    // The four features of the Iris dataset.
                    SepalLength: ctx.LoadFloat(0),
                    SepalWidth: ctx.LoadFloat(1),
                    PetalLength: ctx.LoadFloat(2),
                    PetalWidth: ctx.LoadFloat(3),
                    // Label: kind of iris.
                    Label: ctx.LoadText(4)
                ),
                // Default separator is tab, but the dataset has comma.
                separator: ',');

            // Retrieve the training data.
            var trainData = reader.Read(irisDataPath);

            // Build the training pipeline.
            var learningPipeline = reader.MakeNewEstimator()
                .Append(r => (
                    r.Label,
                    // Concatenate all the features together into one column 'Features'.
                    Features: r.SepalLength.ConcatWith(r.SepalWidth, r.PetalLength, r.PetalWidth)))
                .Append(r => (
                    r.Label,
                    // Train the multi-class SDCA model to predict the label using features.
                    // Note that the label is a text, so it needs to be converted to key using 'ToKey' estimator.
                    Predictions: mlContext.MulticlassClassification.Trainers.Sdca(r.Label.ToKey(), r.Features)))
                // Apply the inverse conversion from 'predictedLabel' key back to string value.
                // Note that the final output column is only one, and we didn't assign a name to it.
                // In this case, ML.NET auto-assigns the name 'Data' to the produced column.
                .Append(r => r.Predictions.predictedLabel.ToValue());

            // Train the model.
            var model = learningPipeline.Fit(trainData).AsDynamic;
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

        private void TrainAndInspectWeights(string dataPath)
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step one: read the data as an IDataView.
            // First, we define the reader: specify the data columns and where to find them in the text file.
            var reader = mlContext.Data.TextReader(ctx => (
                    // The four features of the Iris dataset.
                    SepalLength: ctx.LoadFloat(0),
                    SepalWidth: ctx.LoadFloat(1),
                    PetalLength: ctx.LoadFloat(2),
                    PetalWidth: ctx.LoadFloat(3),
                    // Label: kind of iris.
                    Label: ctx.LoadText(4)
                ),
                // Default separator is tab, but the dataset has comma.
                separator: ',');

            // Retrieve the training data.
            var trainData = reader.Read(dataPath);

            // This is the predictor ('weights collection') that we will train.
            MulticlassLogisticRegressionPredictor predictor = null;
            // And these are the normalizer scales that we will learn.
            ImmutableArray<float> normScales;
            // Build the training pipeline.
            var learningPipeline = reader.MakeNewEstimator()
                .Append(r => (
                    r.Label,
                    // Concatenate all the features together into one column 'Features'.
                    Features: r.SepalLength.ConcatWith(r.SepalWidth, r.PetalLength, r.PetalWidth)))
                .Append(r => (
                    r.Label,
                    // Normalize (rescale) the features to be between -1 and 1. 
                    Features: r.Features.Normalize(
                        // When the normalizer is trained, the below delegate is going to be called.
                        // We use it to memorize the scales.
                        onFit: (scales, offsets) => normScales = scales)))
                .Append(r => (
                    r.Label,
                    // Train the multi-class SDCA model to predict the label using features.
                    // Note that the label is a text, so it needs to be converted to key using 'ToKey' estimator.
                    Predictions: mlContext.MulticlassClassification.Trainers.Sdca(r.Label.ToKey(), r.Features,
                        // When the model is trained, the below delegate is going to be called.
                        // We use that to memorize the predictor object.
                        onFit: p => predictor = p)));

            // Train the model. During this call our 'onFit' delegate will be invoked,
            // and our 'predictor' will be set.
            var model = learningPipeline.Fit(trainData);

            // Now we can use 'predictor' to look at the weights.
            // 'weights' will be an array of weight vectors, one vector per class.
            // Our problem has 3 classes, so numClasses will be 3, and weights will contain
            // 3 vectors (of 4 values each).
            VBuffer<float>[] weights = null;
            predictor.GetWeights(ref weights, out int numClasses);

            // Similarly we can also inspect the biases for the 3 classes.
            var biases = predictor.GetBiases();

            // Inspect the normalizer scales.
            Console.WriteLine(string.Join(" ", normScales));
        }

        [Fact]
        public void InspectModelWeights()
            => TrainAndInspectWeights(GetDataPath("iris.data"));

        private void NormalizationWorkout(string dataPath)
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Define the reader: specify the data columns and where to find them in the text file.
            var reader = mlContext.Data.TextReader(ctx => (
                    // The four features of the Iris dataset will be grouped together as one Features column.
                    Features: ctx.LoadFloat(0, 3),
                    // Label: kind of iris.
                    Label: ctx.LoadText(4)
                ),
                // Default separator is tab, but the dataset has comma.
                separator: ',');

            // Read the training data.
            var trainData = reader.Read(dataPath);

            // Apply all kinds of standard ML.NET normalization to the raw features.
            var pipeline = reader.MakeNewEstimator()
                .Append(r => (
                    MinMaxNormalized: r.Features.Normalize(fixZero: true),
                    MeanVarNormalized: r.Features.NormalizeByMeanVar(fixZero: false),
                    CdfNormalized: r.Features.NormalizeByCumulativeDistribution(),
                    BinNormalized: r.Features.NormalizeByBinning(maxBins: 256)
                ));

            // Let's train our pipeline of normalizers, and then apply it to the same data.
            var normalizedData = pipeline.Fit(trainData).Transform(trainData);

            // Inspect one column of the resulting dataset.
            var meanVarValues = normalizedData.GetColumn(r => r.MeanVarNormalized).ToArray();
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

        [Fact]
        public void TrainOnAutoGeneratedData()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step one: read the data as an IDataView.
            // Let's assume that 'GetChurnData()' fetches and returns the training data from somewhere.
            IEnumerable<CustomerChurnInfo> churnData = GetChurnInfo();

            // Turn the data into the ML.NET data view.
            // We can use CreateDataView or CreateStreamingDataView, depending on whether 'churnData' is an IList, 
            // or merely an IEnumerable.
            var trainData = mlContext.CreateStreamingDataView(churnData);

            // Now note that 'trainData' is just an IDataView, so we face a choice here: either declare the static type
            // and proceed in the statically typed fashion, or keep dynamic types and build a dynamic pipeline.
            // We demonstrate both below.

            // Build the learning pipeline. 
            // In our case, we will one-hot encode the demographic category, and concatenate that with the number of visits.
            // We apply our FastTree binary classifier to predict the 'HasChurned' label.

            var dynamicLearningPipeline = mlContext.Transforms.Categorical.OneHotEncoding("DemographicCategory")
                .Append(new ColumnConcatenatingEstimator (mlContext, "Features", "DemographicCategory", "LastVisits"))
                .Append(mlContext.BinaryClassification.Trainers.FastTree("HasChurned", "Features", numTrees: 20));

            var dynamicModel = dynamicLearningPipeline.Fit(trainData);

            // Build the same learning pipeline, but statically typed.
            // First, transition to the statically-typed data view.
            var staticData = trainData.AssertStatic(mlContext, c => (
                    HasChurned: c.Bool.Scalar,
                    DemographicCategory: c.Text.Scalar,
                    LastVisits: c.R4.Vector));

            // Build the pipeline, same as the one above.
            var staticLearningPipeline = staticData.MakeNewEstimator()
                .Append(r => (
                    r.HasChurned,
                    Features: r.DemographicCategory.OneHotEncoding().ConcatWith(r.LastVisits)))
                .Append(r => mlContext.BinaryClassification.Trainers.FastTree(r.HasChurned, r.Features, numTrees: 20));

            var staticModel = staticLearningPipeline.Fit(staticData);

            // Note that dynamicModel should be the same as staticModel.AsDynamic (give or take random variance from
            // the training procedure).

            var qualityMetrics = mlContext.BinaryClassification.Evaluate(dynamicModel.Transform(trainData), "HasChurned");
        }

        private void TextFeaturizationOn(string dataPath)
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Define the reader: specify the data columns and where to find them in the text file.
            var reader = mlContext.Data.TextReader(ctx => (
                    IsToxic: ctx.LoadBool(0),
                    Message: ctx.LoadText(1)
                ), hasHeader: true);

            // Read the data.
            var data = reader.Read(dataPath);

            // Inspect the message texts that are read from the file.
            var messageTexts = data.GetColumn(x => x.Message).Take(20).ToArray();

            // Apply various kinds of text operations supported by ML.NET.
            var learningPipeline = reader.MakeNewEstimator()
                .Append(r => (
                    // One-stop shop to run the full text featurization.
                    TextFeatures: r.Message.FeaturizeText(),

                    // NLP pipeline 1: bag of words.
                    BagOfWords: r.Message.NormalizeText().ToBagofWords(),

                    // NLP pipeline 2: bag of bigrams, using hashes instead of dictionary indices.
                    BagOfBigrams: r.Message.NormalizeText().ToBagofHashedWords(ngramLength: 2, allLengths: false),

                    // NLP pipeline 3: bag of tri-character sequences with TF-IDF weighting.
                    BagOfTrichar: r.Message.TokenizeIntoCharacters().ToNgrams(ngramLength: 3, weighting: NgramTransform.WeightingCriteria.TfIdf),

                    // NLP pipeline 4: word embeddings.
                    Embeddings: r.Message.NormalizeText().TokenizeText().WordEmbeddings(WordEmbeddingsTransform.PretrainedModelKind.GloVeTwitter25D)
                ));

            // Let's train our pipeline, and then apply it to the same data.
            // Note that even on a small dataset of 70KB the pipeline above can take up to a minute to completely train.
            var transformedData = learningPipeline.Fit(data).Transform(data);

            // Inspect some columns of the resulting dataset.
            var embeddings = transformedData.GetColumn(x => x.Embeddings).Take(10).ToArray();
            var unigrams = transformedData.GetColumn(x => x.BagOfWords).Take(10).ToArray();
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
            var reader = mlContext.Data.TextReader(ctx => (
                    Label: ctx.LoadBool(0),
                    // We will load all the categorical features into one vector column of size 8.
                    CategoricalFeatures: ctx.LoadText(1, 8),
                    // Similarly, load all numerical features into one vector of size 6.
                    NumericalFeatures: ctx.LoadFloat(9, 14),
                    // Let's also separately load the 'Workclass' column.
                    Workclass: ctx.LoadText(1)
                ), hasHeader: true);

            // Read the data.
            var data = reader.Read(dataPath);

            // Inspect the categorical columns to check that they are correctly read.
            var catColumns = data.GetColumn(r => r.CategoricalFeatures).Take(10).ToArray();

            // Build several alternative featurization pipelines.
            var learningPipeline = reader.MakeNewEstimator()
                .Append(r => (
                    r.Label,
                    r.NumericalFeatures,
                    // Convert each categorical feature into one-hot encoding independently.
                    CategoricalOneHot: r.CategoricalFeatures.OneHotEncoding(outputKind: CategoricalStaticExtensions.OneHotVectorOutputKind.Ind),
                    // Convert all categorical features into indices, and build a 'word bag' of these.
                    CategoricalBag: r.CategoricalFeatures.OneHotEncoding(outputKind: CategoricalStaticExtensions.OneHotVectorOutputKind.Bag),
                    // One-hot encode the workclass column, then drop all the categories that have fewer than 10 instances in the train set.
                    WorkclassOneHotTrimmed: r.Workclass.OneHotEncoding().SelectFeaturesBasedOnCount(count: 10)
                ));

            // Let's train our pipeline, and then apply it to the same data.
            var transformedData = learningPipeline.Fit(data).Transform(data);

            // Inspect some columns of the resulting dataset.
            var categoricalBags = transformedData.GetColumn(x => x.CategoricalBag).Take(10).ToArray();
            var workclasses = transformedData.GetColumn(x => x.WorkclassOneHotTrimmed).Take(10).ToArray();

            // Of course, if we want to train the model, we will need to compose a single float vector of all the features.
            // Here's how we could do this:

            var fullLearningPipeline = learningPipeline
                .Append(r => (
                    r.Label,
                    // Concatenate two of the 3 categorical pipelines, and the numeric features.
                    Features: r.NumericalFeatures.ConcatWith(r.CategoricalBag, r.WorkclassOneHotTrimmed)))
                // Now we're ready to train. We chose our FastTree trainer for this classification task.
                .Append(r => mlContext.BinaryClassification.Trainers.FastTree(r.Label, r.Features, numTrees: 50));

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
            var reader = mlContext.Data.TextReader(ctx => (
                    // The four features of the Iris dataset.
                    SepalLength: ctx.LoadFloat(0),
                    SepalWidth: ctx.LoadFloat(1),
                    PetalLength: ctx.LoadFloat(2),
                    PetalWidth: ctx.LoadFloat(3),
                    // Label: kind of iris.
                    Label: ctx.LoadText(4)
                ),
                // Default separator is tab, but the dataset has comma.
                separator: ',');

            // Read the data.
            var data = reader.Read(dataPath);

            // Build the training pipeline.
            var learningPipeline = reader.MakeNewEstimator()
                .Append(r => (
                    // Convert string label to a key.
                    Label: r.Label.ToKey(),
                    // Concatenate all the features together into one column 'Features'.
                    Features: r.SepalLength.ConcatWith(r.SepalWidth, r.PetalLength, r.PetalWidth)))
                .Append(r => (
                    r.Label,
                    // Train the multi-class SDCA model to predict the label using features.
                    Predictions: mlContext.MulticlassClassification.Trainers.Sdca(r.Label, r.Features)));

            // Split the data 90:10 into train and test sets, train and evaluate.
            var (trainData, testData) = mlContext.MulticlassClassification.TrainTestSplit(data, testFraction: 0.1);

            // Train the model.
            var model = learningPipeline.Fit(trainData);
            // Compute quality metrics on the test set.
            var metrics = mlContext.MulticlassClassification.Evaluate(model.Transform(testData), r => r.Label, r => r.Predictions);
            Console.WriteLine(metrics.AccuracyMicro);

            // Now run the 5-fold cross-validation experiment, using the same pipeline.
            var cvResults = mlContext.MulticlassClassification.CrossValidate(data, learningPipeline, r => r.Label, numFolds: 5);

            // The results object is an array of 5 elements. For each of the 5 folds, we have metrics, model and scored test data.
            // Let's compute the average micro-accuracy.
            var microAccuracies = cvResults.Select(r => r.metrics.AccuracyMicro);
            Console.WriteLine(microAccuracies.Average());
        }

        [Fact]
        public void MixAndMatchStaticDynamicOnIris()
            => MixMatch(GetDataPath("iris.data"));

        private void MixMatch(string dataPath)
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Read the data as an IDataView.
            // First, we define the reader: specify the data columns and where to find them in the text file.
            var reader = mlContext.Data.TextReader(ctx => (
                    // The four features of the Iris dataset.
                    SepalLength: ctx.LoadFloat(0),
                    SepalWidth: ctx.LoadFloat(1),
                    PetalLength: ctx.LoadFloat(2),
                    PetalWidth: ctx.LoadFloat(3),
                    // Label: kind of iris.
                    Label: ctx.LoadText(4)
                ),
                // Default separator is tab, but the dataset has comma.
                separator: ',');

            // Read the data.
            var data = reader.Read(dataPath);

            // Build the pre-processing pipeline.
            var learningPipeline = reader.MakeNewEstimator()
                .Append(r => (
                    // Convert string label to a key.
                    Label: r.Label.ToKey(),
                    // Concatenate all the features together into one column 'Features'.
                    Features: r.SepalLength.ConcatWith(r.SepalWidth, r.PetalLength, r.PetalWidth)));

            // Now, at the time of writing, there is no static pipeline for OVA (one-versus-all). So, let's
            // append the OVA learner to the dynamic pipeline.
            IEstimator<ITransformer> dynamicPipe = learningPipeline.AsDynamic;

            // Create a binary classification trainer.
            var binaryTrainer = mlContext.BinaryClassification.Trainers.AveragedPerceptron();

            // Append the OVA learner to the pipeline.
            dynamicPipe = dynamicPipe.Append(new Ova(mlContext, () => binaryTrainer));

            // At this point, we have a choice. We could continue working with the dynamically-typed pipeline, and
            // ultimately call dynamicPipe.Fit(data.AsDynamic) to get the model, or we could go back into the static world.
            // Here's how we go back to the static pipeline:
            var staticFinalPipe = dynamicPipe.AssertStatic(mlContext,
                    // Declare the shape of the input. As you can see, it's identical to the shape of the reader:
                    // four float features and a string label.
                    c => (
                        SepalLength: c.R4.Scalar,
                        SepalWidth: c.R4.Scalar,
                        PetalLength: c.R4.Scalar,
                        PetalWidth: c.R4.Scalar,
                        Label: c.Text.Scalar),
                    // Declare the shape of the output (or a relevant subset of it).
                    // In our case, we care only about the predicted label column (a key type), and scores (vector of floats).
                    c => (
                        Score: c.R4.Vector,
                        // Predicted label is a key backed by uint, with text values (since original labels are text).
                        PredictedLabel: c.KeyU4.TextValues.Scalar))
                // Convert the predicted label from key back to the original string value.
                .Append(r => r.PredictedLabel.ToValue());

            // Train the model in a statically typed way.
            var model = staticFinalPipe.Fit(data);

            // And here is how we could've stayed in the dynamic pipeline and train that way.
            dynamicPipe = dynamicPipe.Append(new KeyToValueEstimator(mlContext, "PredictedLabel"));
            var dynamicModel = dynamicPipe.Fit(data.AsDynamic);

            // Now 'dynamicModel', and 'model.AsDynamic' are equivalent.
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
