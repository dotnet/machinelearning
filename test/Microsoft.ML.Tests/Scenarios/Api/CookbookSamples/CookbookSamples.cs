// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.FastTree;
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
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Learners;

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

        private ITransformer TrainOnIris(string irisDataPath)
        {
            // Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var env = new LocalEnvironment();

            // We know that this is a classification task, so we create a multiclass classification context: it will give us the algorithms
            // we need, as well as the evaluation procedure.
            var classification = new MulticlassClassificationContext(env);

            // Step one: read the data as an IDataView.
            // First, we define the reader: specify the data columns and where to find them in the text file.
            var reader = TextLoader.CreateReader(env, ctx => (
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
            var trainData = reader.Read(new MultiFileSource(irisDataPath));

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
                    Predictions: classification.Trainers.Sdca(r.Label.ToKey(), r.Features)))
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
            // Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var env = new LocalEnvironment();

            // Use the model for one-time prediction.
            // Make the prediction function object. Note that, on average, this call takes around 200x longer
            // than one prediction, so you might want to cache and reuse the prediction function, instead of
            // creating one per prediction.
            var predictionFunc = model.MakePredictionFunction<IrisInput, IrisPrediction>(env);

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
            // Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var env = new LocalEnvironment();

            // We know that this is a classification task, so we create a multiclass classification context: it will give us the algorithms
            // we need, as well as the evaluation procedure.
            var classification = new MulticlassClassificationContext(env);

            // Step one: read the data as an IDataView.
            // First, we define the reader: specify the data columns and where to find them in the text file.
            var reader = TextLoader.CreateReader(env, ctx => (
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
            var trainData = reader.Read(new MultiFileSource(dataPath));

            // This is the predictor ('weights collection') that we will train.
            MulticlassLogisticRegressionPredictor predictor = null;
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
                    Predictions: classification.Trainers.Sdca(r.Label.ToKey(), r.Features,
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
        }

        [Fact]
        public void InspectModelWeights()
            => TrainAndInspectWeights(GetDataPath("iris.data"));

        private void NormalizationWorkout(string dataPath)
        {
            // Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var env = new LocalEnvironment();

            // Define the reader: specify the data columns and where to find them in the text file.
            var reader = TextLoader.CreateReader(env, ctx => (
                    // The four features of the Iris dataset will be grouped together as one Features column.
                    Features: ctx.LoadFloat(0, 3),
                    // Label: kind of iris.
                    Label: ctx.LoadText(4)
                ),
                // Default separator is tab, but the dataset has comma.
                separator: ',');

            // Read the training data.
            var trainData = reader.Read(new MultiFileSource(dataPath));

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
            // Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var env = new LocalEnvironment();

            // Step one: read the data as an IDataView.
            // Let's assume that 'GetChurnData()' fetches and returns the training data from somewhere.
            IEnumerable<CustomerChurnInfo> churnData = GetChurnInfo();

            // Turn the data into the ML.NET data view.
            // We can use CreateDataView or CreateStreamingDataView, depending on whether 'churnData' is an IList, 
            // or merely an IEnumerable.
            var trainData = env.CreateStreamingDataView(churnData);

            // Now note that 'trainData' is just an IDataView, so we face a choice here: either declare the static type
            // and proceed in the statically typed fashion, or keep dynamic types and build a dynamic pipeline.
            // We demonstrate both below.

            // We know that this is a binary classification task, so we create a binary classification context: it will give us the algorithms
            // we need, as well as the evaluation procedure.
            var classification = new BinaryClassificationContext(env);

            // Build the learning pipeline. 
            // In our case, we will one-hot encode the demographic category, and concatenate that with the number of visits.
            // We apply our FastTree binary classifier to predict the 'HasChurned' label.

            var dynamicLearningPipeline = new CategoricalEstimator(env, "DemographicCategory")
                .Append(new ConcatEstimator(env, "Features", "DemographicCategory", "LastVisits"))
                .Append(new FastTreeBinaryClassificationTrainer(env, "HasChurned", "Features", numTrees: 20));

            var dynamicModel = dynamicLearningPipeline.Fit(trainData);

            // Build the same learning pipeline, but statically typed.
            // First, transition to the statically-typed data view.
            var staticData = trainData.AssertStatic(env, c => (
                    HasChurned: c.Bool.Scalar,
                    DemographicCategory: c.Text.Scalar,
                    LastVisits: c.R4.Vector));

            // Build the pipeline, same as the one above.
            var staticLearningPipeline = staticData.MakeNewEstimator()
                .Append(r => (
                    r.HasChurned,
                    Features: r.DemographicCategory.OneHotEncoding().ConcatWith(r.LastVisits)))
                .Append(r => classification.Trainers.FastTree(r.HasChurned, r.Features, numTrees: 20));

            var staticModel = staticLearningPipeline.Fit(staticData);

            // Note that dynamicModel should be the same as staticModel.AsDynamic (give or take random variance from
            // the training procedure).

            var qualityMetrics = classification.Evaluate(dynamicModel.Transform(trainData), "HasChurned");
        }

        private void TextFeaturizationOn(string dataPath)
        {
            // Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var env = new LocalEnvironment();

            // Define the reader: specify the data columns and where to find them in the text file.
            var reader = TextLoader.CreateReader(env, ctx => (
                    IsToxic: ctx.LoadBool(0),
                    Message: ctx.LoadText(1)
                ), hasHeader: true);

            // Read the data.
            var data = reader.Read(new MultiFileSource(dataPath));

            // Inspect the message texts that are read from the file.
            var messageTexts = data.GetColumn(x => x.Message).Take(20).ToArray();

            // Apply various kinds of text operations supported by ML.NET.
            var learningPipeline = reader.MakeNewEstimator()
                .Append(r => (
                    // One-stop shop to run the full text featurization.
                    TextFeatures: r.Message.FeaturizeText(),

                    // NLP pipeline 1: bag of words.
                    BagOfWords: r.Message.NormalizeText().ToBagofWords(),

                    // NLP pipeline 2: bag of bigrams.
                    BagOfBigrams: r.Message.NormalizeText().ToBagofWords(ngramLength: 2, allLengths: false),

                    // NLP pipeline 3: bag of tri-character sequences.
                    BagOfTrichar: r.Message.TokenizeIntoCharacters().ToNgrams(ngramLength: 3),

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
    }
}
