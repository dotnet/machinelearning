// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Functional.Tests.Datasets;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Functional.Tests
{
    public class Training : BaseTestClass
    {
        public Training(ITestOutputHelper output) : base(output)
        {
        }

        /// <summary>
        /// Training: It is easy to compare trainer evaluations on the same dataset.
        /// </summary>
        [Fact]
        public void CompareTrainerEvaluations()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<TweetSentiment>(GetDataPath(TestDatasets.Sentiment.trainFilename),
                separatorChar: TestDatasets.Sentiment.fileSeparator,
                hasHeader: TestDatasets.Sentiment.fileHasHeader,
                allowQuoting: TestDatasets.Sentiment.allowQuoting);
            var trainTestSplit = mlContext.Data.TrainTestSplit(data);
            var trainData = trainTestSplit.TrainSet;
            var testData = trainTestSplit.TestSet;

            // Create a transformation pipeline.
            var featurizationPipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(mlContext);

            // Create a selection of learners.
            var sdcaTrainer = mlContext.BinaryClassification.Trainers.SdcaCalibrated(
                    new SdcaCalibratedBinaryTrainer.Options { NumberOfThreads = 1 });

            var fastTreeTrainer = mlContext.BinaryClassification.Trainers.FastTree(
                    new FastTreeBinaryTrainer.Options { NumberOfThreads = 1 });

            var ffmTrainer = mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine();

            // Fit the data transformation pipeline.
            var featurization = featurizationPipeline.Fit(trainData);
            var featurizedTrain = featurization.Transform(trainData);
            var featurizedTest = featurization.Transform(testData);

            // Fit the trainers.
            var sdca = sdcaTrainer.Fit(featurizedTrain);
            var fastTree = fastTreeTrainer.Fit(featurizedTrain);
            var ffm = ffmTrainer.Fit(featurizedTrain);

            // Evaluate the trainers.
            var sdcaPredictions = sdca.Transform(featurizedTest);
            var sdcaMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(sdcaPredictions);
            var fastTreePredictions = fastTree.Transform(featurizedTest);
            var fastTreeMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(fastTreePredictions);
            var ffmPredictions = sdca.Transform(featurizedTest);
            var ffmMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(ffmPredictions);

            // Validate the results.
            Common.AssertMetrics(sdcaMetrics);
            Common.AssertMetrics(fastTreeMetrics);
            Common.AssertMetrics(ffmMetrics);
        }

        /// <summary>
        /// Training: Models can be trained starting from an existing model.
        /// </summary>
        [Fact]
        public void ContinueTrainingAveragePerceptron()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<TweetSentiment>(GetDataPath(TestDatasets.Sentiment.trainFilename),
                separatorChar: TestDatasets.Sentiment.fileSeparator,
                hasHeader: TestDatasets.Sentiment.fileHasHeader,
                allowQuoting: TestDatasets.Sentiment.allowQuoting);

            // Create a transformation pipeline.
            var featurizationPipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(mlContext);

            var trainer = mlContext.BinaryClassification.Trainers.AveragedPerceptron(
                new AveragedPerceptronTrainer.Options { NumberOfIterations = 1 });

            // Fit the data transformation pipeline.
            var featurization = featurizationPipeline.Fit(data);
            var featurizedData = featurization.Transform(data);

            // Fit the first trainer.
            var firstModel = trainer.Fit(featurizedData);
            var firstModelWeights = firstModel.Model.Weights;

            // Fist the first trainer again.
            var firstModelPrime = trainer.Fit(featurizedData);
            var firstModelWeightsPrime = firstModel.Model.Weights;

            // Fit the second trainer.
            var secondModel = trainer.Fit(featurizedData, firstModel.Model);
            var secondModelWeights = secondModel.Model.Weights;

            // Validate that continued training occurred.
            // Training from the same initial condition, same seed should create the same model.
            Common.AssertEqual(firstModelWeights.ToArray(), firstModelWeightsPrime.ToArray());
            // Continued training should create a different model.
            Common.AssertNotEqual(firstModelWeights.ToArray(), secondModelWeights.ToArray());
        }

        /// <summary>
        /// Training: Models can be trained starting from an existing model.
        /// </summary>
        [Fact]
        public void ContinueTrainingFieldAwareFactorizationMachine()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<TweetSentiment>(GetDataPath(TestDatasets.Sentiment.trainFilename),
                separatorChar: TestDatasets.Sentiment.fileSeparator,
                hasHeader: TestDatasets.Sentiment.fileHasHeader,
                allowQuoting: TestDatasets.Sentiment.allowQuoting);

            // Create a transformation pipeline.
            var featurizationPipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(mlContext);

            var trainer = mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(
                new FieldAwareFactorizationMachineTrainer.Options { NumberOfIterations = 100 });

            // Fit the data transformation pipeline.
            var featurization = featurizationPipeline.Fit(data);
            var featurizedData = featurization.Transform(data);

            // Fit the first trainer.
            var firstModel = trainer.Fit(featurizedData);
            var firstModelWeights = firstModel.Model.GetLinearWeights();

            // Fist the first trainer again.
            var firstModelPrime = trainer.Fit(featurizedData);
            var firstModelWeightsPrime = firstModel.Model.GetLinearWeights();

            // Fit the second trainer.
            var secondModel = trainer.Fit(featurizedData, modelParameters: firstModel.Model);
            var secondModelWeights = secondModel.Model.GetLinearWeights();

            // Validate that continued training occurred.
            // Training from the same initial condition, same seed should create the same model.
            Assert.Equal(firstModelWeights, firstModelWeightsPrime);
            // Continued training should create a different model.
            Assert.NotEqual(firstModelWeights, secondModelWeights);
        }

        /// <summary>
        /// Training: Models can be trained starting from an existing model.
        /// </summary>
        [Fact]
        public void ContinueTrainingLinearSupportVectorMachine()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<TweetSentiment>(GetDataPath(TestDatasets.Sentiment.trainFilename),
                separatorChar: TestDatasets.Sentiment.fileSeparator,
                hasHeader: TestDatasets.Sentiment.fileHasHeader,
                allowQuoting: TestDatasets.Sentiment.allowQuoting);

            // Create a transformation pipeline.
            var featurizationPipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(mlContext);

            var trainer = mlContext.BinaryClassification.Trainers.LinearSvm(
                new LinearSvmTrainer.Options { NumberOfIterations = 1 });

            // Fit the data transformation pipeline.
            var featurization = featurizationPipeline.Fit(data);
            var featurizedData = featurization.Transform(data);

            // Fit the first trainer.
            var firstModel = trainer.Fit(featurizedData);
            var firstModelWeights = firstModel.Model.Weights;

            // Fist the first trainer again.
            var firstModelPrime = trainer.Fit(featurizedData);
            var firstModelWeightsPrime = firstModel.Model.Weights;

            // Fit the second trainer.
            var secondModel = trainer.Fit(featurizedData, firstModel.Model);
            var secondModelWeights = secondModel.Model.Weights;

            // Validate that continued training occurred.
            // Training from the same initial condition, same seed should create the same model.
            Common.AssertEqual(firstModelWeights.ToArray(), firstModelWeightsPrime.ToArray());
            // Continued training should create a different model.
            Common.AssertNotEqual(firstModelWeights.ToArray(), secondModelWeights.ToArray());
        }

        /// <summary>
        /// Training: Models can be trained starting from an existing model.
        /// </summary>
        [Fact]
        public void ContinueTrainingLogisticRegression()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<TweetSentiment>(GetDataPath(TestDatasets.Sentiment.trainFilename),
                separatorChar: TestDatasets.Sentiment.fileSeparator,
                hasHeader: TestDatasets.Sentiment.fileHasHeader,
                allowQuoting: TestDatasets.Sentiment.allowQuoting);

            // Create a transformation pipeline.
            var featurizationPipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(mlContext);

            var trainer = mlContext.BinaryClassification.Trainers.LogisticRegression(
                new LogisticRegressionBinaryTrainer.Options { NumberOfThreads = 1, MaximumNumberOfIterations = 10 });

            // Fit the data transformation pipeline.
            var featurization = featurizationPipeline.Fit(data);
            var featurizedData = featurization.Transform(data);

            // Fit the first trainer.
            var firstModel = trainer.Fit(featurizedData);
            var firstModelWeights = firstModel.Model.SubModel.Weights;

            // Fist the first trainer again.
            var firstModelPrime = trainer.Fit(featurizedData);
            var firstModelWeightsPrime = firstModel.Model.SubModel.Weights;

            // Fit the second trainer.
            var secondModel = trainer.Fit(featurizedData, firstModel.Model.SubModel);
            var secondModelWeights = secondModel.Model.SubModel.Weights;

            // Validate that continued training occurred.
            // Training from the same initial condition, same seed should create the same model.
            Common.AssertEqual(firstModelWeights.ToArray(), firstModelWeightsPrime.ToArray());
            // Continued training should create a different model.
            Common.AssertNotEqual(firstModelWeights.ToArray(), secondModelWeights.ToArray());
        }

        /// <summary>
        /// Training: Models can be trained starting from an existing model.
        /// </summary>
        [Fact]
        public void ContinueTrainingLogisticRegressionMulticlass()
        {
            var mlContext = new MLContext(seed: 1);

            var data = mlContext.Data.LoadFromTextFile<Iris>(GetDataPath(TestDatasets.iris.trainFilename),
                hasHeader: TestDatasets.iris.fileHasHeader,
                separatorChar: TestDatasets.iris.fileSeparator);

            // Create a training pipeline.
            var featurizationPipeline = mlContext.Transforms.Concatenate("Features", Iris.Features)
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .AppendCacheCheckpoint(mlContext);

            var trainer = mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                new LbfgsMaximumEntropyTrainer.Options { NumberOfThreads = 1, MaximumNumberOfIterations = 10 });

            // Fit the data transformation pipeline.
            var featurization = featurizationPipeline.Fit(data);
            var featurizedData = featurization.Transform(data);

            // Fit the first trainer.
            var firstModel = trainer.Fit(featurizedData);
            VBuffer<float>[] firstModelWeights = null;
            firstModel.Model.GetWeights(ref firstModelWeights, out int firstModelNumClasses);

            // Fist the first trainer again.
            var firstModelPrime = trainer.Fit(featurizedData);
            VBuffer<float>[] firstModelWeightsPrime = null;
            firstModel.Model.GetWeights(ref firstModelWeightsPrime, out int firstModelNumClassesPrime);

            // Fit the second trainer.
            var secondModel = trainer.Fit(featurizedData, firstModel.Model);
            VBuffer<float>[] secondModelWeights = null;
            secondModel.Model.GetWeights(ref secondModelWeights, out int secondModelNumClasses);

            // Validate that continued training occurred.
            // Training from the same initial condition, same seed should create the same model.
            Assert.Equal(firstModelNumClasses, firstModelNumClassesPrime);
            for (int i = 0; i < firstModelNumClasses; i++)
                Common.AssertEqual(firstModelWeights[i].DenseValues().ToArray(), firstModelWeightsPrime[i].DenseValues().ToArray());
            // Continued training should create a different model.
            Assert.Equal(firstModelNumClasses, secondModelNumClasses);
            for (int i = 0; i < firstModelNumClasses; i++)
                Common.AssertNotEqual(firstModelWeights[i].DenseValues().ToArray(), secondModelWeights[i].DenseValues().ToArray());
        }

        /// <summary>
        /// Training: Models can be trained starting from an existing model.
        /// </summary>
        [Fact]
        public void ContinueTrainingOnlineGradientDescent()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename),
                separatorChar: TestDatasets.housing.fileSeparator,
                hasHeader: TestDatasets.housing.fileHasHeader);

            // Create a transformation pipeline.
            var featurizationPipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Transforms.Normalize("Features"))
                .AppendCacheCheckpoint(mlContext);

            var trainer = mlContext.Regression.Trainers.OnlineGradientDescent(
                new OnlineGradientDescentTrainer.Options { NumberOfIterations = 10 });

            // Fit the data transformation pipeline.
            var featurization = featurizationPipeline.Fit(data);
            var featurizedData = featurization.Transform(data);

            // Fit the first trainer.
            var firstModel = trainer.Fit(featurizedData);
            var firstModelWeights = firstModel.Model.Weights;

            // Fist the first trainer again.
            var firstModelPrime = trainer.Fit(featurizedData);
            var firstModelWeightsPrime = firstModel.Model.Weights;

            // Fit the second trainer.
            var secondModel = trainer.Fit(featurizedData, firstModel.Model);
            var secondModelWeights = secondModel.Model.Weights;

            // Validate that continued training occurred.
            // Training from the same initial condition, same seed should create the same model.
            Common.AssertEqual(firstModelWeights.ToArray(), firstModelWeightsPrime.ToArray());
            // Continued training should create a different model.
            Common.AssertNotEqual(firstModelWeights.ToArray(), secondModelWeights.ToArray());
        }

        /// <summary>
        /// Training: Models can be trained starting from an existing model.
        /// </summary>
        [Fact]
        public void ContinueTrainingPoissonRegression()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<HousingRegression>(GetDataPath(TestDatasets.housing.trainFilename),
                separatorChar: TestDatasets.housing.fileSeparator,
                hasHeader: TestDatasets.housing.fileHasHeader);

            // Create a transformation pipeline.
            var featurizationPipeline = mlContext.Transforms.Concatenate("Features", HousingRegression.Features)
                .Append(mlContext.Transforms.Normalize("Features"))
                .AppendCacheCheckpoint(mlContext);

            var trainer = mlContext.Regression.Trainers.PoissonRegression(
                new PoissonRegressionTrainer.Options { NumberOfThreads = 1, MaximumNumberOfIterations = 100 });

            // Fit the data transformation pipeline.
            var featurization = featurizationPipeline.Fit(data);
            var featurizedData = featurization.Transform(data);

            // Fit the first trainer.
            var firstModel = trainer.Fit(featurizedData);
            var firstModelWeights = firstModel.Model.Weights;

            // Fist the first trainer again.
            var firstModelPrime = trainer.Fit(featurizedData);
            var firstModelWeightsPrime = firstModel.Model.Weights;

            // Fit the second trainer.
            var secondModel = trainer.Fit(featurizedData, firstModel.Model);
            var secondModelWeights = secondModel.Model.Weights;

            // Validate that continued training occurred.
            // Training from the same initial condition, same seed should create the same model.
            Common.AssertEqual(firstModelWeights.ToArray(), firstModelWeightsPrime.ToArray());
            // Continued training should create a different model.
            Common.AssertNotEqual(firstModelWeights.ToArray(), secondModelWeights.ToArray());
        }

        /// <summary>
        /// Training: Models can be trained starting from an existing model.
        /// </summary>
        [Fact]
        public void ContinueTrainingSymbolicStochasticGradientDescent()
        {
            var mlContext = new MLContext(seed: 1);

            // Get the dataset.
            var data = mlContext.Data.LoadFromTextFile<TweetSentiment>(GetDataPath(TestDatasets.Sentiment.trainFilename),
                separatorChar: TestDatasets.Sentiment.fileSeparator,
                hasHeader: TestDatasets.Sentiment.fileHasHeader,
                allowQuoting: TestDatasets.Sentiment.allowQuoting);

            // Create a transformation pipeline.
            var featurizationPipeline = mlContext.Transforms.Text.FeaturizeText("Features", "SentimentText")
                .AppendCacheCheckpoint(mlContext);

            var trainer = mlContext.BinaryClassification.Trainers.SymbolicSgd(
                new SymbolicSgdTrainer.Options
                {
                    NumberOfThreads = 1,
                    NumberOfIterations = 10
                });

            // Fit the data transformation pipeline.
            var featurization = featurizationPipeline.Fit(data);
            var featurizedData = featurization.Transform(data);

            // Fit the first trainer.
            var firstModel = trainer.Fit(featurizedData);
            var firstModelWeights = firstModel.Model.SubModel.Weights;

            // Fist the first trainer again.
            var firstModelPrime = trainer.Fit(featurizedData);
            var firstModelWeightsPrime = firstModel.Model.SubModel.Weights;

            // Fit the second trainer.
            var secondModel = trainer.Fit(featurizedData, firstModel.Model.SubModel);
            var secondModelWeights = secondModel.Model.SubModel.Weights;

            // Validate that continued training occurred.
            // Training from the same initial condition, same seed should create the same model.
            Common.AssertEqual(firstModelWeights.ToArray(), firstModelWeightsPrime.ToArray());
            // Continued training should create a different model.
            Common.AssertNotEqual(firstModelWeights.ToArray(), secondModelWeights.ToArray());
        }

        /// <summary>
        /// Training: Meta-compononts function as expected. For OVA (one-versus-all), a user will be able to specify only
        /// binary classifier trainers. If they specify a different model class there should be a compile error.
        /// </summary>
        [Fact]
        public void MetacomponentsFunctionAsExpectedOva()
        {
            var mlContext = new MLContext(seed: 1);

            var data = mlContext.Data.LoadFromTextFile<Iris>(GetDataPath(TestDatasets.iris.trainFilename),
                hasHeader: TestDatasets.iris.fileHasHeader,
                separatorChar: TestDatasets.iris.fileSeparator);

            // Create a model training an OVA trainer with a binary classifier.
            var binaryClassificationTrainer = mlContext.BinaryClassification.Trainers.LogisticRegression(
                new LogisticRegressionBinaryTrainer.Options { MaximumNumberOfIterations = 10, NumberOfThreads = 1, });
            var binaryClassificationPipeline = mlContext.Transforms.Concatenate("Features", Iris.Features)
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryClassificationTrainer));

            // Fit the binary classification pipeline.
            var binaryClassificationModel = binaryClassificationPipeline.Fit(data);

            // Transform the data
            var binaryClassificationPredictions = binaryClassificationModel.Transform(data);

            // Evaluate the model.
            var binaryClassificationMetrics = mlContext.MulticlassClassification.Evaluate(binaryClassificationPredictions);
        }
    }
}