// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void SdcaWorkout()
        {
            var dataPath = GetDataPath("breast-cancer.txt");

            var data = TextLoaderStatic.CreateLoader(Env, ctx => (Label: ctx.LoadFloat(0), Features: ctx.LoadFloat(1, 10)))
                .Load(dataPath).Cache();

            var binaryData = ML.Transforms.Conversion.ConvertType("Label", outputKind: DataKind.Boolean)
                .Fit(data.AsDynamic).Transform(data.AsDynamic);

            var binaryTrainer = ML.BinaryClassification.Trainers.SdcaCalibrated(
                new SdcaCalibratedBinaryTrainer.Options { ConvergenceTolerance = 1e-2f, MaximumNumberOfIterations = 10 });
            TestEstimatorCore(binaryTrainer, binaryData);

            var nonCalibratedBinaryTrainer = ML.BinaryClassification.Trainers.SdcaNonCalibrated(
                new SdcaNonCalibratedBinaryTrainer.Options { ConvergenceTolerance = 1e-2f, MaximumNumberOfIterations = 10 });
            TestEstimatorCore(nonCalibratedBinaryTrainer, binaryData);

            var regressionTrainer = ML.Regression.Trainers.Sdca(
                new SdcaRegressionTrainer.Options { ConvergenceTolerance = 1e-2f, MaximumNumberOfIterations = 10 });

            TestEstimatorCore(regressionTrainer, data.AsDynamic);
            var mcData = ML.Transforms.Conversion.MapValueToKey("Label").Fit(data.AsDynamic).Transform(data.AsDynamic);

            var mcTrainer = ML.MulticlassClassification.Trainers.SdcaCalibrated(
                new SdcaCalibratedMulticlassTrainer.Options { ConvergenceTolerance = 1e-2f, MaximumNumberOfIterations = 10 });
            TestEstimatorCore(mcTrainer, mcData);

            Done();
        }

        [Fact]
        public void SdcaLogisticRegression()
        {
            // Generate C# objects as training examples.
            var rawData = SamplesUtils.DatasetUtils.GenerateBinaryLabelFloatFeatureVectorSamples(100);

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step 1: Read the data as an IDataView.
            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // ML.NET doesn't cache data set by default. Caching is very helpful when working with iterative
            // algorithms which needs many data passes. Since SDCA is the case, we cache.
            data = mlContext.Data.Cache(data);

            // Step 2: Create a binary classifier.
            // We set the "Label" column as the label of the dataset, and the "Features" column as the features column.
            var pipeline = mlContext.BinaryClassification.Trainers.SdcaCalibrated(labelColumnName: "Label", featureColumnName: "Features", l2Regularization: 0.001f);

            // Step 3: Train the pipeline created.
            var model = pipeline.Fit(data);

            // Step 4: Make prediction and evaluate its quality (on training set).
            var prediction = model.Transform(data);
            var metrics = mlContext.BinaryClassification.Evaluate(prediction);

            // Check a few metrics to make sure the trained model is ok.
            Assert.InRange(metrics.AreaUnderRocCurve, 0.9, 1);
            Assert.InRange(metrics.LogLoss, 0, 0.5);

            var rawPrediction = mlContext.Data.CreateEnumerable<SamplesUtils.DatasetUtils.CalibratedBinaryClassifierOutput>(prediction, false);

            // Step 5: Inspect the prediction of the first example.
            var first = rawPrediction.First();
            // This is a positive example.
            Assert.True(first.Label);
            // Positive example should have non-negative score. 
            Assert.True(first.Score > 0);
            // Positive example should have high probability of belonging the positive class.
            Assert.InRange(first.Probability, 0.8, 1);
        }

        [Fact]
        public void SdcaSupportVectorMachine()
        {
            // Generate C# objects as training examples.
            var rawData = SamplesUtils.DatasetUtils.GenerateBinaryLabelFloatFeatureVectorSamples(100);

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step 1: Read the data as an IDataView.
            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // ML.NET doesn't cache data set by default. Caching is very helpful when working with iterative
            // algorithms which needs many data passes. Since SDCA is the case, we cache.
            data = mlContext.Data.Cache(data);

            // Step 2: Create a binary classifier.
            // We set the "Label" column as the label of the dataset, and the "Features" column as the features column.
            var pipeline = mlContext.BinaryClassification.Trainers.SdcaNonCalibrated(
                labelColumnName: "Label", featureColumnName: "Features", loss: new HingeLoss(), l2Regularization: 0.001f);

            // Step 3: Train the pipeline created.
            var model = pipeline.Fit(data);

            // Step 4: Make prediction and evaluate its quality (on training set).
            var prediction = model.Transform(data);
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(prediction);

            // Check a few metrics to make sure the trained model is ok.
            Assert.InRange(metrics.AreaUnderRocCurve, 0.9, 1);

            var rawPrediction = mlContext.Data.CreateEnumerable<SamplesUtils.DatasetUtils.NonCalibratedBinaryClassifierOutput>(prediction, false);

            // Step 5: Inspect the prediction of the first example.
            var first = rawPrediction.First();
            // This is a positive example.
            Assert.True(first.Label);
            // Positive example should have non-negative score. 
            Assert.True(first.Score > 0);
        }

        [Fact]
        public void SdcaMulticlassLogisticRegression()
        {
            // Generate C# objects as training examples.
            var rawData = SamplesUtils.DatasetUtils.GenerateFloatLabelFloatFeatureVectorSamples(512);

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step 1: Read the data as an IDataView.
            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // ML.NET doesn't cache data set by default. Caching is very helpful when working with iterative
            // algorithms which needs many data passes. Since SDCA is the case, we cache.
            data = mlContext.Data.Cache(data);

            // Step 2: Create a binary classifier.
            // We set the "Label" column as the label of the dataset, and the "Features" column as the features column.

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("LabelIndex", "Label").
                           Append(mlContext.MulticlassClassification.Trainers.SdcaCalibrated(labelColumnName: "LabelIndex", featureColumnName: "Features", l2Regularization: 0.001f));

            // Step 3: Train the pipeline created.
            var model = pipeline.Fit(data);

            // Step 4: Make prediction and evaluate its quality (on training set).
            var prediction = model.Transform(data);
            var metrics = mlContext.MulticlassClassification.Evaluate(prediction, labelColumnName: "LabelIndex", topK: 1);

            // Check a few metrics to make sure the trained model is ok.
            Assert.InRange(metrics.TopKAccuracy, 0.8, 1);
            Assert.InRange(metrics.LogLoss, 0, 0.5);
        }

        [Fact]
        public void SdcaMulticlassSupportVectorMachine()
        {
            // Generate C# objects as training examples.
            var rawData = SamplesUtils.DatasetUtils.GenerateFloatLabelFloatFeatureVectorSamples(512);

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step 1: Read the data as an IDataView.
            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // ML.NET doesn't cache data set by default. Caching is very helpful when working with iterative
            // algorithms which needs many data passes. Since SDCA is the case, we cache.
            data = mlContext.Data.Cache(data);

            // Step 2: Create a binary classifier.
            // We set the "Label" column as the label of the dataset, and the "Features" column as the features column.
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("LabelIndex", "Label").
                Append(mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated(labelColumnName: "LabelIndex", featureColumnName: "Features", loss: new HingeLoss(), l2Regularization: 0.001f));

            // Step 3: Train the pipeline created.
            var model = pipeline.Fit(data);

            // Step 4: Make prediction and evaluate its quality (on training set).
            var prediction = model.Transform(data);
            var metrics = mlContext.MulticlassClassification.Evaluate(prediction, labelColumnName: "LabelIndex", topK: 1);

            // Check a few metrics to make sure the trained model is ok.
            Assert.InRange(metrics.TopKAccuracy, 0.8, 1);
            Assert.InRange(metrics.MacroAccuracy, 0.8, 1);
        }

    }
}
