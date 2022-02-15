// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Newtonsoft.Json;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class PermutationFeatureImportanceTests : BaseTestPredictors
    {
        public PermutationFeatureImportanceTests(ITestOutputHelper output) : base(output)
        {
        }

        #region Regression Tests

        /// <summary>
        /// Test PFI Regression for Dense Features
        /// </summary>
        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void TestPfiRegressionOnDenseFeatures(bool saveModel)
        {
            var data = GetDenseDataset();
            var model = ML.Regression.Trainers.OnlineGradientDescent().Fit(data);

            ImmutableArray<RegressionMetricsStatistics> pfi;
            ImmutableDictionary<string, RegressionMetricsStatistics> pfiDict;

            if (saveModel)
            {
                var modelAndSchemaPath = GetOutputPath("TestPfiRegressionOnDenseFeatures.zip");
                ML.Model.Save(model, data.Schema, modelAndSchemaPath);

                var loadedModel = ML.Model.Load(modelAndSchemaPath, out var schema);
                var castedModel = loadedModel as RegressionPredictionTransformer<LinearRegressionModelParameters>;

                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                pfi = ML.Regression.PermutationFeatureImportance(castedModel, data);
                pfiDict = ml2.Regression.PermutationFeatureImportance(loadedModel, data);
            }
            else
            {
                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                pfi = ML.Regression.PermutationFeatureImportance(model, data);
                pfiDict = ml2.Regression.PermutationFeatureImportance((ITransformer)model, data);
            }

            // Pfi Indices:
            // X1: 0
            // X2Important: 1
            // X3: 2
            // X4Rand: 3

            // Make sure that PFI from the array and the dictionary both have the same value for each feature.
            Assert.Equal(JsonConvert.SerializeObject(pfi[0]), JsonConvert.SerializeObject(pfiDict["X1"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[1]), JsonConvert.SerializeObject(pfiDict["X2Important"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[2]), JsonConvert.SerializeObject(pfiDict["X3"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[3]), JsonConvert.SerializeObject(pfiDict["X4Rand"]));

            // For the following metrics lower is better, so maximum delta means more important feature, and vice versa
            Assert.Equal(3, MinDeltaIndex(pfi, m => m.MeanAbsoluteError.Mean));
            Assert.Equal(1, MaxDeltaIndex(pfi, m => m.MeanAbsoluteError.Mean));

            Assert.Equal(3, MinDeltaIndex(pfi, m => m.MeanSquaredError.Mean));
            Assert.Equal(1, MaxDeltaIndex(pfi, m => m.MeanSquaredError.Mean));

            Assert.Equal(3, MinDeltaIndex(pfi, m => m.RootMeanSquaredError.Mean));
            Assert.Equal(1, MaxDeltaIndex(pfi, m => m.RootMeanSquaredError.Mean));

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.RSquared.Mean));
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.RSquared.Mean));

            Done();
        }

        /// <summary>
        /// Test PFI Regression for Dense Features in a transformer chain
        /// </summary>
        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void TestPfiRegressionOnDenseFeaturesInTransformerChain(bool saveModel)
        {
            var data = GetDenseDataset();
            var model = ML.Transforms.CopyColumns("Label", "Label").Append(ML.Regression.Trainers.OnlineGradientDescent()).Fit(data);

            ImmutableArray<RegressionMetricsStatistics> pfi;
            ImmutableDictionary<string, RegressionMetricsStatistics> pfiDict;

            if (saveModel)
            {
                var modelAndSchemaPath = GetOutputPath("TestPfiRegressionOnDenseFeatures.zip");
                ML.Model.Save(model, data.Schema, modelAndSchemaPath);

                var loadedModel = ML.Model.Load(modelAndSchemaPath, out var schema);

                ITransformer lastTransformer = null;
                if (loadedModel is ITransformerChainAccessor chain)
                {
                    lastTransformer = chain.Transformers.Last();
                }
                var castedModel = lastTransformer as RegressionPredictionTransformer<LinearRegressionModelParameters>;

                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                pfi = ML.Regression.PermutationFeatureImportance(castedModel, data);
                pfiDict = ml2.Regression.PermutationFeatureImportance(loadedModel, data);
            }
            else
            {
                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                pfi = ML.Regression.PermutationFeatureImportance(model.LastTransformer, data);
                pfiDict = ml2.Regression.PermutationFeatureImportance(model, data);
            }

            // Pfi Indices:
            // X1: 0
            // X2Important: 1
            // X3: 2
            // X4Rand: 3

            // Make sure that PFI from the array and the dictionary both have the same value for each feature.
            Assert.Equal(JsonConvert.SerializeObject(pfi[0]), JsonConvert.SerializeObject(pfiDict["X1"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[1]), JsonConvert.SerializeObject(pfiDict["X2Important"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[2]), JsonConvert.SerializeObject(pfiDict["X3"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[3]), JsonConvert.SerializeObject(pfiDict["X4Rand"]));

            // For the following metrics lower is better, so maximum delta means more important feature, and vice versa
            Assert.Equal(3, MinDeltaIndex(pfi, m => m.MeanAbsoluteError.Mean));
            Assert.Equal(1, MaxDeltaIndex(pfi, m => m.MeanAbsoluteError.Mean));

            Assert.Equal(3, MinDeltaIndex(pfi, m => m.MeanSquaredError.Mean));
            Assert.Equal(1, MaxDeltaIndex(pfi, m => m.MeanSquaredError.Mean));

            Assert.Equal(3, MinDeltaIndex(pfi, m => m.RootMeanSquaredError.Mean));
            Assert.Equal(1, MaxDeltaIndex(pfi, m => m.RootMeanSquaredError.Mean));

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.RSquared.Mean));
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.RSquared.Mean));

            Done();
        }

        /// <summary>
        /// Test PFI Regression Standard Deviation and Standard Error for Dense Features
        /// </summary>
        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void TestPfiRegressionStandardDeviationAndErrorOnDenseFeatures(bool saveModel)
        {
            var data = GetDenseDataset();
            var model = ML.Regression.Trainers.OnlineGradientDescent().Fit(data);

            ImmutableArray<RegressionMetricsStatistics> pfi;
            ImmutableDictionary<string, RegressionMetricsStatistics> pfiDict;

            if (saveModel)
            {
                var modelAndSchemaPath = GetOutputPath("TestPfiRegressionStandardDeviationAndErrorOnDenseFeatures.zip");
                ML.Model.Save(model, data.Schema, modelAndSchemaPath);

                var loadedModel = ML.Model.Load(modelAndSchemaPath, out var schema);
                var castedModel = loadedModel as RegressionPredictionTransformer<LinearRegressionModelParameters>;

                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                pfi = ML.Regression.PermutationFeatureImportance(castedModel, data, permutationCount: 20);
                pfiDict = ml2.Regression.PermutationFeatureImportance(loadedModel, data, permutationCount: 20);
            }
            else
            {
                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                pfi = ML.Regression.PermutationFeatureImportance(model, data, permutationCount: 20);
                pfiDict = ml2.Regression.PermutationFeatureImportance((ITransformer)model, data, permutationCount: 20);
            }

            // Keep the permutation count high so fluctuations are kept to a minimum
            //  but not high enough to slow down the tests
            //  (fluctuations lead to random test failures)

            // Pfi Indices:
            // X1: 0
            // X2Important: 1
            // X3: 2
            // X4Rand: 3

            // Make sure that PFI from the array and the dictionary both have the same value for each feature.
            Assert.Equal(JsonConvert.SerializeObject(pfi[0]), JsonConvert.SerializeObject(pfiDict["X1"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[1]), JsonConvert.SerializeObject(pfiDict["X2Important"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[2]), JsonConvert.SerializeObject(pfiDict["X3"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[3]), JsonConvert.SerializeObject(pfiDict["X4Rand"]));

            // For these metrics, the magnitude of the difference will be greatest for 1, least for 3
            // Stardard Deviation will scale with the magnitude of the measure
            Assert.Equal(3, MinDeltaIndex(pfi, m => m.MeanAbsoluteError.StandardDeviation));
            Assert.Equal(1, MaxDeltaIndex(pfi, m => m.MeanAbsoluteError.StandardDeviation));

            Assert.Equal(3, MinDeltaIndex(pfi, m => m.MeanSquaredError.StandardDeviation));
            Assert.Equal(1, MaxDeltaIndex(pfi, m => m.MeanSquaredError.StandardDeviation));

            Assert.Equal(3, MinDeltaIndex(pfi, m => m.RootMeanSquaredError.StandardDeviation));
            Assert.Equal(1, MaxDeltaIndex(pfi, m => m.RootMeanSquaredError.StandardDeviation));

            Assert.Equal(3, MinDeltaIndex(pfi, m => m.RSquared.StandardDeviation));
            Assert.Equal(1, MaxDeltaIndex(pfi, m => m.RSquared.StandardDeviation));

            // Stardard Error will scale with the magnitude of the measure (as it's SD/sqrt(N))
            Assert.Equal(3, MinDeltaIndex(pfi, m => m.MeanAbsoluteError.StandardError));
            Assert.Equal(1, MaxDeltaIndex(pfi, m => m.MeanAbsoluteError.StandardError));

            Assert.Equal(3, MinDeltaIndex(pfi, m => m.MeanSquaredError.StandardError));
            Assert.Equal(1, MaxDeltaIndex(pfi, m => m.MeanSquaredError.StandardError));

            Assert.Equal(3, MinDeltaIndex(pfi, m => m.RootMeanSquaredError.StandardError));
            Assert.Equal(1, MaxDeltaIndex(pfi, m => m.RootMeanSquaredError.StandardError));

            Assert.Equal(3, MinDeltaIndex(pfi, m => m.RSquared.StandardError));
            Assert.Equal(1, MaxDeltaIndex(pfi, m => m.RSquared.StandardError));

            // And test that the Standard Deviation and Standard Error are related as we expect
            Assert.Equal(pfi[0].RootMeanSquaredError.StandardError, pfi[0].RootMeanSquaredError.StandardDeviation / Math.Sqrt(pfi[0].RootMeanSquaredError.Count));

            Done();
        }

        /// <summary>
        /// Test PFI Regression for Sparse Features
        /// </summary>
        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void TestPfiRegressionOnSparseFeatures(bool saveModel)
        {
            var data = GetSparseDataset();
            var model = ML.Regression.Trainers.OnlineGradientDescent().Fit(data);

            ImmutableArray<RegressionMetricsStatistics> results;
            ImmutableDictionary<string, RegressionMetricsStatistics> pfiDict;

            if (saveModel)
            {
                var modelAndSchemaPath = GetOutputPath("TestPfiRegressionOnSparseFeatures.zip");
                ML.Model.Save(model, data.Schema, modelAndSchemaPath);

                var loadedModel = ML.Model.Load(modelAndSchemaPath, out var schema);
                var castedModel = loadedModel as RegressionPredictionTransformer<LinearRegressionModelParameters>;

                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                results = ML.Regression.PermutationFeatureImportance(castedModel, data);
                pfiDict = ml2.Regression.PermutationFeatureImportance(loadedModel, data);
            }
            else
            {
                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                results = ML.Regression.PermutationFeatureImportance(model, data);
                pfiDict = ml2.Regression.PermutationFeatureImportance((ITransformer)model, data);
            }

            // Pfi Indices:
            // X1: 0
            // X2VBuffer-Slot-0: 1
            // X2VBuffer-Slot-1: 2
            // X2VBuffer-Slot-2: 3
            // X2VBuffer-Slot-3: 4
            // X3Important: 5

            // Make sure that PFI from the array and the dictionary both have the same value for each feature.
            Assert.Equal(JsonConvert.SerializeObject(results[0]), JsonConvert.SerializeObject(pfiDict["X1"]));
            Assert.Equal(JsonConvert.SerializeObject(results[1]), JsonConvert.SerializeObject(pfiDict["Slot 1"]));
            Assert.Equal(JsonConvert.SerializeObject(results[2]), JsonConvert.SerializeObject(pfiDict["Slot 2"]));
            Assert.Equal(JsonConvert.SerializeObject(results[3]), JsonConvert.SerializeObject(pfiDict["Slot 3"]));
            Assert.Equal(JsonConvert.SerializeObject(results[4]), JsonConvert.SerializeObject(pfiDict["Slot 4"]));
            Assert.Equal(JsonConvert.SerializeObject(results[5]), JsonConvert.SerializeObject(pfiDict["X3Important"]));

            // Permuted X2VBuffer-Slot-1 lot (f2) should have min impact on SGD metrics, X3Important -- max impact.
            // For the following metrics lower is better, so maximum delta means more important feature, and vice versa
            Assert.Equal(2, MinDeltaIndex(results, m => m.MeanAbsoluteError.Mean));
            Assert.Equal(5, MaxDeltaIndex(results, m => m.MeanAbsoluteError.Mean));

            Assert.Equal(2, MinDeltaIndex(results, m => m.MeanSquaredError.Mean));
            Assert.Equal(5, MaxDeltaIndex(results, m => m.MeanSquaredError.Mean));

            Assert.Equal(2, MinDeltaIndex(results, m => m.RootMeanSquaredError.Mean));
            Assert.Equal(5, MaxDeltaIndex(results, m => m.RootMeanSquaredError.Mean));

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.Equal(2, MaxDeltaIndex(results, m => m.RSquared.Mean));
            Assert.Equal(5, MinDeltaIndex(results, m => m.RSquared.Mean));
        }

        #endregion

        #region Binary Classification Tests
        /// <summary>
        /// Test PFI Binary Classification for Dense Features
        /// </summary>
        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void TestPfiBinaryClassificationOnDenseFeatures(bool saveModel)
        {
            var data = GetDenseDataset(TaskType.BinaryClassification);
            var model = ML.BinaryClassification.Trainers.LbfgsLogisticRegression(
                new LbfgsLogisticRegressionBinaryTrainer.Options { NumberOfThreads = 1 }).Fit(data);

            ImmutableArray<BinaryClassificationMetricsStatistics> pfi;
            ImmutableDictionary<string, BinaryClassificationMetricsStatistics> pfiDict;

            if (saveModel)
            {
                var modelAndSchemaPath = GetOutputPath("TestPfiBinaryClassificationOnDenseFeatures.zip");
                ML.Model.Save(model, data.Schema, modelAndSchemaPath);

                var loadedModel = ML.Model.Load(modelAndSchemaPath, out var schema);
                var castedModel = loadedModel as BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>;

                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                pfi = ML.BinaryClassification.PermutationFeatureImportance(castedModel, data);
                pfiDict = ml2.BinaryClassification.PermutationFeatureImportanceNonCalibrated(loadedModel, data);
            }
            else
            {
                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                pfi = ML.BinaryClassification.PermutationFeatureImportance(model, data);
                pfiDict = ml2.BinaryClassification.PermutationFeatureImportanceNonCalibrated((ITransformer)model, data);

            }

            // Pfi Indices:
            // X1: 0
            // X2Important: 1
            // X3: 2
            // X4Rand: 3

            // Make sure that PFI from the array and the dictionary both have the same value for each feature.
            Assert.Equal(JsonConvert.SerializeObject(pfi[0]), JsonConvert.SerializeObject(pfiDict["X1"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[1]), JsonConvert.SerializeObject(pfiDict["X2Important"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[2]), JsonConvert.SerializeObject(pfiDict["X3"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[3]), JsonConvert.SerializeObject(pfiDict["X4Rand"]));

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.AreaUnderRocCurve.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.AreaUnderRocCurve.Mean));
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.Accuracy.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.Accuracy.Mean));
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.PositivePrecision.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.PositivePrecision.Mean));
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.PositiveRecall.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.PositiveRecall.Mean));
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.NegativePrecision.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.NegativePrecision.Mean));
            Assert.Equal(0, MaxDeltaIndex(pfi, m => m.NegativeRecall.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.NegativeRecall.Mean));
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.F1Score.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.F1Score.Mean));
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.AreaUnderPrecisionRecallCurve.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.AreaUnderPrecisionRecallCurve.Mean));

            Done();
        }

        /// <summary>
        /// Test PFI Binary Classification for Sparse Features
        /// </summary>
        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void TestPfiBinaryClassificationOnSparseFeatures(bool saveModel)
        {
            var data = GetSparseDataset(TaskType.BinaryClassification);
            var model = ML.BinaryClassification.Trainers.LbfgsLogisticRegression(
                new LbfgsLogisticRegressionBinaryTrainer.Options { NumberOfThreads = 1 }).Fit(data);

            ImmutableArray<BinaryClassificationMetricsStatistics> pfi;
            ImmutableDictionary<string, BinaryClassificationMetricsStatistics> pfiDict;

            if (saveModel)
            {
                var modelAndSchemaPath = GetOutputPath("TestPfiBinaryClassificationOnSparseFeatures.zip");
                ML.Model.Save(model, data.Schema, modelAndSchemaPath);

                var loadedModel = ML.Model.Load(modelAndSchemaPath, out var schema);
                var castedModel = loadedModel as BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>;

                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                pfi = ML.BinaryClassification.PermutationFeatureImportance(castedModel, data);
                pfiDict = ml2.BinaryClassification.PermutationFeatureImportanceNonCalibrated(loadedModel, data);
            }
            else
            {
                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                pfi = ML.BinaryClassification.PermutationFeatureImportance(model, data);
                pfiDict = ml2.BinaryClassification.PermutationFeatureImportanceNonCalibrated((ITransformer)model, data);
            }

            // Pfi Indices:
            // X1: 0
            // X2VBuffer-Slot-0: 1
            // X2VBuffer-Slot-1: 2
            // X2VBuffer-Slot-2: 3
            // X2VBuffer-Slot-3: 4
            // X3Important: 5

            // Make sure that PFI from the array and the dictionary both have the same value for each feature.
            Assert.Equal(JsonConvert.SerializeObject(pfi[0]), JsonConvert.SerializeObject(pfiDict["X1"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[1]), JsonConvert.SerializeObject(pfiDict["Slot 1"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[2]), JsonConvert.SerializeObject(pfiDict["Slot 2"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[3]), JsonConvert.SerializeObject(pfiDict["Slot 3"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[4]), JsonConvert.SerializeObject(pfiDict["Slot 4"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[5]), JsonConvert.SerializeObject(pfiDict["X3Important"]));

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.Equal(2, MaxDeltaIndex(pfi, m => m.AreaUnderRocCurve.Mean));
            Assert.Equal(5, MinDeltaIndex(pfi, m => m.AreaUnderRocCurve.Mean));
            Assert.Equal(2, MaxDeltaIndex(pfi, m => m.Accuracy.Mean));
            Assert.Equal(5, MinDeltaIndex(pfi, m => m.Accuracy.Mean));
            Assert.Equal(2, MaxDeltaIndex(pfi, m => m.PositivePrecision.Mean));
            Assert.Equal(5, MinDeltaIndex(pfi, m => m.PositivePrecision.Mean));
            Assert.Equal(2, MaxDeltaIndex(pfi, m => m.PositiveRecall.Mean));
            Assert.Equal(5, MinDeltaIndex(pfi, m => m.PositiveRecall.Mean));
            Assert.Equal(2, MaxDeltaIndex(pfi, m => m.NegativePrecision.Mean));
            Assert.Equal(5, MinDeltaIndex(pfi, m => m.NegativePrecision.Mean));
            Assert.Equal(2, MaxDeltaIndex(pfi, m => m.NegativeRecall.Mean));
            Assert.Equal(5, MinDeltaIndex(pfi, m => m.NegativeRecall.Mean));
            Assert.Equal(2, MaxDeltaIndex(pfi, m => m.F1Score.Mean));
            Assert.Equal(5, MinDeltaIndex(pfi, m => m.F1Score.Mean));
            Assert.Equal(2, MaxDeltaIndex(pfi, m => m.AreaUnderPrecisionRecallCurve.Mean));
            Assert.Equal(5, MinDeltaIndex(pfi, m => m.AreaUnderPrecisionRecallCurve.Mean));

            Done();
        }

        [Fact]
        public void TestBinaryClassificationWithoutCalibrator()
        {
            var dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var ff = ML.BinaryClassification.Trainers.FastForest();
            var data = ML.Data.LoadFromTextFile(dataPath,
                            new[] { new TextLoader.Column("Label", DataKind.Boolean, 0),
                            new TextLoader.Column("Features", DataKind.Single, 1, 9) });
            var model = ff.Fit(data);
            var pfi = ML.BinaryClassification.PermutationFeatureImportance(model, data);

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.Equal(7, MaxDeltaIndex(pfi, m => m.AreaUnderRocCurve.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.AreaUnderRocCurve.Mean));
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.Accuracy.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.Accuracy.Mean));
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.PositivePrecision.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.PositivePrecision.Mean));
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.PositiveRecall.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.PositiveRecall.Mean));
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.NegativePrecision.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.NegativePrecision.Mean));
            Assert.Equal(2, MaxDeltaIndex(pfi, m => m.NegativeRecall.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.NegativeRecall.Mean));
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.F1Score.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.F1Score.Mean));
            Assert.Equal(7, MaxDeltaIndex(pfi, m => m.AreaUnderPrecisionRecallCurve.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.AreaUnderPrecisionRecallCurve.Mean));
        }
        #endregion

        #region Multiclass Classification Tests
        /// <summary>
        /// Test PFI Multiclass Classification for Dense Features
        /// </summary>
        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void TestPfiMulticlassClassificationOnDenseFeatures(bool saveModel)
        {
            var data = GetDenseDataset(TaskType.MulticlassClassification);
            var model = ML.MulticlassClassification.Trainers.LbfgsMaximumEntropy().Fit(data);

            ImmutableArray<MulticlassClassificationMetricsStatistics> pfi;
            ImmutableDictionary<string, MulticlassClassificationMetricsStatistics> pfiDict;

            if (saveModel)
            {
                var modelAndSchemaPath = GetOutputPath("TestPfiMulticlassClassificationOnDenseFeatures.zip");
                ML.Model.Save(model, data.Schema, modelAndSchemaPath);

                var loadedModel = ML.Model.Load(modelAndSchemaPath, out var schema);
                var castedModel = loadedModel as MulticlassPredictionTransformer<MaximumEntropyModelParameters>;

                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                pfi = ML.MulticlassClassification.PermutationFeatureImportance(castedModel, data);
                pfiDict = ml2.MulticlassClassification.PermutationFeatureImportance(loadedModel, data);
            }
            else
            {
                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                pfi = ML.MulticlassClassification.PermutationFeatureImportance(model, data);
                pfiDict = ml2.MulticlassClassification.PermutationFeatureImportance((ITransformer)model, data);
            }

            // Pfi Indices:
            // X1: 0
            // X2Important: 1
            // X3: 2
            // X4Rand: 3

            // Make sure that PFI from the array and the dictionary both have the same value for each feature.
            Assert.Equal(JsonConvert.SerializeObject(pfi[0]), JsonConvert.SerializeObject(pfiDict["X1"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[1]), JsonConvert.SerializeObject(pfiDict["X2Important"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[2]), JsonConvert.SerializeObject(pfiDict["X3"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[3]), JsonConvert.SerializeObject(pfiDict["X4Rand"]));

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.MicroAccuracy.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.MicroAccuracy.Mean));
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.MacroAccuracy.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.MacroAccuracy.Mean));
            Assert.Equal(3, MaxDeltaIndex(pfi, m => m.LogLossReduction.Mean));
            Assert.Equal(1, MinDeltaIndex(pfi, m => m.LogLossReduction.Mean));

            // For the following metrics-delta lower is better, so maximum delta means more important feature, and vice versa
            //  Because they are _negative_, the difference will be positive for worse classifiers.
            Assert.Equal(1, MaxDeltaIndex(pfi, m => m.LogLoss.Mean));
            Assert.Equal(3, MinDeltaIndex(pfi, m => m.LogLoss.Mean));
            for (int i = 0; i < pfi[0].PerClassLogLoss.Count; i++)
            {
                Assert.True(MaxDeltaIndex(pfi, m => m.PerClassLogLoss[i].Mean) == 1);
                Assert.True(MinDeltaIndex(pfi, m => m.PerClassLogLoss[i].Mean) == 3);
            }

            Done();
        }

        /// <summary>
        /// Test PFI Multiclass Classification for Sparse Features
        /// </summary>
        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void TestPfiMulticlassClassificationOnSparseFeatures(bool saveModel)
        {
            var data = GetSparseDataset(TaskType.MulticlassClassification);
            var model = ML.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                new LbfgsMaximumEntropyMulticlassTrainer.Options { MaximumNumberOfIterations = 1000 }).Fit(data);

            ImmutableArray<MulticlassClassificationMetricsStatistics> pfi;
            ImmutableDictionary<string, MulticlassClassificationMetricsStatistics> pfiDict;

            if (saveModel)
            {
                var modelAndSchemaPath = GetOutputPath("TestPfiMulticlassClassificationOnSparseFeatures.zip");
                ML.Model.Save(model, data.Schema, modelAndSchemaPath);

                var loadedModel = ML.Model.Load(modelAndSchemaPath, out var schema);
                var castedModel = loadedModel as MulticlassPredictionTransformer<MaximumEntropyModelParameters>;

                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                pfi = ML.MulticlassClassification.PermutationFeatureImportance(castedModel, data);
                pfiDict = ml2.MulticlassClassification.PermutationFeatureImportance(loadedModel, data);
            }
            else
            {
                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                pfi = ML.MulticlassClassification.PermutationFeatureImportance(model, data);
                pfiDict = ml2.MulticlassClassification.PermutationFeatureImportance((ITransformer)model, data);
            }

            // Pfi Indices:
            // X1: 0
            // X2VBuffer-Slot-0: 1
            // X2VBuffer-Slot-1: 2 // Least important
            // X2VBuffer-Slot-2: 3
            // X2VBuffer-Slot-3: 4
            // X3Important: 5 // Most important

            // Make sure that PFI from the array and the dictionary both have the same value for each feature.
            Assert.Equal(JsonConvert.SerializeObject(pfi[0]), JsonConvert.SerializeObject(pfiDict["X1"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[1]), JsonConvert.SerializeObject(pfiDict["Slot 1"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[2]), JsonConvert.SerializeObject(pfiDict["Slot 2"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[3]), JsonConvert.SerializeObject(pfiDict["Slot 3"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[4]), JsonConvert.SerializeObject(pfiDict["Slot 4"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[5]), JsonConvert.SerializeObject(pfiDict["X3Important"]));

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            Assert.Equal(2, MaxDeltaIndex(pfi, m => m.MicroAccuracy.Mean));
            Assert.Equal(5, MinDeltaIndex(pfi, m => m.MicroAccuracy.Mean));
            Assert.Equal(2, MaxDeltaIndex(pfi, m => m.MacroAccuracy.Mean));
            Assert.Equal(5, MinDeltaIndex(pfi, m => m.MacroAccuracy.Mean));
            Assert.Equal(2, MaxDeltaIndex(pfi, m => m.LogLossReduction.Mean));
            Assert.Equal(5, MinDeltaIndex(pfi, m => m.LogLossReduction.Mean));

            // For the following metrics-delta lower is better, so maximum delta means more important feature, and vice versa
            //  Because they are negative metrics, the _difference_ will be positive for worse classifiers.
            Assert.Equal(5, MaxDeltaIndex(pfi, m => m.LogLoss.Mean));
            Assert.Equal(2, MinDeltaIndex(pfi, m => m.LogLoss.Mean));
            for (int i = 0; i < pfi[0].PerClassLogLoss.Count; i++)
            {
                Assert.Equal(5, MaxDeltaIndex(pfi, m => m.PerClassLogLoss[i].Mean));
                Assert.Equal(2, MinDeltaIndex(pfi, m => m.PerClassLogLoss[i].Mean));
            }

            Done();
        }
        #endregion

        #region Ranking Tests
        /// <summary>
        /// Test PFI Ranking Classification for Dense Features
        /// </summary>
        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void TestPfiRankingOnDenseFeatures(bool saveModel)
        {
            var data = GetDenseDataset(TaskType.Ranking);
            var model = ML.Ranking.Trainers.FastTree().Fit(data);

            ImmutableArray<RankingMetricsStatistics> pfi;
            ImmutableDictionary<string, RankingMetricsStatistics> pfiDict;

            if (saveModel)
            {
                var modelAndSchemaPath = GetOutputPath("TestPfiRankingOnDenseFeatures.zip");
                ML.Model.Save(model, data.Schema, modelAndSchemaPath);

                var loadedModel = ML.Model.Load(modelAndSchemaPath, out var schema);
                var castedModel = loadedModel as RankingPredictionTransformer<FastTreeRankingModelParameters>;

                // Saving and Loading the model cause the internal random state to change, so we reset the seed
                // here and create another seed for both PFI to match to help the tests pass.
                ML = new MLContext(0);
                var ml2 = new MLContext(0);

                pfi = ML.Ranking.PermutationFeatureImportance(castedModel, data);
                pfiDict = ml2.Ranking.PermutationFeatureImportance(loadedModel, data);
            }
            else
            {
                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(0);
                var ml2 = new MLContext(0);

                pfi = ML.Ranking.PermutationFeatureImportance(model, data);
                pfiDict = ml2.Ranking.PermutationFeatureImportance((ITransformer)model, data);
            }


            // Pfi Indices:
            // X1: 0 // For Ranking, this column won't result in misorderings
            // X2Important: 1
            // X3: 2
            // X4Rand: 3

            // Make sure that PFI from the array and the dictionary both have the same value for each feature.
            Assert.Equal(JsonConvert.SerializeObject(pfi[0]), JsonConvert.SerializeObject(pfiDict["X1"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[1]), JsonConvert.SerializeObject(pfiDict["X2Important"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[2]), JsonConvert.SerializeObject(pfiDict["X3"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[3]), JsonConvert.SerializeObject(pfiDict["X4Rand"]));

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            for (int i = 0; i < pfi[0].DiscountedCumulativeGains.Count; i++)
            {
                Assert.Equal(0, MaxDeltaIndex(pfi, m => m.DiscountedCumulativeGains[i].Mean));
                Assert.Equal(1, MinDeltaIndex(pfi, m => m.DiscountedCumulativeGains[i].Mean));
            }
            for (int i = 0; i < pfi[0].NormalizedDiscountedCumulativeGains.Count; i++)
            {
                Assert.Equal(0, MaxDeltaIndex(pfi, m => m.NormalizedDiscountedCumulativeGains[i].Mean));
                Assert.Equal(1, MinDeltaIndex(pfi, m => m.NormalizedDiscountedCumulativeGains[i].Mean));
            }

            Done();
        }


        /// <summary>
        /// Test PFI Ranking Classification for Sparse Features
        /// </summary>
        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void TestPfiRankingOnSparseFeatures(bool saveModel)
        {
            var data = GetSparseDataset(TaskType.Ranking);
            var model = ML.Ranking.Trainers.FastTree().Fit(data);

            ImmutableArray<RankingMetricsStatistics> pfi;
            ImmutableDictionary<string, RankingMetricsStatistics> pfiDict;

            if (saveModel)
            {
                var modelAndSchemaPath = GetOutputPath("TestPfiRankingOnSparseFeatures.zip");
                ML.Model.Save(model, data.Schema, modelAndSchemaPath);

                var loadedModel = ML.Model.Load(modelAndSchemaPath, out var schema);
                var castedModel = loadedModel as RankingPredictionTransformer<FastTreeRankingModelParameters>;

                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                pfi = ML.Ranking.PermutationFeatureImportance(castedModel, data);
                pfiDict = ml2.Ranking.PermutationFeatureImportance(loadedModel, data);
            }
            else
            {
                // PFI changes the random state, so we need to reset it and create another seed for both PFI to match
                ML = new MLContext(42);
                var ml2 = new MLContext(42);

                pfi = ML.Ranking.PermutationFeatureImportance(model, data);
                pfiDict = ml2.Ranking.PermutationFeatureImportance((ITransformer)model, data);
            }

            // Pfi Indices:
            // X1: 0
            // X2VBuffer-Slot-0: 1
            // X2VBuffer-Slot-1: 2 // Least important
            // X2VBuffer-Slot-2: 3
            // X2VBuffer-Slot-3: 4
            // X3Important: 5 // Most important

            // Make sure that PFI from the array and the dictionary both have the same value for each feature.
            Assert.Equal(JsonConvert.SerializeObject(pfi[0]), JsonConvert.SerializeObject(pfiDict["X1"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[1]), JsonConvert.SerializeObject(pfiDict["Slot 1"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[2]), JsonConvert.SerializeObject(pfiDict["Slot 2"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[3]), JsonConvert.SerializeObject(pfiDict["Slot 3"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[4]), JsonConvert.SerializeObject(pfiDict["Slot 4"]));
            Assert.Equal(JsonConvert.SerializeObject(pfi[5]), JsonConvert.SerializeObject(pfiDict["X3Important"]));

            // For the following metrics higher is better, so minimum delta means more important feature, and vice versa
            for (int i = 0; i < pfi[0].DiscountedCumulativeGains.Count; i++)
            {
                Assert.Equal(2, MaxDeltaIndex(pfi, m => m.DiscountedCumulativeGains[i].Mean));
                Assert.Equal(5, MinDeltaIndex(pfi, m => m.DiscountedCumulativeGains[i].Mean));
            }
            for (int i = 0; i < pfi[0].NormalizedDiscountedCumulativeGains.Count; i++)
            {
                Assert.Equal(2, MaxDeltaIndex(pfi, m => m.NormalizedDiscountedCumulativeGains[i].Mean));
                Assert.Equal(5, MinDeltaIndex(pfi, m => m.NormalizedDiscountedCumulativeGains[i].Mean));
            }

            Done();
        }
        #endregion

        #region Helpers
        /// <summary>
        /// Features: x1, x2, x3, xRand; y = 10*x1 + 20x2 + 5.5x3 + e, xRand- random and Label y is to dependant on xRand.
        /// xRand has the least importance: Evaluation metrics do not change a lot when xRand is permuted.
        /// x2 has the biggest importance.
        /// </summary>
        private IDataView GetDenseDataset(TaskType task = TaskType.Regression)
        {
            // Setup synthetic dataset.
            const int numberOfInstances = 1000;
            var rand = new Random(10);
            float[] yArray = new float[numberOfInstances];
            float[] x1Array = new float[numberOfInstances];
            float[] x2Array = new float[numberOfInstances];
            float[] x3Array = new float[numberOfInstances];
            float[] x4RandArray = new float[numberOfInstances];

            for (var i = 0; i < numberOfInstances; i++)
            {
                var x1 = rand.Next(1000);
                x1Array[i] = x1;
                var x2Important = rand.Next(10000);
                x2Array[i] = x2Important;
                var x3 = rand.Next(5000);
                x3Array[i] = x3;
                var x4Rand = rand.Next(1000);
                x4RandArray[i] = x4Rand;

                var noise = rand.Next(50);

                yArray[i] = (float)(10 * x1 + 20 * x2Important + 5.5 * x3 + noise);
            }

            // If binary classification, modify the labels
            if (task == TaskType.BinaryClassification ||
                task == TaskType.MulticlassClassification)
                GetBinaryClassificationLabels(yArray);
            else if (task == TaskType.Ranking)
                GetRankingLabels(yArray);

            // Create data view.
            var bldr = new ArrayDataViewBuilder(Env);
            bldr.AddColumn("X1", NumberDataViewType.Single, x1Array);
            bldr.AddColumn("X2Important", NumberDataViewType.Single, x2Array);
            bldr.AddColumn("X3", NumberDataViewType.Single, x3Array);
            bldr.AddColumn("X4Rand", NumberDataViewType.Single, x4RandArray);
            bldr.AddColumn("Label", NumberDataViewType.Single, yArray);
            if (task == TaskType.Ranking)
                bldr.AddColumn("GroupId", NumberDataViewType.UInt32, CreateGroupIds(yArray.Length));
            var srcDV = bldr.GetDataView();

            var pipeline = ML.Transforms.Concatenate("Features", "X1", "X2Important", "X3", "X4Rand")
                .Append(ML.Transforms.NormalizeMinMax("Features"));
            if (task == TaskType.BinaryClassification)
                return pipeline.Append(ML.Transforms.Conversion.ConvertType("Label", outputKind: DataKind.Boolean))
                    .Fit(srcDV).Transform(srcDV);
            else if (task == TaskType.MulticlassClassification)
                return pipeline.Append(ML.Transforms.Conversion.MapValueToKey("Label"))
                    .Fit(srcDV).Transform(srcDV);
            else if (task == TaskType.Ranking)
                return pipeline.Append(ML.Transforms.Conversion.MapValueToKey("GroupId"))
                    .Fit(srcDV).Transform(srcDV);

            return pipeline.Fit(srcDV).Transform(srcDV);
        }

        /// <summary>
        /// Features: x1, x2vBuff(sparce vector), x3.
        /// y = 10x1 + 10x2vBuff + 30x3 + e.
        /// Within xBuff feature  2nd slot will be sparse most of the time.
        /// 2nd slot of xBuff has the least importance: Evaluation metrics do not change a lot when this slot is permuted.
        /// x2 has the biggest importance.
        /// </summary>
        private IDataView GetSparseDataset(TaskType task = TaskType.Regression)
        {
            // Setup synthetic dataset.
            const int numberOfInstances = 10000;
            var rand = new Random(10);
            float[] yArray = new float[numberOfInstances];
            float[] x1Array = new float[numberOfInstances];
            float[] x3Array = new float[numberOfInstances];

            VBuffer<float>[] vbArray = new VBuffer<float>[numberOfInstances];

            for (var i = 0; i < numberOfInstances; i++)
            {
                var x1 = rand.Next(1000);
                x1Array[i] = x1;
                var x3Important = rand.Next(10000);
                x3Array[i] = x3Important;

                VBuffer<float> vb;

                if (i % 10 != 0)
                {
                    vb = new VBuffer<float>(4, 3, new float[] { rand.Next(1000), rand.Next(1000), rand.Next(1000) }, new int[] { 0, 2, 3 });
                }
                else
                {
                    vb = new VBuffer<float>(4, 4, new float[] { rand.Next(1000), rand.Next(1000), rand.Next(1000), rand.Next(1000) }, new int[] { 0, 1, 2, 3 });
                }

                vbArray[i] = vb;

                float vbSum = 0;
                foreach (var vbValue in vb.DenseValues())
                {
                    vbSum += vbValue * 10;
                }

                var noise = rand.Next(50);
                yArray[i] = 10 * x1 + vbSum + 20 * x3Important + noise;
            }

            // If binary classification, modify the labels
            if (task == TaskType.BinaryClassification ||
                task == TaskType.MulticlassClassification)
                GetBinaryClassificationLabels(yArray);
            else if (task == TaskType.Ranking)
                GetRankingLabels(yArray);

            // Create data view.
            var bldr = new ArrayDataViewBuilder(Env);
            bldr.AddColumn("X1", NumberDataViewType.Single, x1Array);
            bldr.AddColumn("X2VBuffer", NumberDataViewType.Single, vbArray);
            bldr.AddColumn("X3Important", NumberDataViewType.Single, x3Array);
            bldr.AddColumn("Label", NumberDataViewType.Single, yArray);
            if (task == TaskType.Ranking)
                bldr.AddColumn("GroupId", NumberDataViewType.UInt32, CreateGroupIds(yArray.Length));
            var srcDV = bldr.GetDataView();

            var pipeline = ML.Transforms.Concatenate("Features", "X1", "X2VBuffer", "X3Important")
                .Append(ML.Transforms.NormalizeMinMax("Features"));
            if (task == TaskType.BinaryClassification)
            {
                return pipeline.Append(ML.Transforms.Conversion.ConvertType("Label", outputKind: DataKind.Boolean))
                    .Fit(srcDV).Transform(srcDV);
            }
            else if (task == TaskType.MulticlassClassification)
            {
                return pipeline.Append(ML.Transforms.Conversion.MapValueToKey("Label"))
                    .Fit(srcDV).Transform(srcDV);
            }
            else if (task == TaskType.Ranking)
                return pipeline.Append(ML.Transforms.Conversion.MapValueToKey("GroupId"))
                    .Fit(srcDV).Transform(srcDV);

            return pipeline.Fit(srcDV).Transform(srcDV);
        }

        private int MinDeltaIndex<T>(
            ImmutableArray<T> metricsDelta,
            Func<T, double> metricSelector)
        {
            var min = metricsDelta.OrderBy(m => metricSelector(m)).First();
            return metricsDelta.IndexOf(min);
        }

        private int MaxDeltaIndex<T>(
            ImmutableArray<T> metricsDelta,
            Func<T, double> metricSelector)
        {
            var max = metricsDelta.OrderByDescending(m => metricSelector(m)).First();
            return metricsDelta.IndexOf(max);
        }

        private void GetBinaryClassificationLabels(float[] rawScores)
        {
            float averageScore = GetArrayAverage(rawScores);

            // Center the response and then take the sigmoid to generate the classes
            for (int i = 0; i < rawScores.Length; i++)
                rawScores[i] = MathUtils.Sigmoid(rawScores[i] - averageScore) > 0.5 ? 1 : 0;
        }

        private void GetRankingLabels(float[] rawScores)
        {
            var min = MathUtils.Min(rawScores);
            var max = MathUtils.Max(rawScores);

            for (int i = 0; i < rawScores.Length; i++)
            {
                // Bin from [zero,one), then expand out to [0,5) and truncate
                rawScores[i] = (int)(5 * (rawScores[i] - min) / (max - min));
                if (rawScores[i] == 5)
                    rawScores[i] = 4;
            }
        }

        private float GetArrayAverage(float[] scores)
        {
            // Compute the average so we can center the response
            float averageScore = 0.0f;
            for (int i = 0; i < scores.Length; i++)
                averageScore += scores[i];
            averageScore /= scores.Length;

            return averageScore;
        }

        private uint[] CreateGroupIds(int numRows, int rowsPerGroup = 5)
        {
            var groupIds = new uint[numRows];
            // Construct groups of rowsPerGroup using a modulo counter
            uint group = 0;
            for (int i = 0; i < groupIds.Length; i++)
            {
                if (i % rowsPerGroup == 0)
                    group++;
                groupIds[i] = group;
            }

            return groupIds;
        }

        private enum TaskType
        {
            Regression,
            BinaryClassification,
            MulticlassClassification,
            Ranking
        }
        #endregion
    }
}
