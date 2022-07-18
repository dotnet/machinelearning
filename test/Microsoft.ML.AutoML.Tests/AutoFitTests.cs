// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Xunit;
using Xunit.Abstractions;
using static Microsoft.ML.DataOperationsCatalog;

namespace Microsoft.ML.AutoML.Test
{
    public class AutoFitTests : BaseTestClass
    {
        // Marker necessary for AutoFitContextLogTest to ensure that the wanted logs
        // from Experiment's sub MLContexts were relayed to the main calling MLContext.
        bool _markerAutoFitContextLogTest;
        public AutoFitTests(ITestOutputHelper output) : base(output)
        {
        }

        private void MlContextLog(object sender, LoggingEventArgs e)
        {
            // Log containing ImageClassificationTrainer will only come from AutoML's sub
            // contexts.
            if (!_markerAutoFitContextLogTest && e.Message.Contains("[Source=ImageClassificationTrainer;"))
                _markerAutoFitContextLogTest = true;
        }

        [TensorFlowFact]
        public void AutoFitContextLogTest()
        {
            // This test confirms that logs produced from contexts made during AutoML experiment
            // runs are correctly relayed to the main Experiment MLContext.
            _markerAutoFitContextLogTest = false;
            var context = new MLContext(1);
            context.Log += MlContextLog;
            var datasetPath = DatasetUtil.GetFlowersDataset();
            var columnInference = context.Auto().InferColumns(datasetPath, "Label");
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(datasetPath);
            var result = context.Auto()
                            .CreateMulticlassClassificationExperiment(15)
                            .Execute(trainData, columnInference.ColumnInformation);
            Assert.True(_markerAutoFitContextLogTest, "Image classification trainer logs from Experiment's sub contexts" +
                "were not relayed to the main MLContext.");
        }

        [LightGBMFact]
        public void AutoFit_UCI_Adult_Test()
        {
            var context = new MLContext(1);
            var dataPath = DatasetUtil.GetUciAdultDataset();
            var columnInference = context.Auto().InferColumns(dataPath, DatasetUtil.UciAdultLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(dataPath);
            var result = context.Auto()
                .CreateBinaryClassificationExperiment(1)
                .Execute(trainData, new ColumnInformation() { LabelColumnName = DatasetUtil.UciAdultLabel });
            Assert.True(result.BestRun.ValidationMetrics.Accuracy > 0.70);
            Assert.NotNull(result.BestRun.Estimator);
            Assert.NotNull(result.BestRun.Model);
            Assert.NotNull(result.BestRun.TrainerName);
        }

        [LightGBMFact]
        public void AutoFit_UCI_Adult_Train_Test_Split_Test()
        {
            var context = new MLContext(1);
            var dataPath = DatasetUtil.GetUciAdultDataset();
            var columnInference = context.Auto().InferColumns(dataPath, DatasetUtil.UciAdultLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(dataPath);
            var dataTrainTest = context.Data.TrainTestSplit(trainData);
            var result = context.Auto()
                .CreateBinaryClassificationExperiment(1)
                .Execute(dataTrainTest.TrainSet, dataTrainTest.TestSet, DatasetUtil.UciAdultLabel);
            Assert.True(result.BestRun.ValidationMetrics.Accuracy > 0.70);
            Assert.NotNull(result.BestRun.Estimator);
            Assert.NotNull(result.BestRun.Model);
            Assert.NotNull(result.BestRun.TrainerName);
        }

        [LightGBMFact]
        public void AutoFit_UCI_Adult_CrossValidation_10_Test()
        {
            var context = new MLContext(1);
            var dataPath = DatasetUtil.GetUciAdultDataset();
            var columnInference = context.Auto().InferColumns(dataPath, DatasetUtil.UciAdultLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(dataPath);
            var result = context.Auto()
                .CreateBinaryClassificationExperiment(1)
                .Execute(trainData, 10, DatasetUtil.UciAdultLabel);
            Assert.True(result.BestRun.Results.Select(x => x.ValidationMetrics.Accuracy).Min() > 0.70);
            Assert.NotNull(result.BestRun.Estimator);
            Assert.NotNull(result.BestRun.TrainerName);
        }

        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void AutoFitMultiTest(bool useNumberOfCVFolds)
        {
            var context = new MLContext(0);
            var columnInference = context.Auto().InferColumns(DatasetUtil.TrivialMulticlassDatasetPath, DatasetUtil.TrivialMulticlassDatasetLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(DatasetUtil.TrivialMulticlassDatasetPath);

            if (useNumberOfCVFolds)
            {
                // When setting numberOfCVFolds
                // The results object is a CrossValidationExperimentResults<> object
                uint numberOfCVFolds = 5;
                var result = context.Auto()
                    .CreateMulticlassClassificationExperiment(0)
                    .Execute(trainData, numberOfCVFolds, DatasetUtil.TrivialMulticlassDatasetLabel);

                Assert.True(result.BestRun.Results.First().ValidationMetrics.MicroAccuracy >= 0.7);
                var scoredData = result.BestRun.Results.First().Model.Transform(trainData);
                Assert.Equal(NumberDataViewType.Single, scoredData.Schema[DefaultColumnNames.PredictedLabel].Type);
            }
            else
            {
                // When using this other API, if the trainset is under the
                // crossValRowCountThreshold, AutoML will also perform CrossValidation
                // but through a very different path that the one above,
                // throw a CrossValSummaryRunner and will return
                // a different type of object as "result" which would now be
                // simply a ExperimentResult<> object

                int crossValRowCountThreshold = 15000;
                trainData = context.Data.TakeRows(trainData, crossValRowCountThreshold - 1);
                var result = context.Auto()
                    .CreateMulticlassClassificationExperiment(0)
                    .Execute(trainData, DatasetUtil.TrivialMulticlassDatasetLabel);

                Assert.True(result.BestRun.ValidationMetrics.MicroAccuracy >= 0.7);
                var scoredData = result.BestRun.Model.Transform(trainData);
                Assert.Equal(NumberDataViewType.Single, scoredData.Schema[DefaultColumnNames.PredictedLabel].Type);
            }
        }

        [TensorFlowFact]
        //Skipping test temporarily. This test will be re-enabled once the cause of failures has been determined
        [Trait("Category", "SkipInCI")]
        public void AutoFitImageClassificationTrainTest()
        {
            var context = new MLContext(seed: 1);
            var datasetPath = DatasetUtil.GetFlowersDataset();
            var columnInference = context.Auto().InferColumns(datasetPath, "Label");
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = context.Data.ShuffleRows(textLoader.Load(datasetPath), seed: 1);
            var originalColumnNames = trainData.Schema.Select(c => c.Name);
            TrainTestData trainTestData = context.Data.TrainTestSplit(trainData, testFraction: 0.2, seed: 1);
            IDataView trainDataset = SplitUtil.DropAllColumnsExcept(context, trainTestData.TrainSet, originalColumnNames);
            IDataView testDataset = SplitUtil.DropAllColumnsExcept(context, trainTestData.TestSet, originalColumnNames);
            var result = context.Auto()
                            .CreateMulticlassClassificationExperiment(0)
                            .Execute(trainDataset, testDataset, columnInference.ColumnInformation);

            Assert.Equal(1, result.BestRun.ValidationMetrics.MicroAccuracy, 3);

            var scoredData = result.BestRun.Model.Transform(trainData);
            Assert.Equal(TextDataViewType.Instance, scoredData.Schema[DefaultColumnNames.PredictedLabel].Type);
        }

        [Fact(Skip = "Takes too much time, ~10 minutes.")]
        public void AutoFitImageClassification()
        {
            // This test executes the code path that model builder code will take to get a model using image
            // classification API.

            var context = new MLContext(1);
            context.Log += Context_Log;
            var datasetPath = DatasetUtil.GetFlowersDataset();
            var columnInference = context.Auto().InferColumns(datasetPath, "Label");
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(datasetPath);
            var result = context.Auto()
                            .CreateMulticlassClassificationExperiment(0)
                            .Execute(trainData, columnInference.ColumnInformation);

            Assert.InRange(result.BestRun.ValidationMetrics.MicroAccuracy, 0.80, 0.9);
            var scoredData = result.BestRun.Model.Transform(trainData);
            Assert.Equal(TextDataViewType.Instance, scoredData.Schema[DefaultColumnNames.PredictedLabel].Type);
        }

        private void Context_Log(object sender, LoggingEventArgs e)
        {
            //throw new NotImplementedException();
        }

        [Theory]
        [InlineData("en-US")]
        [InlineData("ar-SA")]
        [InlineData("pl-PL")]
        public void AutoFitRegressionTest(string culture)
        {
            var originalCulture = Thread.CurrentThread.CurrentCulture;
            try
            {
                Thread.CurrentThread.CurrentCulture = new CultureInfo(culture);

                // If users run AutoML with a different locale, sometimes
                // the sweeper encounters problems when parsing some strings.
                // So testing in another culture is necessary.
                // Furthermore, these issues might only occur after ~70
                // iterations, so setting the internal maxModels parameter.
                int maxModels = culture == "en-US" ? 1 : 75;

                var experimentSettings = new RegressionExperimentSettings { MaxModels = maxModels };

                if (!Environment.Is64BitProcess)
                {
                    // LightGBM isn't available on x86 machines
                    experimentSettings.Trainers.Remove(RegressionTrainer.LightGbm);
                }

                var context = new MLContext(1);
                var dataPath = DatasetUtil.GetMlNetGeneratedRegressionDataset();
                var columnInference = context.Auto().InferColumns(dataPath, DatasetUtil.MlNetGeneratedRegressionLabel);
                var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
                var trainData = textLoader.Load(dataPath);
                var validationData = context.Data.TakeRows(trainData, 20);
                trainData = context.Data.SkipRows(trainData, 20);
                var result = context.Auto()
                    .CreateRegressionExperiment(experimentSettings)
                    .Execute(trainData, validationData,
                        new ColumnInformation() { LabelColumnName = DatasetUtil.MlNetGeneratedRegressionLabel });

                Assert.True(result.RunDetails.Max(i => i?.ValidationMetrics?.RSquared) > 0.99);

                // Test the internal maxModels parameter
                Assert.True(culture == "en-US" || result.RunDetails.Count() == 75, $"RunDetails.Count() = {result.RunDetails.Count()}, is not 75");
            }
            finally
            {
                Thread.CurrentThread.CurrentCulture = originalCulture;
            }
        }

        [LightGBMFact]
        public void AutoFitRankingTest()
        {
            string labelColumnName = "Label";
            string scoreColumnName = "Score";
            string groupIdColumnName = "GroupId";
            string featuresColumnVectorNameA = "FeatureVectorA";
            string featuresColumnVectorNameB = "FeatureVectorB";
            var mlContext = new MLContext(1);

            // STEP 1: Load data
            var reader = new TextLoader(mlContext, GetLoaderArgsRank(labelColumnName, groupIdColumnName, featuresColumnVectorNameA, featuresColumnVectorNameB));
            var trainDataView = reader.Load(new MultiFileSource(DatasetUtil.GetMLSRDataset()));
            var testDataView = mlContext.Data.TakeRows(trainDataView, 500);
            trainDataView = mlContext.Data.SkipRows(trainDataView, 500);

            // STEP 2: Run AutoML experiment
            var settings = new RankingExperimentSettings()
            {
                MaxExperimentTimeInSeconds = 5,
                OptimizationMetricTruncationLevel = 3
            };
            var experiment = mlContext.Auto()
                .CreateRankingExperiment(settings);

            ExperimentResult<RankingMetrics>[] experimentResults =
            {
                experiment.Execute(trainDataView, labelColumnName, groupIdColumnName),
                experiment.Execute(trainDataView, testDataView),
                experiment.Execute(trainDataView, testDataView,
                new ColumnInformation()
                {
                    LabelColumnName = labelColumnName,
                    GroupIdColumnName = groupIdColumnName,
                }),
                experiment.Execute(trainDataView, testDataView,
                new ColumnInformation()
                {
                    LabelColumnName = labelColumnName,
                    GroupIdColumnName = groupIdColumnName,
                    SamplingKeyColumnName = groupIdColumnName
                })
            };

            for (int i = 0; i < experimentResults.Length; i++)
            {
                RunDetail<RankingMetrics> bestRun = experimentResults[i].BestRun;
                // The user requested 3, but we always return at least 10.
                Assert.Equal(10, bestRun.ValidationMetrics.DiscountedCumulativeGains.Count);
                Assert.Equal(10, bestRun.ValidationMetrics.NormalizedDiscountedCumulativeGains.Count);
                Assert.True(experimentResults[i].RunDetails.Count() > 0);
                Assert.NotNull(bestRun.ValidationMetrics);
                Assert.True(bestRun.ValidationMetrics.NormalizedDiscountedCumulativeGains.Last() > 0.4);
                Assert.True(bestRun.ValidationMetrics.DiscountedCumulativeGains.Last() > 19);
                var outputSchema = bestRun.Model.GetOutputSchema(trainDataView.Schema);
                var expectedOutputNames = new string[] { labelColumnName, groupIdColumnName, groupIdColumnName, featuresColumnVectorNameA, featuresColumnVectorNameB,
                "Features", scoreColumnName };
                foreach (var col in outputSchema)
                    Assert.True(col.Name == expectedOutputNames[col.Index]);
            }
        }

        [LightGBMFact]
        public void AutoFitRankingCVTest()
        {
            string labelColumnName = "Label";
            string groupIdColumnName = "GroupIdCustom";
            string featuresColumnVectorNameA = "FeatureVectorA";
            string featuresColumnVectorNameB = "FeatureVectorB";
            uint numFolds = 3;

            var mlContext = new MLContext(1);
            var reader = new TextLoader(mlContext, GetLoaderArgsRank(labelColumnName, groupIdColumnName,
                featuresColumnVectorNameA, featuresColumnVectorNameB));
            var trainDataView = reader.Load(DatasetUtil.GetMLSRDataset());
            // Take less than 1500 rows of data to satisfy CrossValSummaryRunner's
            // limit.
            trainDataView = mlContext.Data.TakeRows(trainDataView, 1499);

            var experiment = mlContext.Auto()
                .CreateRankingExperiment(5);
            CrossValidationExperimentResult<RankingMetrics>[] experimentResults =
            {
                experiment.Execute(trainDataView, numFolds,
                    new ColumnInformation()
                    {
                        LabelColumnName = labelColumnName,
                        GroupIdColumnName = groupIdColumnName
                    }),
                experiment.Execute(trainDataView, numFolds, labelColumnName, groupIdColumnName)
            };
            for (int i = 0; i < experimentResults.Length; i++)
            {
                CrossValidationRunDetail<RankingMetrics> bestRun = experimentResults[i].BestRun;
                Assert.True(experimentResults[i].RunDetails.Count() > 0);
                var enumerator = bestRun.Results.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    var model = enumerator.Current;
                    Assert.True(model.ValidationMetrics.NormalizedDiscountedCumulativeGains.Max() > 0.31);
                    Assert.True(model.ValidationMetrics.DiscountedCumulativeGains.Max() > 15);
                }
            }
        }

        [Fact]
        public void AutoFitRecommendationTest()
        {
            // Specific column names of the considered data set
            string labelColumnName = "Label";
            string userColumnName = "User";
            string itemColumnName = "Item";
            string scoreColumnName = "Score";
            MLContext mlContext = new MLContext(1);

            // STEP 1: Load data
            var reader = new TextLoader(mlContext, GetLoaderArgs(labelColumnName, userColumnName, itemColumnName));
            var trainDataView = reader.Load(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.trainFilename)));
            var testDataView = reader.Load(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.testFilename)));

            // STEP 2: Run AutoML experiment
            try
            {
                ExperimentResult<RegressionMetrics> experimentResult = mlContext.Auto()
                    .CreateRecommendationExperiment(5)
                    .Execute(trainDataView, testDataView,
                        new ColumnInformation()
                        {
                            LabelColumnName = labelColumnName,
                            UserIdColumnName = userColumnName,
                            ItemIdColumnName = itemColumnName
                        });

                RunDetail<RegressionMetrics> bestRun = experimentResult.BestRun;
                Assert.True(experimentResult.RunDetails.Count() > 1);
                Assert.NotNull(bestRun.ValidationMetrics);
                Assert.True(experimentResult.RunDetails.Max(i => i?.ValidationMetrics?.RSquared * i?.ValidationMetrics?.RSquared) > 0.5);

                var outputSchema = bestRun.Model.GetOutputSchema(trainDataView.Schema);
                var expectedOutputNames = new string[] { labelColumnName, userColumnName, userColumnName, itemColumnName, itemColumnName, scoreColumnName };
                foreach (var col in outputSchema)
                    Assert.True(col.Name == expectedOutputNames[col.Index]);

                IDataView testDataViewWithBestScore = bestRun.Model.Transform(testDataView);
                // Retrieve label column's index from the test IDataView
                testDataView.Schema.TryGetColumnIndex(labelColumnName, out int labelColumnId);
                // Retrieve score column's index from the IDataView produced by the trained model
                testDataViewWithBestScore.Schema.TryGetColumnIndex(scoreColumnName, out int scoreColumnId);

                var metrices = mlContext.Recommendation().Evaluate(testDataViewWithBestScore, labelColumnName: labelColumnName, scoreColumnName: scoreColumnName);
                Assert.NotEqual(0, metrices.MeanSquaredError);
            }
            catch (AggregateException ae)
            {
                // During CI unit testing, the host machines can run slower than normal, which
                // can increase the run time of unit tests and throw OperationCanceledExceptions
                // from multiple threads in the form of a single AggregateException.
                foreach (var ex in ae.Flatten().InnerExceptions)
                {
                    var ignoredExceptions = new List<Exception>();
                    if (ex is OperationCanceledException)
                        continue;
                    else
                        ignoredExceptions.Add(ex);
                    if (ignoredExceptions.Count > 0)
                        throw new AggregateException(ignoredExceptions);
                }
            }
        }

        [LightGBMFact]
        public void AutoFitWithPresplittedData()
        {
            // Models created in AutoML should work over the same data,
            // no matter how that data is splitted before passing it to the experiment execution
            // or to the model for prediction

            var context = new MLContext(1);
            var dataPath = DatasetUtil.GetUciAdultDataset();
            var columnInference = context.Auto().InferColumns(dataPath, DatasetUtil.UciAdultLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var dataFull = textLoader.Load(dataPath);
            var dataTrainTest = context.Data.TrainTestSplit(dataFull);
            var dataCV = context.Data.CrossValidationSplit(dataFull, numberOfFolds: 2);

            var modelFull = context.Auto()
                .CreateBinaryClassificationExperiment(10)
                .Execute(dataFull,
                    new ColumnInformation() { LabelColumnName = DatasetUtil.UciAdultLabel })
                .BestRun
                .Model;

            // AutoMLExperiment can't run on canceled context.
            // Therefore, we need to create a new context after an experiment is completed.
            context = new MLContext(1);
            var modelTrainTest = context.Auto()
                .CreateBinaryClassificationExperiment(10)
                .Execute(dataTrainTest.TrainSet,
                    new ColumnInformation() { LabelColumnName = DatasetUtil.UciAdultLabel })
                .BestRun
                .Model;

            context = new MLContext(1);
            var modelCV = context.Auto()
                .CreateBinaryClassificationExperiment(10)
                .Execute(dataCV.First().TrainSet,
                    new ColumnInformation() { LabelColumnName = DatasetUtil.UciAdultLabel })
                .BestRun
                .Model;

            var models = new[] { modelFull, modelTrainTest, modelCV };

            foreach (var model in models)
            {
                var resFull = model.Transform(dataFull);
                var resTrainTest = model.Transform(dataTrainTest.TrainSet);
                var resCV = model.Transform(dataCV.First().TrainSet);

                Assert.Equal(31, resFull.Schema.Count);
                Assert.Equal(31, resTrainTest.Schema.Count);
                Assert.Equal(31, resCV.Schema.Count);

                foreach (var col in resFull.Schema)
                {
                    Assert.Equal(col.Name, resTrainTest.Schema[col.Index].Name);
                    Assert.Equal(col.Name, resCV.Schema[col.Index].Name);
                }
            }
        }

        [LightGBMFact]
        public void AutoFitMaxExperimentTimeTest()
        {
            // A single binary classification experiment takes less than 5 seconds.
            // System.OperationCanceledException is thrown when ongoing experiment
            // is canceled and at least one model has been generated.
            // BinaryClassificationExperiment includes LightGBM, which is not 32-bit
            // compatible.
            var context = new MLContext(1);
            var dataPath = DatasetUtil.GetUciAdultDataset();
            var columnInference = context.Auto().InferColumns(dataPath, DatasetUtil.UciAdultLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(dataPath);
            var experiment = context.Auto()
                .CreateBinaryClassificationExperiment(15)
                .Execute(trainData, new ColumnInformation() { LabelColumnName = DatasetUtil.UciAdultLabel });

            // Ensure the (last) model that was training when maximum experiment time was reached has been stopped,
            // and that its MLContext has been canceled. Sometimes during CI unit testing, the host machines can run slower than normal, which
            // can increase the run time of unit tests, and may not produce multiple runs.
            if (experiment.RunDetails.Select(r => r.Exception == null).Count() > 1 && experiment.RunDetails.Last().Exception != null)
            {
                var expectedExceptionMessage = "Operation was canceled";
                var lastException = experiment.RunDetails.Last().Exception;
                var containsMessage = lastException.Message.Contains(expectedExceptionMessage);

                if (lastException is AggregateException lastAggregateException)
                {
                    // Sometimes multiple threads might throw the same "Operation was cancelled"
                    // exception and all of them are grouped inside an AggregateException
                    // Must check that all exceptions are the expected one.
                    containsMessage = true;
                    foreach (var ex in lastAggregateException.Flatten().InnerExceptions)
                    {
                        if (!ex.Message.Contains(expectedExceptionMessage))
                        {
                            containsMessage = false;
                        }
                    }
                }


                Assert.True(containsMessage,
                            $"Did not obtain '{expectedExceptionMessage}' error." +
                            $"Obtained unexpected error of type {lastException.GetType()} with message: {lastException.Message}");

                // Ensure that the best found model can still run after maximum experiment time was reached.
                IDataView predictions = experiment.BestRun.Model.Transform(trainData);
            }
        }

        private TextLoader.Options GetLoaderArgs(string labelColumnName, string userIdColumnName, string itemIdColumnName)
        {
            return new TextLoader.Options()
            {
                Separator = "\t",
                HasHeader = true,
                Columns = new[]
                {
                    new TextLoader.Column(labelColumnName, DataKind.Single, new [] { new TextLoader.Range(0) }),
                    new TextLoader.Column(userIdColumnName, DataKind.UInt32, new [] { new TextLoader.Range(1) }, new KeyCount(20)),
                    new TextLoader.Column(itemIdColumnName, DataKind.UInt32, new [] { new TextLoader.Range(2) }, new KeyCount(40)),
                }
            };
        }

        private TextLoader.Options GetLoaderArgsRank(string labelColumnName, string groupIdColumnName, string featureColumnVectorNameA, string featureColumnVectorNameB)
        {
            return new TextLoader.Options()
            {
                Separator = "\t",
                HasHeader = true,
                Columns = new[]
                {
                    new TextLoader.Column(labelColumnName, DataKind.Single, 0),
                    new TextLoader.Column(groupIdColumnName, DataKind.Int32, 1),
                    new TextLoader.Column(featureColumnVectorNameA, DataKind.Single, 2, 9),
                    new TextLoader.Column(featureColumnVectorNameB, DataKind.Single, 10, 137)
                }
            };
        }
    }
}
