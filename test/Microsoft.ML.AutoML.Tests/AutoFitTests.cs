// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TestFrameworkCommon;
using Xunit;
using Xunit.Abstractions;
using static Microsoft.ML.DataOperationsCatalog;

namespace Microsoft.ML.AutoML.Test
{
    public class AutoFitTests : BaseTestClass
    {
        public AutoFitTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void AutoFitBinaryTest()
        {
            var context = new MLContext(1);
            var dataPath = DatasetUtil.GetUciAdultDataset();
            var columnInference = context.Auto().InferColumns(dataPath, DatasetUtil.UciAdultLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(dataPath);
            var result = context.Auto()
                .CreateBinaryClassificationExperiment(0)
                .Execute(trainData, new ColumnInformation() { LabelColumnName = DatasetUtil.UciAdultLabel });
            Assert.True(result.BestRun.ValidationMetrics.Accuracy > 0.70);
            Assert.NotNull(result.BestRun.Estimator);
            Assert.NotNull(result.BestRun.Model);
            Assert.NotNull(result.BestRun.TrainerName);
        }

        [Fact]
        public void AutoFitMultiTest()
        {
            var context = new MLContext(42);
            var columnInference = context.Auto().InferColumns(DatasetUtil.TrivialMulticlassDatasetPath, DatasetUtil.TrivialMulticlassDatasetLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(DatasetUtil.TrivialMulticlassDatasetPath);
            var result = context.Auto()
                .CreateMulticlassClassificationExperiment(0)
                .Execute(trainData, 5, DatasetUtil.TrivialMulticlassDatasetLabel);
            Assert.True(result.BestRun.Results.First().ValidationMetrics.MicroAccuracy >= 0.7);
            var scoredData = result.BestRun.Results.First().Model.Transform(trainData);
            Assert.Equal(NumberDataViewType.Single, scoredData.Schema[DefaultColumnNames.PredictedLabel].Type);
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

        [Fact(Skip ="Takes too much time, ~10 minutes.")]
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

        [Fact]
        public void AutoFitRegressionTest()
        {
            var context = new MLContext(1);
            var dataPath = DatasetUtil.GetMlNetGeneratedRegressionDataset();
            var columnInference = context.Auto().InferColumns(dataPath, DatasetUtil.MlNetGeneratedRegressionLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(dataPath);
            var validationData = context.Data.TakeRows(trainData, 20);
            trainData = context.Data.SkipRows(trainData, 20);
            var result = context.Auto()
                .CreateRegressionExperiment(0)
                .Execute(trainData, validationData,
                    new ColumnInformation() { LabelColumnName = DatasetUtil.MlNetGeneratedRegressionLabel });

            Assert.True(result.RunDetails.Max(i => i.ValidationMetrics.RSquared > 0.9));
        }

        [LightGBMFact]
        public void AutoFitRankingTest()
        {
            string labelColumnName = "Label";
            string scoreColumnName = "Score";
            string groupIdColumnName = "CustomGroupId";
            string featuresColumnVectorNameA = "FeatureVectorA";
            string featuresColumnVectorNameB = "FeatureVectorB";
            var mlContext = new MLContext(1);

            // STEP 1: Load data
            var reader = new TextLoader(mlContext, GetLoaderArgsRank(labelColumnName, groupIdColumnName, featuresColumnVectorNameA, featuresColumnVectorNameB));
            var trainDataView = reader.Load(new MultiFileSource(DatasetUtil.GetMLSRDataset()));
            var testDataView = mlContext.Data.TakeRows(trainDataView, 500);
            trainDataView = mlContext.Data.SkipRows(trainDataView, 500);
            // STEP 2: Run AutoML experiment
            ExperimentResult<RankingMetrics> experimentResult = mlContext.Auto()
                .CreateRankingExperiment(new RankingExperimentSettings() { GroupIdColumnName = "CustomGroupId", MaxExperimentTimeInSeconds = 5})
                .Execute(trainDataView, testDataView,
                    new ColumnInformation()
                    {
                        LabelColumnName = labelColumnName,
                        GroupIdColumnName = groupIdColumnName
                    });

            RunDetail<RankingMetrics> bestRun = experimentResult.BestRun;
            Assert.True(experimentResult.RunDetails.Count() > 0);
            Assert.NotNull(bestRun.ValidationMetrics);
            Assert.True(experimentResult.RunDetails.Max(i => i.ValidationMetrics.NormalizedDiscountedCumulativeGains.Max() > .5));
            Assert.True(experimentResult.RunDetails.Max(i => i.ValidationMetrics.DiscountedCumulativeGains.Max() > 34));
            var outputSchema = bestRun.Model.GetOutputSchema(trainDataView.Schema);
            var expectedOutputNames = new string[] { labelColumnName, groupIdColumnName, groupIdColumnName, featuresColumnVectorNameA, featuresColumnVectorNameB,
                "Features", scoreColumnName };
            foreach (var col in outputSchema)
                Assert.True(col.Name == expectedOutputNames[col.Index]);
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
            Assert.True(experimentResult.RunDetails.Max(i => i.ValidationMetrics.RSquared != 0));

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

        [Fact]
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
                .CreateBinaryClassificationExperiment(0)
                .Execute(dataFull,
                    new ColumnInformation() { LabelColumnName = DatasetUtil.UciAdultLabel })
                .BestRun
                .Model;

            var modelTrainTest = context.Auto()
                .CreateBinaryClassificationExperiment(0)
                .Execute(dataTrainTest.TrainSet,
                    new ColumnInformation() { LabelColumnName = DatasetUtil.UciAdultLabel })
                .BestRun
                .Model;

            var modelCV = context.Auto()
                .CreateBinaryClassificationExperiment(0)
                .Execute(dataCV.First().TrainSet,
                    new ColumnInformation() { LabelColumnName = DatasetUtil.UciAdultLabel })
                .BestRun
                .Model;

            var models = new[] { modelFull, modelTrainTest, modelCV };

            foreach(var model in models)
            {
                var resFull = model.Transform(dataFull);
                var resTrainTest = model.Transform(dataTrainTest.TrainSet);
                var resCV = model.Transform(dataCV.First().TrainSet);

                Assert.Equal(30, resFull.Schema.Count);
                Assert.Equal(30, resTrainTest.Schema.Count);
                Assert.Equal(30, resCV.Schema.Count);

                foreach (var col in resFull.Schema)
                {
                    Assert.Equal(col.Name, resTrainTest.Schema[col.Index].Name);
                    Assert.Equal(col.Name, resCV.Schema[col.Index].Name);
                }
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
