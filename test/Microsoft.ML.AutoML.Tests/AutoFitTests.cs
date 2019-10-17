// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.AutoML.Test
{
    [TestClass]
    public class AutoFitTests
    {
        [TestMethod]
        public void AutoFitBinaryTest()
        {
            var context = new MLContext();
            var dataPath = DatasetUtil.DownloadUciAdultDataset();
            var columnInference = context.Auto().InferColumns(dataPath, DatasetUtil.UciAdultLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(dataPath);
            var result = context.Auto()
                .CreateBinaryClassificationExperiment(0)
                .Execute(trainData, new ColumnInformation() { LabelColumnName = DatasetUtil.UciAdultLabel });
            Assert.IsTrue(result.BestRun.ValidationMetrics.Accuracy > 0.70);
            Assert.IsNotNull(result.BestRun.Estimator);
            Assert.IsNotNull(result.BestRun.Model);
            Assert.IsNotNull(result.BestRun.TrainerName);
        }

        [TestMethod]
        public void AutoFitMultiTest()
        {
            var context = new MLContext();
            var columnInference = context.Auto().InferColumns(DatasetUtil.TrivialMulticlassDatasetPath, DatasetUtil.TrivialMulticlassDatasetLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(DatasetUtil.TrivialMulticlassDatasetPath);
            var result = context.Auto()
                .CreateMulticlassClassificationExperiment(0)
                .Execute(trainData, 5, DatasetUtil.TrivialMulticlassDatasetLabel);
            Assert.IsTrue(result.BestRun.Results.First().ValidationMetrics.MicroAccuracy >= 0.7);
            var scoredData = result.BestRun.Results.First().Model.Transform(trainData);
            Assert.AreEqual(NumberDataViewType.Single, scoredData.Schema[DefaultColumnNames.PredictedLabel].Type);
        }

        [TestMethod]
        public void AutoFitRegressionTest()
        {
            var context = new MLContext();
            var dataPath = DatasetUtil.DownloadMlNetGeneratedRegressionDataset();
            var columnInference = context.Auto().InferColumns(dataPath, DatasetUtil.MlNetGeneratedRegressionLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(dataPath);
            var validationData = context.Data.TakeRows(trainData, 20);
            trainData = context.Data.SkipRows(trainData, 20);
            var result = context.Auto()
                .CreateRegressionExperiment(0)
                .Execute(trainData, validationData,
                    new ColumnInformation() { LabelColumnName = DatasetUtil.MlNetGeneratedRegressionLabel });

            Assert.IsTrue(result.RunDetails.Max(i => i.ValidationMetrics.RSquared > 0.9));
        }

        [TestMethod]
        public void AutoFitRecommendationTest()
        {
            // Specific column names of the considered data set
            string labelColumnName = "Label";
            string userColumnName = "User";
            string itemColumnName = "Item";
            string scoreColumnName = "Score";
            MLContext mlContext = new MLContext();

            // STEP 1: Load data
            var reader = new TextLoader(mlContext, GetLoaderArgs(labelColumnName, userColumnName, itemColumnName));
            var trainDataView = reader.Load(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.trainFilename)));
            var testDataView = reader.Load(new MultiFileSource(GetDataPath(TestDatasets.trivialMatrixFactorization.testFilename)));

            // STEP 2: Run AutoML experiment
            ExperimentResult<RegressionMetrics> experimentResult = mlContext.Auto()
                .CreateRecommendationExperiment(5)
                .Execute(trainDataView, testDataView,
                    new ColumnInformation() { 
                        LabelColumnName = labelColumnName,
                        UserIdColumnName = userColumnName,
                        ItemIdColumnName = itemColumnName
                    });

            RunDetail<RegressionMetrics> bestRun = experimentResult.BestRun;
            Assert.IsTrue(experimentResult.RunDetails.Count() > 1);
            Assert.IsNotNull(bestRun.ValidationMetrics);
            Assert.IsTrue(experimentResult.RunDetails.Max(i => i.ValidationMetrics.RSquared != 0));

            var outputSchema = bestRun.Model.GetOutputSchema(trainDataView.Schema);
            var expectedOutputNames = new string[] { labelColumnName, userColumnName, userColumnName, itemColumnName, itemColumnName, scoreColumnName };
            foreach (var col in outputSchema)
                Assert.IsTrue(col.Name == expectedOutputNames[col.Index]);

            IDataView testDataViewWithBestScore = bestRun.Model.Transform(testDataView);
            // Retrieve label column's index from the test IDataView
            testDataView.Schema.TryGetColumnIndex(labelColumnName, out int labelColumnId);
            // Retrieve score column's index from the IDataView produced by the trained model
            testDataViewWithBestScore.Schema.TryGetColumnIndex(scoreColumnName, out int scoreColumnId);

            var metrices = mlContext.Recommendation().Evaluate(testDataViewWithBestScore, labelColumnName: labelColumnName, scoreColumnName: scoreColumnName);
            Assert.AreNotEqual(0, metrices.MeanSquaredError);
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

        private static string GetRepoRoot()
        {
#if NETFRAMEWORK
            string directory = AppDomain.CurrentDomain.BaseDirectory;
#else
            string directory = AppContext.BaseDirectory;
#endif

            while (!Directory.Exists(Path.Combine(directory, ".git")) && directory != null)
            {
                directory = Directory.GetParent(directory).FullName;
            }

            if (directory == null)
            {
                return null;
            }
            return directory;
        }

        public static string GetDataPath(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.GetFullPath(Path.Combine(Path.Combine(GetRepoRoot(), "test", "data"), name));
        }
    }
}
