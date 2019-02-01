// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class AutoFitTests
    {
        [TestMethod]
        public void AutoFitBinaryTest()
        {
            var context = new MLContext();
            var dataPath = DatasetUtil.DownloadUciAdultDataset();
            var columnInference = context.Data.InferColumns(dataPath, DatasetUtil.UciAdultLabel, true);
            var textLoader = context.Data.CreateTextReader(columnInference);
            var trainData = textLoader.Read(dataPath);
            var validationData = trainData.Take(100);
            trainData = trainData.Skip(100);
            var result = context.BinaryClassification.AutoFit(trainData, DatasetUtil.UciAdultLabel, validationData, settings:
                new AutoFitSettings()
                {
                    StoppingCriteria = new ExperimentStoppingCriteria()
                    {
                        MaxIterations = 2,
                        TimeOutInMinutes = 1000000000
                    }
                }, debugLogger: null);

            Assert.IsTrue(result.Max(i => i.Metrics.Accuracy) > 0.80);
        }

        [TestMethod]
        public void AutoFitMultiTest()
        {
            var context = new MLContext();
            var dataPath = DatasetUtil.DownloadTrivialDataset();
            var columnInference = context.Data.InferColumns(dataPath, DatasetUtil.TrivialDatasetLabel, true);
            var textLoader = context.Data.CreateTextReader(columnInference);
            var trainData = textLoader.Read(dataPath);
            var validationData = trainData.Take(20);
            trainData = trainData.Skip(20);
            var result = context.MulticlassClassification.AutoFit(trainData, DatasetUtil.TrivialDatasetLabel, validationData, settings:
                new AutoFitSettings()
                {
                    StoppingCriteria = new ExperimentStoppingCriteria()
                    {
                        MaxIterations = 1,
                        TimeOutInMinutes = 1000000000
                    }
                }, debugLogger: null);

            Assert.IsTrue(result.Max(i => i.Metrics.AccuracyMacro) > 0.80);
        }

        [TestMethod]
        public void AutoFitRegressionTest()
        {
            var context = new MLContext();
            var dataPath = DatasetUtil.DownloadMlNetGeneratedRegressionDataset();
            var columnInference = context.Data.InferColumns(dataPath, DatasetUtil.MlNetGeneratedRegressionLabel, true);
            var textLoader = context.Data.CreateTextReader(columnInference);
            var trainData = textLoader.Read(dataPath);
            var validationData = trainData.Take(20);
            trainData = trainData.Skip(20);
            var result = context.Regression.AutoFit(trainData, DatasetUtil.MlNetGeneratedRegressionLabel, validationData, settings:
                new AutoFitSettings()
                {
                    StoppingCriteria = new ExperimentStoppingCriteria()
                    {
                        MaxIterations = 1,
                        TimeOutInMinutes = 1000000000
                    }
                }, debugLogger: null);

            Assert.IsTrue(result.Max(i => i.Metrics.RSquared > 0.9));
        }
    }
}
