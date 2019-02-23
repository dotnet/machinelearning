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
            var columnInference = context.AutoInference().InferColumns(dataPath, DatasetUtil.UciAdultLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderArgs);
            var trainData = textLoader.Read(dataPath);
            var validationData = context.Data.TakeRows(trainData, 100);
            trainData = context.Data.SkipRows(trainData, 100);
            var result = context.AutoInference()
                .CreateBinaryClassificationExperiment(0)
                .Execute(trainData, validationData, new ColumnInformation() { LabelColumn = DatasetUtil.UciAdultLabel });

            Assert.IsTrue(result.Max(i => i.Metrics.Accuracy) > 0.80);
        }

        [TestMethod]
        public void AutoFitMultiTest()
        {
            var context = new MLContext();
            var columnInference = context.AutoInference().InferColumns(DatasetUtil.TrivialMulticlassDatasetPath, DatasetUtil.TrivialMulticlassDatasetLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderArgs);
            var trainData = textLoader.Read(DatasetUtil.TrivialMulticlassDatasetPath);
            var validationData = context.Data.TakeRows(trainData, 20);
            trainData = context.Data.SkipRows(trainData, 20);
            var result = context.AutoInference()
                .CreateMulticlassClassificationExperiment(0)
                .Execute(trainData, validationData, new ColumnInformation() { LabelColumn = DatasetUtil.TrivialMulticlassDatasetLabel });

            Assert.IsTrue(result.Max(i => i.Metrics.AccuracyMacro) > 0.80);
        }

        [TestMethod]
        public void AutoFitRegressionTest()
        {
            var context = new MLContext();
            var dataPath = DatasetUtil.DownloadMlNetGeneratedRegressionDataset();
            var columnInference = context.AutoInference().InferColumns(dataPath, DatasetUtil.MlNetGeneratedRegressionLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderArgs);
            var trainData = textLoader.Read(dataPath);
            var validationData = context.Data.TakeRows(trainData, 20);
            trainData = context.Data.SkipRows(trainData, 20);
            var results = context.AutoInference()
                .CreateRegressionExperiment(0)
                .Execute(trainData, validationData,
                    new ColumnInformation() { LabelColumn = DatasetUtil.MlNetGeneratedRegressionLabel });

            Assert.IsTrue(results.Max(i => i.Metrics.RSquared > 0.9));
        }
    }
}
