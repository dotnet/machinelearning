using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class AutoFitTests
    {
        [TestMethod]
        public void Hello()
        {
            var context = new MLContext();
            var dataPath = DatasetUtil.DownloadUciAdultDataset();
            var columnInference = context.Data.InferColumns(dataPath, DatasetUtil.UciAdultLabel, true);
            var textLoader = context.Data.CreateTextReader(columnInference);
            var trainData = textLoader.Read(dataPath);
            var validationData = trainData.Take(100);
            trainData = trainData.Skip(100);
            var best = context.BinaryClassification.AutoFit(trainData, DatasetUtil.UciAdultLabel, validationData, settings:
                new AutoFitSettings()
                {
                    StoppingCriteria = new ExperimentStoppingCriteria()
                    {
                        MaxIterations = 2,
                        TimeOutInMinutes = 1000000000
                    }
                }, debugLogger: null);

            Assert.IsNotNull(best?.BestPipeline?.Model);
            Assert.IsTrue(best.BestPipeline.Metrics.Accuracy > 0.80);
        }
    }
}
