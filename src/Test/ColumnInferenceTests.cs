using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class ColumnInferenceTests
    {
        [TestMethod]
        public void UnGroupColumnsTest()
        {
            var dataPath = DatasetUtil.DownloadUciAdultDataset();
            var context = new MLContext();
            var columnInferenceWithoutGrouping = context.Data.InferColumns(dataPath, DatasetUtil.UciAdultLabel, groupColumns: false);
            foreach (var col in columnInferenceWithoutGrouping.TextLoaderArgs.Column)
            {
                Assert.IsFalse(col.Source.Length > 1 || col.Source[0].Min != col.Source[0].Max);
            }

            var columnInferenceWithGrouping = context.Data.InferColumns(dataPath, DatasetUtil.UciAdultLabel, groupColumns: true);
            Assert.IsTrue(columnInferenceWithGrouping.TextLoaderArgs.Column.Count() < columnInferenceWithoutGrouping.TextLoaderArgs.Column.Count());
        }

        [TestMethod]
        public void IncorrectLabelColumnTest()
        {
            var dataPath = DatasetUtil.DownloadUciAdultDataset();
            var context = new MLContext();
            Assert.ThrowsException<InferenceException>(new System.Action(() => context.Data.InferColumns(dataPath, "Junk", groupColumns: false)));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentOutOfRangeException))]
        public void InferColumnsLabelIndexOutOfBounds()
        {
            new MLContext().Data.InferColumns(DatasetUtil.DownloadUciAdultDataset(), 100);
        }

        [TestMethod]
        public void InferColumnsLabelIndex()
        {
            var result = new MLContext().Data.InferColumns(DatasetUtil.DownloadUciAdultDataset(), 14, hasHeader: true);
            Assert.AreEqual(true, result.TextLoaderArgs.HasHeader);
            var labelCol = result.TextLoaderArgs.Column.First(c => c.Source[0].Min == 14 && c.Source[0].Max == 14);
            Assert.AreEqual("hours_per_week", labelCol.Name);
            var labelPurposes = result.ColumnPurpopses.Where(c => c.Purpose == ColumnPurpose.Label);
            Assert.AreEqual(1, labelPurposes.Count());
            Assert.AreEqual("hours_per_week", labelPurposes.First().Name);
        }

        [TestMethod]
        public void InferColumnsLabelIndexNoHeaders()
        {
            var result = new MLContext().Data.InferColumns(DatasetUtil.DownloadIrisDataset(), DatasetUtil.IrisDatasetLabelColIndex);
            Assert.AreEqual(false, result.TextLoaderArgs.HasHeader);
            var labelCol = result.TextLoaderArgs.Column.First(c => c.Source[0].Min == DatasetUtil.IrisDatasetLabelColIndex &&
                c.Source[0].Max == DatasetUtil.IrisDatasetLabelColIndex);
            Assert.AreEqual(DefaultColumnNames.Label, labelCol.Name);
            var labelPurposes = result.ColumnPurpopses.Where(c => c.Purpose == ColumnPurpose.Label);
            Assert.AreEqual(1, labelPurposes.Count());
            Assert.AreEqual(DefaultColumnNames.Label, labelPurposes.First().Name);
        }
    }
}