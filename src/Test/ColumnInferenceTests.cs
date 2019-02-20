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
            var columnInferenceWithoutGrouping = context.AutoInference().InferColumns(dataPath, DatasetUtil.UciAdultLabel, groupColumns: false);
            foreach (var col in columnInferenceWithoutGrouping.TextLoaderArgs.Column)
            {
                Assert.IsFalse(col.Source.Length > 1 || col.Source[0].Min != col.Source[0].Max);
            }

            var columnInferenceWithGrouping = context.AutoInference().InferColumns(dataPath, DatasetUtil.UciAdultLabel, groupColumns: true);
            Assert.IsTrue(columnInferenceWithGrouping.TextLoaderArgs.Column.Count() < columnInferenceWithoutGrouping.TextLoaderArgs.Column.Count());
        }

        [TestMethod]
        public void IncorrectLabelColumnTest()
        {
            var dataPath = DatasetUtil.DownloadUciAdultDataset();
            var context = new MLContext();
            Assert.ThrowsException<ArgumentException>(new System.Action(() => context.AutoInference().InferColumns(dataPath, "Junk", groupColumns: false)));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentOutOfRangeException))]
        public void InferColumnsLabelIndexOutOfBounds()
        {
            new MLContext().AutoInference().InferColumns(DatasetUtil.DownloadUciAdultDataset(), 100);
        }

        [TestMethod]
        public void InferColumnsLabelIndex()
        {
            var result = new MLContext().AutoInference().InferColumns(DatasetUtil.DownloadUciAdultDataset(), 14, hasHeader: true);
            Assert.AreEqual(true, result.TextLoaderArgs.HasHeader);
            var labelCol = result.TextLoaderArgs.Column.First(c => c.Source[0].Min == 14 && c.Source[0].Max == 14);
            Assert.AreEqual("hours-per-week", labelCol.Name);
            Assert.AreEqual("hours-per-week", result.ColumnInformation.LabelColumn);
        }

        [TestMethod]
        public void InferColumnsLabelIndexNoHeaders()
        {
            var result = new MLContext().AutoInference().InferColumns(DatasetUtil.DownloadIrisDataset(), DatasetUtil.IrisDatasetLabelColIndex);
            Assert.AreEqual(false, result.TextLoaderArgs.HasHeader);
            var labelCol = result.TextLoaderArgs.Column.First(c => c.Source[0].Min == DatasetUtil.IrisDatasetLabelColIndex &&
                c.Source[0].Max == DatasetUtil.IrisDatasetLabelColIndex);
            Assert.AreEqual(DefaultColumnNames.Label, labelCol.Name);
            Assert.AreEqual(DefaultColumnNames.Label, result.ColumnInformation.LabelColumn);
        }

        [TestMethod]
        public void InferColumnsWithDatasetWithEmptyColumn()
        {
            var result = new MLContext().AutoInference().InferColumns(@".\TestData\DatasetWithEmptyColumn.txt", DefaultColumnNames.Label);
            var emptyColumn = result.TextLoaderArgs.Column.First(c => c.Name == "Empty");
            Assert.AreEqual(DataKind.TX, emptyColumn.Type);
        }

        [TestMethod]
        public void InferColumnsWithDatasetWithBoolColumn()
        {
            var result = new MLContext().AutoInference().InferColumns(@".\TestData\BinaryDatasetWithBoolColumn.txt", DefaultColumnNames.Label);
            Assert.AreEqual(2, result.TextLoaderArgs.Column.Count());

            var boolColumn = result.TextLoaderArgs.Column.First(c => c.Name == "Bool");
            var labelColumn = result.TextLoaderArgs.Column.First(c => c.Name == DefaultColumnNames.Label);
            // ensure non-label Boolean column is detected as R4
            Assert.AreEqual(DataKind.R4, boolColumn.Type);
            Assert.AreEqual(DataKind.BL, labelColumn.Type);

            // ensure non-label Boolean column is detected as R4
            Assert.AreEqual(1, result.ColumnInformation.NumericColumns.Count());
            Assert.AreEqual("Bool", result.ColumnInformation.NumericColumns.First());
            Assert.AreEqual(DefaultColumnNames.Label, result.ColumnInformation.LabelColumn);
        }

        [TestMethod]
        public void InferColumnsWhereNameColumnIsOnlyFeature()
        {
            var result = new MLContext().AutoInference().InferColumns(@".\TestData\NameColumnIsOnlyFeatureDataset.txt", DefaultColumnNames.Label);
            Assert.AreEqual(2, result.TextLoaderArgs.Column.Count());

            var nameColumn = result.TextLoaderArgs.Column.First(c => c.Name == "Username");
            var labelColumn = result.TextLoaderArgs.Column.First(c => c.Name == DefaultColumnNames.Label);
            Assert.AreEqual(DataKind.TX, nameColumn.Type);
            Assert.AreEqual(DataKind.BL, labelColumn.Type);
            
            Assert.AreEqual(1, result.ColumnInformation.TextColumns.Count());
            Assert.AreEqual("Username", result.ColumnInformation.TextColumns.First());
            Assert.AreEqual(DefaultColumnNames.Label, result.ColumnInformation.LabelColumn);
        }
    }
}