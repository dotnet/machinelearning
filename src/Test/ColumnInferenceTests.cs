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
            Assert.ThrowsException<ArgumentException>(new System.Action(() => context.Data.InferColumns(dataPath, "Junk", groupColumns: false)));
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
            Assert.AreEqual("hours-per-week", labelCol.Name);
            var labelPurposes = result.ColumnPurpopses.Where(c => c.Purpose == ColumnPurpose.Label);
            Assert.AreEqual(1, labelPurposes.Count());
            Assert.AreEqual("hours-per-week", labelPurposes.First().Name);
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

        [TestMethod]
        public void InferColumnsWithDatasetWithEmptyColumn()
        {
            var result = new MLContext().Data.InferColumns(@".\TestData\DatasetWithEmptyColumn.txt", DefaultColumnNames.Label);
            var emptyColumn = result.TextLoaderArgs.Column.First(c => c.Name == "Empty");
            Assert.AreEqual(DataKind.TX, emptyColumn.Type);
        }

        [TestMethod]
        public void InferColumnsWithDatasetWithBoolColumn()
        {
            var result = new MLContext().Data.InferColumns(@".\TestData\BinaryDatasetWithBoolColumn.txt", DefaultColumnNames.Label);
            Assert.AreEqual(2, result.TextLoaderArgs.Column.Count());
            Assert.AreEqual(2, result.ColumnPurpopses.Count());

            var boolColumn = result.TextLoaderArgs.Column.First(c => c.Name == "Bool");
            var labelColumn = result.TextLoaderArgs.Column.First(c => c.Name == DefaultColumnNames.Label);
            // ensure non-label Boolean column is detected as R4
            Assert.AreEqual(DataKind.R4, boolColumn.Type);
            Assert.AreEqual(DataKind.BL, labelColumn.Type);

            // ensure non-label Boolean column is detected as R4
            var boolPurpose = result.ColumnPurpopses.First(c => c.Name == "Bool").Purpose;
            var labelPurpose = result.ColumnPurpopses.First(c => c.Name == DefaultColumnNames.Label).Purpose;
            Assert.AreEqual(ColumnPurpose.NumericFeature, boolPurpose);
            Assert.AreEqual(ColumnPurpose.Label, labelPurpose);
        }

        [TestMethod]
        public void InferColumnsWhereNameColumnIsOnlyFeature()
        {
            var result = new MLContext().Data.InferColumns(@".\TestData\NameColumnIsOnlyFeatureDataset.txt", DefaultColumnNames.Label);
            Assert.AreEqual(2, result.TextLoaderArgs.Column.Count());
            Assert.AreEqual(2, result.ColumnPurpopses.Count());

            var nameColumn = result.TextLoaderArgs.Column.First(c => c.Name == DefaultColumnNames.Name);
            var labelColumn = result.TextLoaderArgs.Column.First(c => c.Name == DefaultColumnNames.Label);
            Assert.AreEqual(DataKind.TX, nameColumn.Type);
            Assert.AreEqual(DataKind.BL, labelColumn.Type);

            var namePurpose = result.ColumnPurpopses.First(c => c.Name == DefaultColumnNames.Name).Purpose;
            var labelPurpose = result.ColumnPurpopses.First(c => c.Name == DefaultColumnNames.Label).Purpose;
            Assert.AreEqual(ColumnPurpose.TextFeature, namePurpose);
            Assert.AreEqual(ColumnPurpose.Label, labelPurpose);
        }
    }
}