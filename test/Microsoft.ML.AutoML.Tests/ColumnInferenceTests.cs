using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.AutoML.Test
{
    [TestClass]
    public class ColumnInferenceTests
    {
        [TestMethod]
        public void UnGroupReturnsMoreColumnsThanGroup()
        {
            var dataPath = DatasetUtil.DownloadUciAdultDataset();
            var context = new MLContext();
            var columnInferenceWithoutGrouping = context.Auto().InferColumns(dataPath, DatasetUtil.UciAdultLabel, groupColumns: false);
            foreach (var col in columnInferenceWithoutGrouping.TextLoaderOptions.Columns)
            {
                Assert.IsFalse(col.Source.Length > 1 || col.Source[0].Min != col.Source[0].Max);
            }

            var columnInferenceWithGrouping = context.Auto().InferColumns(dataPath, DatasetUtil.UciAdultLabel, groupColumns: true);
            Assert.IsTrue(columnInferenceWithGrouping.TextLoaderOptions.Columns.Count() < columnInferenceWithoutGrouping.TextLoaderOptions.Columns.Count());
        }

        [TestMethod]
        public void IncorrectLabelColumnThrows()
        {
            var dataPath = DatasetUtil.DownloadUciAdultDataset();
            var context = new MLContext();
            Assert.ThrowsException<ArgumentException>(new System.Action(() => context.Auto().InferColumns(dataPath, "Junk", groupColumns: false)));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentOutOfRangeException))]
        public void LabelIndexOutOfBoundsThrows()
        {
            new MLContext().Auto().InferColumns(DatasetUtil.DownloadUciAdultDataset(), 100);
        }

        [TestMethod]
        public void IdentifyLabelColumnThroughIndexWithHeader()
        {
            var result = new MLContext().Auto().InferColumns(DatasetUtil.DownloadUciAdultDataset(), 14, hasHeader: true);
            Assert.AreEqual(true, result.TextLoaderOptions.HasHeader);
            var labelCol = result.TextLoaderOptions.Columns.First(c => c.Source[0].Min == 14 && c.Source[0].Max == 14);
            Assert.AreEqual("hours-per-week", labelCol.Name);
            Assert.AreEqual("hours-per-week", result.ColumnInformation.LabelColumnName);
        }

        [TestMethod]
        public void IdentifyLabelColumnThroughIndexWithoutHeader()
        {
            var result = new MLContext().Auto().InferColumns(DatasetUtil.DownloadIrisDataset(), DatasetUtil.IrisDatasetLabelColIndex);
            Assert.AreEqual(false, result.TextLoaderOptions.HasHeader);
            var labelCol = result.TextLoaderOptions.Columns.First(c => c.Source[0].Min == DatasetUtil.IrisDatasetLabelColIndex &&
                c.Source[0].Max == DatasetUtil.IrisDatasetLabelColIndex);
            Assert.AreEqual(DefaultColumnNames.Label, labelCol.Name);
            Assert.AreEqual(DefaultColumnNames.Label, result.ColumnInformation.LabelColumnName);
        }

        [TestMethod]
        public void DatasetWithEmptyColumn()
        {
            var result = new MLContext().Auto().InferColumns(Path.Combine("TestData", "DatasetWithEmptyColumn.txt"), DefaultColumnNames.Label, groupColumns: false);
            var emptyColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "Empty");
            Assert.AreEqual(DataKind.Single, emptyColumn.DataKind);
        }

        [TestMethod]
        public void DatasetWithBoolColumn()
        {
            var result = new MLContext().Auto().InferColumns(Path.Combine("TestData", "BinaryDatasetWithBoolColumn.txt"), DefaultColumnNames.Label);
            Assert.AreEqual(2, result.TextLoaderOptions.Columns.Count());

            var boolColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "Bool");
            var labelColumn = result.TextLoaderOptions.Columns.First(c => c.Name == DefaultColumnNames.Label);
            // ensure non-label Boolean column is detected as R4
            Assert.AreEqual(DataKind.Single, boolColumn.DataKind);
            Assert.AreEqual(DataKind.Boolean, labelColumn.DataKind);

            // ensure non-label Boolean column is detected as R4
            Assert.AreEqual(1, result.ColumnInformation.NumericColumnNames.Count());
            Assert.AreEqual("Bool", result.ColumnInformation.NumericColumnNames.First());
            Assert.AreEqual(DefaultColumnNames.Label, result.ColumnInformation.LabelColumnName);
        }

        [TestMethod]
        public void WhereNameColumnIsOnlyFeature()
        {
            var result = new MLContext().Auto().InferColumns(Path.Combine("TestData", "NameColumnIsOnlyFeatureDataset.txt"), DefaultColumnNames.Label);
            Assert.AreEqual(2, result.TextLoaderOptions.Columns.Count());

            var nameColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "Username");
            var labelColumn = result.TextLoaderOptions.Columns.First(c => c.Name == DefaultColumnNames.Label);
            Assert.AreEqual(DataKind.String, nameColumn.DataKind);
            Assert.AreEqual(DataKind.Boolean, labelColumn.DataKind);
            
            Assert.AreEqual(1, result.ColumnInformation.TextColumnNames.Count());
            Assert.AreEqual("Username", result.ColumnInformation.TextColumnNames.First());
            Assert.AreEqual(DefaultColumnNames.Label, result.ColumnInformation.LabelColumnName);
        }

        [TestMethod]
        public void DefaultColumnNamesInferredCorrectly()
        {
            var result = new MLContext().Auto().InferColumns(Path.Combine("TestData", "DatasetWithDefaultColumnNames.txt"),
                new ColumnInformation()
                {
                    LabelColumnName = DefaultColumnNames.Label,
                    ExampleWeightColumnName = DefaultColumnNames.Weight,
                },
                groupColumns : false);

            Assert.AreEqual(DefaultColumnNames.Label, result.ColumnInformation.LabelColumnName);
            Assert.AreEqual(DefaultColumnNames.Weight, result.ColumnInformation.ExampleWeightColumnName);
            Assert.AreEqual(result.ColumnInformation.NumericColumnNames.Count(), 3);
        }

        [TestMethod]
        public void DefaultColumnNamesNoGrouping()
        {
            var result = new MLContext().Auto().InferColumns(Path.Combine("TestData", "DatasetWithDefaultColumnNames.txt"),
                new ColumnInformation()
                {
                    LabelColumnName = DefaultColumnNames.Label,
                    ExampleWeightColumnName = DefaultColumnNames.Weight,
                });

            Assert.AreEqual(DefaultColumnNames.Label, result.ColumnInformation.LabelColumnName);
            Assert.AreEqual(DefaultColumnNames.Weight, result.ColumnInformation.ExampleWeightColumnName);
            Assert.AreEqual(1, result.ColumnInformation.NumericColumnNames.Count());
            Assert.AreEqual(DefaultColumnNames.Features, result.ColumnInformation.NumericColumnNames.First());
        }

        [TestMethod]
        public void InferColumnsColumnInfoParam()
        {
            var columnInfo = new ColumnInformation() { LabelColumnName = DatasetUtil.MlNetGeneratedRegressionLabel };
            var result = new MLContext().Auto().InferColumns(DatasetUtil.DownloadMlNetGeneratedRegressionDataset(), 
                columnInfo);
            var labelCol = result.TextLoaderOptions.Columns.First(c => c.Name == DatasetUtil.MlNetGeneratedRegressionLabel);
            Assert.AreEqual(DataKind.Single, labelCol.DataKind);
            Assert.AreEqual(DatasetUtil.MlNetGeneratedRegressionLabel, result.ColumnInformation.LabelColumnName);
            Assert.AreEqual(1, result.ColumnInformation.NumericColumnNames.Count());
            Assert.AreEqual(DefaultColumnNames.Features, result.ColumnInformation.NumericColumnNames.First());
            Assert.AreEqual(null, result.ColumnInformation.ExampleWeightColumnName);
        }
    }
}