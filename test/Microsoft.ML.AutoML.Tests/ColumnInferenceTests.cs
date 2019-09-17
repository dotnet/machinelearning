using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Xunit;

namespace Microsoft.ML.AutoML.Test
{
    
    public class ColumnInferenceTests
    {
        [Fact]
        public void UnGroupReturnsMoreColumnsThanGroup()
        {
            var dataPath = DatasetUtil.DownloadUciAdultDataset();
            var context = new MLContext();
            var columnInferenceWithoutGrouping = context.Auto().InferColumns(dataPath, DatasetUtil.UciAdultLabel, groupColumns: false);
            foreach (var col in columnInferenceWithoutGrouping.TextLoaderOptions.Columns)
            {
                Assert.False(col.Source.Length > 1 || col.Source[0].Min != col.Source[0].Max);
            }

            var columnInferenceWithGrouping = context.Auto().InferColumns(dataPath, DatasetUtil.UciAdultLabel, groupColumns: true);
            Assert.True(columnInferenceWithGrouping.TextLoaderOptions.Columns.Count() < columnInferenceWithoutGrouping.TextLoaderOptions.Columns.Count());
        }

        [Fact]
        public void IncorrectLabelColumnThrows()
        {
            var dataPath = DatasetUtil.DownloadUciAdultDataset();
            var context = new MLContext();
            Assert.Throws<ArgumentException>(new System.Action(() => context.Auto().InferColumns(dataPath, "Junk", groupColumns: false)));
        }

        [Fact]
        public void LabelIndexOutOfBoundsThrows()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new MLContext().Auto().InferColumns(DatasetUtil.DownloadUciAdultDataset(), 100));
        }

        [Fact]
        public void IdentifyLabelColumnThroughIndexWithHeader()
        {
            var result = new MLContext().Auto().InferColumns(DatasetUtil.DownloadUciAdultDataset(), 14, hasHeader: true);
            Assert.True(result.TextLoaderOptions.HasHeader);
            var labelCol = result.TextLoaderOptions.Columns.First(c => c.Source[0].Min == 14 && c.Source[0].Max == 14);
            Assert.Equal("hours-per-week", labelCol.Name);
            Assert.Equal("hours-per-week", result.ColumnInformation.LabelColumnName);
        }

        [Fact]
        public void IdentifyLabelColumnThroughIndexWithoutHeader()
        {
            var result = new MLContext().Auto().InferColumns(DatasetUtil.DownloadIrisDataset(), DatasetUtil.IrisDatasetLabelColIndex);
            Assert.False(result.TextLoaderOptions.HasHeader);
            var labelCol = result.TextLoaderOptions.Columns.First(c => c.Source[0].Min == DatasetUtil.IrisDatasetLabelColIndex &&
                c.Source[0].Max == DatasetUtil.IrisDatasetLabelColIndex);
            Assert.Equal(DefaultColumnNames.Label, labelCol.Name);
            Assert.Equal(DefaultColumnNames.Label, result.ColumnInformation.LabelColumnName);
        }

        [Fact]
        public void DatasetWithEmptyColumn()
        {
            var result = new MLContext().Auto().InferColumns(Path.Combine("TestData", "DatasetWithEmptyColumn.txt"), DefaultColumnNames.Label, groupColumns: false);
            var emptyColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "Empty");
            Assert.Equal(DataKind.Single, emptyColumn.DataKind);
        }

        [Fact]
        public void DatasetWithBoolColumn()
        {
            var result = new MLContext().Auto().InferColumns(Path.Combine("TestData", "BinaryDatasetWithBoolColumn.txt"), DefaultColumnNames.Label);
            Assert.Equal(2, result.TextLoaderOptions.Columns.Count());

            var boolColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "Bool");
            var labelColumn = result.TextLoaderOptions.Columns.First(c => c.Name == DefaultColumnNames.Label);
            // ensure non-label Boolean column is detected as R4
            Assert.Equal(DataKind.Single, boolColumn.DataKind);
            Assert.Equal(DataKind.Boolean, labelColumn.DataKind);

            // ensure non-label Boolean column is detected as R4
            Assert.Single(result.ColumnInformation.NumericColumnNames);
            Assert.Equal("Bool", result.ColumnInformation.NumericColumnNames.First());
            Assert.Equal(DefaultColumnNames.Label, result.ColumnInformation.LabelColumnName);
        }

        [Fact]
        public void WhereNameColumnIsOnlyFeature()
        {
            var result = new MLContext().Auto().InferColumns(Path.Combine("TestData", "NameColumnIsOnlyFeatureDataset.txt"), DefaultColumnNames.Label);
            Assert.Equal(2, result.TextLoaderOptions.Columns.Count());

            var nameColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "Username");
            var labelColumn = result.TextLoaderOptions.Columns.First(c => c.Name == DefaultColumnNames.Label);
            Assert.Equal(DataKind.String, nameColumn.DataKind);
            Assert.Equal(DataKind.Boolean, labelColumn.DataKind);
            
            Assert.Single(result.ColumnInformation.TextColumnNames);
            Assert.Equal("Username", result.ColumnInformation.TextColumnNames.First());
            Assert.Equal(DefaultColumnNames.Label, result.ColumnInformation.LabelColumnName);
        }

        [Fact]
        public void DefaultColumnNamesInferredCorrectly()
        {
            var result = new MLContext().Auto().InferColumns(Path.Combine("TestData", "DatasetWithDefaultColumnNames.txt"),
                new ColumnInformation()
                {
                    LabelColumnName = DefaultColumnNames.Label,
                    ExampleWeightColumnName = DefaultColumnNames.Weight,
                },
                groupColumns : false);

            Assert.Equal(DefaultColumnNames.Label, result.ColumnInformation.LabelColumnName);
            Assert.Equal(DefaultColumnNames.Weight, result.ColumnInformation.ExampleWeightColumnName);
            Assert.Equal(3, result.ColumnInformation.NumericColumnNames.Count());
        }

        [Fact]
        public void DefaultColumnNamesNoGrouping()
        {
            var result = new MLContext().Auto().InferColumns(Path.Combine("TestData", "DatasetWithDefaultColumnNames.txt"),
                new ColumnInformation()
                {
                    LabelColumnName = DefaultColumnNames.Label,
                    ExampleWeightColumnName = DefaultColumnNames.Weight,
                });

            Assert.Equal(DefaultColumnNames.Label, result.ColumnInformation.LabelColumnName);
            Assert.Equal(DefaultColumnNames.Weight, result.ColumnInformation.ExampleWeightColumnName);
            Assert.Single(result.ColumnInformation.NumericColumnNames);
            Assert.Equal(DefaultColumnNames.Features, result.ColumnInformation.NumericColumnNames.First());
        }

        [Fact]
        public void InferColumnsColumnInfoParam()
        {
            var columnInfo = new ColumnInformation() { LabelColumnName = DatasetUtil.MlNetGeneratedRegressionLabel };
            var result = new MLContext().Auto().InferColumns(DatasetUtil.DownloadMlNetGeneratedRegressionDataset(), 
                columnInfo);
            var labelCol = result.TextLoaderOptions.Columns.First(c => c.Name == DatasetUtil.MlNetGeneratedRegressionLabel);
            Assert.Equal(DataKind.Single, labelCol.DataKind);
            Assert.Equal(DatasetUtil.MlNetGeneratedRegressionLabel, result.ColumnInformation.LabelColumnName);
            Assert.Single(result.ColumnInformation.NumericColumnNames);
            Assert.Equal(DefaultColumnNames.Features, result.ColumnInformation.NumericColumnNames.First());
            Assert.Null(result.ColumnInformation.ExampleWeightColumnName);
        }
    }
}