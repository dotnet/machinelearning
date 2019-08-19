// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Data;
using Xunit;

namespace Microsoft.ML.AutoML.Test
{
    
    public class UserInputValidationTests
    {
        private static readonly IDataView Data = DatasetUtil.GetUciAdultDataView();

        [Fact]
        public void ValidateExperimentExecuteNullTrainData()
        {
            var ex = Assert.Throws<ArgumentNullException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(null, new ColumnInformation(), null, TaskKind.Regression));
            Assert.StartsWith("Training data cannot be null", ex.Message);
        }

        [Fact]
        public void ValidateExperimentExecuteNullLabel()
        {
            var ex = Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(Data, 
                new ColumnInformation() { LabelColumnName = null }, null, TaskKind.Regression));

            Assert.Equal("Provided label column cannot be null", ex.Message);
        }

        [Fact]
        public void ValidateExperimentExecuteLabelNotInTrain()
        {
            var ex = Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(Data,
                new ColumnInformation() { LabelColumnName = "L" }, null, TaskKind.Regression));

            Assert.Equal("Provided label column 'L' not found in training data.", ex.Message);
        }

        [Fact]
        public void ValidateExperimentExecuteNumericColNotInTrain()
        {
            var columnInfo = new ColumnInformation();
            columnInfo.NumericColumnNames.Add("N");

            var ex = Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(Data, columnInfo, null, TaskKind.Regression));
            Assert.Equal("Provided label column 'Label' was of type Boolean, but only type Single is allowed.", ex.Message);
        }

        [Fact]
        public void ValidateExperimentExecuteNullNumericCol()
        {
            var columnInfo = new ColumnInformation();
            columnInfo.NumericColumnNames.Add(null);

            var ex = Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(Data, columnInfo, null, TaskKind.Regression));
            Assert.Equal("Null column string was specified as numeric in column information", ex.Message);
        }

        [Fact]
        public void ValidateExperimentExecuteDuplicateCol()
        {
            var columnInfo = new ColumnInformation();
            columnInfo.NumericColumnNames.Add(DefaultColumnNames.Label);

            var ex = Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(Data, columnInfo, null, TaskKind.Regression));
        }

        [Fact]
        public void ValidateExperimentExecuteArgsTrainValidColCountMismatch()
        {
            var context = new MLContext();

            var trainDataBuilder = new ArrayDataViewBuilder(context);
            trainDataBuilder.AddColumn("0", NumberDataViewType.Single, new float[] { 1 });
            trainDataBuilder.AddColumn("1", new string[] { "1" });
            var trainData = trainDataBuilder.GetDataView();

            var validDataBuilder = new ArrayDataViewBuilder(context);
            validDataBuilder.AddColumn("0", NumberDataViewType.Single, new float[] { 1 });
            var validData = validDataBuilder.GetDataView();

            var ex = Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(trainData, 
                new ColumnInformation() { LabelColumnName = "0" }, validData, TaskKind.Regression));
            Assert.StartsWith("Training data and validation data schemas do not match. Train data has '2' columns,and validation data has '1' columns.", ex.Message);
        }

        [Fact]
        public void ValidateExperimentExecuteArgsTrainValidColNamesMismatch()
        {
            var context = new MLContext();

            var trainDataBuilder = new ArrayDataViewBuilder(context);
            trainDataBuilder.AddColumn("0", NumberDataViewType.Single, new float[] { 1 });
            trainDataBuilder.AddColumn("1", new string[] { "1" });
            var trainData = trainDataBuilder.GetDataView();

            var validDataBuilder = new ArrayDataViewBuilder(context);
            validDataBuilder.AddColumn("0", NumberDataViewType.Single, new float[] { 1 });
            validDataBuilder.AddColumn("2", new string[] { "2" });
            var validData = validDataBuilder.GetDataView();

            var ex = Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(trainData,
                new ColumnInformation() { LabelColumnName = "0" }, validData, TaskKind.Regression));
            Assert.StartsWith("Training data and validation data schemas do not match. Column '1' exsits in train data, but not in validation data.", ex.Message);
        }

        [Fact]
        public void ValidateExperimentExecuteArgsTrainValidColTypeMismatch()
        {
            var context = new MLContext();

            var trainDataBuilder = new ArrayDataViewBuilder(context);
            trainDataBuilder.AddColumn("0", NumberDataViewType.Single, new float[] { 1 });
            trainDataBuilder.AddColumn("1", new string[] { "1" });
            var trainData = trainDataBuilder.GetDataView();

            var validDataBuilder = new ArrayDataViewBuilder(context);
            validDataBuilder.AddColumn("0", NumberDataViewType.Single, new float[] { 1 });
            validDataBuilder.AddColumn("1", NumberDataViewType.Single, new float[] { 1 });
            var validData = validDataBuilder.GetDataView();

            var ex = Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(trainData,
                new ColumnInformation() { LabelColumnName = "0" }, validData, TaskKind.Regression));
            Assert.StartsWith("Training data and validation data schemas do not match. Column '1' is of type String in train data, and type Single in validation data.", ex.Message);
        }

        [Fact]
        public void ValidateInferColumnsArgsNullPath()
        {
            var ex = Assert.Throws<ArgumentNullException>(() => UserInputValidationUtil.ValidateInferColumnsArgs(null, "Label"));
            Assert.StartsWith("Provided path cannot be null", ex.Message);
        }

        [Fact]
        public void ValidateInferColumnsArgsPathNotExist()
        {
            var ex = Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateInferColumnsArgs("idontexist", "Label"));
            Assert.StartsWith("File 'idontexist' does not exist", ex.Message);
        }

        [Fact]
        public void ValidateInferColumnsArgsEmptyFile()
        {
            const string emptyFilePath = "empty";
            File.Create(emptyFilePath).Dispose();
            var ex = Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateInferColumnsArgs(emptyFilePath, "Label"));
            Assert.StartsWith("File at path 'empty' cannot be empty", ex.Message);
        }

        [Fact]
        public void ValidateInferColsPath()
        {
            UserInputValidationUtil.ValidateInferColumnsArgs(DatasetUtil.DownloadUciAdultDataset());
        }

        [Fact]
        public void ValidateFeaturesColInvalidType()
        {
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn(DefaultColumnNames.Features, NumberDataViewType.Double);
            schemaBuilder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single);
            var schema = schemaBuilder.ToSchema();
            var dataView = DataViewTestFixture.BuildDummyDataView(schema);

            var ex = Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, new ColumnInformation(), null, TaskKind.Regression));
            Assert.StartsWith("Features column must be of data type Single", ex.Message);
        }

        [Fact]
        public void ValidateTextColumnNotText()
        {
            const string TextPurposeColName = "TextColumn";
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn(DefaultColumnNames.Features, NumberDataViewType.Single);
            schemaBuilder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single);
            schemaBuilder.AddColumn(TextPurposeColName, NumberDataViewType.Single);
            var schema = schemaBuilder.ToSchema();
            var dataView = DataViewTestFixture.BuildDummyDataView(schema);

            var columnInfo = new ColumnInformation();
            columnInfo.TextColumnNames.Add(TextPurposeColName);

            var ex = Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, columnInfo, null, TaskKind.Regression));
            Assert.Equal("Provided text column 'TextColumn' was of type Single, but only type String is allowed.", ex.Message);
        }

        [Fact]
        public void ValidateRegressionLabelTypes()
        {
            ValidateLabelTypeTestCore<float>(TaskKind.Regression, NumberDataViewType.Single, true);
            ValidateLabelTypeTestCore<bool>(TaskKind.Regression, BooleanDataViewType.Instance, false);
            ValidateLabelTypeTestCore<double>(TaskKind.Regression, NumberDataViewType.Double, false);
            ValidateLabelTypeTestCore<string>(TaskKind.Regression, TextDataViewType.Instance, false);
        }

        [Fact]
        public void ValidateBinaryClassificationLabelTypes()
        {
            ValidateLabelTypeTestCore<float>(TaskKind.BinaryClassification, NumberDataViewType.Single, false);
            ValidateLabelTypeTestCore<bool>(TaskKind.BinaryClassification, BooleanDataViewType.Instance, true);
        }

        [Fact]
        public void ValidateMulticlassLabelTypes()
        {
            ValidateLabelTypeTestCore<float>(TaskKind.MulticlassClassification, NumberDataViewType.Single, true);
            ValidateLabelTypeTestCore<bool>(TaskKind.MulticlassClassification, BooleanDataViewType.Instance, true);
            ValidateLabelTypeTestCore<double>(TaskKind.MulticlassClassification, NumberDataViewType.Double, true);
            ValidateLabelTypeTestCore<string>(TaskKind.MulticlassClassification, TextDataViewType.Instance, true);
        }

        [Fact]
        public void ValidateAllowedFeatureColumnTypes()
        {
            var dataViewBuilder = new ArrayDataViewBuilder(new MLContext());
            dataViewBuilder.AddColumn("Boolean", BooleanDataViewType.Instance, false);
            dataViewBuilder.AddColumn("Number", NumberDataViewType.Single, 0f);
            dataViewBuilder.AddColumn("Text", "a");
            dataViewBuilder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single, 0f);
            var dataView = dataViewBuilder.GetDataView();
            UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, new ColumnInformation(),
                null, TaskKind.Regression);
        }

        [Fact]
        public void ValidateProhibitedFeatureColumnType()
        {
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("UInt64", NumberDataViewType.UInt64);
            schemaBuilder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single);
            var schema = schemaBuilder.ToSchema();
            var dataView = DataViewTestFixture.BuildDummyDataView(schema);

            var ex = Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, new ColumnInformation(),
                null, TaskKind.Regression));
            Assert.StartsWith("Only supported feature column types are Boolean, Single, and String. Please change the feature column UInt64 of type UInt64 to one of the supported types.", ex.Message);
        }

        [Fact]
        public void ValidateEmptyTrainingDataThrows()
        {
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("Number", NumberDataViewType.Single);
            schemaBuilder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single);
            var schema = schemaBuilder.ToSchema();
            var dataView = DataViewTestFixture.BuildDummyDataView(schema, createDummyRow: false);
            var ex = Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, new ColumnInformation(),
                null, TaskKind.Regression));
            Assert.StartsWith("Training data has 0 rows", ex.Message);
        }

        [Fact]
        public void ValidateEmptyValidationDataThrows()
        {
            // Training data
            var dataViewBuilder = new ArrayDataViewBuilder(new MLContext());
            dataViewBuilder.AddColumn("Number", NumberDataViewType.Single, 0f);
            dataViewBuilder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single, 0f);
            var trainingData = dataViewBuilder.GetDataView();

            // Validation data
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("Number", NumberDataViewType.Single);
            schemaBuilder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single);
            var schema = schemaBuilder.ToSchema();
            var validationData = DataViewTestFixture.BuildDummyDataView(schema, createDummyRow: false);

            var ex = Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(trainingData, new ColumnInformation(),
                validationData, TaskKind.Regression));
            Assert.StartsWith("Validation data has 0 rows", ex.Message);
        }

        private static void ValidateLabelTypeTestCore<LabelRawType>(TaskKind task, PrimitiveDataViewType labelType, bool labelTypeShouldBeValid)
        {
            var dataViewBuilder = new ArrayDataViewBuilder(new MLContext());
            dataViewBuilder.AddColumn(DefaultColumnNames.Features, NumberDataViewType.Single, 0f);
            if (labelType == TextDataViewType.Instance)
            {
                dataViewBuilder.AddColumn(DefaultColumnNames.Label, string.Empty);
            }
            else
            {
                dataViewBuilder.AddColumn(DefaultColumnNames.Label, labelType, Activator.CreateInstance<LabelRawType>());
            }
            var dataView = dataViewBuilder.GetDataView();
            var validationExceptionThrown = false;
            try
            {
                UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, new ColumnInformation(), null, task);
            }
            catch
            {
                validationExceptionThrown = true;
            }
            Assert.Equal(labelTypeShouldBeValid, !validationExceptionThrown);
        }
    }
}
