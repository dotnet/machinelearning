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
            Assert.Throws<ArgumentNullException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(null, new ColumnInformation(), null, TaskKind.Regression));
        }

        [Fact]
        public void ValidateExperimentExecuteNullLabel()
        {
            Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(Data, 
                new ColumnInformation() { LabelColumnName = null }, null, TaskKind.Regression));
        }

        [Fact]
        public void ValidateExperimentExecuteLabelNotInTrain()
        {
            Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(Data,
                new ColumnInformation() { LabelColumnName = "L" }, null, TaskKind.Regression));
        }

        [Fact]
        public void ValidateExperimentExecuteNumericColNotInTrain()
        {
            var columnInfo = new ColumnInformation();
            columnInfo.NumericColumnNames.Add("N");

            Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(Data, columnInfo, null, TaskKind.Regression));
        }

        [Fact]
        public void ValidateExperimentExecuteNullNumericCol()
        {
            var columnInfo = new ColumnInformation();
            columnInfo.NumericColumnNames.Add(null);
            Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(Data, columnInfo, null, TaskKind.Regression));
        }

        [Fact]
        public void ValidateExperimentExecuteDuplicateCol()
        {
            var columnInfo = new ColumnInformation();
            columnInfo.NumericColumnNames.Add(DefaultColumnNames.Label);

            Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(Data, columnInfo, null, TaskKind.Regression));
        }

        [Fact]
        public void ValidateExperimentExecuteArgsTrainValidColCountMismatch()
        {
            var context = new MLContext();

            var trainDataBuilder = new ArrayDataViewBuilder(context);
            trainDataBuilder.AddColumn("0", new string[] { "0" });
            trainDataBuilder.AddColumn("1", new string[] { "1" });
            var trainData = trainDataBuilder.GetDataView();

            var validDataBuilder = new ArrayDataViewBuilder(context);
            validDataBuilder.AddColumn("0", new string[] { "0" });
            var validData = validDataBuilder.GetDataView();

            Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(trainData, 
                new ColumnInformation() { LabelColumnName = "0" }, validData, TaskKind.Regression));
        }

        [Fact]
        public void ValidateExperimentExecuteArgsTrainValidColNamesMismatch()
        {
            var context = new MLContext();

            var trainDataBuilder = new ArrayDataViewBuilder(context);
            trainDataBuilder.AddColumn("0", new string[] { "0" });
            trainDataBuilder.AddColumn("1", new string[] { "1" });
            var trainData = trainDataBuilder.GetDataView();

            var validDataBuilder = new ArrayDataViewBuilder(context);
            validDataBuilder.AddColumn("0", new string[] { "0" });
            validDataBuilder.AddColumn("2", new string[] { "2" });
            var validData = validDataBuilder.GetDataView();

            Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(trainData,
                new ColumnInformation() { LabelColumnName = "0" }, validData, TaskKind.Regression));
        }

        [Fact]
        public void ValidateExperimentExecuteArgsTrainValidColTypeMismatch()
        {
            var context = new MLContext();

            var trainDataBuilder = new ArrayDataViewBuilder(context);
            trainDataBuilder.AddColumn("0", new string[] { "0" });
            trainDataBuilder.AddColumn("1", new string[] { "1" });
            var trainData = trainDataBuilder.GetDataView();

            var validDataBuilder = new ArrayDataViewBuilder(context);
            validDataBuilder.AddColumn("0", new string[] { "0" });
            validDataBuilder.AddColumn("1", NumberDataViewType.Single, new float[] { 1 });
            var validData = validDataBuilder.GetDataView();

            Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(trainData,
                new ColumnInformation() { LabelColumnName = "0" }, validData, TaskKind.Regression));
        }

        [Fact]
        public void ValidateInferColumnsArgsNullPath()
        {
            Assert.Throws<ArgumentNullException>(() => UserInputValidationUtil.ValidateInferColumnsArgs(null, "Label"));
        }

        [Fact]
        public void ValidateInferColumnsArgsPathNotExist()
        {
            Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateInferColumnsArgs("idontexist", "Label"));
        }

        [Fact]
        public void ValidateInferColumnsArgsEmptyFile()
        {
            const string emptyFilePath = "empty";
            File.Create(emptyFilePath).Dispose();
            Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateInferColumnsArgs(emptyFilePath, "Label"));
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
            var dataView = new EmptyDataView(new MLContext(), schema);
            Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, new ColumnInformation(), null, TaskKind.Regression));
        }

        [Fact]
        public void ValidateTextColumnNotText()
        {
            const string TextPurposeColName = "TextColumn";
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn(DefaultColumnNames.Features, NumberDataViewType.Single);
            schemaBuilder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single);
            schemaBuilder.AddColumn(TextPurposeColName, NumberDataViewType.Double);
            var schema = schemaBuilder.ToSchema();
            var dataView = new EmptyDataView(new MLContext(), schema);

            var columnInfo = new ColumnInformation();
            columnInfo.NumericColumnNames.Add(TextPurposeColName);

            Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, columnInfo, null, TaskKind.Regression));
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
            var dataView = new EmptyDataView(new MLContext(), schema);
            Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, new ColumnInformation(),
                null, TaskKind.Regression));
        }

        [Fact]
        public void ValidateEmptyTrainingDataThrows()
        {
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("Number", NumberDataViewType.Single);
            schemaBuilder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single);
            var schema = schemaBuilder.ToSchema();
            var dataView = new EmptyDataView(new MLContext(), schema);
            Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, new ColumnInformation(),
                null, TaskKind.Regression));
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
            var validationData = new EmptyDataView(new MLContext(), schema);

            Assert.Throws<ArgumentException>(() => UserInputValidationUtil.ValidateExperimentExecuteArgs(trainingData, new ColumnInformation(),
                validationData, TaskKind.Regression));
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
