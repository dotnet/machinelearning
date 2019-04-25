// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class UserInputValidationTests
    {
        private static readonly IDataView Data = DatasetUtil.GetUciAdultDataView();

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ValidateExperimentExecuteNullTrainData()
        {
            UserInputValidationUtil.ValidateExperimentExecuteArgs(null, new ColumnInformation(), null, TaskKind.Regression);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateExperimentExecuteNullLabel()
        {
            UserInputValidationUtil.ValidateExperimentExecuteArgs(Data, 
                new ColumnInformation() { LabelColumnName = null }, null, TaskKind.Regression);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateExperimentExecuteLabelNotInTrain()
        {
            UserInputValidationUtil.ValidateExperimentExecuteArgs(Data,
                new ColumnInformation() { LabelColumnName = "L" }, null, TaskKind.Regression);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateExperimentExecuteNumericColNotInTrain()
        {
            var columnInfo = new ColumnInformation();
            columnInfo.NumericColumnNames.Add("N");

            UserInputValidationUtil.ValidateExperimentExecuteArgs(Data, columnInfo, null, TaskKind.Regression);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateExperimentExecuteNullNumericCol()
        {
            var columnInfo = new ColumnInformation();
            columnInfo.NumericColumnNames.Add(null);
            UserInputValidationUtil.ValidateExperimentExecuteArgs(Data, columnInfo, null, TaskKind.Regression);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateExperimentExecuteDuplicateCol()
        {
            var columnInfo = new ColumnInformation();
            columnInfo.NumericColumnNames.Add(DefaultColumnNames.Label);

            UserInputValidationUtil.ValidateExperimentExecuteArgs(Data, columnInfo, null, TaskKind.Regression);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
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

            UserInputValidationUtil.ValidateExperimentExecuteArgs(trainData, 
                new ColumnInformation() { LabelColumnName = "0" }, validData, TaskKind.Regression);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
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

            UserInputValidationUtil.ValidateExperimentExecuteArgs(trainData,
                new ColumnInformation() { LabelColumnName = "0" }, validData, TaskKind.Regression);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
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

            UserInputValidationUtil.ValidateExperimentExecuteArgs(trainData,
                new ColumnInformation() { LabelColumnName = "0" }, validData, TaskKind.Regression);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ValidateInferColumnsArgsNullPath()
        {
            UserInputValidationUtil.ValidateInferColumnsArgs(null, "Label");
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateInferColumnsArgsPathNotExist()
        {
            UserInputValidationUtil.ValidateInferColumnsArgs("idontexist", "Label");
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateInferColumnsArgsEmptyFile()
        {
            const string emptyFilePath = "empty";
            File.Create(emptyFilePath).Dispose();
            UserInputValidationUtil.ValidateInferColumnsArgs(emptyFilePath, "Label");
        }

        [TestMethod]
        public void ValidateInferColsPath()
        {
            UserInputValidationUtil.ValidateInferColumnsArgs(DatasetUtil.DownloadUciAdultDataset());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateFeaturesColInvalidType()
        {
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn(DefaultColumnNames.Features, NumberDataViewType.Double);
            schemaBuilder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single);
            var schema = schemaBuilder.ToSchema();
            var dataView = new EmptyDataView(new MLContext(), schema);
            UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, new ColumnInformation(), null, TaskKind.Regression);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
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

            UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, columnInfo, null, TaskKind.Regression);
        }

        [TestMethod]
        public void ValidateRegressionLabelTypes()
        {
            ValidateLabelTypeTestCore(TaskKind.Regression, NumberDataViewType.Single, true);
            ValidateLabelTypeTestCore(TaskKind.Regression, BooleanDataViewType.Instance, false);
            ValidateLabelTypeTestCore(TaskKind.Regression, NumberDataViewType.Double, false);
            ValidateLabelTypeTestCore(TaskKind.Regression, TextDataViewType.Instance, false);
        }

        [TestMethod]
        public void ValidateBinaryClassificationLabelTypes()
        {
            ValidateLabelTypeTestCore(TaskKind.BinaryClassification, NumberDataViewType.Single, false);
            ValidateLabelTypeTestCore(TaskKind.BinaryClassification, BooleanDataViewType.Instance, true);
        }

        [TestMethod]
        public void ValidateMulticlassLabelTypes()
        {
            ValidateLabelTypeTestCore(TaskKind.MulticlassClassification, NumberDataViewType.Single, true);
            ValidateLabelTypeTestCore(TaskKind.MulticlassClassification, BooleanDataViewType.Instance, true);
            ValidateLabelTypeTestCore(TaskKind.MulticlassClassification, NumberDataViewType.Double, true);
            ValidateLabelTypeTestCore(TaskKind.MulticlassClassification, TextDataViewType.Instance, true);
        }

        [TestMethod]
        public void ValidateAllowedFeatureColumnTypes()
        {
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("Boolean", BooleanDataViewType.Instance);
            schemaBuilder.AddColumn("Number", NumberDataViewType.Single);
            schemaBuilder.AddColumn("Text", TextDataViewType.Instance);
            schemaBuilder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single);
            var schema = schemaBuilder.ToSchema();
            var dataView = new EmptyDataView(new MLContext(), schema);
            UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, new ColumnInformation(),
                null, TaskKind.Regression);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateProhibitedFeatureColumnType()
        {
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn("UInt64", NumberDataViewType.UInt64);
            schemaBuilder.AddColumn(DefaultColumnNames.Label, NumberDataViewType.Single);
            var schema = schemaBuilder.ToSchema();
            var dataView = new EmptyDataView(new MLContext(), schema);
            UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, new ColumnInformation(),
                null, TaskKind.Regression);
        }

        private static void ValidateLabelTypeTestCore(TaskKind task, DataViewType labelType, bool labelTypeShouldBeValid)
        {
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumn(DefaultColumnNames.Features, NumberDataViewType.Single);
            schemaBuilder.AddColumn(DefaultColumnNames.Label, labelType);
            var schema = schemaBuilder.ToSchema();
            var dataView = new EmptyDataView(new MLContext(), schema);
            var validationExceptionThrown = false;
            try
            {
                UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, new ColumnInformation(), null, task);
            }
            catch
            {
                validationExceptionThrown = true;
            }
            Assert.AreEqual(labelTypeShouldBeValid, !validationExceptionThrown);
        }
    }
}
