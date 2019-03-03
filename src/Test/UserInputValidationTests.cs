// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.Data.DataView;
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
            UserInputValidationUtil.ValidateExperimentExecuteArgs(null, new ColumnInformation(), null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateExperimentExecuteNullLabel()
        {
            UserInputValidationUtil.ValidateExperimentExecuteArgs(Data, 
                new ColumnInformation() { LabelColumn = null }, null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateExperimentExecuteLabelNotInTrain()
        {
            UserInputValidationUtil.ValidateExperimentExecuteArgs(Data,
                new ColumnInformation() { LabelColumn = "L" }, null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateExperimentExecuteNumericColNotInTrain()
        {
            var columnInfo = new ColumnInformation();
            columnInfo.NumericColumns.Add("N");

            UserInputValidationUtil.ValidateExperimentExecuteArgs(Data, columnInfo, null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateExperimentExecuteNullNumericCol()
        {
            var columnInfo = new ColumnInformation();
            columnInfo.NumericColumns.Add(null);
            UserInputValidationUtil.ValidateExperimentExecuteArgs(Data, columnInfo, null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateExperimentExecuteDuplicateCol()
        {
            var columnInfo = new ColumnInformation();
            columnInfo.NumericColumns.Add(DefaultColumnNames.Label);

            UserInputValidationUtil.ValidateExperimentExecuteArgs(Data, columnInfo, null);
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
                new ColumnInformation() { LabelColumn = "0" }, validData);
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
                new ColumnInformation() { LabelColumn = "0" }, validData);
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
                new ColumnInformation() { LabelColumn = "0" }, validData);
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
            UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, new ColumnInformation(), null);
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
            columnInfo.NumericColumns.Add(TextPurposeColName);

            UserInputValidationUtil.ValidateExperimentExecuteArgs(dataView, columnInfo, null);
        }
    }
}
