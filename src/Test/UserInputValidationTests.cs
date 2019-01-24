using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    [TestClass]
    public class UserInputValidationTests
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ValidateCreateTextReaderArgsNullInput()
        {
            UserInputValidationUtil.ValidateCreateTextReaderArgs(null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateCreateTextReaderArgsNoColumns()
        {
            var input = new ColumnInferenceResult(new List<(TextLoader.Column, ColumnPurpose)>(),
                false, false, "\t", false, false);
            UserInputValidationUtil.ValidateCreateTextReaderArgs(input);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateCreateTextReaderArgsNullColumn()
        {
            var input = new ColumnInferenceResult(
                new List<(TextLoader.Column, ColumnPurpose)>() { (null, ColumnPurpose.CategoricalFeature) },
                false, false, "\t", false, false);
            UserInputValidationUtil.ValidateCreateTextReaderArgs(input);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateCreateTextReaderArgsColumnWithNullSoure()
        {
            var input = new ColumnInferenceResult(
                new List<(TextLoader.Column, ColumnPurpose)>() { (new TextLoader.Column() { Name = "Column", Type = DataKind.R4 } , ColumnPurpose.CategoricalFeature) },
                false, false, "\t", false, false);
            UserInputValidationUtil.ValidateCreateTextReaderArgs(input);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateCreateTextReaderArgsNullSeparator()
        {
            var input = new ColumnInferenceResult(
                new List<(TextLoader.Column, ColumnPurpose)>() { (new TextLoader.Column("Column", DataKind.R4, 4), ColumnPurpose.CategoricalFeature) },
                false, false, null, false, false);
            UserInputValidationUtil.ValidateCreateTextReaderArgs(input);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ValidateAutoFitNullTrainData()
        {
            UserInputValidationUtil.ValidateAutoFitArgs(null, DatasetUtil.UciAdultLabel, 
                DatasetUtil.GetUciAdultDataView(), null, null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ValidateAutoFitArgsNullValidData()
        {
            UserInputValidationUtil.ValidateAutoFitArgs(DatasetUtil.GetUciAdultDataView(),
                DatasetUtil.UciAdultLabel, null, null, null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ValidateAutoFitArgsNullLabel()
        {
            UserInputValidationUtil.ValidateAutoFitArgs(DatasetUtil.GetUciAdultDataView(),
                null, DatasetUtil.GetUciAdultDataView(), null, null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateAutoFitArgsLabelNotInTrain()
        {
            UserInputValidationUtil.ValidateAutoFitArgs(DatasetUtil.GetUciAdultDataView(),
                "Label1", DatasetUtil.GetUciAdultDataView(), null, null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentOutOfRangeException))]
        public void ValidateAutoFitArgsZeroMaxIterations()
        {
            UserInputValidationUtil.ValidateAutoFitArgs(DatasetUtil.GetUciAdultDataView(),
                DatasetUtil.UciAdultLabel, DatasetUtil.GetUciAdultDataView(),
                new AutoFitSettings() {
                    StoppingCriteria = new ExperimentStoppingCriteria() {
                        MaxIterations = 0,
                    }
                }, null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateAutoFitArgsPurposeOverrideNullCol()
        {
            UserInputValidationUtil.ValidateAutoFitArgs(DatasetUtil.GetUciAdultDataView(),
                DatasetUtil.UciAdultLabel, DatasetUtil.GetUciAdultDataView(),
                null, new List<(string, ColumnPurpose)>()
                {
                    (null, ColumnPurpose.TextFeature)
                });
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateAutoFitArgsPurposeOverrideColNotExist()
        {
            UserInputValidationUtil.ValidateAutoFitArgs(DatasetUtil.GetUciAdultDataView(),
                DatasetUtil.UciAdultLabel, DatasetUtil.GetUciAdultDataView(),
                null, new List<(string, ColumnPurpose)>()
                {
                    ("IDontExist", ColumnPurpose.TextFeature)
                });
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateAutoFitArgsPurposeOverrideLabelMismatch()
        {
            UserInputValidationUtil.ValidateAutoFitArgs(DatasetUtil.GetUciAdultDataView(),
                DatasetUtil.UciAdultLabel, DatasetUtil.GetUciAdultDataView(),
                null, new List<(string, ColumnPurpose)>()
                {
                    ("Workclass", ColumnPurpose.Label)
                });
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateAutoFitArgsPurposeOverrideDuplicateCol()
        {
            UserInputValidationUtil.ValidateAutoFitArgs(DatasetUtil.GetUciAdultDataView(),
                DatasetUtil.UciAdultLabel, DatasetUtil.GetUciAdultDataView(),
                null, new List<(string, ColumnPurpose)>()
                {
                    ("Workclass", ColumnPurpose.CategoricalFeature),
                    ("Workclass", ColumnPurpose.CategoricalFeature)
                });
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateAutoFitArgsTrainValidColCountMismatch()
        {
            var context = new MLContext();

            var trainDataBuilder = new ArrayDataViewBuilder(context);
            trainDataBuilder.AddColumn("0", new string[] { "0" });
            trainDataBuilder.AddColumn("1", new string[] { "1" });
            var trainData = trainDataBuilder.GetDataView();

            var validDataBuilder = new ArrayDataViewBuilder(context);
            validDataBuilder.AddColumn("0", new string[] { "0" });
            var validData = validDataBuilder.GetDataView();

            UserInputValidationUtil.ValidateAutoFitArgs(trainData, "0", validData, null, null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateAutoFitArgsTrainValidColNamesMismatch()
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

            UserInputValidationUtil.ValidateAutoFitArgs(trainData, "0", validData, null, null);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateAutoFitArgsTrainValidColTypeMismatch()
        {
            var context = new MLContext();

            var trainDataBuilder = new ArrayDataViewBuilder(context);
            trainDataBuilder.AddColumn("0", new string[] { "0" });
            trainDataBuilder.AddColumn("1", new string[] { "1" });
            var trainData = trainDataBuilder.GetDataView();

            var validDataBuilder = new ArrayDataViewBuilder(context);
            validDataBuilder.AddColumn("0", new string[] { "0" });
            validDataBuilder.AddColumn("1", NumberType.R4, new float[] { 1 });
            var validData = validDataBuilder.GetDataView();

            UserInputValidationUtil.ValidateAutoFitArgs(trainData, "0", validData, null, null);
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
    }
}
