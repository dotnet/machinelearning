// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.ML.Auto.Test
{
    /*[TestClass]
    public class UserInputValidationTests
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ValidateAutoFitNullTrainData()
        {
            UserInputValidationUtil.ValidateAutoFitArgs(null, DatasetUtil.UciAdultLabel,
                DatasetUtil.GetUciAdultDataView(), null, null);
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
                new AutoFitSettings()
                {
                    StoppingCriteria = new ExperimentStoppingCriteria()
                    {
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
        public void ValidateAutoFitArgsPurposeOverrideSuccess()
        {
            UserInputValidationUtil.ValidateAutoFitArgs(DatasetUtil.GetUciAdultDataView(),
                DatasetUtil.UciAdultLabel, DatasetUtil.GetUciAdultDataView(),
                null, new List<(string, ColumnPurpose)>()
                {
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

        [TestMethod]
        public void ValidateInferColsPath()
        {
            UserInputValidationUtil.ValidateInferColumnsArgs(DatasetUtil.DownloadUciAdultDataset());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ValidateFeaturesColInvalidType()
        {
            var schemaBuilder = new SchemaBuilder();
            schemaBuilder.AddColumn(DefaultColumnNames.Features, NumberType.R8);
            schemaBuilder.AddColumn(DefaultColumnNames.Label, NumberType.R4);
            var schema = schemaBuilder.GetSchema();
            var dataView = new EmptyDataView(new MLContext(), schema);
            UserInputValidationUtil.ValidateAutoFitArgs(dataView, DefaultColumnNames.Label, null, null, null);
        }
    }*/
}
