// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Runtime.CompilerServices;
using ApprovalTests;
using ApprovalTests.Reporters;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.CLI.CodeGenerator.CSharp;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace mlnet.Test
{
    [TestClass]
    [UseReporter(typeof(DiffReporter))]
    public class ConsoleCodeGeneratorTests
    {
        private Pipeline pipeline;
        private ColumnInferenceResults columnInference = default;
        private string namespaceValue = "TestNamespace";


        [TestMethod]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ConsoleHelperFileContentTest()
        {
            (Pipeline pipeline,
                        ColumnInferenceResults columnInference) = GetMockedPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, new CodeGeneratorSettings()
            {
                MlTask = TaskKind.BinaryClassification,
                OutputBaseDir = null,
                OutputName = "MyNamespace",
                TrainDataset = "x:\\dummypath\\dummy_train.csv",
                TestDataset = "x:\\dummypath\\dummy_test.csv",
                LabelName = "Label",
                ModelPath = "x:\\models\\model.zip"
            });
            var result = consoleCodeGen.GenerateTrainProjectContents(namespaceValue, typeof(float));

            Approvals.Verify(result.Item3);
        }

        [TestMethod]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void TrainProgramCSFileContentTest()
        {
            (Pipeline pipeline,
                        ColumnInferenceResults columnInference) = GetMockedPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, new CodeGeneratorSettings()
            {
                MlTask = TaskKind.BinaryClassification,
                OutputBaseDir = null,
                OutputName = "MyNamespace",
                TrainDataset = "x:\\dummypath\\dummy_train.csv",
                TestDataset = "x:\\dummypath\\dummy_test.csv",
                LabelName = "Label",
                ModelPath = "x:\\models\\model.zip"
            });
            var result = consoleCodeGen.GenerateTrainProjectContents(namespaceValue, typeof(float));

            Approvals.Verify(result.Item1);
        }


        [TestMethod]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void TrainProjectFileContentTest()
        {
            (Pipeline pipeline,
                        ColumnInferenceResults columnInference) = GetMockedPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, new CodeGeneratorSettings()
            {
                MlTask = TaskKind.BinaryClassification,
                OutputBaseDir = null,
                OutputName = "MyNamespace",
                TrainDataset = "x:\\dummypath\\dummy_train.csv",
                TestDataset = "x:\\dummypath\\dummy_test.csv",
                LabelName = "Label",
                ModelPath = "x:\\models\\model.zip"
            });
            var result = consoleCodeGen.GenerateTrainProjectContents(namespaceValue, typeof(float));

            Approvals.Verify(result.Item2);
        }

        [TestMethod]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ModelProjectFileContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, new CodeGeneratorSettings()
            {
                MlTask = TaskKind.BinaryClassification,
                OutputBaseDir = null,
                OutputName = "MyNamespace",
                TrainDataset = "x:\\dummypath\\dummy_train.csv",
                TestDataset = "x:\\dummypath\\dummy_test.csv",
                LabelName = "Label",
                ModelPath = "x:\\models\\model.zip"
            });
            var result = consoleCodeGen.GenerateModelProjectContents(namespaceValue, typeof(float));

            Approvals.Verify(result.ModelProjectFileContent);
        }

        [TestMethod]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ObservationCSFileContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, new CodeGeneratorSettings()
            {
                MlTask = TaskKind.BinaryClassification,
                OutputBaseDir = null,
                OutputName = "MyNamespace",
                TrainDataset = "x:\\dummypath\\dummy_train.csv",
                TestDataset = "x:\\dummypath\\dummy_test.csv",
                LabelName = "Label",
                ModelPath = "x:\\models\\model.zip"
            });
            var result = consoleCodeGen.GenerateModelProjectContents(namespaceValue, typeof(float));

            Approvals.Verify(result.ObservationCSFileContent);
        }


        [TestMethod]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void PredictionCSFileContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, new CodeGeneratorSettings()
            {
                MlTask = TaskKind.BinaryClassification,
                OutputBaseDir = null,
                OutputName = "MyNamespace",
                TrainDataset = "x:\\dummypath\\dummy_train.csv",
                TestDataset = "x:\\dummypath\\dummy_test.csv",
                LabelName = "Label",
                ModelPath = "x:\\models\\model.zip"
            });
            var result = consoleCodeGen.GenerateModelProjectContents(namespaceValue, typeof(float));

            Approvals.Verify(result.PredictionCSFileContent);
        }

        [TestMethod]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void PredictProgramCSFileContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, new CodeGeneratorSettings()
            {
                MlTask = TaskKind.BinaryClassification,
                OutputBaseDir = null,
                OutputName = "MyNamespace",
                TrainDataset = "x:\\dummypath\\dummy_train.csv",
                TestDataset = "x:\\dummypath\\dummy_test.csv",
                LabelName = "Label",
                ModelPath = "x:\\models\\model.zip"
            });
            var result = consoleCodeGen.GeneratePredictProjectContents(namespaceValue);

            Approvals.Verify(result.PredictProgramCSFileContent);
        }

        [TestMethod]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void PredictProjectFileContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, new CodeGeneratorSettings()
            {
                MlTask = TaskKind.BinaryClassification,
                OutputBaseDir = null,
                OutputName = "MyNamespace",
                TrainDataset = "x:\\dummypath\\dummy_train.csv",
                TestDataset = "x:\\dummypath\\dummy_test.csv",
                LabelName = "Label",
                ModelPath = "x:\\models\\model.zip"
            });
            var result = consoleCodeGen.GeneratePredictProjectContents(namespaceValue);

            Approvals.Verify(result.PredictProjectFileContent);
        }

        private (Pipeline, ColumnInferenceResults) GetMockedPipelineAndInference()
        {
            if (pipeline == null)
            {
                MLContext context = new MLContext();
                // same learners with different hyperparams
                var hyperparams1 = new Microsoft.ML.Auto.ParameterSet(new List<Microsoft.ML.Auto.IParameterValue>() { new LongParameterValue("NumLeaves", 2) });
                var hyperparams2 = new Microsoft.ML.Auto.ParameterSet(new List<Microsoft.ML.Auto.IParameterValue>() { new LongParameterValue("NumLeaves", 6) });
                var trainer1 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), new ColumnInformation(), hyperparams1);
                var trainer2 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), new ColumnInformation(), hyperparams2);
                var transforms1 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
                var transforms2 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
                var inferredPipeline1 = new SuggestedPipeline(transforms1, new List<SuggestedTransform>(), trainer1, context, true);
                var inferredPipeline2 = new SuggestedPipeline(transforms2, new List<SuggestedTransform>(), trainer2, context, false);

                this.pipeline = inferredPipeline1.ToPipeline();
                var textLoaderArgs = new TextLoader.Options()
                {
                    Columns = new[] {
                        new TextLoader.Column("Label", DataKind.Boolean, 0),
                        new TextLoader.Column("col1", DataKind.Single, 1),
                        new TextLoader.Column("col2", DataKind.Single, 0),
                        new TextLoader.Column("col3", DataKind.String, 0),
                        new TextLoader.Column("col4", DataKind.Int32, 0),
                        new TextLoader.Column("col5", DataKind.UInt32, 0),
                    },
                    AllowQuoting = true,
                    AllowSparse = true,
                    HasHeader = true,
                    Separators = new[] { ',' }
                };

                this.columnInference = new ColumnInferenceResults()
                {
                    TextLoaderOptions = textLoaderArgs,
                    ColumnInformation = new ColumnInformation() { LabelColumn = "Label" }
                };
            }
            return (pipeline, columnInference);
        }
    }
}
