// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Runtime.CompilerServices;
using ApprovalTests;
using ApprovalTests.Reporters;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CSharp;
using Microsoft.ML.Data;
using Xunit;

namespace mlnet.Tests
{
    [UseReporter(typeof(DiffReporter))]
    public class ConsoleCodeGeneratorTests
    {
        private Pipeline mockedPipeline;
        private Pipeline mockedOvaPipeline;
        private ColumnInferenceResults columnInference = default;
        private string namespaceValue = "TestNamespace";


        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ConsoleAppModelBuilderCSFileContentOvaTest()
        {
            (Pipeline pipeline,
                        ColumnInferenceResults columnInference) = GetMockedOvaPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, new CodeGeneratorSettings()
            {
                MlTask = TaskKind.MulticlassClassification,
                OutputBaseDir = null,
                OutputName = "MyNamespace",
                TrainDataset = "x:\\dummypath\\dummy_train.csv",
                TestDataset = "x:\\dummypath\\dummy_test.csv",
                LabelName = "Label",
                ModelPath = "x:\\models\\model.zip"
            });
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(namespaceValue, typeof(float), true, true, false);

            Approvals.Verify(result.modelBuilderCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ConsoleAppModelBuilderCSFileContentBinaryTest()
        {
            (Pipeline pipeline,
                        ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

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
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(namespaceValue, typeof(float), true, true, false);

            Approvals.Verify(result.modelBuilderCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ConsoleAppModelBuilderCSFileContentRegressionTest()
        {
            (Pipeline pipeline,
                        ColumnInferenceResults columnInference) = GetMockedRegressionPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, new CodeGeneratorSettings()
            {
                MlTask = TaskKind.Regression,
                OutputBaseDir = null,
                OutputName = "MyNamespace",
                TrainDataset = "x:\\dummypath\\dummy_train.csv",
                TestDataset = "x:\\dummypath\\dummy_test.csv",
                LabelName = "Label",
                ModelPath = "x:\\models\\model.zip"
            });
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(namespaceValue, typeof(float), true, true, false);

            Approvals.Verify(result.modelBuilderCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ModelProjectFileContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

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
            var result = consoleCodeGen.GenerateModelProjectContents(namespaceValue, typeof(float), true, true, true);

            Approvals.Verify(result.ModelProjectFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ConsumeModelContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, new CodeGeneratorSettings()
            {
                MlTask = TaskKind.BinaryClassification,
                OutputBaseDir = null,
                OutputName = "MyNamespace",
                TrainDataset = "x:\\dummypath\\dummy_train.csv",
                TestDataset = "x:\\dummypath\\dummy_test.csv",
                LabelName = "Label",
                ModelPath = "x:\\models\\model.zip",
                Target = GenerateTarget.Cli,
            });
            var result = consoleCodeGen.GenerateModelProjectContents(namespaceValue, typeof(float), true, true, false);

            Approvals.Verify(result.ConsumeModelCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ObservationCSFileContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

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
            var result = consoleCodeGen.GenerateModelProjectContents(namespaceValue, typeof(float), true, true, false);

            Approvals.Verify(result.ModelInputCSFileContent);
        }


        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void PredictionCSFileContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

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
            var result = consoleCodeGen.GenerateModelProjectContents(namespaceValue, typeof(float), true, true, false);

            Approvals.Verify(result.ModelOutputCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void PredictionProgramCSFileContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

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
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(namespaceValue, typeof(float), true, true, false);

            Approvals.Verify(result.ConsoleAppProgramCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]

        public void ConsoleAppProgramCSFileContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

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
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(namespaceValue, typeof(float), true, true, false);

            Approvals.Verify(result.ConsoleAppProgramCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ConsoleAppProjectFileContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

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
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(namespaceValue, typeof(float), true, true, false);

            Approvals.Verify(result.ConsoleAppProjectFileContent);
        }

        private (Pipeline, ColumnInferenceResults) GetMockedBinaryPipelineAndInference()
        {
            if (mockedPipeline == null)
            {
                MLContext context = new MLContext();
                // same learners with different hyperparams
                var hyperparams1 = new Microsoft.ML.AutoML.ParameterSet(new List<Microsoft.ML.AutoML.IParameterValue>() { new LongParameterValue("NumLeaves", 2) });
                var trainer1 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), new ColumnInformation(), hyperparams1);
                var transforms1 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
                var inferredPipeline1 = new SuggestedPipeline(transforms1, new List<SuggestedTransform>(), trainer1, context, true);

                this.mockedPipeline = inferredPipeline1.ToPipeline();
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
                    ColumnInformation = new ColumnInformation() { LabelColumnName = "Label" }
                };
            }
            return (mockedPipeline, columnInference);
        }

        private (Pipeline, ColumnInferenceResults) GetMockedRegressionPipelineAndInference()
        {
            if (mockedPipeline == null)
            {
                MLContext context = new MLContext();
                // same learners with different hyperparams
                var hyperparams1 = new Microsoft.ML.AutoML.ParameterSet(new List<Microsoft.ML.AutoML.IParameterValue>() { new LongParameterValue("NumLeaves", 2) });
                var trainer1 = new SuggestedTrainer(context, new LightGbmRegressionExtension(), new ColumnInformation(), hyperparams1);
                var transforms1 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
                var inferredPipeline1 = new SuggestedPipeline(transforms1, new List<SuggestedTransform>(), trainer1, context, true);

                this.mockedPipeline = inferredPipeline1.ToPipeline();
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
                    ColumnInformation = new ColumnInformation() { LabelColumnName = "Label" }
                };
            }
            return (mockedPipeline, columnInference);
        }
        private (Pipeline, ColumnInferenceResults) GetMockedOvaPipelineAndInference()
        {
            if (mockedOvaPipeline == null)
            {
                MLContext context = new MLContext();
                // same learners with different hyperparams
                var hyperparams1 = new Microsoft.ML.AutoML.ParameterSet(new List<Microsoft.ML.AutoML.IParameterValue>() { new LongParameterValue("NumLeaves", 2) });
                var trainer1 = new SuggestedTrainer(context, new FastForestOvaExtension(), new ColumnInformation(), hyperparams1);
                var transforms1 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
                var inferredPipeline1 = new SuggestedPipeline(transforms1, new List<SuggestedTransform>(), trainer1, context, true);

                this.mockedOvaPipeline = inferredPipeline1.ToPipeline();
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
                    ColumnInformation = new ColumnInformation() { LabelColumnName = "Label" }
                };

            }
            return (mockedOvaPipeline, columnInference);
        }
    }
}
