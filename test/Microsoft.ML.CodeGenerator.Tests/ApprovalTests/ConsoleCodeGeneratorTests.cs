// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator;
using Microsoft.ML.CodeGenerator.CodeGenerator;
using Microsoft.ML.CodeGenerator.CodeGenerator.CSharp;
using Microsoft.ML.CodeGenerator.CodeGenerator.CSharp.AzureCodeGenerator;
using Microsoft.ML.CodeGenerator.CSharp;
using Microsoft.ML.CodeGenerator.Templates.Azure.Console;
using Microsoft.ML.CodeGenerator.Templates.Console;
using Microsoft.ML.CodeGenerator.Utilities;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers;
using Xunit;
using Xunit.Abstractions;
using CodeGenerator = Microsoft.ML.CodeGenerator.CSharp.CodeGenerator;

namespace mlnet.Tests
{
    [UseReporter(typeof(DiffReporter))]
    public class ConsoleCodeGeneratorTests : BaseTestClass
    {
        private Pipeline _mockedPipeline;
        private Pipeline _mockedOvaPipeline;
        private ColumnInferenceResults _columnInference = default;
        private string _namespaceValue = "TestNamespace";
        private const string StablePackageVersion = "1.4.0-preview3-28229-2";
        private const string UnstablePackageVersion = "0.16.0-preview3-28229-2";

        public ConsoleCodeGeneratorTests(ITestOutputHelper output) : base(output)
        {
            if (System.Environment.GetEnvironmentVariable("HELIX_CORRELATION_ID") != null)
            {
                Approvals.UseAssemblyLocationForApprovedFiles();
            }
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ConsoleAppModelBuilderCSFileContentOvaTest()
        {
            (Pipeline pipeline,
                        ColumnInferenceResults columnInference) = GetMockedOvaPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, CreateCodeGeneratorSettingsFor(TaskKind.MulticlassClassification));
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(_namespaceValue, typeof(float), true, true,
                false, false, false, false);

            Approvals.Verify(result.modelBuilderCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ConsoleAppModelBuilderCSFileContentBinaryTest()
        {
            (Pipeline pipeline,
                        ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, CreateCodeGeneratorSettingsFor(TaskKind.BinaryClassification));
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(_namespaceValue, typeof(float), true, true,
                false, false, false, false);

            Approvals.Verify(result.modelBuilderCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ConsoleAppModelBuilderCSFileContentRegressionTest()
        {
            (Pipeline pipeline,
                        ColumnInferenceResults columnInference) = GetMockedRegressionPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, CreateCodeGeneratorSettingsFor(TaskKind.Regression));
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(_namespaceValue, typeof(float), true, true,
                false, false, false, false);

            Approvals.Verify(result.modelBuilderCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ConsoleAppModelBuilderCSFileContentRankingTest()
        {
            (Pipeline pipeline,
                        ColumnInferenceResults columnInference) = GetMockedRankingPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, CreateCodeGeneratorSettingsFor(TaskKind.Ranking));
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(_namespaceValue, typeof(float), true, true,
                false, false, false, false);

            Approvals.Verify(result.modelBuilderCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ModelProjectFileContentTestOnlyStableProjects()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, CreateCodeGeneratorSettingsFor(TaskKind.BinaryClassification));
            var result = consoleCodeGen.GenerateModelProjectContents(_namespaceValue, typeof(float), true, true, true,
                false, false, false);

            Approvals.Verify(result.ModelProjectFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ConsumeModelContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, CreateCodeGeneratorSettingsFor(TaskKind.BinaryClassification));
            var result = consoleCodeGen.GenerateModelProjectContents(_namespaceValue, typeof(float), true, true, false,
                false, false, false);

            Approvals.Verify(result.ConsumeModelCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ObservationCSFileContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, CreateCodeGeneratorSettingsFor(TaskKind.BinaryClassification));
            var result = consoleCodeGen.GenerateModelProjectContents(_namespaceValue, typeof(float), true, true, false,
                false, false, false);

            Approvals.Verify(result.ModelInputCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void PredictionCSFileContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, CreateCodeGeneratorSettingsFor(TaskKind.BinaryClassification));
            var result = consoleCodeGen.GenerateModelProjectContents(_namespaceValue, typeof(float), true, true, false,
                false, false, false);

            Approvals.Verify(result.ModelOutputCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void PredictionProgramCSFileContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, CreateCodeGeneratorSettingsFor(TaskKind.BinaryClassification));
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(_namespaceValue, typeof(float), true, true,
                false, false, false, false);

            Approvals.Verify(result.ConsoleAppProgramCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]

        public void ConsoleAppProgramCSFileContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, CreateCodeGeneratorSettingsFor(TaskKind.BinaryClassification));
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(_namespaceValue, typeof(float), true, true,
                false, false, false, false);

            Approvals.Verify(result.ConsoleAppProgramCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ConsoleAppProjectFileContentTest()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, CreateCodeGeneratorSettingsFor(TaskKind.BinaryClassification));
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(_namespaceValue, typeof(float), true, true,
                false, false, false, false);

            Approvals.Verify(result.ConsoleAppProjectFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void AzureImageCodeGeneratorTest()
        {
            (var pipeline, var columnInference) = GetMockedAzureImagePipelineAndInference();
            var setting = new CodeGeneratorSettings()
            {
                TrainDataset = @"/path/to/dataset",
                ModelName = @"/path/to/model",
                MlTask = TaskKind.MulticlassClassification,
                OutputName = @"CodeGenTest",
                OutputBaseDir = @"/path/to/codegen",
                LabelName = "Label",
                Target = GenerateTarget.ModelBuilder,
                StablePackageVersion = "stableversion",
                UnstablePackageVersion = "unstableversion",
                OnnxModelName = @"/path/to/onnxModel",
                OnnxRuntimePackageVersion = "1.2.3",
                IsAzureAttach = true,
                IsObjectDetection = false,
                IsImage = true,
            };
            var codeGen = new AzureAttachCodeGenenrator(pipeline, columnInference, setting);
            foreach (var project in codeGen.ToSolution())
            {
                foreach (var projectFile in project)
                {
                    NamerFactory.AdditionalInformation = projectFile.Name;
                    Approvals.Verify(((ICSharpFile)projectFile).File);
                }
            }
        }


        // Tevin: added to test OD codeGen working
        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void AzureObjectDetectionCodeGeneratorTest()
        {
            (var pipeline, var columnInference) = GetMockedAzureObjectDetectionPipelineAndInference(@"/path/to/onnxModel");
            var setting = new CodeGeneratorSettings()
            {
                TrainDataset = @"/path/to/dataset",
                ModelName = @"/path/to/model",
                MlTask = TaskKind.ObjectDetection,
                OutputName = @"CodeGenTest",
                OutputBaseDir = @"/path/to/codegen",
                LabelName = "Label",
                Target = GenerateTarget.ModelBuilder,
                StablePackageVersion = "stableversion",
                UnstablePackageVersion = "unstableversion",
                OnnxModelName = @"/path/to/onnxModel",
                OnnxRuntimePackageVersion = @"1.2.3",
                IsAzureAttach = true,
                IsImage = false,
                IsObjectDetection = true,
                ClassificationLabel = new string[] { "label1", "label2", "label3" },
                ObjectLabel = new string[] { "label1", "label2", "label3" },
            };
            var codeGen = new AzureAttachCodeGenenrator(pipeline, columnInference, setting);
            //codeGen.GenerateOutput(); // use this to see output in project.
            foreach (var project in codeGen.ToSolution())
            {
                foreach (var projectFile in project)
                {

                    NamerFactory.AdditionalInformation = projectFile.Name;
                    Approvals.Verify(((ICSharpFile)projectFile).File);
                }
            }
        }



        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void AzureCodeGeneratorTest()
        {
            (var pipeline, var columnInference, var mapping) = GetMockedAzurePipelineAndInference();
            var setting = new CodeGeneratorSettings()
            {
                TrainDataset = @"\path\to\file",
                ModelName = @"\path\to\model",
                MlTask = TaskKind.MulticlassClassification,
                OutputName = @"test",
                OutputBaseDir = @"\path\to\test",
                LabelName = "label",
                Target = GenerateTarget.ModelBuilder,
                StablePackageVersion = "StablePackageVersion",
                UnstablePackageVersion = "UnstablePackageVersion",
                OnnxModelName = @"\path\to\onnx",
                OnnxRuntimePackageVersion = "1.2.3",
                IsAzureAttach = true,
                IsImage = false,
                IsObjectDetection = false,
                OnnxInputMapping = mapping,
            };
            var codeGen = new AzureAttachCodeGenenrator(pipeline, columnInference, setting);
            foreach (var project in codeGen.ToSolution())
            {
                foreach (var projectFile in project)
                {
                    NamerFactory.AdditionalInformation = projectFile.Name;
                    Approvals.Verify(((ICSharpFile)projectFile).File);
                }
            }
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ModelInputClassTest()
        {
            // Test with datasets whose columns are sanitized and not sanitized. The columns of a dataset are considered
            // sanitized if the column names are all unique and distinct, irrespective of capitalization.
            (var pipelineSanitized, var columnInferenceSanitized, var mappingSanitized) = this.GetMockedAzurePipelineAndInference();
            TestModelInput(pipelineSanitized, columnInferenceSanitized, mappingSanitized, "sanitized");
            (var pipelineUnsatinized, var columnInferenceUnsatinized, var mappingUnsatinized) = this.GetMockedAzurePipelineAndInferenceUnsanitizedColumnNames();
            TestModelInput(pipelineUnsatinized, columnInferenceUnsatinized, mappingUnsatinized, "unsanitized");

        }

        private void TestModelInput(Pipeline pipeline, ColumnInferenceResults columnInference,
                                    IDictionary<string, CodeGeneratorSettings.ColumnMapping> mapping, string info)
        {
            // test with null map case
            var columnMappingStringList = Utils.GenerateClassLabels(columnInference);
            var modelInputProject = new CSharpCodeFile()
            {
                File = new ModelInputClass()
                {
                    Namespace = "test",
                    ClassLabels = columnMappingStringList,
                    Target = GenerateTarget.Cli,
                }.TransformText(),
                Name = "ModelInput.cs",
            };
            NamerFactory.AdditionalInformation = info + "_null_map";
            Approvals.Verify(modelInputProject.File);

            // test with map case
            columnMappingStringList = Utils.GenerateClassLabels(columnInference, mapping);
            modelInputProject = new CSharpCodeFile()
            {
                File = new ModelInputClass()
                {
                    Namespace = "test",
                    ClassLabels = columnMappingStringList,
                    Target = GenerateTarget.Cli,
                }.TransformText(),
                Name = "ModelInput.cs",
            };
            NamerFactory.AdditionalInformation = info + "_map";
            Approvals.Verify(modelInputProject.File);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void AzureModelBuilderTest()
        {
            // test Azure Image Case
            (var bestPipeLine, var columnInference) = GetMockedAzureImagePipelineAndInference();
            var (_, _, PreTrainerTransforms, _) = bestPipeLine.GenerateTransformsAndTrainers();

            var azureModelBuilder = new CSharpCodeFile()
            {
                File = new AzureModelBuilder()
                {
                    Path = "/path/to/file",
                    HasHeader = true,
                    Separator = ',',
                    PreTrainerTransforms = PreTrainerTransforms,
                    AllowQuoting = true,
                    AllowSparse = true,
                    Namespace = "test",
                    Target = GenerateTarget.ModelBuilder,
                    OnnxModelPath = "/path/to/onnxModel",
                    MLNetModelpath = "/path/to/MLNet",
                }.TransformText(),
                Name = "ModelBuilder.cs",
            };
            NamerFactory.AdditionalInformation = "Image";
            Approvals.Verify(azureModelBuilder.File);

            // test Azure Non-Image Case
            (bestPipeLine, columnInference, _) = GetMockedAzurePipelineAndInference();
            (_, _, PreTrainerTransforms, _) = bestPipeLine.GenerateTransformsAndTrainers();
            azureModelBuilder = new CSharpCodeFile()
            {
                File = new AzureModelBuilder()
                {
                    Path = "/path/to/file",
                    HasHeader = true,
                    Separator = ',',
                    PreTrainerTransforms = PreTrainerTransforms,
                    AllowQuoting = true,
                    AllowSparse = true,
                    Namespace = "test",
                    Target = GenerateTarget.ModelBuilder,
                    OnnxModelPath = "/path/to/onnxModel",
                    MLNetModelpath = "/path/to/MLNet",
                }.TransformText(),
                Name = "ModelBuilder.cs",
            };
            NamerFactory.AdditionalInformation = "NonImage";
            Approvals.Verify(azureModelBuilder.File);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void Recommendation_GenerateModelProjectContents_VerifyModelInput()
        {
            (string ModelInputCSFileContent, string ModelOutputCSFileContent, string ConsumeModelCSFileContent, string ModelProjectFileContent) codeGenResult
                = GenerateModelProjectContentsForRecommendation();

            Approvals.Verify(codeGenResult.ModelInputCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void Recommendation_GenerateModelProjectContents_VerifyModelOutput()
        {
            (string ModelInputCSFileContent, string ModelOutputCSFileContent, string ConsumeModelCSFileContent, string ModelProjectFileContent) codeGenResult
                = GenerateModelProjectContentsForRecommendation();

            Approvals.Verify(codeGenResult.ModelOutputCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void Recommendation_GenerateModelProjectContents_VerifyConsumeModel()
        {
            (string ModelInputCSFileContent, string ModelOutputCSFileContent, string ConsumeModelCSFileContent, string ModelProjectFileContent) codeGenResult
                = GenerateModelProjectContentsForRecommendation();

            Approvals.Verify(codeGenResult.ConsumeModelCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void Recommendation_GenerateModelProjectContents_VerifyModelProject()
        {
            (string ModelInputCSFileContent, string ModelOutputCSFileContent, string ConsumeModelCSFileContent, string ModelProjectFileContent) codeGenResult
                = GenerateModelProjectContentsForRecommendation();

            Approvals.Verify(codeGenResult.ModelProjectFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void Recommendation_GenerateConsoleAppProjectContents_VerifyPredictProgram()
        {
            (string ConsoleAppProgramCSFileContent, string ConsoleAppProjectFileContent, string modelBuilderCSFileContent) codeGenResult
                = GenerateConsoleAppProjectContentsForRecommendation();

            Approvals.Verify(codeGenResult.ConsoleAppProgramCSFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void Recommendation_GenerateConsoleAppProjectContents_VerifyPredictProject()
        {
            (string ConsoleAppProgramCSFileContent, string ConsoleAppProjectFileContent, string modelBuilderCSFileContent) codeGenResult
                = GenerateConsoleAppProjectContentsForRecommendation();

            Approvals.Verify(codeGenResult.ConsoleAppProjectFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void Recommendation_GenerateConsoleAppProjectContents_VerifyModelBuilder()
        {
            (string ConsoleAppProgramCSFileContent, string ConsoleAppProjectFileContent, string modelBuilderCSFileContent) codeGenResult
                = GenerateConsoleAppProjectContentsForRecommendation();

            Approvals.Verify(codeGenResult.modelBuilderCSFileContent);
        }

        private (string ModelInputCSFileContent, string ModelOutputCSFileContent, string ConsumeModelCSFileContent, string ModelProjectFileContent) GenerateModelProjectContentsForRecommendation()
        {
            CodeGenerator consoleCodeGen = PrepareForRecommendationTask();
            return consoleCodeGen.GenerateModelProjectContents(
                _namespaceValue,
                labelTypeCsharp: typeof(float),
                includeLightGbmPackage: false,
                includeMklComponentsPackage: false,
                includeFastTreePackage: false,
                includeImageTransformerPackage: false,
                includeImageClassificationPackage: false,
                includeRecommenderPackage: true);
        }

        private (string ConsoleAppProgramCSFileContent, string ConsoleAppProjectFileContent, string modelBuilderCSFileContent) GenerateConsoleAppProjectContentsForRecommendation()
        {
            CodeGenerator consoleCodeGen = PrepareForRecommendationTask();
            return consoleCodeGen.GenerateConsoleAppProjectContents(
                _namespaceValue,
                labelTypeCsharp: typeof(float),
                includeLightGbmPackage: false,
                includeMklComponentsPackage: false,
                includeFastTreePackage: false,
                includeImageTransformerPackage: false,
                includeImageClassificationPackage: false,
                includeRecommenderPackage: true);
        }

        private CodeGenerator PrepareForRecommendationTask()
        {
            (Pipeline pipeline,
                       ColumnInferenceResults columnInference) = GetMockedRecommendationPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, CreateCodeGeneratorSettingsFor(TaskKind.Recommendation));
            return consoleCodeGen;
        }

        private (Pipeline, ColumnInferenceResults) GetMockedRecommendationPipelineAndInference()
        {
            if (_mockedPipeline == null)
            {
                MLContext context = new MLContext();
                var hyperParam = new Dictionary<string, object>()
                {
                    {"MatrixColumnIndexColumnName","userId" },
                    {"MatrixRowIndexColumnName","movieId" },
                    {"LabelColumnName","Label" },
                    {nameof(MatrixFactorizationTrainer.Options.NumberOfIterations), 10 },
                    {nameof(MatrixFactorizationTrainer.Options.LearningRate), 0.01f },
                    {nameof(MatrixFactorizationTrainer.Options.ApproximationRank), 8 },
                    {nameof(MatrixFactorizationTrainer.Options.Lambda), 0.01f },
                    {nameof(MatrixFactorizationTrainer.Options.LossFunction), MatrixFactorizationTrainer.LossFunctionType.SquareLossRegression },
                    {nameof(MatrixFactorizationTrainer.Options.Alpha), 1f },
                    {nameof(MatrixFactorizationTrainer.Options.C), 0.00001f },
                };
                var valueToKeyPipelineNode1 = new PipelineNode(nameof(EstimatorName.ValueToKeyMapping), PipelineNodeType.Transform, "userId", "userId");
                var valueToKeyPipelineNode2 = new PipelineNode(nameof(EstimatorName.ValueToKeyMapping), PipelineNodeType.Transform, "movieId", "movieId");
                var matrixPipelineNode = new PipelineNode(nameof(TrainerName.MatrixFactorization), PipelineNodeType.Trainer, "Features", "Score", hyperParam);
                var pipeline = new Pipeline(new PipelineNode[]
                {
                    valueToKeyPipelineNode1,
                    valueToKeyPipelineNode2,
                    matrixPipelineNode
                });

                _mockedPipeline = pipeline;
                var textLoaderArgs = new TextLoader.Options()
                {
                    Columns = new[] {
                        new TextLoader.Column("Label", DataKind.String, 0),
                        new TextLoader.Column("userId", DataKind.String, 1),
                        new TextLoader.Column("movieId", DataKind.String, 2),
                    },
                    AllowQuoting = true,
                    AllowSparse = true,
                    HasHeader = true,
                    Separators = new[] { ',' }
                };

                this._columnInference = new ColumnInferenceResults()
                {
                    TextLoaderOptions = textLoaderArgs,
                    ColumnInformation = new ColumnInformation()
                    {
                        LabelColumnName = "Label",
                        UserIdColumnName = "userId",
                        ItemIdColumnName = "movieId"
                    }
                };
            }
            return (_mockedPipeline, _columnInference);
        }

        private (Pipeline, ColumnInferenceResults) GetMockedBinaryPipelineAndInference()
        {
            if (_mockedPipeline == null)
            {
                MLContext context = new MLContext();
                // same learners with different hyperparams
                var hyperparams1 = new Microsoft.ML.AutoML.ParameterSet(new List<Microsoft.ML.AutoML.IParameterValue>() { new LongParameterValue("NumLeaves", 2) });
                var trainer1 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), new ColumnInformation(), hyperparams1);
                var transforms1 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
                var inferredPipeline1 = new SuggestedPipeline(transforms1, new List<SuggestedTransform>(), trainer1, context, true);

                this._mockedPipeline = inferredPipeline1.ToPipeline();
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

                this._columnInference = new ColumnInferenceResults()
                {
                    TextLoaderOptions = textLoaderArgs,
                    ColumnInformation = new ColumnInformation() { LabelColumnName = "Label" }
                };
            }
            return (_mockedPipeline, _columnInference);
        }

        private (Pipeline, ColumnInferenceResults) GetMockedRegressionPipelineAndInference()
        {
            if (_mockedPipeline == null)
            {
                MLContext context = new MLContext();
                // same learners with different hyperparams
                var hyperparams1 = new Microsoft.ML.AutoML.ParameterSet(new List<Microsoft.ML.AutoML.IParameterValue>() { new LongParameterValue("NumLeaves", 2) });
                var trainer1 = new SuggestedTrainer(context, new LightGbmRegressionExtension(), new ColumnInformation(), hyperparams1);
                var transforms1 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
                var inferredPipeline1 = new SuggestedPipeline(transforms1, new List<SuggestedTransform>(), trainer1, context, true);

                this._mockedPipeline = inferredPipeline1.ToPipeline();
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

                this._columnInference = new ColumnInferenceResults()
                {
                    TextLoaderOptions = textLoaderArgs,
                    ColumnInformation = new ColumnInformation() { LabelColumnName = "Label" }
                };
            }
            return (_mockedPipeline, _columnInference);
        }

        private (Pipeline, ColumnInferenceResults) GetMockedAzureObjectDetectionPipelineAndInference(string onnxModelPath)
        {
            var onnxPipeLineNode = new PipelineNode(
                nameof(SpecialTransformer.ApplyOnnxModel),
                PipelineNodeType.Transform,
                new string[] { },
                new string[] { },
                null);
            var loadImageNode = new PipelineNode(EstimatorName.ImageLoading.ToString(), PipelineNodeType.Transform, "ImagePath", "input");
            var resizeImageNode = new PipelineNode(
                nameof(SpecialTransformer.ObjectDetectionResizeImage),
                PipelineNodeType.Transform,
                "input",
                "input",
                new Dictionary<string, object>()
                {
                    { "imageWidth", 800 },
                    { "imageHeight", 600 },
                });
            var extractPixelsNode = new PipelineNode(nameof(SpecialTransformer.ExtractPixel), PipelineNodeType.Transform, "input", "input");
            var bestPipeLine = new Pipeline(new PipelineNode[]
            {
                loadImageNode,
                resizeImageNode,
                extractPixelsNode,
                onnxPipeLineNode,
            });

            var textLoaderArgs = new TextLoader.Options()
            {
                Columns = new[] {
                        new TextLoader.Column("Label", DataKind.String, 0),
                        new TextLoader.Column("ImagePath", DataKind.String, 1), // 0?
                    },
                AllowQuoting = true,
                AllowSparse = true,
                HasHeader = true,
                Separators = new[] { '\t' }
            };

            var columnInference = new ColumnInferenceResults()
            {
                TextLoaderOptions = textLoaderArgs,
                ColumnInformation = new ColumnInformation() { LabelColumnName = "Label" }
            };

            return (bestPipeLine, columnInference);
        }

        private (Pipeline, ColumnInferenceResults) GetMockedRankingPipelineAndInference()
        {
            if (_mockedPipeline == null)
            {
                var hyperParam = new Dictionary<string, object>()
                {
                    {"RowGroupColumnName","GroupId" },
                    {"LabelColumnName","Label" },
                };
                var hashPipelineNode = new PipelineNode(nameof(EstimatorName.Hashing), PipelineNodeType.Transform, "GroupId", "GroupId");
                var lightGbmPipelineNode = new PipelineNode(nameof(TrainerName.LightGbmRanking), PipelineNodeType.Trainer, "Features", "Score", hyperParam);
                var pipeline = new Pipeline(new PipelineNode[]
                {
                    hashPipelineNode,
                    lightGbmPipelineNode
                });
                _mockedPipeline = pipeline;
                var textLoaderArgs = new TextLoader.Options()
                {
                    Columns = new[] {
                        new TextLoader.Column("Label", DataKind.Boolean, 0),
                        new TextLoader.Column("GroupId", DataKind.Single, 1),
                        new TextLoader.Column("col1", DataKind.Single, 0),
                        new TextLoader.Column("col2", DataKind.String, 0),
                        new TextLoader.Column("col3", DataKind.Int32, 0),
                        new TextLoader.Column("col4", DataKind.UInt32, 0),
                    },
                    AllowQuoting = true,
                    AllowSparse = true,
                    HasHeader = true,
                    Separators = new[] { ',' }
                };

                this._columnInference = new ColumnInferenceResults()
                {
                    TextLoaderOptions = textLoaderArgs,
                    ColumnInformation = new ColumnInformation() { LabelColumnName = "Label", GroupIdColumnName = "GroupId" }
                };
            }

            return (_mockedPipeline, _columnInference);
        }

        private (Pipeline, ColumnInferenceResults) GetMockedAzureImagePipelineAndInference()
        {
            // construct pipeline
            var onnxPipeLineNode = new PipelineNode(nameof(SpecialTransformer.ApplyOnnxModel), PipelineNodeType.Transform, string.Empty, string.Empty);
            var loadImageNode = new PipelineNode(EstimatorName.ImageLoading.ToString(), PipelineNodeType.Transform, "ImageSource", "ImageSource_featurized");
            var resizeImageNode = new PipelineNode(
                nameof(SpecialTransformer.ResizeImage),
                PipelineNodeType.Transform,
                "ImageSource_featurized",
                "ImageSource_featurized",
                new Dictionary<string, object>()
                {
                    { "imageWidth", 224 },
                    { "imageHeight", 224 },
                });
            var extractPixelsNode = new PipelineNode(nameof(SpecialTransformer.ExtractPixel), PipelineNodeType.Transform, "ImageSource_featurized", "input1");
            var bestPipeLine = new Pipeline(new PipelineNode[]
            {
                loadImageNode,
                resizeImageNode,
                extractPixelsNode,
                onnxPipeLineNode,
            });

            // construct column inference
            var textLoaderArgs = new TextLoader.Options()
            {
                Columns = new[] {
                        new TextLoader.Column("Label", DataKind.String, 0),
                        new TextLoader.Column("ImageSource", DataKind.String, 1), // 0?
                    },
                AllowQuoting = true,
                AllowSparse = true,
                HasHeader = true,
                Separators = new[] { '\t' }
            };

            var columnInference = new ColumnInferenceResults()
            {
                TextLoaderOptions = textLoaderArgs,
                ColumnInformation = new ColumnInformation() { LabelColumnName = "Label" }
            };

            return (bestPipeLine, columnInference);
        }

        private (Pipeline, ColumnInferenceResults, IDictionary<string, CodeGeneratorSettings.ColumnMapping>) GetMockedAzurePipelineAndInference()
        {
            // construct pipeline
            var onnxPipeLineNode = new PipelineNode(nameof(SpecialTransformer.ApplyOnnxModel), PipelineNodeType.Transform, string.Empty, string.Empty);
            var bestPipeLine = new Pipeline(new PipelineNode[]
            {
                onnxPipeLineNode,
            });

            // construct column inference
            var textLoaderArgs = new TextLoader.Options()
            {
                Columns = new[] {
                        new TextLoader.Column("Age", DataKind.Double, 0),
                        new TextLoader.Column("Workclass", DataKind.String, 1), // 0?
                        new TextLoader.Column("Fnlwgt", DataKind.Double, 2),
                        new TextLoader.Column("Education", DataKind.String, 3),
                        new TextLoader.Column("Education_num", DataKind.Double, 4),
                        new TextLoader.Column("Marital_status", DataKind.String, 5),
                        new TextLoader.Column("Occupation", DataKind.String, 6),
                        new TextLoader.Column("Relationship", DataKind.String, 7),
                        new TextLoader.Column("Race", DataKind.String, 8),
                        new TextLoader.Column("Sex", DataKind.String, 9),
                        new TextLoader.Column("Capital_gain", DataKind.Double, 10),
                        new TextLoader.Column("Capital_loss", DataKind.Double, 11),
                        new TextLoader.Column("Hours_per_week", DataKind.Double, 12),
                        new TextLoader.Column("Native_country", DataKind.String, 13),
                        new TextLoader.Column("label", DataKind.Boolean, 14),
                    },
                AllowQuoting = true,
                AllowSparse = true,
                HasHeader = true,
                Separators = new[] { ',' }
            };

            var columnInference = new ColumnInferenceResults()
            {
                TextLoaderOptions = textLoaderArgs,
                ColumnInformation = new ColumnInformation() { LabelColumnName = "Label" }
            };

            // construct columnMapping
            // mock columnMapping
            var mapping = new Dictionary<string, CodeGeneratorSettings.ColumnMapping>()
            {
                {
                    "Age",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_0",
                        ColumnType = DataKind.Int64,
                    }
                },
                {
                    "Workclass",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_1",
                        ColumnType = DataKind.String,
                    }
                },
                {
                    "Fnlwgt",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_2",
                        ColumnType = DataKind.Int64,
                    }
                },
                {
                    "Education",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_3",
                        ColumnType = DataKind.String,
                    }
                },
                {
                    "Education_num",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_4",
                        ColumnType = DataKind.Int64,
                    }
                },
                {
                    "Marital_status",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_5",
                        ColumnType = DataKind.String,
                    }
                },
                {
                    "Occupation",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_6",
                        ColumnType = DataKind.String,
                    }
                },
                {
                    "Relationship",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_7",
                        ColumnType = DataKind.String,
                    }
                },
                {
                    "Race",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_8",
                        ColumnType = DataKind.String,
                    }
                },
                {
                    "Sex",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_9",
                        ColumnType = DataKind.String,
                    }
                },
                {
                    "Capital_gain",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_10",
                        ColumnType = DataKind.Int64,
                    }
                },
                {
                    "Capital_loss",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_11",
                        ColumnType = DataKind.Int64,
                    }
                },
                {
                    "Hours_per_week",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_12",
                        ColumnType = DataKind.Int64,
                    }
                },
                {
                    "Native_country",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_13",
                        ColumnType = DataKind.String,
                    }
                },
                {
                    "label",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "label(IsOver50K)",
                        ColumnType = DataKind.Boolean,
                    }
                }
            };

            return (bestPipeLine, columnInference, mapping);
        }

        private (Pipeline, ColumnInferenceResults, IDictionary<string, CodeGeneratorSettings.ColumnMapping>) GetMockedAzurePipelineAndInferenceUnsanitizedColumnNames()
        {
            // construct pipeline
            var onnxPipeLineNode = new PipelineNode(nameof(SpecialTransformer.ApplyOnnxModel), PipelineNodeType.Transform, new[] { "input.1" }, new[] { "output.1" },
                new Dictionary<string, object>()
                {
                    { "outputColumnNames", "output1" },
                    { "inputColumnNames", "input1"},
                });
            var bestPipeLine = new Pipeline(new PipelineNode[]
            {
                onnxPipeLineNode,
            });

            // construct column inference
            var textLoaderArgs = new TextLoader.Options()
            {
                Columns = new[] {
                        new TextLoader.Column("id", DataKind.Int32, 0),
                        new TextLoader.Column("MsAssetNum", DataKind.Int32, 1),
                        new TextLoader.Column("Make", DataKind.String, 2),
                        new TextLoader.Column("Model", DataKind.String, 3),
                        new TextLoader.Column("model", DataKind.Double, 4),
                        new TextLoader.Column("work category", DataKind.String, 5),
                        new TextLoader.Column("Work category", DataKind.Int32, 6),
                        new TextLoader.Column("IsDetachable", DataKind.Boolean, 7),
                    },
                AllowQuoting = true,
                AllowSparse = true,
                HasHeader = true,
                Separators = new[] { ',' }
            };

            var columnInference = new ColumnInferenceResults()
            {
                TextLoaderOptions = textLoaderArgs,
                ColumnInformation = new ColumnInformation() { LabelColumnName = "Label" }
            };

            // construct columnMapping
            // mock columnMapping
            var mapping = new Dictionary<string, CodeGeneratorSettings.ColumnMapping>()
            {
                {
                    "id",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_0",
                        ColumnType = DataKind.Int32,
                    }
                },
                {
                    "MsAssetNum",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_1",
                        ColumnType = DataKind.Int32,
                    }
                },
                {
                    "Make",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_2",
                        ColumnType = DataKind.String,
                    }
                },
                {
                    "Model",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_3",
                        ColumnType = DataKind.String,
                    }
                },
                {
                    "model",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_4",
                        ColumnType = DataKind.Double,
                    }
                },
                {
                    "work category",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_5",
                        ColumnType = DataKind.String,
                    }
                },
                {
                    "Work Category",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_6",
                        ColumnType = DataKind.Int32,
                    }
                },
                {
                    "IsDetachable",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "input_7",
                        ColumnType = DataKind.Boolean,
                    }
                }
            };

            return (bestPipeLine, columnInference, mapping);
        }


        private (Pipeline, ColumnInferenceResults) GetMockedOvaPipelineAndInference()
        {
            if (_mockedOvaPipeline == null)
            {
                MLContext context = new MLContext();
                // same learners with different hyperparameters
                var hyperparams1 = new Microsoft.ML.AutoML.ParameterSet(new List<Microsoft.ML.AutoML.IParameterValue>() { new LongParameterValue("NumLeaves", 2) });
                var trainer1 = new SuggestedTrainer(context, new FastForestOvaExtension(), new ColumnInformation(), hyperparams1);
                var transforms1 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
                var inferredPipeline1 = new SuggestedPipeline(transforms1, new List<SuggestedTransform>(), trainer1, context, true);

                this._mockedOvaPipeline = inferredPipeline1.ToPipeline();
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


                this._columnInference = new ColumnInferenceResults()
                {
                    TextLoaderOptions = textLoaderArgs,
                    ColumnInformation = new ColumnInformation() { LabelColumnName = "Label" }
                };

            }
            return (_mockedOvaPipeline, _columnInference);
        }

        private static CodeGeneratorSettings CreateCodeGeneratorSettingsFor(TaskKind task)
        {
            return new CodeGeneratorSettings()
            {
                MlTask = task,
                OutputBaseDir = null,
                OutputName = "MyNamespace",
                TrainDataset = "x:\\dummypath\\dummy_train.csv",
                TestDataset = "x:\\dummypath\\dummy_test.csv",
                LabelName = "Label",
                ModelName = "x:\\models\\model.zip",
                StablePackageVersion = StablePackageVersion,
                UnstablePackageVersion = UnstablePackageVersion,
                OnnxRuntimePackageVersion = "1.2.3",
            };
        }
    }
}
