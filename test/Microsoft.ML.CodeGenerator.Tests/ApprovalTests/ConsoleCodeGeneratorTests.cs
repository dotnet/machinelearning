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
using Microsoft.ML.CodeGenerator.CodeGenerator;
using Microsoft.ML.CodeGenerator.CodeGenerator.CSharp;
using Microsoft.ML.CodeGenerator.CodeGenerator.CSharp.AzureCodeGenerator;
using Microsoft.ML.CodeGenerator.CSharp;
using Microsoft.ML.CodeGenerator.Templates.Console;
using Microsoft.ML.CodeGenerator.Utilities;
using Microsoft.ML.Data;
using Xunit;
using CodeGenerator = Microsoft.ML.CodeGenerator.CSharp.CodeGenerator;

namespace mlnet.Tests
{
    [UseReporter(typeof(DiffReporter))]
    public class ConsoleCodeGeneratorTests
    {
        private Pipeline mockedPipeline;
        private Pipeline mockedOvaPipeline;
        private ColumnInferenceResults columnInference = default;
        private string namespaceValue = "TestNamespace";
        private const string StablePackageVersion = "1.4.0-preview3-28229-2";
        private const string UnstablePackageVersion = "0.16.0-preview3-28229-2";


        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ConsoleAppModelBuilderCSFileContentOvaTest()
        {
            (Pipeline pipeline,
                        ColumnInferenceResults columnInference) = GetMockedOvaPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, CreateCodeGeneratorSettingsFor(TaskKind.MulticlassClassification));
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(namespaceValue, typeof(float), true, true,
                false, false, false, false);

            Approvals.Verify(result.modelBuilderCSFileContent);
        }

        [Fact]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void IgniteDemoTest()
        {
            (var bestPipeLine, var columnInference) = GetMockedAzureImagePipelineAndInference();

            // construct CodeGen option
            var setting = new CodeGeneratorSettings()
            {
                TrainDataset = @"C:\Users\xiaoyuz\Desktop\flower_photos_tiny_set_for_unit_tests\data.tsv",
                ModelPath = @"C:\Users\xiaoyuz\Desktop\flower_photos_tiny_set_for_unit_tests\CodeGenTest\MLModel.zip",
                MlTask = TaskKind.MulticlassClassification,
                OutputName = @"CodeGenTest",
                OutputBaseDir = @"C:\Users\xiaoyuz\Desktop\CodeGenTest",
                LabelName = "Label",
                Target = GenerateTarget.ModelBuilder,
                StablePackageVersion = "1.3.1",
                UnstablePackageVersion = "0.16.0-preview3-28231-2",
                IsAzureAttach = true,
            };

            // generate project
            var codeGen = new AzureAttachImageCodeGenerator(bestPipeLine, columnInference, setting);
            codeGen.ToSolution().WriteToDisk(setting.OutputBaseDir);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ConsoleAppModelBuilderCSFileContentBinaryTest()
        {
            (Pipeline pipeline,
                        ColumnInferenceResults columnInference) = GetMockedBinaryPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, CreateCodeGeneratorSettingsFor(TaskKind.BinaryClassification));
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(namespaceValue, typeof(float), true, true,
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
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(namespaceValue, typeof(float), true, true,
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
            var result = consoleCodeGen.GenerateModelProjectContents(namespaceValue, typeof(float), true, true, true,
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
            var result = consoleCodeGen.GenerateModelProjectContents(namespaceValue, typeof(float), true, true, false,
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
            var result = consoleCodeGen.GenerateModelProjectContents(namespaceValue, typeof(float), true, true, false,
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
            var result = consoleCodeGen.GenerateModelProjectContents(namespaceValue, typeof(float), true, true, false,
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
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(namespaceValue, typeof(float), true, true,
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
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(namespaceValue, typeof(float), true, true,
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
            var result = consoleCodeGen.GenerateConsoleAppProjectContents(namespaceValue, typeof(float), true, true,
                false, false, false, false);

            Approvals.Verify(result.ConsoleAppProjectFileContent);
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void AzureImageCodeGeneratorTest()
        {
            // That's the hammer I want
            (var pipeline, var columnInference) = GetMockedAzureImagePipelineAndInference();
            var setting = new CodeGeneratorSettings()
            {
                TrainDataset = @"/path/to/dataset",
                ModelPath = @"/path/to/model",
                MlTask = TaskKind.MulticlassClassification,
                OutputName = @"CodeGenTest",
                OutputBaseDir = @"/path/to/codegen",
                LabelName = "Label",
                Target = GenerateTarget.ModelBuilder,
                StablePackageVersion = "stableversion",
                UnstablePackageVersion = "unstableversion",
                IsAzureAttach = true,
                IsImage = true,
            };
            var codeGen = new AzureAttachCodeGenenrator(pipeline, columnInference, setting);
            foreach (var project in codeGen.ToSolution())
            {
                foreach(var projectFile in project)
                {
                    NamerFactory.AdditionalInformation = projectFile.Name;
                    Approvals.Verify(((ProjectFile)projectFile).Data);
                }
            }
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ModelInputClassTest()
        {
            // mock ColumnInferenceResults
            var textLoaderArgs = new TextLoader.Options()
            {
                Columns = new[] {
                        new TextLoader.Column("Label", DataKind.Boolean, 0),
                        new TextLoader.Column("col1", DataKind.Single, 1),
                        new TextLoader.Column("col2", DataKind.Single, 2),
                        new TextLoader.Column("col3", DataKind.String, 3),
                        new TextLoader.Column("col4", DataKind.Int32, 4),
                        new TextLoader.Column("col5", DataKind.UInt32, 5),
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

            // mock columnMapping
            var mapping = new Dictionary<string, CodeGeneratorSettings.ColumnMapping>()
            {
                {
                    "col1",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "col1_map",
                        ColumnType = DataKind.Int64,
                    }
                },
                {
                    "col2",
                    new CodeGeneratorSettings.ColumnMapping()
                    {
                        ColumnName = "col2_map",
                        ColumnType = DataKind.UInt32,
                    }
                }
            };

            // test with null map case
            var columnMappingStringList = Utils.GenerateClassLabels(columnInference);
            var modelInputProject = new ModelInputClass()
            {
                Namespace = "test",
                ClassLabels = columnMappingStringList,
                Target = GenerateTarget.Cli,
            }.ToProjectFile() as ProjectFile;
            NamerFactory.AdditionalInformation = "null_map";
            Approvals.Verify(modelInputProject.Data);

            // test with map case
            columnMappingStringList = Utils.GenerateClassLabels(columnInference, mapping);
            modelInputProject = new ModelInputClass()
            {
                Namespace = "test",
                ClassLabels = columnMappingStringList,
                Target = GenerateTarget.Cli,
            }.ToProjectFile() as ProjectFile;
            NamerFactory.AdditionalInformation = "map";
            Approvals.Verify(modelInputProject.Data);
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
                namespaceValue,
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
                namespaceValue,
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
            if (mockedPipeline == null)
            {
                MLContext context = new MLContext();

                var trainer1 = new SuggestedTrainer(context, new MatrixFactorizationExtension(), new ColumnInformation() {
                    LabelColumnName = "Label",
                    UserIdColumnName = "userId",
                    ItemIdColumnName = "movieId",
                }, hyperParamSet: null);
                var transforms1 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
                var inferredPipeline1 = new SuggestedPipeline(transforms1, new List<SuggestedTransform>(), trainer1, context, false);

                mockedPipeline = inferredPipeline1.ToPipeline();
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

                this.columnInference = new ColumnInferenceResults()
                {
                    TextLoaderOptions = textLoaderArgs,
                    ColumnInformation = new ColumnInformation() {
                        LabelColumnName = "Label",
                        UserIdColumnName = "userId",
                        ItemIdColumnName = "movieId"
                    }
                };
            }
            return (mockedPipeline, columnInference);
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

        private (Pipeline, ColumnInferenceResults) GetMockedAzureImagePipelineAndInference()
        {
            // construct pipeline
            var onnxPipeLineNode = new PipelineNode(nameof(SpecialTransformer.ApplyOnnxModel), PipelineNodeType.Transform, new[] { "input.1" }, new[] { "output.1" },
                new Dictionary<string, object>()
                {
                    { "outputColumnNames", "output1" },
                    { "inputColumnNames", "input1"},
                    { "modelFile" , "awesomeModel.onnx"},
                });
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
            var extractPixelsNode = new PipelineNode(nameof(SpecialTransformer.ExtractPixel), PipelineNodeType.Transform, "ImageSource_featurized", "ImageSource_featurized");
            var customePipeline = new PipelineNode(nameof(SpecialTransformer.NormalizeMapping), PipelineNodeType.Transform, string.Empty, string.Empty);
            var bestPipeLine = new Pipeline(new PipelineNode[]
            {
                loadImageNode,
                resizeImageNode,
                extractPixelsNode,
                customePipeline,
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
                ModelPath = "x:\\models\\model.zip",
                StablePackageVersion = StablePackageVersion,
                UnstablePackageVersion = UnstablePackageVersion
            };
        }
    }
}
