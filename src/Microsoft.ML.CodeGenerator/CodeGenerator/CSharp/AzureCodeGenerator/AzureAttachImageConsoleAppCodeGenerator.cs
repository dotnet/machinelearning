using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CSharp;
using Microsoft.ML.CodeGenerator.Templates.Console;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.CSharp
{
    internal class AzureAttachImageConsoleAppCodeGenerator : IProjectGenerator
    {
        private readonly Pipeline _pipeline;
        private readonly CodeGeneratorSettings _settings;
        private readonly ColumnInferenceResults _columnInferenceResult;
        private readonly string _nameSpaceValue;

        public IProjectFile ModelBuilder { get; private set; }
        public IProjectFile PredictProject { get; private set; }
        public IProjectFile PredictProgram { get; private set; }

        public AzureAttachImageConsoleAppCodeGenerator(Pipeline pipeline, ColumnInferenceResults columnInferenceResults, CodeGeneratorSettings options, string namespaceValue)
        {
            _pipeline = pipeline;
            _settings = options;
            _columnInferenceResult = columnInferenceResults;
            _nameSpaceValue = namespaceValue;
        }

        public void GenerateOutput()
        {
            var consoleAppProjectDir = Path.Combine(_settings.OutputBaseDir, $"{_settings.OutputName}.ConsoleApp");
            ToProjectFiles().WriteToDisk(consoleAppProjectDir);
        }

        public IProjectGenerator ToProjectFiles()
        {
            var (Usings, TrainerMethod, PreTrainerTransforms, PostTrainerTransforms) = _pipeline.GenerateTransformsAndTrainers();

            ModelBuilder = new ModelBuilder()
            {
                Path = _settings.TrainDataset,
                TestPath = _settings.TestDataset,
                HasHeader = _columnInferenceResult.TextLoaderOptions.HasHeader,
                Separator = _columnInferenceResult.TextLoaderOptions.Separators.FirstOrDefault(),
                PreTrainerTransforms = PreTrainerTransforms,
                Trainer = TrainerMethod,
                TaskType = _settings.MlTask.ToString(),
                GeneratedUsings = Usings,
                AllowQuoting = _columnInferenceResult.TextLoaderOptions.AllowQuoting,
                AllowSparse = _columnInferenceResult.TextLoaderOptions.AllowSparse,
                Namespace = _nameSpaceValue,
                LabelName = _settings.LabelName,
                CacheBeforeTrainer = _pipeline.CacheBeforeTrainer,
                Target = _settings.Target,
                HasOnnxModel = true,
            }.ToProjectFile();

            PredictProject = new PredictProject()
            {
                Namespace = _nameSpaceValue,
                IncludeMklComponentsPackage = false,
                IncludeLightGBMPackage = false,
                IncludeFastTreePackage = false,
                IncludeImageTransformerPackage = true,
                IncludeImageClassificationPackage = true,
                IncludeOnnxPackage = true,
                IncludeResNet18Package = true,
                IncludeRecommenderPackage = false,
                StablePackageVersion = _settings.StablePackageVersion,
                UnstablePackageVersion = _settings.UnstablePackageVersion
            }.ToProjectFile();

            var columns = _columnInferenceResult.TextLoaderOptions.Columns;
            var featuresList = columns.Where((str) => str.Name != _settings.LabelName).Select((str) => str.Name).ToList();
            PredictProgram = new PredictProgram()
            {
                TaskType = _settings.MlTask.ToString(),
                LabelName = _settings.LabelName,
                Namespace = _nameSpaceValue,
                TestDataPath = _settings.TestDataset,
                TrainDataPath = _settings.TrainDataset,
                AllowQuoting = _columnInferenceResult.TextLoaderOptions.AllowQuoting,
                AllowSparse = _columnInferenceResult.TextLoaderOptions.AllowSparse,
                HasHeader = _columnInferenceResult.TextLoaderOptions.HasHeader,
                Separator = _columnInferenceResult.TextLoaderOptions.Separators.FirstOrDefault(),
                Target = _settings.Target,
                IsAzureAttach = true,
                Features = featuresList,
            }.ToProjectFile();
            return this;
        }

        public void WriteToDisk(string folder)
        {
            if(ModelBuilder == null)
            {
                throw new Exception($"{nameof(AzureAttachImageConsoleAppCodeGenerator)}: Call ToProjectFiles First");
            }
            ModelBuilder.WriteToDisk(Path.Combine(folder, "ModelBuilder.cs"));
            PredictProject.WriteToDisk(Path.Combine(folder, $"{_settings.OutputName}.ConsoleApp.csproj"));
            PredictProgram.WriteToDisk(Path.Combine(folder, "Program.cs"));
        }
    }
}
