using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CSharp;
using Microsoft.ML.CodeGenerator.Templates.Console;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.CSharp
{
    internal class AzureAttachConsoleAppCodeGenerator : IProjectGenerator
    {
        private readonly Pipeline _pipeline;
        private readonly CodeGeneratorSettings _settings;
        private readonly ColumnInferenceResults _columnInferenceResult;
        private readonly string _nameSpaceValue;

        public IProjectFileGenerator ModelBuilder { get; private set; }
        public IProjectFileGenerator PredictProject { get; private set; }
        public IProjectFileGenerator PredictProgram { get; private set; }
        public string Name { get; set; }

        public AzureAttachConsoleAppCodeGenerator(Pipeline pipeline, ColumnInferenceResults columnInferenceResults, CodeGeneratorSettings options, string namespaceValue)
        {
            _pipeline = pipeline;
            _settings = options;
            _columnInferenceResult = columnInferenceResults;
            _nameSpaceValue = namespaceValue;
            Name = $"{_settings.OutputName}.ConsoleApp";

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
            };

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
                UnstablePackageVersion = _settings.UnstablePackageVersion,
                OutputName = _settings.OutputName,
            };

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
            };
        }

        public IProject ToProject()
        {
            var project = new Project()
            {
                ModelBuilder.ToProjectFile(),
                PredictProject.ToProjectFile(),
                PredictProgram.ToProjectFile(),
            };

            project.Name = Name;
            return project;
        }

        public void GenerateOutput()
        {
            throw new NotImplementedException();
        }
    }
}
