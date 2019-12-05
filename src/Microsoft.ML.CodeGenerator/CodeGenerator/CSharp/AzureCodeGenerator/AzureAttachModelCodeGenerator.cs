using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CSharp;
using Microsoft.ML.CodeGenerator.Templates.AzureImageClassification.Model;
using Microsoft.ML.CodeGenerator.Templates.Console;
using Microsoft.ML.CodeGenerator.Utilities;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.CSharp.AzureCodeGenerator
{
    internal class AzureAttachModelCodeGenerator : IProjectGenerator
    {
        private readonly Pipeline _pipeline;
        private readonly CodeGeneratorSettings _settings;
        private readonly ColumnInferenceResults _columnInferenceResult;
        private readonly string _nameSpaceValue;

        public IProjectFileGenerator ModelInputClass { get; private set; }
        public IProjectFileGenerator ModelOutputClass { get; private set; }
        public IProjectFileGenerator NormalizeMapping { get; private set; }
        public IProjectFileGenerator ModelProject { get; private set; }
        public IProjectFileGenerator ConsumeModel { get; private set; }
        public IProjectFileGenerator LabelMapping { get; private set; }
        public IProjectFileGenerator ImageLabelMapping { get; private set; }
        public string Name { get; set; }

        public AzureAttachModelCodeGenerator(Pipeline pipeline, ColumnInferenceResults columnInferenceResults, CodeGeneratorSettings options, string namespaceValue)
        {
            _pipeline = pipeline;
            _settings = options;
            _columnInferenceResult = columnInferenceResults;
            _nameSpaceValue = namespaceValue;
            Name = $"{_settings.OutputName}.Model";

            ModelInputClass = new ModelInputClass()
            {
                Namespace = _nameSpaceValue,
                ClassLabels = Utilities.Utils.GenerateClassLabels(_columnInferenceResult),
                Target = _settings.Target
            };

            ModelOutputClass = new OnnxModelOutputClass()
            {
                Namespace = _nameSpaceValue,
                Target = _settings.Target
            };

            NormalizeMapping = new NormalizeMapping()
            {
                Target = _settings.Target,
                Namespace = _nameSpaceValue,
            };

            ModelProject = new ModelProject()
            {
                IncludeFastTreePackage = false,
                IncludeImageClassificationPackage = true,
                IncludeImageTransformerPackage = true,
                IncludeLightGBMPackage = false,
                IncludeMklComponentsPackage = false,
                IncludeOnnxModel = true,
                IncludeRecommenderPackage = false,
                StablePackageVersion = _settings.StablePackageVersion,
                UnstablePackageVersion = _settings.UnstablePackageVersion,
                OutputName = _settings.OutputName,
            };

            var labelType = _columnInferenceResult.TextLoaderOptions.Columns.Where(t => t.Name == _settings.LabelName).First().DataKind;
            Type labelTypeCsharp = Utils.GetCSharpType(labelType);

            LabelMapping = new LabelMapping()
            {
                Target = _settings.Target,
                Namespace = _nameSpaceValue,
                LabelMappingInputLabelType = typeof(Int64).Name,
                PredictionLabelType = labelTypeCsharp.Name,
                TaskType = _settings.MlTask.ToString(),
            };

            ConsumeModel = new AzureAttachImageConsumeModel()
            {
                Namespace = _nameSpaceValue,
                Target = _settings.Target
            };
        }

        public IProject ToProject()
        {
            Project project = default;

            if (_settings.IsImage)
            {
                project = new Project()
                {
                    ModelInputClass.ToProjectFile(),
                    ModelOutputClass.ToProjectFile(),
                    ConsumeModel.ToProjectFile(),
                    ModelProject.ToProjectFile(),
                    NormalizeMapping.ToProjectFile(),
                };
            }
            else
            {
                project = new Project()
                {
                    ModelInputClass.ToProjectFile(),
                    ModelOutputClass.ToProjectFile(),
                    ConsumeModel.ToProjectFile(),
                    ModelProject.ToProjectFile(),
                    LabelMapping.ToProjectFile(),
                };
            }
            project.Name = Name;
            return project;
        }

        public void GenerateOutput()
        {
            throw new NotImplementedException();
        }
    }
}
