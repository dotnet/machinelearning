using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CSharp;
using Microsoft.ML.CodeGenerator.Templates.AzureImageClassification.Model;
using Microsoft.ML.CodeGenerator.Templates.Console;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.CSharp.AzureCodeGenerator
{
    internal class AzureAttachImageModelCodeGenerator : IProjectGenerator
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
        public string Name { get; set; }

        public AzureAttachImageModelCodeGenerator(Pipeline pipeline, ColumnInferenceResults columnInferenceResults, CodeGeneratorSettings options, string namespaceValue)
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

            ModelOutputClass = new AzureAttachImageModelOutputClass()
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

            ConsumeModel = new AzureAttachImageConsumeModel()
            {
                Namespace = _nameSpaceValue,
                Target = _settings.Target
            };
        }

        public IProject ToProject()
        {
            var project = new Project()
            {
                ModelInputClass.ToProjectFile(),
                ModelOutputClass.ToProjectFile(),
                ConsumeModel.ToProjectFile(),
                ModelProject.ToProjectFile(),
                NormalizeMapping.ToProjectFile(),
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
