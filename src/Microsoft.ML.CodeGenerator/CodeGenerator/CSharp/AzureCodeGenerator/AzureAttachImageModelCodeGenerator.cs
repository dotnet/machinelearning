using System;
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

        public IProjectFile ModelInputClass { get; private set; }
        public IProjectFile ModelOutputClass { get; private set; }
        public IProjectFile NormalizeMapping { get; private set; }
        public IProjectFile ModelProject { get; private set; }
        public IProjectFile ConsumeModel { get; private set; }

        public AzureAttachImageModelCodeGenerator(Pipeline pipeline, ColumnInferenceResults columnInferenceResults, CodeGeneratorSettings options, string namespaceValue)
        {
            _pipeline = pipeline;
            _settings = options;
            _columnInferenceResult = columnInferenceResults;
            _nameSpaceValue = namespaceValue;
        }

        public void GenerateOutput()
        {
            var modelProjectDir = Path.Combine(_settings.OutputBaseDir, $"{_settings.OutputName}.Model");
            ToProjectFiles().WriteToDisk(modelProjectDir);
        }

        public IProjectGenerator ToProjectFiles()
        {
            ModelInputClass = new ModelInputClass()
            {
                Namespace = _nameSpaceValue,
                ClassLabels = Utilities.Utils.GenerateClassLabels(_columnInferenceResult),
                Target = _settings.Target
            }.ToProjectFile();

            ModelOutputClass = new AzureAttachImageModelOutputClass()
            {
                Namespace = _nameSpaceValue,
                Target = _settings.Target
            }.ToProjectFile();

            NormalizeMapping = new NormalizeMapping()
            {
                Target = _settings.Target,
                Namespace = _nameSpaceValue,
            }.ToProjectFile();

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
            }.ToProjectFile();

            ConsumeModel = new AzureAttachImageConsumeModel()
            {
                Namespace = _nameSpaceValue,
                Target = _settings.Target
            }.ToProjectFile();
            return this;
        }

        public void WriteToDisk(string folder)
        {
            if (ModelInputClass == null)
            {
                throw new Exception($"{nameof(AzureAttachImageModelCodeGenerator)}: Call ToProjectFiles First");
            }
            ModelInputClass.WriteToDisk(Path.Combine(folder, "ModelInput.cs"));
            ModelOutputClass.WriteToDisk(Path.Combine(folder, "ModelOutput.cs"));
            ConsumeModel.WriteToDisk(Path.Combine(folder, "ConsumeModel.cs"));
            ModelProject.WriteToDisk(Path.Combine(folder, $"{_settings.OutputName}.Model.csproj"));
            NormalizeMapping.WriteToDisk(Path.Combine(folder, "NormalizeMapping.cs"));
        }
    }
}
