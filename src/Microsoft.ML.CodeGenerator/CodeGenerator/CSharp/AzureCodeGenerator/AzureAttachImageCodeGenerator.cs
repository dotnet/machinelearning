using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Xml.Serialization;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CodeGenerator.CSharp.AzureCodeGenerator;
using Microsoft.ML.CodeGenerator.CSharp;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.CSharp
{
    internal class AzureAttachImageCodeGenerator : IProjectGenerator
    {
        public IProjectGenerator AzureAttachImageConsoleApp { get; private set; }
        public IProjectGenerator AzureAttachImageModel { get; private set; }

        private readonly Pipeline _pipeline;
        private readonly CodeGeneratorSettings _settings;
        private readonly ColumnInferenceResults _columnInferenceResult;
        public AzureAttachImageCodeGenerator(Pipeline pipeline, ColumnInferenceResults columnInferenceResults, CodeGeneratorSettings options)
        {
            _pipeline = pipeline;
            _settings = options;
            _columnInferenceResult = columnInferenceResults;
        }
        public void GenerateOutput()
        {
            if (AzureAttachImageConsoleApp == null)
            {
                throw new Exception($"{nameof(AzureAttachImageCodeGenerator)}: Call ToProjectFiles First");
            }

            AzureAttachImageConsoleApp.GenerateOutput();
            AzureAttachImageModel.GenerateOutput();
        }

        public IProjectGenerator ToProjectFiles()
        {
            var namespaceValue = Utilities.Utils.Normalize(_settings.OutputName);

            AzureAttachImageConsoleApp = new AzureAttachImageConsoleAppCodeGenerator(_pipeline, _columnInferenceResult, _settings, namespaceValue).ToProjectFiles();
            AzureAttachImageModel = new AzureAttachImageModelCodeGenerator(_pipeline, _columnInferenceResult, _settings, namespaceValue).ToProjectFiles();
            return this;
        }

        public void WriteToDisk(string folder)
        {
            if (AzureAttachImageConsoleApp == null)
            {
                throw new Exception($"{nameof(AzureAttachImageCodeGenerator)}: Call ToProjectFiles First");
            }
            AzureAttachImageConsoleApp.WriteToDisk(Path.Combine(folder, $"{_settings.OutputName}.ConsoleApp"));
            AzureAttachImageModel.WriteToDisk(Path.Combine(folder, $"{_settings.OutputName}.Model"));
        }
    }
}
