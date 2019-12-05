using System;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CodeGenerator.CSharp.AzureCodeGenerator;
using Microsoft.ML.CodeGenerator.CSharp;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.CSharp
{
    internal class AzureAttachImageCodeGenerator : ISolutionGenerator
    {
        public IProjectGenerator AzureAttachImageConsoleApp { get; private set; }
        public IProjectGenerator AzureAttachImageModel { get; private set; }
        public string Name { get; set; }

        private readonly Pipeline _pipeline;
        private readonly CodeGeneratorSettings _settings;
        private readonly ColumnInferenceResults _columnInferenceResult;

        public AzureAttachImageCodeGenerator(Pipeline pipeline, ColumnInferenceResults columnInferenceResults, CodeGeneratorSettings options)
        {
            _pipeline = pipeline;
            _settings = options;
            _columnInferenceResult = columnInferenceResults;
            Name = _settings.OutputName;
            var namespaceValue = Utilities.Utils.Normalize(_settings.OutputName);
            AzureAttachImageConsoleApp = new AzureAttachConsoleAppCodeGenerator(_pipeline, _columnInferenceResult, _settings, namespaceValue);
            AzureAttachImageModel = new AzureAttachImageModelCodeGenerator(_pipeline, _columnInferenceResult, _settings, namespaceValue);
        }

        public ISolution ToSolution()
        {
            var solution = new Solution()
            {
                AzureAttachImageConsoleApp.ToProject(),
                AzureAttachImageModel.ToProject()
            };

            solution.Name = _settings.OutputName;
            return solution;
        }
    }
}
