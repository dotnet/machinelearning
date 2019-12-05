using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CSharp;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.CSharp.AzureCodeGenerator
{
    internal class AzureAttachCodeGenenrator : ISolutionGenerator
    {
        public IProjectGenerator AzureAttachConsoleApp { get; private set; }
        public IProjectGenerator AzureAttachModel { get; private set; }
        public string Name { get; set; }

        private readonly Pipeline _pipeline;
        private readonly CodeGeneratorSettings _settings;
        private readonly ColumnInferenceResults _columnInferenceResult;

        public AzureAttachCodeGenenrator(Pipeline pipeline, ColumnInferenceResults columnInferenceResults, CodeGeneratorSettings options)
        {
            _pipeline = pipeline;
            _settings = options;
            _columnInferenceResult = columnInferenceResults;
            Name = _settings.OutputName;
            var namespaceValue = Utilities.Utils.Normalize(_settings.OutputName);
            AzureAttachConsoleApp = new AzureAttachConsoleAppCodeGenerator(_pipeline, _columnInferenceResult, _settings, namespaceValue);
            AzureAttachModel = new AzureAttachModelCodeGenerator(_pipeline, _columnInferenceResult, _settings, namespaceValue);
        }

        public ISolution ToSolution()
        {
            var solution = new Solution()
            {
                AzureAttachConsoleApp.ToProject(),
                AzureAttachModel.ToProject()
            };

            solution.Name = _settings.OutputName;
            return solution;
        }
    }
}
