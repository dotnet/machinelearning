// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CodeGenerator.CSharp.Interface;
using Microsoft.ML.CodeGenerator.CSharp;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.CSharp.AzureCodeGenerator
{
    internal class AzureAttachCodeGenenrator : ICSharpSolutionGenerator
    {
        public ICSharpProjectGenerator AzureAttachConsoleApp { get; private set; }
        public ICSharpProjectGenerator AzureAttachModel { get; private set; }
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

        public ICSharpSolution ToSolution()
        {
            var solution = new CSharpSolution()
            {
                AzureAttachConsoleApp.ToProject(),
                AzureAttachModel.ToProject()
            };

            solution.Name = _settings.OutputName;
            return solution;
        }

        public void GenerateOutput()
        {
            var folder = Path.Combine(_settings.OutputBaseDir, _settings.OutputName);
            ToSolution().WriteToDisk(folder);
        }
    }
}
