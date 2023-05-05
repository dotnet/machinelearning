// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CodeGenerator.CSharp.Interface;
using Microsoft.ML.CodeGenerator.CSharp;
using Microsoft.ML.CodeGenerator.Templates.Azure.Model;
using Microsoft.ML.CodeGenerator.Templates.Console;
using Microsoft.ML.CodeGenerator.Utilities;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.CSharp.AzureCodeGenerator
{
    internal class AzureAttachModelCodeGenerator : ICSharpProjectGenerator
    {
        private readonly Pipeline _pipeline;
        private readonly CodeGeneratorSettings _settings;
        private readonly ColumnInferenceResults _columnInferenceResult;
        private readonly string _nameSpaceValue;

        public ICSharpFile ModelInputClass { get; private set; }
        public ICSharpFile AzureImageModelOutputClass { get; private set; }
        public ICSharpFile AzureObjectDetectionModelOutputClass { get; private set; }
        public ICSharpFile ModelProject { get; private set; }
        public ICSharpFile ConsumeModel { get; private set; }
        public ICSharpFile LabelMapping { get; private set; }
        public ICSharpFile ImageLabelMapping { get; private set; }
        public ICSharpFile ObjectDetectionConsumeModel { get; private set; }
        public string Name { get; set; }

        public AzureAttachModelCodeGenerator(Pipeline pipeline, ColumnInferenceResults columnInferenceResults, CodeGeneratorSettings options, string namespaceValue)
        {
            _pipeline = pipeline;
            _settings = options;
            _columnInferenceResult = columnInferenceResults;
            _nameSpaceValue = namespaceValue;
            Name = $"{_settings.OutputName}.Model";

            ModelInputClass = new CSharpCodeFile()
            {
                File = new ModelInputClass()
                {
                    Namespace = _nameSpaceValue,
                    ClassLabels = Utilities.Utils.GenerateClassLabels(_columnInferenceResult, _settings.OnnxInputMapping),
                    Target = _settings.Target
                }.TransformText(),
                Name = "ModelInput.cs",
            };

            var labelType = _columnInferenceResult.TextLoaderOptions.Columns.Where(t => t.Name == _settings.LabelName).First().DataKind;
            Type labelTypeCsharp = Utils.GetCSharpType(labelType);

            AzureImageModelOutputClass = new CSharpCodeFile()
            {
                File = new AzureImageModelOutputClass()
                {
                    Namespace = _nameSpaceValue,
                    Target = _settings.Target,
                    Labels = _settings.ClassificationLabel,
                }.TransformText(),
                Name = "ModelOutput.cs",
            };

            AzureObjectDetectionModelOutputClass = new CSharpCodeFile()
            {
                File = new AzureObjectDetectionModelOutputClass()
                {
                    Namespace = _nameSpaceValue,
                    Target = _settings.Target,
                    Labels = _settings.ObjectLabel,
                }.TransformText(),
                Name = "ModelOutput.cs",
            };

            ModelProject = new CSharpProjectFile()
            {
                File = new ModelProject()
                {
                    IncludeFastTreePackage = false,
                    IncludeImageClassificationPackage = false,
                    IncludeImageTransformerPackage = _settings.IsImage,
                    IncludeLightGBMPackage = false,
                    IncludeMklComponentsPackage = false,
                    IncludeOnnxModel = true,
                    IncludeOnnxRuntime = _settings.IsObjectDetection,
                    IncludeRecommenderPackage = false,
                    StablePackageVersion = _settings.StablePackageVersion,
                    UnstablePackageVersion = _settings.UnstablePackageVersion,
                    OnnxRuntimePackageVersion = _settings.OnnxRuntimePackageVersion,
                    Target = _settings.Target,
                }.TransformText(),
                Name = $"{_settings.OutputName}.Model.csproj",
            };

            ConsumeModel = new CSharpCodeFile()
            {
                File = new ConsumeModel()
                {
                    Namespace = _nameSpaceValue,
                    Target = _settings.Target,
                    MLNetModelName = _settings.ModelName,
                    OnnxModelName = _settings.OnnxModelName,
                    IsAzureImage = _settings.IsAzureAttach && _settings.IsImage,
                    IsAzureObjectDetection = _settings.IsObjectDetection && _settings.IsAzureAttach,
                }.TransformText(),
                Name = "ConsumeModel.cs",
            };
        }

        public ICSharpProject ToProject()
        {
            CSharpProject project;
            if (_settings.IsImage)
            {
                project = new CSharpProject()
                {
                    ModelInputClass,
                    AzureImageModelOutputClass,
                    ConsumeModel,
                    ModelProject,
                };
            }
            else if (_settings.IsObjectDetection)
            {
                project = new CSharpProject()
                {
                    ModelInputClass,
                    AzureObjectDetectionModelOutputClass,
                    ConsumeModel,
                    ModelProject,
                };
            }
            else
            {
                project = new CSharpProject()
                {
                    ModelInputClass,
                    AzureImageModelOutputClass,
                    ConsumeModel,
                    ModelProject,
                };
            }
            project.Name = Name;
            return project;
        }
    }
}
