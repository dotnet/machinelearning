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
        public ICSharpFile ModelOutputClass { get; private set; }
        public ICSharpFile NormalizeMapping { get; private set; }
        public ICSharpFile ModelProject { get; private set; }
        public ICSharpFile ConsumeModel { get; private set; }
        public ICSharpFile LabelMapping { get; private set; }
        public ICSharpFile ImageLabelMapping { get; private set; }
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

            ModelOutputClass = new CSharpCodeFile()
            {
                File = new ModelOutputClass()
                {
                    Namespace = _nameSpaceValue,
                    Target = _settings.Target,
                    TaskType = _settings.MlTask.ToString(),
                    PredictionLabelType = labelTypeCsharp.Name,
                }.TransformText(),
                Name = "ModelOutput.cs",
            };

            NormalizeMapping = new CSharpCodeFile()
            {
                File = new NormalizeMapping()
                {
                    Target = _settings.Target,
                    Namespace = _nameSpaceValue,
                }.TransformText(),
                Name = "NormalizeMapping.cs",
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
                    IncludeRecommenderPackage = false,
                    StablePackageVersion = _settings.StablePackageVersion,
                    UnstablePackageVersion = _settings.UnstablePackageVersion,
                }.TransformText(),
                Name = $"{ _settings.OutputName }.Model.csproj",
            };

            LabelMapping = new CSharpCodeFile()
            {
                File = new LabelMapping()
                {
                    Target = _settings.Target,
                    Namespace = _nameSpaceValue,
                    LabelMappingInputLabelType = typeof(Int64).Name,
                    PredictionLabelType = labelTypeCsharp.Name,
                    TaskType = _settings.MlTask.ToString(),
                }.TransformText(),
                Name = "LabelMapping.cs",
            };

            ImageLabelMapping = new CSharpCodeFile()
            {
                File = new ImageLabelMapping()
                {
                    Target = _settings.Target,
                    Namespace = _nameSpaceValue,
                    Labels = _settings.ClassificationLabel,
                }.TransformText(),
                Name = "LabelMapping.cs",
            };

            ConsumeModel = new CSharpCodeFile()
            {
                File = new ConsumeModel()
                {
                    Namespace = _nameSpaceValue,
                    Target = _settings.Target,
                    HasLabelMapping = true,
                    HasNormalizeMapping = _settings.IsImage,
                    MLNetModelpath = _settings.ModelPath,
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
                    ModelOutputClass,
                    ConsumeModel,
                    ModelProject,
                    NormalizeMapping,
                    ImageLabelMapping,
                };
            }
            else
            {
                project = new CSharpProject()
                {
                    ModelInputClass,
                    ModelOutputClass,
                    ConsumeModel,
                    ModelProject,
                    LabelMapping,
                };
            }
            project.Name = Name;
            return project;
        }
    }
}
