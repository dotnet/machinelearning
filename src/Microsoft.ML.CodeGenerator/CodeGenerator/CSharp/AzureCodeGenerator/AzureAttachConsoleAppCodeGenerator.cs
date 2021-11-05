// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CodeGenerator.CSharp.Interface;
using Microsoft.ML.CodeGenerator.CSharp;
using Microsoft.ML.CodeGenerator.Templates.Azure.Console;
using Microsoft.ML.CodeGenerator.Templates.Console;
using Microsoft.ML.CodeGenerator.Utilities;
using Microsoft.ML.Transforms;
using Tensorflow.Operations.Losses;

namespace Microsoft.ML.CodeGenerator.CodeGenerator.CSharp
{
    internal class AzureAttachConsoleAppCodeGenerator : ICSharpProjectGenerator
    {
        private readonly Pipeline _pipeline;
        private readonly CodeGeneratorSettings _settings;
        private readonly ColumnInferenceResults _columnInferenceResult;
        private readonly string _nameSpaceValue;

        public ICSharpFile ModelBuilder { get; private set; }
        public ICSharpFile PredictProject { get; private set; }
        public ICSharpFile PredictProgram { get; private set; }
        public string Name { get; set; }

        public AzureAttachConsoleAppCodeGenerator(Pipeline pipeline, ColumnInferenceResults columnInferenceResults, CodeGeneratorSettings options, string namespaceValue)
        {
            _pipeline = pipeline;
            _settings = options;
            _columnInferenceResult = columnInferenceResults;
            _nameSpaceValue = namespaceValue;
            Name = $"{_settings.OutputName}.ConsoleApp";

            var (_, _, PreTrainerTransforms, _) = _pipeline.GenerateTransformsAndTrainers();

            ModelBuilder = new CSharpCodeFile()
            {
                File = new AzureModelBuilder()
                {
                    Path = _settings.TrainDataset,
                    HasHeader = _columnInferenceResult.TextLoaderOptions.HasHeader,
                    Separator = _columnInferenceResult.TextLoaderOptions.Separators.FirstOrDefault(),
                    PreTrainerTransforms = PreTrainerTransforms,
                    AllowQuoting = _columnInferenceResult.TextLoaderOptions.AllowQuoting,
                    AllowSparse = _columnInferenceResult.TextLoaderOptions.AllowSparse,
                    Namespace = _nameSpaceValue,
                    Target = _settings.Target,
                    OnnxModelPath = _settings.OnnxModelName,
                    MLNetModelpath = _settings.ModelName,
                }.TransformText(),
                Name = "ModelBuilder.cs",
            };

            PredictProject = new CSharpProjectFile()
            {
                File = new PredictProject()
                {
                    Namespace = _nameSpaceValue,
                    IncludeMklComponentsPackage = false,
                    IncludeLightGBMPackage = false,
                    IncludeFastTreePackage = false,
                    IncludeImageTransformerPackage = _settings.IsImage || _settings.IsObjectDetection,
                    IncludeImageClassificationPackage = false,
                    IncludeOnnxPackage = true,
                    IncludeOnnxRuntime = _settings.IsObjectDetection,
                    IncludeResNet18Package = false,
                    IncludeRecommenderPackage = false,
                    StablePackageVersion = _settings.StablePackageVersion,
                    UnstablePackageVersion = _settings.UnstablePackageVersion,
                    OnnxRuntimePackageVersion = _settings.OnnxRuntimePackageVersion,
                }.TransformText(),
                Name = $"{_settings.OutputName}.ConsoleApp.csproj",
            };

            var sampleResult = Utils.GenerateSampleData(_settings.TrainDataset, _columnInferenceResult);
            PredictProgram = new CSharpCodeFile()
            {
                File = new PredictProgram()
                {
                    TaskType = _settings.MlTask.ToString(),
                    LabelName = _settings.LabelName,
                    Namespace = _nameSpaceValue,
                    AllowQuoting = _columnInferenceResult.TextLoaderOptions.AllowQuoting,
                    AllowSparse = _columnInferenceResult.TextLoaderOptions.AllowSparse,
                    HasHeader = _columnInferenceResult.TextLoaderOptions.HasHeader,
                    Separator = _columnInferenceResult.TextLoaderOptions.Separators.FirstOrDefault(),
                    Target = _settings.Target,
                    SampleData = sampleResult,
                    IsObjectDetection = _settings.IsObjectDetection,
                }.TransformText(),
                Name = "Program.cs",
            };
        }

        public ICSharpProject ToProject()
        {
            var project = new CSharpProject()
            {
                ModelBuilder,
                PredictProject,
                PredictProgram,
            };

            project.Name = Name;
            return project;
        }
    }
}
