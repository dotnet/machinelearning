// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Data.Common;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CodeGenerator;
using Microsoft.ML.CodeGenerator.Templates.Azure.Model;
using Microsoft.ML.CodeGenerator.Templates.Console;
using Microsoft.ML.CodeGenerator.Utilities;
using Microsoft.ML.Data;

namespace Microsoft.ML.CodeGenerator.CSharp
{
    internal class CodeGenerator : IProjectGenerator
    {
        private readonly Pipeline _pipeline;
        private readonly CodeGeneratorSettings _settings;
        private readonly ColumnInferenceResults _columnInferenceResult;
        private static readonly HashSet<string> _recommendationTrainers = new HashSet<string>() { TrainerName.MatrixFactorization.ToString() };
        private static readonly HashSet<string> _lightGbmTrainers = new HashSet<string>() { TrainerName.LightGbmBinary.ToString(), TrainerName.LightGbmMulti.ToString(),
            TrainerName.LightGbmRegression.ToString(), TrainerName.LightGbmRanking.ToString() };
        private static readonly HashSet<string> _mklComponentsTrainers = new HashSet<string>() { TrainerName.OlsRegression.ToString(), TrainerName.SymbolicSgdLogisticRegressionBinary.ToString() };
        private static readonly HashSet<string> _fastTreeTrainers = new HashSet<string>() { TrainerName.FastForestBinary.ToString(), TrainerName.FastForestRegression.ToString(),
            TrainerName.FastTreeBinary.ToString(), TrainerName.FastTreeRegression.ToString(), TrainerName.FastTreeTweedieRegression.ToString(), TrainerName.FastTreeRanking.ToString() };
        private static readonly HashSet<string> _imageTransformers = new HashSet<string>() { EstimatorName.RawByteImageLoading.ToString() };
        private static readonly HashSet<string> _imageClassificationTrainers = new HashSet<string>() { TrainerName.ImageClassification.ToString() };

        internal CodeGenerator(Pipeline pipeline, ColumnInferenceResults columnInferenceResult, CodeGeneratorSettings settings)
        {
            _pipeline = pipeline;
            _columnInferenceResult = columnInferenceResult;
            _settings = settings;
        }

        public void GenerateOutput()
        {
            // Get the extra nuget packages to be included in the generated project.
            var trainerNodes = _pipeline.Nodes.Where(t => t.NodeType == PipelineNodeType.Trainer);

            bool includeLightGbmPackage = false;
            bool includeMklComponentsPackage = false;
            bool includeFastTreeePackage = false;
            bool includeImageTransformerPackage = false;
            bool includeImageClassificationPackage = false;
            bool includeRecommenderPackage = false;
            // Get the extra nuget packages to be included in the generated project.
            SetRequiredNugetPackages(_pipeline.Nodes, ref includeLightGbmPackage, ref includeMklComponentsPackage,
                ref includeFastTreeePackage, ref includeImageTransformerPackage, ref includeImageClassificationPackage, ref includeRecommenderPackage);

            // Get Namespace
            var namespaceValue = Utils.Normalize(_settings.OutputName);
            var labelType = _columnInferenceResult.TextLoaderOptions.Columns.Where(t => t.Name == _settings.LabelName).First().DataKind;
            Type labelTypeCsharp = Utils.GetCSharpType(labelType);

            // Generate Model Project
            var modelProjectContents = GenerateModelProjectContents(namespaceValue, labelTypeCsharp,
                includeLightGbmPackage, includeMklComponentsPackage, includeFastTreeePackage,
                includeImageTransformerPackage, includeImageClassificationPackage, includeRecommenderPackage);

            // Write files to disk.
            var modelprojectDir = Path.Combine(_settings.OutputBaseDir, $"{_settings.OutputName}.Model");
            var dataModelsDir = modelprojectDir;
            var modelProjectName = $"{_settings.OutputName}.Model.csproj";

            Utils.WriteOutputToFiles(modelProjectContents.ModelInputCSFileContent, "ModelInput.cs", dataModelsDir);
            Utils.WriteOutputToFiles(modelProjectContents.ModelOutputCSFileContent, "ModelOutput.cs", dataModelsDir);
            Utils.WriteOutputToFiles(modelProjectContents.ConsumeModelCSFileContent, "ConsumeModel.cs", dataModelsDir);
            Utils.WriteOutputToFiles(modelProjectContents.ModelProjectFileContent, modelProjectName, modelprojectDir);

            // Generate ConsoleApp Project
            var consoleAppProjectContents = GenerateConsoleAppProjectContents(namespaceValue, labelTypeCsharp,
                includeLightGbmPackage, includeMklComponentsPackage, includeFastTreeePackage,
                includeImageTransformerPackage, includeImageClassificationPackage, includeRecommenderPackage);

            // Write files to disk.
            var consoleAppProjectDir = Path.Combine(_settings.OutputBaseDir, $"{_settings.OutputName}.ConsoleApp");
            var consoleAppProjectName = $"{_settings.OutputName}.ConsoleApp.csproj";

            Utils.WriteOutputToFiles(consoleAppProjectContents.ConsoleAppProgramCSFileContent, "Program.cs", consoleAppProjectDir);
            Utils.WriteOutputToFiles(consoleAppProjectContents.modelBuilderCSFileContent, "ModelBuilder.cs", consoleAppProjectDir);
            Utils.WriteOutputToFiles(consoleAppProjectContents.ConsoleAppProjectFileContent, consoleAppProjectName, consoleAppProjectDir);

            // New solution file.
            Utils.CreateSolutionFile(_settings.OutputName, _settings.OutputBaseDir);

            // Add projects to solution
            var solutionPath = Path.Combine(_settings.OutputBaseDir, $"{_settings.OutputName}.sln");
            Utils.AddProjectsToSolution(modelprojectDir, modelProjectName, consoleAppProjectDir, consoleAppProjectName, solutionPath);
        }

        private void SetRequiredNugetPackages(IEnumerable<PipelineNode> trainerNodes, ref bool includeLightGbmPackage,
            ref bool includeMklComponentsPackage, ref bool includeFastTreePackage,
            ref bool includeImageTransformerPackage, ref bool includeImageClassificationPackage, ref bool includeRecommenderPackage)
        {
            foreach (var node in trainerNodes)
            {
                PipelineNode currentNode = node;
                if (currentNode.Name == TrainerName.Ova.ToString())
                {
                    currentNode = (PipelineNode)currentNode.Properties["BinaryTrainer"];
                }

                if (_lightGbmTrainers.Contains(currentNode.Name))
                {
                    includeLightGbmPackage = true;
                }
                else if (_mklComponentsTrainers.Contains(currentNode.Name))
                {
                    includeMklComponentsPackage = true;
                }
                else if (_fastTreeTrainers.Contains(currentNode.Name))
                {
                    includeFastTreePackage = true;
                }
                else if (_imageTransformers.Contains(currentNode.Name))
                {
                    includeImageTransformerPackage = true;
                }
                else if (_imageClassificationTrainers.Contains(currentNode.Name))
                {
                    includeImageClassificationPackage = true;
                }
                else if (_recommendationTrainers.Contains(currentNode.Name))
                {
                    includeRecommenderPackage = true;
                }
            }
        }

        internal (string ConsoleAppProgramCSFileContent, string ConsoleAppProjectFileContent,
            string modelBuilderCSFileContent) GenerateConsoleAppProjectContents(string namespaceValue,
                Type labelTypeCsharp, bool includeLightGbmPackage, bool includeMklComponentsPackage,
                bool includeFastTreePackage, bool includeImageTransformerPackage,
                bool includeImageClassificationPackage, bool includeRecommenderPackage, bool includeOnnxPackage = false, bool includeResNet18Package = false)
        {
            var predictProgramCSFileContent = GeneratePredictProgramCSFileContent(namespaceValue);
            predictProgramCSFileContent = Utils.FormatCode(predictProgramCSFileContent);

            var predictProjectFileContent = GeneratPredictProjectFileContent(_settings.OutputName,
                includeLightGbmPackage, includeMklComponentsPackage, includeFastTreePackage,
                includeImageTransformerPackage, includeImageClassificationPackage, includeRecommenderPackage, includeOnnxPackage, includeResNet18Package,
                _settings.StablePackageVersion, _settings.UnstablePackageVersion, _settings.Target);

            var transformsAndTrainers = GenerateTransformsAndTrainers();
            var modelBuilderCSFileContent = GenerateModelBuilderCSFileContent(transformsAndTrainers.Usings, transformsAndTrainers.TrainerMethod,
                transformsAndTrainers.PreTrainerTransforms, transformsAndTrainers.PostTrainerTransforms, namespaceValue, _pipeline.CacheBeforeTrainer, labelTypeCsharp.Name, includeOnnxPackage);
            modelBuilderCSFileContent = Utils.FormatCode(modelBuilderCSFileContent);

            return (predictProgramCSFileContent, predictProjectFileContent, modelBuilderCSFileContent);
        }

        internal (string ModelInputCSFileContent, string ModelOutputCSFileContent, string ConsumeModelCSFileContent,
            string ModelProjectFileContent) GenerateModelProjectContents(string namespaceValue,
                Type labelTypeCsharp, bool includeLightGbmPackage, bool includeMklComponentsPackage,
                bool includeFastTreePackage, bool includeImageTransformerPackage,
                bool includeImageClassificationPackage, bool includeRecommenderPackage, bool includeOnnxModel = false)
        {
            var classLabels = GenerateClassLabels();

            // generate ModelInput.cs
            var modelInputCSFileContent = GenerateModelInputCSFileContent(namespaceValue, classLabels);
            modelInputCSFileContent = Utils.FormatCode(modelInputCSFileContent);

            // generate ModelOutput.cs
            var modelOutputCSFileContent = GenerateModelOutputCSFileContent(labelTypeCsharp.Name, namespaceValue);
            modelOutputCSFileContent = Utils.FormatCode(modelOutputCSFileContent);

            // generate ConsumeModel.cs
            var consumeModelCSFileContent = GenerateConsumeModelCSFileContent(namespaceValue);
            consumeModelCSFileContent = Utils.FormatCode(consumeModelCSFileContent);
            var modelProjectFileContent = GenerateModelProjectFileContent(includeLightGbmPackage,
                includeMklComponentsPackage, includeFastTreePackage, includeImageTransformerPackage,
                includeImageClassificationPackage, includeRecommenderPackage, includeOnnxModel,
                _settings.StablePackageVersion, _settings.UnstablePackageVersion, _settings.Target);

            return (modelInputCSFileContent, modelOutputCSFileContent, consumeModelCSFileContent, modelProjectFileContent);
        }

        private string GenerateConsumeModelCSFileContent(string namespaceValue)
        {
            ConsumeModel consumeModel = new ConsumeModel()
            {
                Namespace = namespaceValue,
                Target = _settings.Target,
                MLNetModelName = _settings.ModelName,
            };
            return consumeModel.TransformText();
        }

        internal (string Usings, string TrainerMethod, List<string> PreTrainerTransforms, List<string> PostTrainerTransforms) GenerateTransformsAndTrainers()
        {
            StringBuilder usingsBuilder = new StringBuilder();
            var usings = new List<string>();

            // Get pre-trainer transforms
            var nodes = _pipeline.Nodes.TakeWhile(t => t.NodeType == PipelineNodeType.Transform);
            var preTrainerTransformsAndUsings = GenerateTransformsAndUsings(nodes);

            // Get post trainer transforms
            nodes = _pipeline.Nodes.SkipWhile(t => t.NodeType == PipelineNodeType.Transform)
                .SkipWhile(t => t.NodeType == PipelineNodeType.Trainer) //skip the trainer
                .TakeWhile(t => t.NodeType == PipelineNodeType.Transform); //post trainer transforms
            var postTrainerTransformsAndUsings = GenerateTransformsAndUsings(nodes);

            //Get trainer code and its associated usings.
            (string trainerMethod, string[] trainerUsings) = GenerateTrainerAndUsings();
            if (trainerUsings != null)
            {
                usings.AddRange(trainerUsings);
            }

            //Get transforms code and its associated (unique) usings.
            var preTrainerTransforms = preTrainerTransformsAndUsings?.Select(t => t.Item1).ToList();
            var postTrainerTransforms = postTrainerTransformsAndUsings?.Select(t => t.Item1).ToList();
            usings.AddRange(preTrainerTransformsAndUsings.Where(t => t.Item2 != null).SelectMany(t => t.Item2));
            usings.AddRange(postTrainerTransformsAndUsings.Where(t => t.Item2 != null).SelectMany(t => t.Item2));
            usings = usings.Distinct().ToList();

            //Combine all using statements to actual text.
            usingsBuilder = new StringBuilder();
            usings.ForEach(t =>
            {
                if (t != null)
                    usingsBuilder.Append(t);
            });

            return (usingsBuilder.ToString(), trainerMethod, preTrainerTransforms, postTrainerTransforms);
        }

        internal IList<(string, string[])> GenerateTransformsAndUsings(IEnumerable<PipelineNode> nodes)
        {
            //var nodes = pipeline.Nodes.TakeWhile(t => t.NodeType == PipelineNodeType.Transform);
            //var nodes = pipeline.Nodes.Where(t => t.NodeType == PipelineNodeType.Transform);
            var results = new List<(string, string[])>();
            foreach (var node in nodes)
            {
                ITransformGenerator generator = TransformGeneratorFactory.GetInstance(node);
                results.Add((generator.GenerateTransformer(), generator.GenerateUsings()));
            }

            return results;
        }

        internal (string, string[]) GenerateTrainerAndUsings()
        {
            if (_pipeline == null)
                throw new ArgumentNullException(nameof(_pipeline));
            var node = _pipeline.Nodes.Where(t => t.NodeType == PipelineNodeType.Trainer).First();
            ITrainerGenerator generator = TrainerGeneratorFactory.GetInstance(node);
            var trainerString = generator.GenerateTrainer();
            var trainerUsings = generator.GenerateUsings();
            return (trainerString, trainerUsings);
        }

        /// <summary>
        /// Utilize <see cref="ML.CodeGenerator.Utilities.Utils.GenerateClassLabels(ColumnInferenceResults, IDictionary{string, CodeGeneratorSettings.ColumnMapping})"/>
        /// </summary>
        internal IList<string> GenerateClassLabels(IDictionary<string, CodeGeneratorSettings.ColumnMapping> columnMapping = default)
        {
            return Utils.GenerateClassLabels(_columnInferenceResult, columnMapping);
        }

        #region Model project
        private string GenerateModelProjectFileContent(bool includeLightGbmPackage,
            bool includeMklComponentsPackage, bool includeFastTreePackage, bool includeImageTransformerPackage,
                bool includeImageClassificationPackage, bool includeRecommenderPackage, bool includeOnnxModel,
                string stablePackageVersion, string unstablePackageVersion, GenerateTarget target)
        {
            ModelProject modelProject = new ModelProject()
            {
                IncludeLightGBMPackage = includeLightGbmPackage,
                IncludeMklComponentsPackage = includeMklComponentsPackage,
                IncludeFastTreePackage = includeFastTreePackage,
                IncludeImageTransformerPackage = includeImageTransformerPackage,
                IncludeImageClassificationPackage = includeImageClassificationPackage,
                IncludeOnnxModel = includeOnnxModel,
                IncludeRecommenderPackage = includeRecommenderPackage,
                StablePackageVersion = stablePackageVersion,
                UnstablePackageVersion = unstablePackageVersion,
                OnnxRuntimePackageVersion = _settings.OnnxRuntimePackageVersion,
                Target = target,
            };

            return modelProject.TransformText();
        }

        private string GenerateModelOutputCSFileContent(string predictionLabelType, string namespaceValue)
        {
            ModelOutputClass modelOutputClass = new ModelOutputClass()
            {
                TaskType = _settings.MlTask.ToString(),
                PredictionLabelType = predictionLabelType,
                Namespace = namespaceValue,
                Target = _settings.Target,
            };
            return modelOutputClass.TransformText();
        }

        private string GenerateModelInputCSFileContent(string namespaceValue, IList<string> classLabels)
        {
            ModelInputClass modelInputClass = new ModelInputClass()
            {
                Namespace = namespaceValue,
                ClassLabels = classLabels,
                Target = _settings.Target,
            };
            return modelInputClass.TransformText();
        }
        #endregion

        #region Predict Project
        private string GeneratPredictProjectFileContent(string namespaceValue, bool includeLightGbmPackage,
            bool includeMklComponentsPackage, bool includeFastTreePackage, bool includeImageTransformerPackage,
                bool includeImageClassificationPackage, bool includeRecommenderPackage,
                bool includeOnnxPackage, bool includeResNet18Package,
                string stablePackageVersion, string unstablePackageVersion, GenerateTarget target)
        {
            var predictProjectFileContent = new PredictProject()
            {
                Namespace = namespaceValue,
                IncludeMklComponentsPackage = includeMklComponentsPackage,
                IncludeLightGBMPackage = includeLightGbmPackage,
                IncludeFastTreePackage = includeFastTreePackage,
                IncludeImageTransformerPackage = includeImageTransformerPackage,
                IncludeImageClassificationPackage = includeImageClassificationPackage,
                IncludeOnnxPackage = includeOnnxPackage,
                IncludeResNet18Package = includeResNet18Package,
                IncludeRecommenderPackage = includeRecommenderPackage,
                StablePackageVersion = stablePackageVersion,
                UnstablePackageVersion = unstablePackageVersion,
                OnnxRuntimePackageVersion = _settings.OnnxRuntimePackageVersion,
                Target = target,
            };
            return predictProjectFileContent.TransformText();
        }

        private string GeneratePredictProgramCSFileContent(string namespaceValue)
        {
            var sampleData = Utils.GenerateSampleData(_settings.TrainDataset, _columnInferenceResult);
            PredictProgram predictProgram = new PredictProgram()
            {
                TaskType = _settings.MlTask.ToString(),
                LabelName = _settings.LabelName,
                Namespace = namespaceValue,
                HasHeader = _columnInferenceResult.TextLoaderOptions.HasHeader,
                Separator = _columnInferenceResult.TextLoaderOptions.Separators.FirstOrDefault(),
                AllowQuoting = _columnInferenceResult.TextLoaderOptions.AllowQuoting,
                AllowSparse = _columnInferenceResult.TextLoaderOptions.AllowSparse,
                Target = _settings.Target,
                SampleData = sampleData,
            };
            return predictProgram.TransformText();
        }

        private string GenerateModelBuilderCSFileContent(string usings,
            string trainerMethod,
            List<string> preTrainerTransforms,
            List<string> postTrainerTransforms,
            string namespaceValue,
            bool cacheBeforeTrainer,
            string predictionLabelType,
            bool hasOnnxModel = false)
        {
            var modelBuilder = new ModelBuilder()
            {
                PreTrainerTransforms = preTrainerTransforms,
                PostTrainerTransforms = postTrainerTransforms,
                HasHeader = _columnInferenceResult.TextLoaderOptions.HasHeader,
                Separator = _columnInferenceResult.TextLoaderOptions.Separators.FirstOrDefault(),
                AllowQuoting = _columnInferenceResult.TextLoaderOptions.AllowQuoting,
                AllowSparse = _columnInferenceResult.TextLoaderOptions.AllowSparse,
                Trainer = trainerMethod,
                GeneratedUsings = usings,
                Path = _settings.TrainDataset,
                TestPath = _settings.TestDataset,
                TaskType = _settings.MlTask.ToString(),
                Namespace = namespaceValue,
                LabelName = _settings.LabelName,
                CacheBeforeTrainer = cacheBeforeTrainer,
                Target = _settings.Target,
                HasOnnxModel = hasOnnxModel,
                MLNetModelName = _settings.ModelName,
            };

            return modelBuilder.TransformText();
        }
        #endregion
    }
}
