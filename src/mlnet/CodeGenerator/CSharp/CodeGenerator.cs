// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.ML.AutoML;
using Microsoft.ML.CLI.Templates.Console;
using Microsoft.ML.CLI.Utilities;

namespace Microsoft.ML.CLI.CodeGenerator.CSharp
{
    internal class CodeGenerator : IProjectGenerator
    {
        private readonly Pipeline pipeline;
        private readonly CodeGeneratorSettings settings;
        private readonly ColumnInferenceResults columnInferenceResult;
        private readonly HashSet<string> LightGBMTrainers = new HashSet<string>() { TrainerName.LightGbmBinary.ToString(), TrainerName.LightGbmMulti.ToString(), TrainerName.LightGbmRegression.ToString() };
        private readonly HashSet<string> mklComponentsTrainers = new HashSet<string>() { TrainerName.OlsRegression.ToString(), TrainerName.SymbolicSgdLogisticRegressionBinary.ToString() };
        private readonly HashSet<string> FastTreeTrainers = new HashSet<string>() { TrainerName.FastForestBinary.ToString(), TrainerName.FastForestRegression.ToString(), TrainerName.FastTreeBinary.ToString(), TrainerName.FastTreeRegression.ToString(), TrainerName.FastTreeTweedieRegression.ToString() };


        internal CodeGenerator(Pipeline pipeline, ColumnInferenceResults columnInferenceResult, CodeGeneratorSettings settings)
        {
            this.pipeline = pipeline;
            this.columnInferenceResult = columnInferenceResult;
            this.settings = settings;
        }

        public void GenerateOutput()
        {
            // Get the extra nuget packages to be included in the generated project.
            var trainerNodes = pipeline.Nodes.Where(t => t.NodeType == PipelineNodeType.Trainer);

            bool includeLightGbmPackage = false;
            bool includeMklComponentsPackage = false;
            bool includeFastTreeePackage = false;
            SetRequiredNugetPackages(trainerNodes, ref includeLightGbmPackage, ref includeMklComponentsPackage, ref includeFastTreeePackage);

            // Get Namespace
            var namespaceValue = Utils.Normalize(settings.OutputName);
            var labelType = columnInferenceResult.TextLoaderOptions.Columns.Where(t => t.Name == settings.LabelName).First().DataKind;
            Type labelTypeCsharp = Utils.GetCSharpType(labelType);

            // Generate Model Project
            var modelProjectContents = GenerateModelProjectContents(namespaceValue, labelTypeCsharp, includeLightGbmPackage, includeMklComponentsPackage, includeFastTreeePackage);

            // Write files to disk. 
            var modelprojectDir = Path.Combine(settings.OutputBaseDir, $"{settings.OutputName}.Model");
            var dataModelsDir = Path.Combine(modelprojectDir, "DataModels");
            var modelProjectName = $"{settings.OutputName}.Model.csproj";

            Utils.WriteOutputToFiles(modelProjectContents.ObservationCSFileContent, "SampleObservation.cs", dataModelsDir);
            Utils.WriteOutputToFiles(modelProjectContents.PredictionCSFileContent, "SamplePrediction.cs", dataModelsDir);
            Utils.WriteOutputToFiles(modelProjectContents.ModelProjectFileContent, modelProjectName, modelprojectDir);

            // Generate ConsoleApp Project
            var consoleAppProjectContents = GenerateConsoleAppProjectContents(namespaceValue, labelTypeCsharp, includeLightGbmPackage, includeMklComponentsPackage, includeFastTreeePackage);

            // Write files to disk. 
            var consoleAppProjectDir = Path.Combine(settings.OutputBaseDir, $"{settings.OutputName}.ConsoleApp");
            var consoleAppProjectName = $"{settings.OutputName}.ConsoleApp.csproj";

            Utils.WriteOutputToFiles(consoleAppProjectContents.ConsoleAppProgramCSFileContent, "Program.cs", consoleAppProjectDir);
            Utils.WriteOutputToFiles(consoleAppProjectContents.modelBuilderCSFileContent, "ModelBuilder.cs", consoleAppProjectDir);
            Utils.WriteOutputToFiles(consoleAppProjectContents.ConsoleAppProjectFileContent, consoleAppProjectName, consoleAppProjectDir);

            // New solution file.
            Utils.CreateSolutionFile(settings.OutputName, settings.OutputBaseDir);

            // Add projects to solution
            var solutionPath = Path.Combine(settings.OutputBaseDir, $"{settings.OutputName}.sln");
            Utils.AddProjectsToSolution(modelprojectDir, modelProjectName, consoleAppProjectDir, consoleAppProjectName, solutionPath);
        }

        private void SetRequiredNugetPackages(IEnumerable<PipelineNode> trainerNodes, ref bool includeLightGbmPackage, ref bool includeMklComponentsPackage, ref bool includeFastTreePackage)
        {
            foreach (var node in trainerNodes)
            {
                PipelineNode currentNode = node;
                if (currentNode.Name == TrainerName.Ova.ToString())
                {
                    currentNode = (PipelineNode)currentNode.Properties["BinaryTrainer"];
                }

                if (LightGBMTrainers.Contains(currentNode.Name))
                {
                    includeLightGbmPackage = true;
                }
                else if (mklComponentsTrainers.Contains(currentNode.Name))
                {
                    includeMklComponentsPackage = true;
                }
                else if (FastTreeTrainers.Contains(currentNode.Name))
                {
                    includeFastTreePackage = true;
                }
            }
        }

        internal (string ConsoleAppProgramCSFileContent, string ConsoleAppProjectFileContent, string modelBuilderCSFileContent) GenerateConsoleAppProjectContents(string namespaceValue, Type labelTypeCsharp, bool includeLightGbmPackage, bool includeMklComponentsPackage, bool includeFastTreePackage)
        {
            var predictProgramCSFileContent = GeneratePredictProgramCSFileContent(namespaceValue);
            predictProgramCSFileContent = Utils.FormatCode(predictProgramCSFileContent);

            var predictProjectFileContent = GeneratPredictProjectFileContent(namespaceValue, includeLightGbmPackage, includeMklComponentsPackage, includeFastTreePackage);

            var transformsAndTrainers = GenerateTransformsAndTrainers();
            var modelBuilderCSFileContent = GenerateModelBuilderCSFileContent(transformsAndTrainers.Usings, transformsAndTrainers.TrainerMethod, transformsAndTrainers.PreTrainerTransforms, transformsAndTrainers.PostTrainerTransforms, namespaceValue, pipeline.CacheBeforeTrainer, labelTypeCsharp.Name);
            modelBuilderCSFileContent = Utils.FormatCode(modelBuilderCSFileContent);

            return (predictProgramCSFileContent, predictProjectFileContent, modelBuilderCSFileContent);
        }

        internal (string ObservationCSFileContent, string PredictionCSFileContent, string ModelProjectFileContent) GenerateModelProjectContents(string namespaceValue, Type labelTypeCsharp, bool includeLightGbmPackage, bool includeMklComponentsPackage, bool includeFastTreePackage)
        {
            var classLabels = this.GenerateClassLabels();
            var observationCSFileContent = GenerateObservationCSFileContent(namespaceValue, classLabels);
            observationCSFileContent = Utils.FormatCode(observationCSFileContent);
            var predictionCSFileContent = GeneratePredictionCSFileContent(labelTypeCsharp.Name, namespaceValue);
            predictionCSFileContent = Utils.FormatCode(predictionCSFileContent);
            var modelProjectFileContent = GenerateModelProjectFileContent(includeLightGbmPackage, includeMklComponentsPackage, includeFastTreePackage);
            return (observationCSFileContent, predictionCSFileContent, modelProjectFileContent);
        }

        internal (string Usings, string TrainerMethod, List<string> PreTrainerTransforms, List<string> PostTrainerTransforms) GenerateTransformsAndTrainers()
        {
            StringBuilder usingsBuilder = new StringBuilder();
            var usings = new List<string>();

            // Get pre-trainer transforms
            var nodes = pipeline.Nodes.TakeWhile(t => t.NodeType == PipelineNodeType.Transform);
            var preTrainerTransformsAndUsings = this.GenerateTransformsAndUsings(nodes);

            // Get post trainer transforms
            nodes = pipeline.Nodes.SkipWhile(t => t.NodeType == PipelineNodeType.Transform)
                .SkipWhile(t => t.NodeType == PipelineNodeType.Trainer) //skip the trainer
                .TakeWhile(t => t.NodeType == PipelineNodeType.Transform); //post trainer transforms
            var postTrainerTransformsAndUsings = this.GenerateTransformsAndUsings(nodes);

            //Get trainer code and its associated usings.
            (string trainerMethod, string[] trainerUsings) = this.GenerateTrainerAndUsings();
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
            if (pipeline == null)
                throw new ArgumentNullException(nameof(pipeline));
            var node = pipeline.Nodes.Where(t => t.NodeType == PipelineNodeType.Trainer).First();
            if (node == null)
                throw new ArgumentException($"The trainer was not found.");

            ITrainerGenerator generator = TrainerGeneratorFactory.GetInstance(node);
            var trainerString = generator.GenerateTrainer();
            var trainerUsings = generator.GenerateUsings();
            return (trainerString, trainerUsings);
        }

        internal IList<string> GenerateClassLabels()
        {
            IList<string> result = new List<string>();
            foreach (var column in columnInferenceResult.TextLoaderOptions.Columns)
            {
                StringBuilder sb = new StringBuilder();
                int range = (column.Source[0].Max - column.Source[0].Min).Value;
                bool isArray = range > 0;
                sb.Append(Symbols.PublicSymbol);
                sb.Append(Symbols.Space);
                switch (column.DataKind)
                {
                    case Microsoft.ML.Data.DataKind.String:
                        sb.Append(Symbols.StringSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.Boolean:
                        sb.Append(Symbols.BoolSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.Single:
                        sb.Append(Symbols.FloatSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.Double:
                        sb.Append(Symbols.DoubleSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.Int32:
                        sb.Append(Symbols.IntSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.UInt32:
                        sb.Append(Symbols.UIntSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.Int64:
                        sb.Append(Symbols.LongSymbol);
                        break;
                    case Microsoft.ML.Data.DataKind.UInt64:
                        sb.Append(Symbols.UlongSymbol);
                        break;
                    default:
                        throw new ArgumentException($"The data type '{column.DataKind}' is not handled currently.");

                }

                if (range > 0)
                {
                    result.Add($"[ColumnName(\"{column.Name}\"),LoadColumn({column.Source[0].Min}, {column.Source[0].Max}) VectorType({(range + 1)})]");
                    sb.Append("[]");
                }
                else
                {
                    result.Add($"[ColumnName(\"{column.Name}\"), LoadColumn({column.Source[0].Min})]");
                }
                sb.Append(" ");
                sb.Append(Utils.Normalize(column.Name));
                sb.Append("{get; set;}");
                result.Add(sb.ToString());
                result.Add("\r\n");
            }
            return result;
        }

        #region Model project
        private static string GenerateModelProjectFileContent(bool includeLightGbmPackage, bool includeMklComponentsPackage, bool includeFastTreePackage)
        {
            ModelProject modelProject = new ModelProject() { IncludeLightGBMPackage = includeLightGbmPackage, IncludeMklComponentsPackage = includeMklComponentsPackage, IncludeFastTreePackage = includeFastTreePackage };
            return modelProject.TransformText();
        }

        private string GeneratePredictionCSFileContent(string predictionLabelType, string namespaceValue)
        {
            PredictionClass predictionClass = new PredictionClass() { TaskType = settings.MlTask.ToString(), PredictionLabelType = predictionLabelType, Namespace = namespaceValue };
            return predictionClass.TransformText();
        }

        private string GenerateObservationCSFileContent(string namespaceValue, IList<string> classLabels)
        {
            ObservationClass observationClass = new ObservationClass() { Namespace = namespaceValue, ClassLabels = classLabels };
            return observationClass.TransformText();
        }
        #endregion

        #region Predict Project
        private static string GeneratPredictProjectFileContent(string namespaceValue, bool includeLightGbmPackage, bool includeMklComponentsPackage, bool includeFastTreePackage)
        {
            var predictProjectFileContent = new PredictProject() { Namespace = namespaceValue, IncludeMklComponentsPackage = includeMklComponentsPackage, IncludeLightGBMPackage = includeLightGbmPackage, IncludeFastTreePackage = includeFastTreePackage };
            return predictProjectFileContent.TransformText();
        }

        private string GeneratePredictProgramCSFileContent(string namespaceValue)
        {
            PredictProgram predictProgram = new PredictProgram()
            {
                TaskType = settings.MlTask.ToString(),
                LabelName = settings.LabelName,
                Namespace = namespaceValue,
                TestDataPath = settings.TestDataset,
                TrainDataPath = settings.TrainDataset,
                HasHeader = columnInferenceResult.TextLoaderOptions.HasHeader,
                Separator = columnInferenceResult.TextLoaderOptions.Separators.FirstOrDefault(),
                AllowQuoting = columnInferenceResult.TextLoaderOptions.AllowQuoting,
                AllowSparse = columnInferenceResult.TextLoaderOptions.AllowSparse,
            };
            return predictProgram.TransformText();
        }

        private string GenerateModelBuilderCSFileContent(string usings,
            string trainerMethod,
            List<string> preTrainerTransforms,
            List<string> postTrainerTransforms,
            string namespaceValue,
            bool cacheBeforeTrainer,
            string predictionLabelType)
        {
            var modelBuilder = new ModelBuilder()
            {
                PreTrainerTransforms = preTrainerTransforms,
                PostTrainerTransforms = postTrainerTransforms,
                HasHeader = columnInferenceResult.TextLoaderOptions.HasHeader,
                Separator = columnInferenceResult.TextLoaderOptions.Separators.FirstOrDefault(),
                AllowQuoting = columnInferenceResult.TextLoaderOptions.AllowQuoting,
                AllowSparse = columnInferenceResult.TextLoaderOptions.AllowSparse,
                Trainer = trainerMethod,
                GeneratedUsings = usings,
                Path = settings.TrainDataset,
                TestPath = settings.TestDataset,
                TaskType = settings.MlTask.ToString(),
                Namespace = namespaceValue,
                LabelName = settings.LabelName,
                CacheBeforeTrainer = cacheBeforeTrainer,
            };

            return modelBuilder.TransformText();
        }
        #endregion
    }
}
