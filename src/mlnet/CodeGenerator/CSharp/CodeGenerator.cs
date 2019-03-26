// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.ML.Auto;
using Microsoft.ML.CLI.Templates.Console;
using Microsoft.ML.CLI.Utilities;

namespace Microsoft.ML.CLI.CodeGenerator.CSharp
{
    internal class CodeGenerator : IProjectGenerator
    {
        private readonly Pipeline pipeline;
        private readonly CodeGeneratorSettings settings;
        private readonly ColumnInferenceResults columnInferenceResult;

        internal CodeGenerator(Pipeline pipeline, ColumnInferenceResults columnInferenceResult, CodeGeneratorSettings settings)
        {
            this.pipeline = pipeline;
            this.columnInferenceResult = columnInferenceResult;
            this.settings = settings;
        }

        public void GenerateOutput()
        {
            // Get Namespace
            var namespaceValue = Utils.Normalize(settings.OutputName);
            var labelType = columnInferenceResult.TextLoaderOptions.Columns.Where(t => t.Name == columnInferenceResult.ColumnInformation.LabelColumn).First().DataKind;
            Type labelTypeCsharp = Utils.GetCSharpType(labelType);

            // Generate Model Project
            var modelProjectContents = GenerateModelProjectContents(namespaceValue, labelTypeCsharp);

            // Write files to disk. 
            var modelprojectDir = Path.Combine(settings.OutputBaseDir, $"{settings.OutputName}.Model");
            var dataModelsDir = Path.Combine(modelprojectDir, "DataModels");
            var modelProjectName = $"{settings.OutputName}.Model.csproj";

            Utils.WriteOutputToFiles(modelProjectContents.ObservationCSFileContent, "Observation.cs", dataModelsDir);
            Utils.WriteOutputToFiles(modelProjectContents.PredictionCSFileContent, "Prediction.cs", dataModelsDir);
            Utils.WriteOutputToFiles(modelProjectContents.ModelProjectFileContent, modelProjectName, modelprojectDir);

            // Generate Predict Project
            var predictProjectContents = GeneratePredictProjectContents(namespaceValue);

            // Write files to disk. 
            var predictProjectDir = Path.Combine(settings.OutputBaseDir, $"{settings.OutputName}.Predict");
            var predictProjectName = $"{settings.OutputName}.Predict.csproj";

            Utils.WriteOutputToFiles(predictProjectContents.PredictProgramCSFileContent, "Program.cs", predictProjectDir);
            Utils.WriteOutputToFiles(predictProjectContents.PredictProjectFileContent, predictProjectName, predictProjectDir);

            // Generate Train Project
            (string trainProgramCSFileContent, string trainProjectFileContent, string consoleHelperCSFileContent) = GenerateTrainProjectContents(namespaceValue, labelTypeCsharp);

            // Write files to disk. 
            var trainProjectDir = Path.Combine(settings.OutputBaseDir, $"{settings.OutputName}.Train");
            var trainProjectName = $"{settings.OutputName}.Train.csproj";

            Utils.WriteOutputToFiles(trainProgramCSFileContent, "Program.cs", trainProjectDir);
            Utils.WriteOutputToFiles(consoleHelperCSFileContent, "ConsoleHelper.cs", trainProjectDir);
            Utils.WriteOutputToFiles(trainProjectFileContent, trainProjectName, trainProjectDir);

            // New solution file.
            Utils.CreateSolutionFile(settings.OutputName, settings.OutputBaseDir);

            // Add projects to solution
            var solutionPath = Path.Combine(settings.OutputBaseDir, $"{settings.OutputName}.sln");
            Utils.AddProjectsToSolution(modelprojectDir, modelProjectName, predictProjectDir, predictProjectName, trainProjectDir, trainProjectName, solutionPath);
        }

        internal (string, string, string) GenerateTrainProjectContents(string namespaceValue, Type labelTypeCsharp)
        {
            var result = GenerateTransformsAndTrainers();

            var trainProgramCSFileContent = GenerateTrainProgramCSFileContent(result.Usings, result.Trainer, result.PreTrainerTransforms, result.PostTrainerTransforms, namespaceValue, pipeline.CacheBeforeTrainer, labelTypeCsharp.Name);
            trainProgramCSFileContent = Utils.FormatCode(trainProgramCSFileContent);

            var trainProjectFileContent = GeneratTrainProjectFileContent(namespaceValue);
            var consoleHelperCSFileContent = GenerateConsoleHelper(namespaceValue);

            return (trainProgramCSFileContent, trainProjectFileContent, consoleHelperCSFileContent);
        }

        internal (string PredictProgramCSFileContent, string PredictProjectFileContent) GeneratePredictProjectContents(string namespaceValue)
        {
            var predictProgramCSFileContent = GeneratePredictProgramCSFileContent(namespaceValue);
            predictProgramCSFileContent = Utils.FormatCode(predictProgramCSFileContent);

            var predictProjectFileContent = GeneratPredictProjectFileContent(namespaceValue, true, true);
            return (predictProgramCSFileContent, predictProjectFileContent);
        }

        internal (string ObservationCSFileContent, string PredictionCSFileContent, string ModelProjectFileContent) GenerateModelProjectContents(string namespaceValue, Type labelTypeCsharp)
        {
            var classLabels = this.GenerateClassLabels();
            var observationCSFileContent = GenerateObservationCSFileContent(namespaceValue, classLabels);
            observationCSFileContent = Utils.FormatCode(observationCSFileContent);
            var predictionCSFileContent = GeneratePredictionCSFileContent(labelTypeCsharp.Name, namespaceValue);
            predictionCSFileContent = Utils.FormatCode(predictionCSFileContent);
            var modelProjectFileContent = GenerateModelProjectFileContent();
            return (observationCSFileContent, predictionCSFileContent, modelProjectFileContent);
        }

        internal (string Usings, string Trainer, List<string> PreTrainerTransforms, List<string> PostTrainerTransforms) GenerateTransformsAndTrainers()
        {
            StringBuilder usingsBuilder = new StringBuilder();
            var usings = new List<string>();
            var trainerAndUsings = this.GenerateTrainerAndUsings();

            // Get pre-trainer transforms
            var nodes = pipeline.Nodes.TakeWhile(t => t.NodeType == PipelineNodeType.Transform);
            var preTrainerTransformsAndUsings = this.GenerateTransformsAndUsings(nodes);

            // Get post trainer transforms
            nodes = pipeline.Nodes.SkipWhile(t => t.NodeType == PipelineNodeType.Transform)
                .SkipWhile(t => t.NodeType == PipelineNodeType.Trainer) //skip the trainer
                .TakeWhile(t => t.NodeType == PipelineNodeType.Transform); //post trainer transforms
            var postTrainerTransformsAndUsings = this.GenerateTransformsAndUsings(nodes);

            //Get trainer code and its associated usings.
            var trainer = trainerAndUsings.Item1;
            usings.Add(trainerAndUsings.Item2);

            //Get transforms code and its associated (unique) usings.
            var preTrainerTransforms = preTrainerTransformsAndUsings.Select(t => t.Item1).ToList();
            var postTrainerTransforms = postTrainerTransformsAndUsings.Select(t => t.Item1).ToList();
            usings.AddRange(preTrainerTransformsAndUsings.Select(t => t.Item2));
            usings.AddRange(postTrainerTransformsAndUsings.Select(t => t.Item2));
            usings = usings.Distinct().ToList();

            //Combine all using statements to actual text.
            usingsBuilder = new StringBuilder();
            usings.ForEach(t =>
            {
                if (t != null)
                    usingsBuilder.Append(t);
            });

            return (usingsBuilder.ToString(), trainer, preTrainerTransforms, postTrainerTransforms);
        }

        internal IList<(string, string)> GenerateTransformsAndUsings(IEnumerable<PipelineNode> nodes)
        {
            //var nodes = pipeline.Nodes.TakeWhile(t => t.NodeType == PipelineNodeType.Transform);
            //var nodes = pipeline.Nodes.Where(t => t.NodeType == PipelineNodeType.Transform);
            var results = new List<(string, string)>();
            foreach (var node in nodes)
            {
                ITransformGenerator generator = TransformGeneratorFactory.GetInstance(node);
                results.Add((generator.GenerateTransformer(), generator.GenerateUsings()));
            }

            return results;
        }

        internal (string, string) GenerateTrainerAndUsings()
        {
            ITrainerGenerator generator = TrainerGeneratorFactory.GetInstance(pipeline);
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

        #region Train Project
        private string GenerateTrainProgramCSFileContent(string usings,
            string trainer,
            List<string> preTrainerTransforms,
            List<string> postTrainerTransforms,
            string namespaceValue,
            bool cacheBeforeTrainer,
            string predictionLabelType)
        {
            var trainProgram = new TrainProgram()
            {
                PreTrainerTransforms = preTrainerTransforms,
                PostTrainerTransforms = postTrainerTransforms,
                HasHeader = columnInferenceResult.TextLoaderOptions.HasHeader,
                Separator = columnInferenceResult.TextLoaderOptions.Separators.FirstOrDefault(),
                AllowQuoting = columnInferenceResult.TextLoaderOptions.AllowQuoting,
                AllowSparse = columnInferenceResult.TextLoaderOptions.AllowSparse,
                Trainer = trainer,
                GeneratedUsings = usings,
                Path = settings.TrainDataset,
                TestPath = settings.TestDataset,
                TaskType = settings.MlTask.ToString(),
                Namespace = namespaceValue,
                LabelName = settings.LabelName,
                CacheBeforeTrainer = cacheBeforeTrainer,
            };

            return trainProgram.TransformText();
        }

        private string GeneratTrainProjectFileContent(string namespaceValue)
        {
            var trainProjectFileContent = new TrainProject() { Namespace = namespaceValue,/*The following args need to dynamic*/ IncludeHalLearnersPackage = true, IncludeLightGBMPackage = true };
            return trainProjectFileContent.TransformText();
        }

        private static string GenerateConsoleHelper(string namespaceValue)
        {
            var consoleHelperCodeGen = new ConsoleHelper() { Namespace = namespaceValue };
            return consoleHelperCodeGen.TransformText();
        }
        #endregion

        #region Model project
        private static string GenerateModelProjectFileContent()
        {
            ModelProject modelProject = new ModelProject();
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
        private static string GeneratPredictProjectFileContent(string namespaceValue, bool includeHalLearnersPackage, bool includeLightGBMPackage)
        {
            var predictProjectFileContent = new PredictProject() { Namespace = namespaceValue, IncludeHalLearnersPackage = includeHalLearnersPackage, IncludeLightGBMPackage = includeLightGBMPackage };
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
        #endregion

    }
}
