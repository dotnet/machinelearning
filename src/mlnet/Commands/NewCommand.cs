// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using mlnet.Templates;

namespace Microsoft.ML.CLI
{
    internal class NewCommand
    {
        internal static void Run(Options options)
        {
            var context = new MLContext();
            var label = options.LabelName;

            // For Version 0.1 It is required that the data set has header. 
            var columnInference = context.Data.InferColumns(options.TrainDataset.FullName, label, true, groupColumns: false);
            var textLoader = context.Data.CreateTextReader(columnInference);
            var trainData = textLoader.Read(options.TrainDataset.FullName);

            var validationData = textLoader.Read(options.TestDataset.FullName);
            Pipeline pipelineToDeconstruct = null;

            var result = ExploreModels(options, context, label, trainData, validationData, pipelineToDeconstruct);
            pipelineToDeconstruct = result.Item1;
            var model = result.Item2;
            //Path can be overriden from args
            GenerateModel(model, @"./BestModel", "model.zip", context);
            RunCodeGen(options, columnInference, pipelineToDeconstruct);
        }

        private static void GenerateModel(ITransformer model, string ModelPath, string modelName, MLContext mlContext)
        {
            if (!Directory.Exists(ModelPath))
            {
                Directory.CreateDirectory(ModelPath);
            }
            ModelPath = ModelPath + "/" + modelName;
            using (var fs = File.Create(ModelPath))
                model.SaveTo(mlContext, fs);
        }

        private static (Pipeline, ITransformer) ExploreModels(
            Options options, MLContext context,
            string label,
            IDataView trainData,
            IDataView validationData,
            Pipeline pipelineToDeconstruct)
        {
            ITransformer model = null;

            if (options.MlTask == TaskKind.BinaryClassification)
            {
                var result = context.BinaryClassification.AutoFit(trainData, label, validationData, settings:
                    new AutoFitSettings()
                    {
                        StoppingCriteria = new ExperimentStoppingCriteria()
                        {
                            //default need to have a way to override
                            TimeOutInMinutes = 10
                        }
                    });
                result = result.OrderByDescending(t => t.Metrics.Accuracy);
                var bestIteration = result.FirstOrDefault();
                pipelineToDeconstruct = bestIteration.Pipeline;
                model = bestIteration.Model;
            }

            if (options.MlTask == TaskKind.Regression)
            {
                var result = context.Regression.AutoFit(trainData, label, validationData, settings:
                    new AutoFitSettings()
                    {
                        StoppingCriteria = new ExperimentStoppingCriteria()
                        {
                            //default need to have a way to override
                            TimeOutInMinutes = 10
                        }
                    });
                result = result.OrderByDescending(t => t.Metrics.RSquared);
                var bestIteration = result.FirstOrDefault();
                pipelineToDeconstruct = bestIteration.Pipeline;
                model = bestIteration.Model;
            }

            if (options.MlTask == TaskKind.Regression)
            {
                throw new NotImplementedException();
            }
            //Multi-class exploration here

            return (pipelineToDeconstruct, model);
        }

        private static void RunCodeGen(Options options, ColumnInferenceResult columnInference, Pipeline pipelineToDeconstruct)
        {
            var codeGenerator = new CodeGenerator(pipelineToDeconstruct, columnInference);
            MLCodeGen codeGen = new MLCodeGen()
            {
                Path = options.TrainDataset.FullName,
                TestPath = options.TestDataset.FullName,
                Columns = codeGenerator.GenerateColumns(),
                Transforms = codeGenerator.GenerateTransforms(),
                HasHeader = columnInference.HasHeader,
                Separator = columnInference.Separator,
                Trainer = codeGenerator.GenerateTrainer(),
                TaskType = options.MlTask.ToString(),
                ClassLabels = codeGenerator.GenerateClassLabels(),
                GeneratedUsings = codeGenerator.GenerateUsings()
            };

            MLProjectGen csProjGenerator = new MLProjectGen();
            ConsoleHelper consoleHelper = new ConsoleHelper();
            var trainScoreCode = codeGen.TransformText();
            var projectSourceCode = csProjGenerator.TransformText();
            var consoleHelperCode = consoleHelper.TransformText();
            if (!Directory.Exists("./BestModel"))
            {
                Directory.CreateDirectory("./BestModel");
            }
            File.WriteAllText("./BestModel/Train.cs", trainScoreCode);
            File.WriteAllText("./BestModel/MyML.csproj", projectSourceCode);
            File.WriteAllText("./BestModel/ConsoleHelper.cs", consoleHelperCode);
        }

    }
}
