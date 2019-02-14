// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML.Auto;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using mlnet;
using mlnet.Utilities;
using NLog;

namespace Microsoft.ML.CLI
{
    internal class NewCommand
    {
        private Options options;
        private static Logger logger = LogManager.GetCurrentClassLogger();

        internal NewCommand(Options options)
        {
            this.options = options;
        }

        internal void Run()
        {
            if (options.MlTask == TaskKind.MulticlassClassification)
            {
                Console.WriteLine($"{Strings.UnsupportedMlTask}: {options.MlTask}");
            }

            var context = new MLContext();

            //Check what overload method of InferColumns needs to be called.
            logger.Log(LogLevel.Info, Strings.InferColumns);
            (TextLoader.Arguments TextLoaderArgs, IEnumerable<(string Name, ColumnPurpose Purpose)> ColumnPurpopses) columnInference = default((TextLoader.Arguments TextLoaderArgs, IEnumerable<(string Name, ColumnPurpose Purpose)> ColumnPurpopses));
            if (options.LabelName != null)
            {
                columnInference = context.Data.InferColumns(options.TrainDataset.FullName, options.LabelName, groupColumns: false);
            }
            else
            {
                columnInference = context.Data.InferColumns(options.TrainDataset.FullName, options.LabelIndex, groupColumns: false);
            }

            logger.Log(LogLevel.Info, Strings.CreateDataLoader);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderArgs);

            logger.Log(LogLevel.Info, Strings.LoadData);
            IDataView trainData = textLoader.Read(options.TrainDataset.FullName);
            IDataView validationData = options.ValidationDataset == null ? null : textLoader.Read(options.ValidationDataset.FullName);

            //Explore the models
            (Pipeline, ITransformer) result = default;
            Console.WriteLine($"{Strings.ExplorePipeline}: {options.MlTask}");
            try
            {
                result = ExploreModels(context, trainData, validationData);
            }
            catch (Exception e)
            {
                logger.Log(LogLevel.Error, $"{Strings.ExplorePipelineException}:");
                logger.Log(LogLevel.Error, e.StackTrace);
                logger.Log(LogLevel.Error, Strings.Exiting);
                return;
            }

            //Get the best pipeline
            Pipeline pipeline = null;
            pipeline = result.Item1;
            var model = result.Item2;

            //Save the model
            logger.Log(LogLevel.Info, Strings.SavingBestModel);
            var modelPath = Path.Combine(@options.OutputBaseDir, options.OutputName);
            SaveModel(model, modelPath, $"{options.OutputName}_model.zip", context);


            //Generate code
            logger.Log(LogLevel.Info, Strings.GenerateProject);
            var codeGenerator = new CodeGenerator(
                pipeline,
                columnInference,
                new CodeGeneratorOptions()
                {
                    TrainDataset = options.TrainDataset,
                    MlTask = options.MlTask,
                    TestDataset = options.TestDataset,
                    OutputName = options.OutputName,
                    OutputBaseDir = options.OutputBaseDir
                });
            codeGenerator.GenerateOutput();
        }

        private (Pipeline, ITransformer) ExploreModels(
            MLContext context,
            IDataView trainData,
            IDataView validationData)
        {
            ITransformer model = null;
            string label = options.LabelName ?? "Label"; // It is guaranteed training dataview to have Label column
            Pipeline pipeline = null;

            if (options.MlTask == TaskKind.BinaryClassification)
            {
                var progressReporter = new ProgressHandlers.BinaryClassificationHandler();
                var result = context.BinaryClassification.AutoFit(trainData, label, validationData, options.Timeout, progressCallback: progressReporter);
                logger.Log(LogLevel.Info, Strings.RetrieveBestPipeline);
                var bestIteration = result.Best();
                pipeline = bestIteration.Pipeline;
                model = bestIteration.Model;
            }

            if (options.MlTask == TaskKind.Regression)
            {
                var progressReporter = new ProgressHandlers.RegressionHandler();
                var result = context.Regression.AutoFit(trainData, label, validationData, options.Timeout, progressCallback: progressReporter);
                logger.Log(LogLevel.Info, Strings.RetrieveBestPipeline);
                var bestIteration = result.Best();
                pipeline = bestIteration.Pipeline;
                model = bestIteration.Model;
            }

            if (options.MlTask == TaskKind.MulticlassClassification)
            {
                throw new NotImplementedException();
            }
            //Multi-class exploration here

            return (pipeline, model);
        }

        private static void SaveModel(ITransformer model, string ModelPath, string modelName, MLContext mlContext)
        {
            if (!Directory.Exists(ModelPath))
            {
                Directory.CreateDirectory(ModelPath);
            }
            ModelPath = Path.Combine(ModelPath, modelName);
            using (var fs = File.Create(ModelPath))
                model.SaveTo(mlContext, fs);
        }
    }
}
