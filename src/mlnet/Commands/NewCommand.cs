// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Auto;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace Microsoft.ML.CLI
{
    internal class NewCommand
    {
        private Options options;

        internal NewCommand(Options options)
        {
            this.options = options;
        }

        internal void Run()
        {
            if (options.MlTask == TaskKind.MulticlassClassification)
            {
                Console.WriteLine($"Unsupported ml-task: {options.MlTask}");
            }

            var context = new MLContext();

            //Check what overload method of InferColumns needs to be called.
            (TextLoader.Arguments TextLoaderArgs, IEnumerable<(string Name, ColumnPurpose Purpose)> ColumnPurpopses) columnInference = default((TextLoader.Arguments TextLoaderArgs, IEnumerable<(string Name, ColumnPurpose Purpose)> ColumnPurpopses));
            if (options.LabelName != null)
            {
                columnInference = context.Data.InferColumns(options.TrainDataset.FullName, options.LabelName, groupColumns: false);
            }
            else
            {
                columnInference = context.Data.InferColumns(options.TrainDataset.FullName, options.LabelIndex, groupColumns: false);
            }

            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderArgs);

            IDataView trainData = textLoader.Read(options.TrainDataset.FullName);
            IDataView validationData = options.ValidationDataset == null ? null : textLoader.Read(options.ValidationDataset.FullName);

            //Explore the models
            Pipeline pipeline = null;
            var result = ExploreModels(context, trainData, validationData, pipeline);

            //Get the best pipeline
            pipeline = result.Item1;
            var model = result.Item2;

            //Generate code
            var codeGenerator = new CodeGenerator(pipeline, columnInference, new CodeGeneratorOptions() { TrainDataset = options.TrainDataset, MlTask = options.MlTask, TestDataset = options.TestDataset });
            codeGenerator.GenerateOutput();

            //Save the model
            SaveModel(model, @"./BestModel", "model.zip", context);
        }

        private (Pipeline, ITransformer) ExploreModels(
            MLContext context,
            IDataView trainData,
            IDataView validationData,
            Pipeline pipeline)
        {
            ITransformer model = null;
            string label = options.LabelName ?? "Label"; // It is guaranteed training dataview to have Label column

            if (options.MlTask == TaskKind.BinaryClassification)
            {
                var result = context.BinaryClassification.AutoFit(trainData, label, validationData, options.Timeout);
                result = result.OrderByDescending(t => t.Metrics.Accuracy).ToList();
                var bestIteration = result.FirstOrDefault();
                pipeline = bestIteration.Pipeline;
                model = bestIteration.Model;
            }

            if (options.MlTask == TaskKind.Regression)
            {
                var result = context.Regression.AutoFit(trainData, label, validationData, options.Timeout);
                result = result.OrderByDescending(t => t.Metrics.RSquared).ToList();
                var bestIteration = result.FirstOrDefault();
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
            ModelPath = ModelPath + "/" + modelName;
            using (var fs = File.Create(ModelPath))
                model.SaveTo(mlContext, fs);
        }
    }
}
