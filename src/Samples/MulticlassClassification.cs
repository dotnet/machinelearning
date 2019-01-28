// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.Auto;

namespace Samples
{
    public class MulticlassClassification
    {
        public static void Run()
        {
            const string trainDataPath = @"C:\data\train.csv";
            const string validationDataPath = @"C:\data\valid.csv";
            const string testDataPath = @"C:\data\test.csv";
            const string label = "Label";

            var mlContext = new MLContext();

            // auto-load data from disk
            var trainData = mlContext.Data.AutoRead(trainDataPath, label);
            var validationData = mlContext.Data.AutoRead(validationDataPath, label);
            var testData = mlContext.Data.AutoRead(testDataPath, label);

            // run AutoML & train model
            var autoMlResult = mlContext.MulticlassClassification.AutoFit(trainData, "Label", validationData,
                settings: new AutoFitSettings()
                {
                    StoppingCriteria = new ExperimentStoppingCriteria() { MaxIterations = 10 }
                });
            // get best AutoML model
            var model = autoMlResult.BestPipeline.Model;

            // run AutoML on test data
            var transformedOutput = model.Transform(testData);
            var results = mlContext.BinaryClassification.Evaluate(transformedOutput);
            Console.WriteLine($"Model Accuracy: {results.Accuracy}\r\n");

            Console.ReadLine();
        }
    }
}
