// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// This file contains code examples that currently do not even compile. 
// They serve as the reference point of the 'desired user-facing API' for the future work.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public class AspirationalExamples
    {
        public class IrisPrediction
        {
            public string PredictedLabel;
        }

        public class IrisExample
        {
            public float SepalWidth { get; set; }
            public float SepalLength { get; set; }
            public float PetalWidth { get; set; }
            public float PetalLength { get; set; }
        }

        public void FirstExperienceWithML()
        {
            // Load the data into the system.
            string dataPath = "iris-data.txt";
            var data = TextReader.FitAndRead(env, dataPath, row => (
                Label: row.ReadString(0),
                SepalWidth: row.ReadFloat(1),
                SepalLength: row.ReadFloat(2),
                PetalWidth: row.ReadFloat(3),
                PetalLength: row.ReadFloat(4)));


            var preprocess = data.Schema.MakeEstimator(row => (
                // Convert string label to key.
                Label: row.Label.DictionarizeLabel(),
                // Concatenate all features into a vector.
                Features: row.SepalWidth.ConcatWith(row.SepalLength, row.PetalWidth, row.PetalLength)));

            var pipeline = preprocess
                // Append the trainer to the training pipeline.
                .AppendEstimator(row => row.Label.PredictWithSdca(row.Features))
                .AppendEstimator(row => row.PredictedLabel.KeyToValue());

            // Train the model and make some predictions.
            var model = pipeline.Fit<IrisExample, IrisPrediction>(data);

            IrisPrediction prediction = model.Predict(new IrisExample
            {
                SepalWidth = 3.3f,
                SepalLength = 1.6f,
                PetalWidth = 0.2f,
                PetalLength = 5.1f
            });
        }
    }
}
