// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace Samples.Dynamic
{
    public static class DetectIidChangePoint
    {
        class ChangePointPrediction
        {
            [VectorType(4)]
            public double[] Prediction { get; set; }
        }

        class IidChangePointData
        {
            public float Value;

            public IidChangePointData(float value)
            {
                Value = value;
            }
        }

        // This example creates a time series (list of Data with the i-th element corresponding to the i-th time slot). 
        // The estimator is applied then to identify points where data distribution changed.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with a change
            const int Size = 16;
            var data = new List<IidChangePointData>(Size);
            for (int i = 0; i < Size / 2; i++)
                data.Add(new IidChangePointData(5));
            // This is a change point
            for (int i = 0; i < Size / 2; i++)
                data.Add(new IidChangePointData(7));

            // Convert data to IDataView.
            var dataView = ml.Data.LoadFromEnumerable(data);

            // Setup IidSpikeDetector arguments
            string outputColumnName = nameof(ChangePointPrediction.Prediction);
            string inputColumnName = nameof(IidChangePointData.Value);

            // Time Series model.
            ITransformer model = ml.Transforms.DetectIidChangePoint(outputColumnName, inputColumnName, 95, Size / 4).Fit(dataView);

            // Create a time series prediction engine from the model.
            var engine = model.CreateTimeSeriesPredictionFunction<IidChangePointData, ChangePointPrediction>(ml);
            for (int index = 0; index < 8; index++)
            {
                // Anomaly change point detection.
                var prediction = engine.Predict(new IidChangePointData(5));
                Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}\t{4:0.00}", 5, prediction.Prediction[0],
                    prediction.Prediction[1], prediction.Prediction[2], prediction.Prediction[3]);
            }

            // Change point
            var changePointPrediction = engine.Predict(new IidChangePointData(7));
            Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}\t{4:0.00}", 7, changePointPrediction.Prediction[0],
                changePointPrediction.Prediction[1], changePointPrediction.Prediction[2], changePointPrediction.Prediction[3]);

            // Checkpoint the model.
            var modelPath = "temp.zip";
            engine.CheckPoint(ml, modelPath);

            // Reference to current time series engine because in the next step "engine" will point to the
            // checkpointed model being loaded from disk.
            var timeseries1 = engine;

            // Load the model.
            using (var file = File.OpenRead(modelPath))
                model = ml.Model.Load(file, out DataViewSchema schema);

            // Create a time series prediction engine from the checkpointed model.
            engine = model.CreateTimeSeriesPredictionFunction<IidChangePointData, ChangePointPrediction>(ml);
            for (int index = 0; index < 8; index++)
            {
                // Anomaly change point detection.
                var prediction = engine.Predict(new IidChangePointData(7));
                Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}\t{4:0.00}", 7, prediction.Prediction[0],
                    prediction.Prediction[1], prediction.Prediction[2], prediction.Prediction[3]);
            }

            // Prediction from the original time series engine should match the prediction from 
            // check pointed model.
            engine = timeseries1;
            for (int index = 0; index < 8; index++)
            {
                // Anomaly change point detection.
                var prediction = engine.Predict(new IidChangePointData(7));
                Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}\t{4:0.00}", 7, prediction.Prediction[0],
                    prediction.Prediction[1], prediction.Prediction[2], prediction.Prediction[3]);
            }

            // Data Alert      Score   P-Value Martingale value
            // 5       0       5.00    0.50    0.00       <-- Time Series 1.
            // 5       0       5.00    0.50    0.00
            // 5       0       5.00    0.50    0.00
            // 5       0       5.00    0.50    0.00
            // 5       0       5.00    0.50    0.00
            // 5       0       5.00    0.50    0.00
            // 5       0       5.00    0.50    0.00
            // 5       0       5.00    0.50    0.00
            // 7       1       7.00    0.00    10298.67   <-- alert is on, predicted changepoint (and model is checkpointed).

            // 7       0       7.00    0.13    33950.16   <-- Time Series 2 : Model loaded back from disk and prediction is made.
            // 7       0       7.00    0.26    60866.34
            // 7       0       7.00    0.38    78362.04
            // 7       0       7.00    0.50    0.01
            // 7       0       7.00    0.50    0.00
            // 7       0       7.00    0.50    0.00
            // 7       0       7.00    0.50    0.00

            // 7       0       7.00    0.13    33950.16   <-- Time Series 1 and prediction is made.
            // 7       0       7.00    0.26    60866.34
            // 7       0       7.00    0.38    78362.04
            // 7       0       7.00    0.50    0.01
            // 7       0       7.00    0.50    0.00
            // 7       0       7.00    0.50    0.00
            // 7       0       7.00    0.50    0.00
        }
    }
}
