// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Transforms.TimeSeries;
using Xunit;

namespace Microsoft.ML.Tests
{
    public sealed class TimeSeries
    {

        private sealed class Prediction
        {
#pragma warning disable CS0649
            [VectorType(4)]
            public double[] Change;
#pragma warning restore CS0649
        }

        public class Prediction1
        {
            public float Random;
        }

        private sealed class Data
        {
            public string Text;
            public float Random;
            public float Value;

            public Data(float value)
            {
                Text = "random123value";
                Random = -1;
                Value = value;
            }
        }

        [Fact]
        public void ChangeDetection()
        {
            var env = new MLContext();
            const int size = 10;
            List<Data> data = new List<Data>(size);
            var dataView = env.Data.LoadFromEnumerable(data);
            for (int i = 0; i < size / 2; i++)
                data.Add(new Data(5));

            for (int i = 0; i < size / 2; i++)
                data.Add(new Data((float)(5 + i * 1.1)));

            var args = new IidChangePointDetector.Options()
            {
                Confidence = 80,
                Source = "Value",
                Name = "Change",
                ChangeHistoryLength = size
            };
            // Train
            var detector = new IidChangePointEstimator(env, args).Fit(dataView);
            // Transform
            var output = detector.Transform(dataView);
            // Get predictions
            var enumerator = env.Data.CreateEnumerable<Prediction>(output, true).GetEnumerator();
            Prediction row = null;
            List<double> expectedValues = new List<double>() { 0, 5, 0.5, 5.1200000000000114E-08, 0, 5, 0.4999999995, 5.1200000046080209E-08, 0, 5, 0.4999999995, 5.1200000092160303E-08,
                0, 5, 0.4999999995, 5.12000001382404E-08};
            int index = 0;
            while (enumerator.MoveNext() && index < expectedValues.Count)
            {
                row = enumerator.Current;

                Assert.Equal(expectedValues[index++], row.Change[0]);
                Assert.Equal(expectedValues[index++], row.Change[1]);
                Assert.Equal(expectedValues[index++], row.Change[2]);
                Assert.Equal(expectedValues[index++], row.Change[3]);
            }
        }

        [LessThanNetCore30OrNotNetCoreFact("netcoreapp3.0 output differs from Baseline")]
        public void ChangePointDetectionWithSeasonality()
        {
            var env = new MLContext();
            const int ChangeHistorySize = 10;
            const int SeasonalitySize = 10;
            const int NumberOfSeasonsInTraining = 5;
            const int MaxTrainingSize = NumberOfSeasonsInTraining * SeasonalitySize;

            List<Data> data = new List<Data>();
            var dataView = env.Data.LoadFromEnumerable(data);

            var args = new SsaChangePointDetector.Options()
            {
                Confidence = 95,
                Source = "Value",
                Name = "Change",
                ChangeHistoryLength = ChangeHistorySize,
                TrainingWindowSize = MaxTrainingSize,
                SeasonalWindowSize = SeasonalitySize
            };

            for (int j = 0; j < NumberOfSeasonsInTraining; j++)
                for (int i = 0; i < SeasonalitySize; i++)
                    data.Add(new Data(i));

            for (int i = 0; i < ChangeHistorySize; i++)
                data.Add(new Data(i * 100));

            // Train
            var detector = new SsaChangePointEstimator(env, args).Fit(dataView);
            // Transform
            var output = detector.Transform(dataView);
            // Get predictions
            var enumerator = env.Data.CreateEnumerable<Prediction>(output, true).GetEnumerator();
            Prediction row = null;
            List<double> expectedValues = new List<double>() { 0, -3.31410598754883, 0.5, 5.12000000000001E-08, 0, 1.5700820684432983, 5.2001145245395008E-07,
                    0.012414560443710681, 0, 1.2854313254356384, 0.28810801662678009, 0.02038940454467935, 0, -1.0950627326965332, 0.36663890634019225, 0.026956459625565483};

            int index = 0;
            while (enumerator.MoveNext() && index < expectedValues.Count)
            {
                row = enumerator.Current;
                Assert.Equal(expectedValues[index++], row.Change[0], precision: 7);  // Alert
                Assert.Equal(expectedValues[index++], row.Change[1], precision: 7);  // Raw score
                Assert.Equal(expectedValues[index++], row.Change[2], precision: 7);  // P-Value score
                Assert.Equal(expectedValues[index++], row.Change[3], precision: 7);  // Martingale score
            }
        }

        [LessThanNetCore30OrNotNetCoreFact("netcoreapp3.0 output differs from Baseline")]
        public void ChangePointDetectionWithSeasonalityPredictionEngineNoColumn()
        {
            const int ChangeHistorySize = 10;
            const int SeasonalitySize = 10;
            const int NumberOfSeasonsInTraining = 5;
            const int MaxTrainingSize = NumberOfSeasonsInTraining * SeasonalitySize;

            List<Data> data = new List<Data>();

            var ml = new MLContext(seed: 1);
            var dataView = ml.Data.LoadFromEnumerable(data);

            for (int j = 0; j < NumberOfSeasonsInTraining; j++)
                for (int i = 0; i < SeasonalitySize; i++)
                    data.Add(new Data(i));

            for (int i = 0; i < ChangeHistorySize; i++)
                data.Add(new Data(i * 100));


            // Pipeline.
            var pipeline = ml.Transforms.Text.FeaturizeText("Text_Featurized", "Text")
                .Append(new SsaChangePointEstimator(ml, new SsaChangePointDetector.Options()
                {
                    Confidence = 95,
                    Source = "Value",
                    Name = "Change",
                    ChangeHistoryLength = ChangeHistorySize,
                    TrainingWindowSize = MaxTrainingSize,
                    SeasonalWindowSize = SeasonalitySize
                }));

            // Train.
            var model = pipeline.Fit(dataView);

            //Create prediction function.
            var engine = model.CreateTimeSeriesPredictionFunction<Data, Prediction1>(ml);

            //Checkpoint with no inputs passed at prediction.
            var modelPath = "temp.zip";
            engine.CheckPoint(ml, modelPath);

            //Load time series model and we will use this to pass two inputs and compare the raw score 
            //with "engine".
            ITransformer model2 = null;
            using (var file = File.OpenRead(modelPath))
                model2 = ml.Model.Load(file, out var schema);

            //Raw score after state gets updated with two inputs.
            var engine2 = model2.CreateTimeSeriesPredictionFunction<Data, Prediction>(ml);
            var prediction2 = engine2.Predict(new Data(1));
            //Raw score after first input.
            Assert.Equal(1.1661833524703979, prediction2.Change[1], precision: 5); // Raw score
            prediction2 = engine2.Predict(new Data(1));
            //Raw score after second input.
            Assert.Equal(0.12216401100158691, prediction2.Change[1], precision: 5); // Raw score

            //Even though time series column is not requested it will 
            // pass the observation through time series transform and update the state with the first input.
            var prediction = engine.Predict(new Data(1));
            Assert.Equal(-1, prediction.Random);

            //Save the model with state updated with just one input.
            engine.CheckPoint(ml, modelPath + 1);
            ITransformer model3 = null;
            using (var file = File.OpenRead(modelPath + 1))
                model3 = ml.Model.Load(file, out var schema);

            //Load the model with state updated with just one input, then pass in the second input
            //and raw score should match the raw score obtained by passing the two input in the first model.
            var engine3 = model3.CreateTimeSeriesPredictionFunction<Data, Prediction>(ml);
            var prediction3 = engine3.Predict(new Data(1));
            Assert.Equal(0.12216401100158691, prediction2.Change[1], precision: 5); // Raw score
        }

        [LessThanNetCore30OrNotNetCoreFact("netcoreapp3.0 output differs from Baseline")]
        public void ChangePointDetectionWithSeasonalityPredictionEngine()
        {
            const int ChangeHistorySize = 10;
            const int SeasonalitySize = 10;
            const int NumberOfSeasonsInTraining = 5;
            const int MaxTrainingSize = NumberOfSeasonsInTraining * SeasonalitySize;

            List<Data> data = new List<Data>();

            var ml = new MLContext(seed: 1);
            var dataView = ml.Data.LoadFromEnumerable(data);

            for (int j = 0; j < NumberOfSeasonsInTraining; j++)
                for (int i = 0; i < SeasonalitySize; i++)
                    data.Add(new Data(i));

            for (int i = 0; i < ChangeHistorySize; i++)
                data.Add(new Data(i * 100));


            // Pipeline.
            var pipeline = ml.Transforms.Text.FeaturizeText("Text_Featurized", "Text")
                .Append(new SsaChangePointEstimator(ml, new SsaChangePointDetector.Options()
                {
                    Confidence = 95,
                    Source = "Value",
                    Name = "Change",
                    ChangeHistoryLength = ChangeHistorySize,
                    TrainingWindowSize = MaxTrainingSize,
                    SeasonalWindowSize = SeasonalitySize
                }));

            // Train.
            var model = pipeline.Fit(dataView);
            
            //Model 1: Prediction #1.
            var engine = model.CreateTimeSeriesPredictionFunction<Data, Prediction>(ml);
            var prediction = engine.Predict(new Data(1));
            Assert.Equal(0, prediction.Change[0], precision: 7); // Alert
            Assert.Equal(1.1661833524703979, prediction.Change[1], precision: 5); // Raw score
            Assert.Equal(0.5, prediction.Change[2], precision: 7); // P-Value score
            Assert.Equal(5.1200000000000114E-08, prediction.Change[3], precision: 7); // Martingale score

            //Model 1: Checkpoint.
            var modelPath = "temp.zip";
            engine.CheckPoint(ml, modelPath);

            //Model 1: Prediction #2
            prediction = engine.Predict(new Data(1));
            Assert.Equal(0, prediction.Change[0], precision: 7); // Alert
            Assert.Equal(0.12216401100158691, prediction.Change[1], precision: 5); // Raw score
            Assert.Equal(0.14823824685192111, prediction.Change[2], precision: 5); // P-Value score
            Assert.Equal(1.5292508189989167E-07, prediction.Change[3], precision: 7); // Martingale score

            // Load Model 1.
            ITransformer model2 = null;
            using (var file = File.OpenRead(modelPath))
                model2 = ml.Model.Load(file, out var schema);

            //Predict and expect the same result after checkpointing(Prediction #2).
            engine = model2.CreateTimeSeriesPredictionFunction<Data, Prediction>(ml);
            prediction = engine.Predict(new Data(1));
            Assert.Equal(0, prediction.Change[0], precision: 7); // Alert
            Assert.Equal(0.12216401100158691, prediction.Change[1], precision: 5); // Raw score
            Assert.Equal(0.14823824685192111, prediction.Change[2], precision: 5); // P-Value score
            Assert.Equal(1.5292508189989167E-07, prediction.Change[3], precision: 5); // Martingale score
        }
    }
}
