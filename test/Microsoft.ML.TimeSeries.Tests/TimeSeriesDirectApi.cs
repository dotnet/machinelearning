// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.TimeSeriesProcessing;
using Microsoft.ML.TimeSeries;
using Xunit;

namespace Microsoft.ML.Tests
{
    public sealed class TimeSeries
    {

        public class Prediction
        {
            [VectorType(4)] public double[] Change;
        }

        public class Prediction1
        {
            public float Random;
        }

        sealed class Data
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
            using (var env = new ConsoleEnvironment(conc: 1))
            {
                const int size = 10;
                List<Data> data = new List<Data>(size);
                var dataView = env.CreateStreamingDataView(data);
                for (int i = 0; i < size / 2; i++)
                    data.Add(new Data(5));

                for (int i = 0; i < size / 2; i++)
                    data.Add(new Data((float)(5 + i * 1.1)));

                var args = new IidChangePointDetector.Arguments()
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
                var enumerator = output.AsEnumerable<Prediction>(env, true).GetEnumerator();
                Prediction row = null;
                List<double> expectedValues = new List<double>()
                {
                    0, 5, 0.5, 5.1200000000000114E-08, 0, 5, 0.4999999995, 5.1200000046080209E-08, 0, 5, 0.4999999995,
                    5.1200000092160303E-08,
                    0, 5, 0.4999999995, 5.12000001382404E-08
                };
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
        }

        [Fact]
        public void ChangePointDetectionWithSeasonality()
        {
            using (var env = new ConsoleEnvironment(conc: 1))
            {
                const int ChangeHistorySize = 10;
                const int SeasonalitySize = 10;
                const int NumberOfSeasonsInTraining = 5;
                const int MaxTrainingSize = NumberOfSeasonsInTraining * SeasonalitySize;

                List<Data> data = new List<Data>();
                var dataView = env.CreateStreamingDataView(data);

                var args = new SsaChangePointDetector.Arguments()
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
                var enumerator = output.AsEnumerable<Prediction>(env, true).GetEnumerator();
                Prediction row = null;
                List<double> expectedValues = new List<double>()
                {
                    0, -3.31410598754883, 0.5, 5.12000000000001E-08, 0, 1.5700820684432983, 5.2001145245395008E-07,
                    0.012414560443710681, 0, 1.2854313254356384, 0.28810801662678009, 0.02038940454467935, 0,
                    -1.0950627326965332, 0.36663890634019225, 0.026956459625565483
                };

                int index = 0;
                while (enumerator.MoveNext() && index < expectedValues.Count)
                {
                    row = enumerator.Current;
                    Assert.Equal(expectedValues[index++], row.Change[0], precision: 7); // Alert
                    Assert.Equal(expectedValues[index++], row.Change[1], precision: 7); // Raw score
                    Assert.Equal(expectedValues[index++], row.Change[2], precision: 7); // P-Value score
                    Assert.Equal(expectedValues[index++], row.Change[3], precision: 7); // Martingale score
                }
            }
        }

        [Fact]
        public void ChangePointDetectionWithSeasonalityPredictionEngineNoColumn()
        {
            const int ChangeHistorySize = 10;
            const int SeasonalitySize = 10;
            const int NumberOfSeasonsInTraining = 5;
            const int MaxTrainingSize = NumberOfSeasonsInTraining * SeasonalitySize;

            List<Data> data = new List<Data>();

            var ml = new MLContext(seed: 1, conc: 1);
            var dataView = ml.CreateStreamingDataView(data);

            for (int j = 0; j < NumberOfSeasonsInTraining; j++)
                for (int i = 0; i < SeasonalitySize; i++)
                    data.Add(new Data(i));

            for (int i = 0; i < ChangeHistorySize; i++)
                data.Add(new Data(i * 100));


            // Pipeline.
            var pipeline = ml.Transforms.Text.FeaturizeText("Text", "Text_Featurized")
                .Append(new SsaChangePointEstimator(ml, new SsaChangePointDetector.Arguments()
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

            //Predict.
            var engine = model.MakeTimeSeriesPredictionFunction<Data, Prediction1>(ml);
            //Even though time series column is not requested it will pass the observation through time series transform.
            var prediction = engine.Predict(new Data(1));
            Assert.Equal(-1, prediction.Random);
            prediction = engine.Predict(new Data(2));
            Assert.Equal(-1, prediction.Random);
        }

        [Fact]
        public void ChangePointDetectionWithSeasonalityPredictionEngine()
        {
            const int ChangeHistorySize = 10;
            const int SeasonalitySize = 10;
            const int NumberOfSeasonsInTraining = 5;
            const int MaxTrainingSize = NumberOfSeasonsInTraining * SeasonalitySize;

            List<Data> data = new List<Data>();

            var ml = new MLContext(seed: 1, conc: 1);
            var dataView = ml.CreateStreamingDataView(data);

            for (int j = 0; j < NumberOfSeasonsInTraining; j++)
                for (int i = 0; i < SeasonalitySize; i++)
                    data.Add(new Data(i));

            for (int i = 0; i < ChangeHistorySize; i++)
                data.Add(new Data(i * 100));


            // Pipeline.
            var pipeline = ml.Transforms.Text.FeaturizeText("Text", "Text_Featurized")
                .Append(new SsaChangePointEstimator(ml, new SsaChangePointDetector.Arguments()
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

            //Predict.
            var engine = model.MakeTimeSeriesPredictionFunction<Data, Prediction>(ml);
            var prediction = engine.Predict(new Data(1));
            Assert.Equal(0, prediction.Change[0], precision: 7); // Alert
            Assert.Equal(-2.3141059875488281, prediction.Change[1], precision: 7); // Raw score
            Assert.Equal(0.5, prediction.Change[2], precision: 7); // P-Value score
            Assert.Equal(5.1200000000000114E-08, prediction.Change[3], precision: 7); // Martingale score
        }
    }
}
