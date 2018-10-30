// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.TimeSeriesProcessing;
using Xunit;

namespace Microsoft.ML.Tests
{
    public sealed class TimeSeries
    {

        public class Prediction
        {
            [VectorType(4)]
            public double[] Change;
        }

        sealed class Data
        {
            public float Value;

            public Data(float value)
            {
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
                List<Data> tempData = new List<Data>();
                for (int i = 0; i < size / 2; i++)
                    tempData.Add(new Data(5));

                for (int i = 0; i < size / 2; i++)
                    tempData.Add(new Data((float)(5 + i * 1.1)));

                foreach (var d in tempData)
                    data.Add(new Data(d.Value));

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
        }

        [Fact]
        public void ChangePointDetectionWithSeasonality()
        {
            using (var env = new ConsoleEnvironment(conc: 1))
            {
                const int ChangeHistorySize = 2000;
                const int SeasonalitySize = 1000;
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
                List<double> expectedValues = new List<double>() { 0, -341.27374267578125, 0.5, 0, 0, 143.96232604980469, 0,
                    0, 0, 142.27732849121094, 0.26294916025125559, 0, 0, -84.966033935546875, 0.39567696623630655, 0};

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
        }
    }
}
