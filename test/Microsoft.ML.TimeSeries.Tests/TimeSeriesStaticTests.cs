// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.StaticPipe;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public sealed class TimeSeriesStaticTests : BaseTestBaseline
    {
        public TimeSeriesStaticTests(ITestOutputHelper output) : base(output)
        {
        }
#pragma warning disable CS0649 // Ignore unintialized field warning
        private sealed class ChangePointPrediction
        {
            // Note that this field must be named "Data"; we ultimately convert
            // to a dynamic IDataView in order to extract AsEnumerable
            // predictions and that process uses "Data" as the default column
            // name for an output column from a static pipeline.
            [VectorType(4)]
            public double[] Data;
        }

        private sealed class SpikePrediction
        {
            [VectorType(3)]
            public double[] Data;
        }
#pragma warning restore CS0649

        private sealed class Data
        {
            public float Value;

            public Data(float value) => Value = value;
        }

        [Fact]
        public void ChangeDetection()
        {
            var env = new MLContext();
            const int Size = 10;
            var data = new List<Data>(Size);
            var dataView = env.Data.LoadFromEnumerable(data);
            for (int i = 0; i < Size / 2; i++)
                data.Add(new Data(5));

            for (int i = 0; i < Size / 2; i++)
                data.Add(new Data((float)(5 + i * 1.1)));

            // Convert to statically-typed data view.
            var staticData = dataView.AssertStatic(env, c => new { Value = c.R4.Scalar });
            // Build the pipeline
            var staticLearningPipeline = staticData.MakeNewEstimator()
                .Append(r => r.Value.DetectIidChangePoint(80, Size));
            // Train
            var detector = staticLearningPipeline.Fit(staticData);
            // Transform
            var output = detector.Transform(staticData);

            // Get predictions
            var enumerator = env.Data.CreateEnumerable<ChangePointPrediction>(output.AsDynamic, true).GetEnumerator();
            ChangePointPrediction row = null;
            List<double> expectedValues = new List<double>() { 0, 5, 0.5, 5.1200000000000114E-08, 0, 5, 0.4999999995, 5.1200000046080209E-08, 0, 5, 0.4999999995, 5.1200000092160303E-08,
                0, 5, 0.4999999995, 5.12000001382404E-08};
            int index = 0;
            while (enumerator.MoveNext() && index < expectedValues.Count)
            {
                row = enumerator.Current;

                Assert.Equal(expectedValues[index++], row.Data[0], precision: 7);
                Assert.Equal(expectedValues[index++], row.Data[1], precision: 7);
                Assert.Equal(expectedValues[index++], row.Data[2], precision: 7);
                Assert.Equal(expectedValues[index++], row.Data[3], precision: 7);
            }
        }

        [Fact]
        public void ChangePointDetectionWithSeasonality()
        {
            var env = new MLContext();
            const int ChangeHistorySize = 10;
            const int SeasonalitySize = 10;
            const int NumberOfSeasonsInTraining = 5;
            const int MaxTrainingSize = NumberOfSeasonsInTraining * SeasonalitySize;

            var data = new List<Data>();
            var dataView = env.Data.LoadFromEnumerable(data);

            for (int j = 0; j < NumberOfSeasonsInTraining; j++)
                for (int i = 0; i < SeasonalitySize; i++)
                    data.Add(new Data(i));

            for (int i = 0; i < ChangeHistorySize; i++)
                data.Add(new Data(i * 100));

            // Convert to statically-typed data view.
            var staticData = dataView.AssertStatic(env, c => new { Value = c.R4.Scalar });
            // Build the pipeline
            var staticLearningPipeline = staticData.MakeNewEstimator()
                .Append(r => r.Value.DetectChangePointBySsa(95, ChangeHistorySize, MaxTrainingSize, SeasonalitySize));
            // Train
            var detector = staticLearningPipeline.Fit(staticData);
            // Transform
            var output = detector.Transform(staticData);

            // Get predictions
            var enumerator = env.Data.CreateEnumerable<ChangePointPrediction>(output.AsDynamic, true).GetEnumerator();
            ChangePointPrediction row = null;
            List<double> expectedValues = new List<double>() { 0, -3.31410598754883, 0.5, 5.12000000000001E-08, 0, 1.5700820684432983, 5.2001145245395008E-07,
                    0.012414560443710681, 0, 1.2854313254356384, 0.28810801662678009, 0.02038940454467935, 0, -1.0950627326965332, 0.36663890634019225, 0.026956459625565483};

            int index = 0;
            while (enumerator.MoveNext() && index < expectedValues.Count)
            {
                row = enumerator.Current;

                CompareNumbersWithTolerance(expectedValues[index++], row.Data[0], digitsOfPrecision: 5);  // Alert
                CompareNumbersWithTolerance(expectedValues[index++], row.Data[1], digitsOfPrecision: 5);  // Raw score
                CompareNumbersWithTolerance(expectedValues[index++], row.Data[2], digitsOfPrecision: 5);  // P-Value score
                CompareNumbersWithTolerance(expectedValues[index++], row.Data[3], digitsOfPrecision: 5);  // Martingale score
            }
        }

        [Fact]
        public void SpikeDetection()
        {
            var env = new MLContext();
            const int Size = 10;
            const int PvalHistoryLength = Size / 4;

            // Generate sample series data with a spike
            List<Data> data = new List<Data>(Size);
            var dataView = env.Data.LoadFromEnumerable(data);
            for (int i = 0; i < Size / 2; i++)
                data.Add(new Data(5));
            data.Add(new Data(10)); // This is the spike
            for (int i = 0; i < Size / 2 - 1; i++)
                data.Add(new Data(5));

            // Convert to statically-typed data view.
            var staticData = dataView.AssertStatic(env, c => new { Value = c.R4.Scalar });
            // Build the pipeline
            var staticLearningPipeline = staticData.MakeNewEstimator()
                .Append(r => r.Value.DetectIidSpike(80, PvalHistoryLength));
            // Train
            var detector = staticLearningPipeline.Fit(staticData);
            // Transform
            var output = detector.Transform(staticData);

            // Get predictions
            var enumerator = env.Data.CreateEnumerable<SpikePrediction>(output.AsDynamic, true).GetEnumerator();
            var expectedValues = new List<double[]>() {
                //            Alert   Score   P-Value
                new double[] {0,      5,      0.5},
                new double[] {0,      5,      0.5},
                new double[] {0,      5,      0.5},
                new double[] {0,      5,      0.5},
                new double[] {0,      5,      0.5},
                new double[] {1,      10,     0.0},     // alert is on, predicted spike
                new double[] {0,      5,      0.261375},
                new double[] {0,      5,      0.261375},
                new double[] {0,      5,      0.50},
                new double[] {0,      5,      0.50}
            };

            SpikePrediction row = null;
            for (var i = 0; enumerator.MoveNext() && i < expectedValues.Count; i++)
            {
                row = enumerator.Current;

                CompareNumbersWithTolerance(expectedValues[i][0], row.Data[0], digitsOfPrecision: 7);
                CompareNumbersWithTolerance(expectedValues[i][1], row.Data[1], digitsOfPrecision: 7);
                CompareNumbersWithTolerance(expectedValues[i][2], row.Data[2], digitsOfPrecision: 7);
            }
        }

        [Fact]
        public void SsaSpikeDetection()
        {
            var env = new MLContext();
            const int Size = 16;
            const int ChangeHistoryLength = Size / 4;
            const int TrainingWindowSize = Size / 2;
            const int SeasonalityWindowSize = Size / 8;

            // Generate sample series data with a spike
            List<Data> data = new List<Data>(Size);
            var dataView = env.Data.LoadFromEnumerable(data);
            for (int i = 0; i < Size / 2; i++)
                data.Add(new Data(5));
            data.Add(new Data(10)); // This is the spike
            for (int i = 0; i < Size / 2 - 1; i++)
                data.Add(new Data(5));

            // Convert to statically-typed data view.
            var staticData = dataView.AssertStatic(env, c => new { Value = c.R4.Scalar });
            // Build the pipeline
            var staticLearningPipeline = staticData.MakeNewEstimator()
                .Append(r => r.Value.DetectSpikeBySsa(80, ChangeHistoryLength, TrainingWindowSize, SeasonalityWindowSize));
            // Train
            var detector = staticLearningPipeline.Fit(staticData);
            // Transform
            var output = detector.Transform(staticData);

            // Get predictions
            var enumerator = env.Data.CreateEnumerable<SpikePrediction>(output.AsDynamic, true).GetEnumerator();
            var expectedValues = new List<double[]>() {
                //            Alert   Score   P-Value
                new double[] {0,      0.0,    0.5},
                new double[] {0,      0.0,    0.5},
                new double[] {0,      0.0,    0.5},
                new double[] {0,      0.0,    0.5},
                new double[] {0,      0.0,    0.5},
                new double[] {0,      0.0,    0.5},
                new double[] {0,      0.0,    0.5},
                new double[] {0,      0.0,    0.5},
                new double[] {1,      5.0,    0.0},     // alert is on, predicted spike
                new double[] {1,     -2.5,    0.093146},
                new double[] {0,     -2.5,    0.215437},
                new double[] {0,      0.0,    0.465745},
                new double[] {0,      0.0,    0.465745},
                new double[] {0,      0.0,    0.261375},
                new double[] {0,      0.0,    0.377615},
                new double[] {0,      0.0,    0.50}
            };

            SpikePrediction row = null;
            for (var i = 0; enumerator.MoveNext() && i < expectedValues.Count; i++)
            {
                row = enumerator.Current;

                CompareNumbersWithTolerance(expectedValues[i][0], row.Data[0], digitsOfPrecision: 6);
                CompareNumbersWithTolerance(expectedValues[i][1], row.Data[1], digitsOfPrecision: 6);
                CompareNumbersWithTolerance(expectedValues[i][2], row.Data[2], digitsOfPrecision: 6);
            }
        }
    }
}
