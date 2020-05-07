// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Microsoft.ML.Transforms.TimeSeries;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class TimeSeriesEstimatorTests : TestDataPipeBase
    {
        private const int InputSize = 150528;

        private class Data
        {
            public float Value;

            public Data(float value)
            {
                Value = value;
            }
        }

        private class TestDataXY
        {
            [VectorType(InputSize)]
            public float[] A;
        }
        private class TestDataDifferntType
        {
            [VectorType(InputSize)]
            public string[] data_0;
        }

        public TimeSeriesEstimatorTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TestSsaChangePointEstimator()
        {
            int confidence = 95;
            int changeHistorySize = 10;
            int seasonalitySize = 10;
            int numberOfSeasonsInTraining = 5;
            int maxTrainingSize = numberOfSeasonsInTraining * seasonalitySize;

            List<Data> data = new List<Data>();
            var dataView = ML.Data.LoadFromEnumerable(data);

            for (int j = 0; j < numberOfSeasonsInTraining; j++)
                for (int i = 0; i < seasonalitySize; i++)
                    data.Add(new Data(i));

            for (int i = 0; i < changeHistorySize; i++)
                data.Add(new Data(i * 100));

            var pipe = new SsaChangePointEstimator(Env, "Change", 
                confidence, changeHistorySize, maxTrainingSize, seasonalitySize, "Value");

            var xyData = new List<TestDataXY> { new TestDataXY() { A = new float[InputSize] } };
            var stringData = new List<TestDataDifferntType> { new TestDataDifferntType() { data_0 = new string[InputSize] } };

            var invalidDataWrongNames = ML.Data.LoadFromEnumerable(xyData);
            var invalidDataWrongTypes = ML.Data.LoadFromEnumerable(stringData);

            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongTypes);
            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongNames);

            Done();
        }

        [Fact]
        public void TestSsaForecastingEstimator()
        {
            const int changeHistorySize = 10;
            const int seasonalitySize = 10;
            const int numberOfSeasonsInTraining = 5;

            List<Data> data = new List<Data>();

            var ml = new MLContext(seed: 1);
            var dataView = ml.Data.LoadFromEnumerable(data);

            for (int j = 0; j < numberOfSeasonsInTraining; j++)
                for (int i = 0; i < seasonalitySize; i++)
                    data.Add(new Data(i));

            for (int i = 0; i < changeHistorySize; i++)
                data.Add(new Data(i * 100));

            // Train
            var pipe = new SsaForecastingEstimator(Env, "Forecast", "Value", 10, 11, 22, 4,
                    confidenceLowerBoundColumn: "ConfidenceLowerBound",
                    confidenceUpperBoundColumn: "ConfidenceUpperBound");

            var xyData = new List<TestDataXY> { new TestDataXY() { A = new float[InputSize] } };
            var stringData = new List<TestDataDifferntType> { new TestDataDifferntType() { data_0 = new string[InputSize] } };

            var invalidDataWrongNames = ML.Data.LoadFromEnumerable(xyData);
            var invalidDataWrongTypes = ML.Data.LoadFromEnumerable(stringData);

            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongTypes);
            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongNames);

            Done();
        }

        [Fact]
        public void TestSsaSpikeEstimator()
        {
            int confidence = 95;
            int pValueHistorySize = 10;
            int seasonalitySize = 10;
            int numberOfSeasonsInTraining = 5;
            int maxTrainingSize = numberOfSeasonsInTraining * seasonalitySize;

            List<Data> data = new List<Data>();
            var dataView = ML.Data.LoadFromEnumerable(data);

            for (int j = 0; j < numberOfSeasonsInTraining; j++)
                for (int i = 0; i < seasonalitySize; i++)
                    data.Add(new Data(i));

            for (int i = 0; i < pValueHistorySize; i++)
                data.Add(new Data(i * 100));

            var pipe = new SsaSpikeEstimator(Env, "Change",
                confidence, pValueHistorySize, maxTrainingSize, seasonalitySize, "Value");

            var xyData = new List<TestDataXY> { new TestDataXY() { A = new float[InputSize] } };
            var stringData = new List<TestDataDifferntType> { new TestDataDifferntType() { data_0 = new string[InputSize] } };

            var invalidDataWrongNames = ML.Data.LoadFromEnumerable(xyData);
            var invalidDataWrongTypes = ML.Data.LoadFromEnumerable(stringData);

            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongTypes);
            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongNames);

            Done();
        }

        [Fact]
        public void TestIidChangePointEstimator()
        {
            int confidence = 95;
            int changeHistorySize = 10;

            List<Data> data = new List<Data>();
            var dataView = ML.Data.LoadFromEnumerable(data);

            for (int i = 0; i < changeHistorySize; i++)
                data.Add(new Data(i * 100));

            var pipe = new IidChangePointEstimator(Env,
                "Change", confidence, changeHistorySize, "Value");

            var xyData = new List<TestDataXY> { new TestDataXY() { A = new float[InputSize] } };
            var stringData = new List<TestDataDifferntType> { new TestDataDifferntType() { data_0 = new string[InputSize] } };

            var invalidDataWrongNames = ML.Data.LoadFromEnumerable(xyData);
            var invalidDataWrongTypes = ML.Data.LoadFromEnumerable(stringData);

            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongTypes);
            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongNames);

            Done();
        }

        [Theory]
        [IterationData(iterations: 20)]
        [Trait("Category", "RunSpecificTest")]
        public void CompleteTestIidSpikeEstimator(int iterations)
        {
            Output.WriteLine($"{iterations} - th");

            int timeout = 20 * 60 * 1000;

            var runTask = Task.Run(TestIidSpikeEstimator);
            var timeoutTask = Task.Delay(timeout + iterations);
            var finishedTask = Task.WhenAny(timeoutTask, runTask).Result;
            if (finishedTask == timeoutTask)
            {
                Console.WriteLine("TestIidSpikeEstimator test Hanging: fail to complete in 20 minutes");
                Environment.FailFast("Fail here to take memory dump");
            }
        }

        [Fact]
        public void TestIidSpikeEstimator()
        {
            int confidence = 95;
            int pValueHistorySize = 10;

            List<Data> data = new List<Data>();
            var dataView = ML.Data.LoadFromEnumerable(data);

            for (int i = 0; i < pValueHistorySize; i++)
                data.Add(new Data(i * 100));

            var pipe = new IidSpikeEstimator(Env,
                "Change", confidence, pValueHistorySize, "Value");

            var xyData = new List<TestDataXY> { new TestDataXY() { A = new float[InputSize] } };
            var stringData = new List<TestDataDifferntType> { new TestDataDifferntType() { data_0 = new string[InputSize] } };

            var invalidDataWrongNames = ML.Data.LoadFromEnumerable(xyData);
            var invalidDataWrongTypes = ML.Data.LoadFromEnumerable(stringData);

            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongTypes);
            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongNames);

            Done();
        }
    }
}
