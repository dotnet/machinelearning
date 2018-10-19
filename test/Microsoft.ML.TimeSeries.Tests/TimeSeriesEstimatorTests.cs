// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.TimeSeriesProcessing;
using System.Collections.Generic;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class TimeSeriesEstimatorTests : TestDataPipeBase
    {
        private const int inputSize = 150528;

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
            [VectorType(inputSize)]
            public float[] A;
        }
        private class TestDataDifferntType
        {
            [VectorType(inputSize)]
            public string[] data_0;
        }

        public TimeSeriesEstimatorTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        void TestSsaChangePointEstimator()
        {
            int Confidence = 95;
            int ChangeHistorySize = 2000;
            int SeasonalitySize = 1000;
            int NumberOfSeasonsInTraining = 5;
            int MaxTrainingSize = NumberOfSeasonsInTraining * SeasonalitySize;

            List<Data> data = new List<Data>();
            var dataView = Env.CreateStreamingDataView(data);

            for (int j = 0; j < NumberOfSeasonsInTraining; j++)
                for (int i = 0; i < SeasonalitySize; i++)
                    data.Add(new Data(i));

            for (int i = 0; i < ChangeHistorySize; i++)
                data.Add(new Data(i * 100));

            var pipe = new SsaChangePointEstimator(Env, "Change", "Value",
                Confidence, ChangeHistorySize, MaxTrainingSize, SeasonalitySize);

            var xyData = new List<TestDataXY> { new TestDataXY() { A = new float[inputSize] } };
            var stringData = new List<TestDataDifferntType> { new TestDataDifferntType() { data_0 = new string[inputSize] } };

            var invalidDataWrongNames = ComponentCreation.CreateDataView(Env, xyData);
            var invalidDataWrongTypes = ComponentCreation.CreateDataView(Env, stringData);

            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongTypes);
            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongNames);
        }

        [Fact]
        void TestSsaSpikeEstimator()
        {
            int Confidence = 95;
            int PValueHistorySize = 2000;
            int SeasonalitySize = 1000;
            int NumberOfSeasonsInTraining = 5;
            int MaxTrainingSize = NumberOfSeasonsInTraining * SeasonalitySize;

            List<Data> data = new List<Data>();
            var dataView = Env.CreateStreamingDataView(data);

            for (int j = 0; j < NumberOfSeasonsInTraining; j++)
                for (int i = 0; i < SeasonalitySize; i++)
                    data.Add(new Data(i));

            for (int i = 0; i < PValueHistorySize; i++)
                data.Add(new Data(i * 100));

            var pipe = new SsaSpikeEstimator(Env, "Change", "Value",
                Confidence, PValueHistorySize, MaxTrainingSize, SeasonalitySize);

            var xyData = new List<TestDataXY> { new TestDataXY() { A = new float[inputSize] } };
            var stringData = new List<TestDataDifferntType> { new TestDataDifferntType() { data_0 = new string[inputSize] } };

            var invalidDataWrongNames = ComponentCreation.CreateDataView(Env, xyData);
            var invalidDataWrongTypes = ComponentCreation.CreateDataView(Env, stringData);

            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongTypes);
            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongNames);
        }

        [Fact]
        void TestIidChangePointEstimator()
        {
            int Confidence = 95;
            int ChangeHistorySize = 2000;

            List<Data> data = new List<Data>();
            var dataView = Env.CreateStreamingDataView(data);

            for (int i = 0; i < ChangeHistorySize; i++)
                data.Add(new Data(i * 100));

            var pipe = new IidChangePointEstimator(Env,
                "Change", "Value", Confidence, ChangeHistorySize);

            var xyData = new List<TestDataXY> { new TestDataXY() { A = new float[inputSize] } };
            var stringData = new List<TestDataDifferntType> { new TestDataDifferntType() { data_0 = new string[inputSize] } };

            var invalidDataWrongNames = ComponentCreation.CreateDataView(Env, xyData);
            var invalidDataWrongTypes = ComponentCreation.CreateDataView(Env, stringData);

            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongTypes);
            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongNames);
        }

        [Fact]
        void TestIidSpikeEstimator()
        {
            int Confidence = 95;
            int PValueHistorySize = 2000;

            List<Data> data = new List<Data>();
            var dataView = Env.CreateStreamingDataView(data);

            for (int i = 0; i < PValueHistorySize; i++)
                data.Add(new Data(i * 100));

            var pipe = new IidSpikeEstimator(Env, 
                "Change", "Value", Confidence, PValueHistorySize);

            var xyData = new List<TestDataXY> { new TestDataXY() { A = new float[inputSize] } };
            var stringData = new List<TestDataDifferntType> { new TestDataDifferntType() { data_0 = new string[inputSize] } };

            var invalidDataWrongNames = ComponentCreation.CreateDataView(Env, xyData);
            var invalidDataWrongTypes = ComponentCreation.CreateDataView(Env, stringData);

            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongTypes);
            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongNames);
        }
    }
}
