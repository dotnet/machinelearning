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

            var pipe = new SsaChangePointEstimator(Env, 
                Confidence, ChangeHistorySize, MaxTrainingSize, SeasonalitySize, "Value", "Change");
            TestEstimatorCore(pipe, dataView);
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

            var pipe = new SsaSpikeEstimator(Env,
                Confidence, PValueHistorySize, MaxTrainingSize, SeasonalitySize, "Value", "Change");
            TestEstimatorCore(pipe, dataView);
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
                Confidence, ChangeHistorySize, "Value", "Change");
            TestEstimatorCore(pipe, dataView);
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
                Confidence, PValueHistorySize, "Value", "Change");
            TestEstimatorCore(pipe, dataView);
        }
    }
}
