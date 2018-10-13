// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.TimeSeriesProcessing;
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
        void TestSsaChangePointEstimatorWithSeasonality()
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
    }
}
