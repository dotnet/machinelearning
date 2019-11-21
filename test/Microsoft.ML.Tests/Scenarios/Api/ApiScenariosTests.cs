// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    /// <summary>
    /// Common utility functions for API scenarios tests.
    /// </summary>
    public partial class ApiScenariosTests : BaseTestClass
    {
        public ApiScenariosTests(ITestOutputHelper output) : base(output)
        {
        }

        public class IrisData : IrisDataNoLabel
        {
            [LoadColumn(4), ColumnName("Label")]
            public string Label;
        }

        public class IrisDataNoLabel
        {
            [LoadColumn(0)]
            public float SepalLength;

            [LoadColumn(1)]
            public float SepalWidth;

            [LoadColumn(2)]
            public float PetalLength;

            [LoadColumn(3)]
            public float PetalWidth;
        }

        public class IrisPrediction
        {
            public string PredictedLabel;
            public float[] Score;
        }
        public class IrisPredictionNotCasted
        {
            public uint PredictedLabel;
            public float[] Score;
        }

        public class SentimentData
        {
            [LoadColumn(0), ColumnName("Label")]
            public bool Sentiment;

            [LoadColumn(1)]
            public string SentimentText;
        }

        public class SentimentPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool Sentiment;

            public float Score;
        }
    }
}
