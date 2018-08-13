using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.TestFramework;
using System;
using System.Collections.Generic;
using System.Text;
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

        public const string IrisDataPath = "iris.data";
        public const string SentimentDataPath = "wikipedia-detox-250-line-data.tsv";
        public const string SentimentTestPath = "wikipedia-detox-250-line-test.tsv";

        public class IrisData: IrisDataNoLabel
        {
            public string Label;
        }

        public class IrisDataNoLabel
        {
            public float SepalLength;
            public float SepalWidth;
            public float PetalLength;
            public float PetalWidth;
        }

        public class IrisPrediction
        {
            public string PredictedLabel;
            public float[] Score;
        }

        public class SentimentData
        {
            [ColumnName("Label")]
            public bool Sentiment;
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
