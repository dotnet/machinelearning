// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.TestFramework;
using System;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Scenarios.PipelineApi
{
    public partial class PipelineApiScenarioTests : BaseTestClass
    {
        public PipelineApiScenarioTests(ITestOutputHelper output) : base(output)
        {
        }

        public const string IrisDataPath = "iris.data";
        public const string SentimentDataPath = "wikipedia-detox-250-line-data.tsv";
        public const string SentimentTestPath = "wikipedia-detox-250-line-test.tsv";

        public class IrisData : IrisDataNoLabel
        {
            [Column("0")]
            public string Label;
        }

        public class IrisDataNoLabel
        {
            [Column("1")]
            public float SepalLength;

            [Column("2")]
            public float SepalWidth;

            [Column("3")]
            public float PetalLength;

            [Column("4")]
            public float PetalWidth;
        }

        public class IrisPrediction
        {
            public float[] Score;
        }

        public class SentimentData
        {
            [Column("0", name: "Label")]
            public bool Sentiment;
            [Column("1")]
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
