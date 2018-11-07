// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
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

        private static TextLoader.Arguments MakeIrisTextLoaderArgs()
        {
            return new TextLoader.Arguments()
            {
                Separator = "comma",
                Column = new[]
                {
                    new TextLoader.Column("SepalLength", DataKind.R4, 0),
                    new TextLoader.Column("SepalWidth", DataKind.R4, 1),
                    new TextLoader.Column("PetalLength", DataKind.R4, 2),
                    new TextLoader.Column("PetalWidth",DataKind.R4, 3),
                    new TextLoader.Column("Label", DataKind.Text, 4)
                }
            };
        }

        private static TextLoader.Arguments MakeSentimentTextLoaderArgs()
        {
            return new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.BL, 0),
                    new TextLoader.Column("SentimentText", DataKind.Text, 1)
                }
            };
        }
    }
}
