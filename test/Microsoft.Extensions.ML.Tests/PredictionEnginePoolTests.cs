// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Threading;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Primitives;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFrameworkCommon;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.Extensions.ML
{
    public class PredictionEnginePoolTests : BaseTestClass
    {
        public PredictionEnginePoolTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void can_load_namedmodel()
        {
            var services = new ServiceCollection()
                .AddOptions()
                .AddLogging();

            services.AddPredictionEnginePool<SentimentData, SentimentPrediction>()
               .FromFile(modelName: "model1", filePath: Path.Combine("TestModels", "SentimentModel.zip"), watchForChanges: false);

            var sp = services.BuildServiceProvider();

            var pool = sp.GetRequiredService<PredictionEnginePool<SentimentData, SentimentPrediction>>();
            var model = pool.GetModel("model1");

            Assert.NotNull(model);
        }

        public class SentimentData
        {
            [ColumnName("Label"), LoadColumn(0)]
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
