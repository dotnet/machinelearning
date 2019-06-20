// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Primitives;
using Microsoft.ML.Data;
using Xunit;

namespace Microsoft.Extensions.ML
{
    public class FileLoaderTests
    {
        [Fact]
        public void throw_until_started()
        {
            var services = new ServiceCollection()
                .AddOptions()
                .AddLogging();
            var sp = services.BuildServiceProvider();

            var loaderUnderTest = ActivatorUtilities.CreateInstance<FileModelLoader>(sp);
            Assert.Throws<InvalidOperationException>(() => loaderUnderTest.GetModel());
            Assert.Throws<InvalidOperationException>(() => loaderUnderTest.GetReloadToken());
        }

        [Fact]
        public void can_load_model()
        {
            var services = new ServiceCollection()
                .AddOptions()
                .AddLogging();
            var sp = services.BuildServiceProvider();

            var loaderUnderTest = ActivatorUtilities.CreateInstance<FileModelLoader>(sp);
            loaderUnderTest.Start(Path.Combine("TestModels", "SentimentModel.zip"), false);

            var model = loaderUnderTest.GetModel();
            var context = sp.GetRequiredService<IOptions<MLOptions>>().Value.MLContext;
            var engine = context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            var prediction = engine.Predict(new SentimentData() { SentimentText = "This is great" });
            Assert.True(prediction.Sentiment);
        }

        //TODO: This is a quick test to give coverage of the main scenarios. Refactoring and re-implementing of tests should happen.
        //Right now this screams of probably flakeyness
        [Fact]
        public async Task can_reload_model()
        {
            var services = new ServiceCollection()
                .AddOptions()
                .AddLogging();
            var sp = services.BuildServiceProvider();

            var loaderUnderTest = ActivatorUtilities.CreateInstance<FileLoaderMock>(sp);
            loaderUnderTest.Start("testdata.txt", true);

            var changed = false;
            var changeTokenRegistration = ChangeToken.OnChange(
                        () => loaderUnderTest.GetReloadToken(),
                        () => changed = true);

            File.WriteAllText("testdata.txt", "test");

            await Task.Delay(1000);

            Assert.True(changed);
        }


        private class FileLoaderMock : FileModelLoader
        {
            public FileLoaderMock(IOptions<MLOptions> contextOptions, ILogger<FileModelLoader> logger)
                : base(contextOptions, logger)
            {
            }

            internal override void LoadModel()
            {
            }
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
