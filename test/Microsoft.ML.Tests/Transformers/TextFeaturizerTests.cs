// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.RunTests;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public sealed class TextFeaturizerTests : TestDataPipeBase
    {
        public TextFeaturizerTests(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact]
        public void TextFeaturizerWorkout()
        {
            string sentimentDataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var data = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Read(new MultiFileSource(sentimentDataPath))
                .AsDynamic;

            var invalidData = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadFloat(1)), hasHeader: true)
                .Read(new MultiFileSource(sentimentDataPath))
                .AsDynamic;

            var feat = new TextTransform(Env, "text", outputTokens: true);
            TestEstimatorCore(feat, data, invalidInput: invalidData);

            var outputPath = GetOutputPath("Text", "featurized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, TakeFilter.Create(Env, feat.Fit(data).Transform(data), 4), fs, keepHidden: true);
            }

            CheckEquality("Text", "featurized.tsv");
            Done();
        }
    }
}
