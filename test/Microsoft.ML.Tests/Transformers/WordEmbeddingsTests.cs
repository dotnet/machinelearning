﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Scenarios;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Transforms.Text;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public sealed class WordEmbeddingsTests : TestDataPipeBase
    {
        public WordEmbeddingsTests(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact]
        public void TestWordEmbeddings()
        {
            var dataPath = GetDataPath(TestDatasets.Sentiment.trainFilename);
            var testDataPath = GetDataPath(TestDatasets.Sentiment.testFilename);

            var data = TextLoaderStatic.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    SentimentText: ctx.LoadText(1)), hasHeader: true)
                .Read(dataPath);
            var dynamicData = new TextFeaturizingEstimator(Env, "SentimentText", "SentimentText_Features", args =>
            {
                args.OutputTokens = true;
                args.KeepPunctuations = false;
                args.UseStopRemover = true;
                args.VectorNormalizer = TextFeaturizingEstimator.TextNormKind.None;
                args.UseCharExtractor = false;
                args.UseWordExtractor = false;
            }).Fit(data.AsDynamic).Transform(data.AsDynamic);
            var data2 = dynamicData.AssertStatic(Env, ctx => (
                SentimentText_Features_TransformedText: ctx.Text.VarVector,
                SentimentText: ctx.Text.Scalar,
                label: ctx.Bool.Scalar));

            var est = data2.MakeNewEstimator()
                .Append(row => row.SentimentText_Features_TransformedText.WordEmbeddings());
            TestEstimatorCore(est.AsDynamic, data2.AsDynamic, invalidInput: data.AsDynamic);
            Done();
        }
    }
}
