﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
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
            var dataPath = GetDataPath(ScenariosTests.SentimentDataPath);
            var testDataPath = GetDataPath(ScenariosTests.SentimentTestPath);

            var data = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    SentimentText: ctx.LoadText(1)), hasHeader: true)
                .Read(dataPath);

            var dynamicData = TextFeaturizingEstimator.Create(Env, new TextFeaturizingEstimator.Arguments()
            {
                Column = new TextFeaturizingEstimator.Column
                {
                    Name = "SentimentText_Features",
                    Source = new[] { "SentimentText" }
                },
                OutputTokens = true,
                KeepPunctuations = false,
                StopWordsRemover = new Runtime.TextAnalytics.PredefinedStopWordsRemoverFactory(),
                VectorNormalizer = TextFeaturizingEstimator.TextNormKind.None,
                CharFeatureExtractor = null,
                WordFeatureExtractor = null,
            }, data.AsDynamic);

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
