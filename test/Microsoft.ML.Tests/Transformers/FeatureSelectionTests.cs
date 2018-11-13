// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public sealed class FeatureSelectionTests : TestDataPipeBase
    {
        public FeatureSelectionTests(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact(Skip = "FeatureSeclection transform cannot be trained on empty data, schema propagation fails")]
        public void FeatureSelectionWorkout()
        {
            string sentimentDataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var data = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Read(sentimentDataPath);

            var invalidData = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadFloat(1)), hasHeader: true)
                .Read(sentimentDataPath);

            var est = new WordBagEstimator(Env, "text", "bag_of_words")
                .Append(new CountFeatureSelector(Env, "bag_of_words", "bag_of_words_count", 10)
                .Append(new MutualInformationFeatureSelector(Env, "bag_of_words", "bag_of_words_mi", labelColumn: "label")));

            var outputPath = GetOutputPath("FeatureSelection", "featureselection.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                    IDataView savedData = TakeFilter.Create(Env, est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                    savedData = SelectColumnsTransformer.CreateKeep(Env, savedData, new[] { "bag_of_words_count", "bag_of_words_mi" });

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("FeatureSelection", "featureselection.tsv");
            Done();
        }
    }
}
