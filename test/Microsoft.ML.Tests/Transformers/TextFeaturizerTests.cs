// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data.StaticPipe;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Transforms;
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
                .Read(new MultiFileSource(sentimentDataPath));

            var invalidData = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadFloat(1)), hasHeader: true)
                .Read(new MultiFileSource(sentimentDataPath))
                .AsDynamic;

            var feat = data.MakeNewEstimator()
                 .Append(row => row.text.FeaturizeText(advancedSettings: s => { s.OutputTokens = true; }));

            TestEstimatorCore(feat.AsDynamic, data.AsDynamic, invalidInput: invalidData);

            var outputPath = GetOutputPath("Text", "featurized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                IDataView savedData = TakeFilter.Create(Env, feat.Fit(data).Transform(data).AsDynamic, 4);
                savedData = new ChooseColumnsTransform(Env, savedData, "Data", "Data_TransformedText");

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("Text", "featurized.tsv");
            Done();
        }

        [Fact]
        public void TextTokenizationWorkout()
        {
            string sentimentDataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var data = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Read(new MultiFileSource(sentimentDataPath));

            var invalidData = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadFloat(1)), hasHeader: true)
                .Read(new MultiFileSource(sentimentDataPath));

            var est = new WordTokenizer(Env, "text", "words")
                .Append(new CharacterTokenizer(Env, "text", "chars"))
                .Append(new KeyToValueEstimator(Env, "chars"));
            TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("Text", "tokenized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                IDataView savedData = TakeFilter.Create(Env, est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                savedData = new ChooseColumnsTransform(Env, savedData, "text", "words", "chars");

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("Text", "tokenized.tsv");
            Done();
        }


        [Fact]
        public void TextNormalizationAndStopwordRemoverWorkout()
        {
            string sentimentDataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var data = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Read(new MultiFileSource(sentimentDataPath));

            var invalidData = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadFloat(1)), hasHeader: true)
                .Read(new MultiFileSource(sentimentDataPath));

            var est = new TextNormalizer(Env,"text")
                .Append(new WordTokenizer(Env, "text", "words"))
                .Append(new StopwordRemover(Env, "words", "words_without_stopwords"));
            TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("Text", "words_without_stopwords.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                IDataView savedData = TakeFilter.Create(Env, est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                savedData = new ChooseColumnsTransform(Env, savedData, "text", "words_without_stopwords");

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("Text", "words_without_stopwords.tsv");
            Done();
        }

        [Fact]
        public void WordBagWorkout()
        {
            string sentimentDataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var data = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Read(new MultiFileSource(sentimentDataPath));

            var invalidData = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadFloat(1)), hasHeader: true)
                .Read(new MultiFileSource(sentimentDataPath));

            var est = new WordBagEstimator(Env, "text", "bag_of_words").
                Append(new WordHashBagEstimator(Env, "text", "bag_of_wordshash"));
            //TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("Text", "bag_of_words.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                IDataView savedData = TakeFilter.Create(Env, est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                savedData = new ChooseColumnsTransform(Env, savedData, "text", "bag_of_words", "bag_of_wordshash");

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("Text", "bag_of_words.tsv");
            Done();
        }

        [Fact]
        public void NgramWorkout()
        {
            string sentimentDataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var data = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Read(new MultiFileSource(sentimentDataPath));

            var invalidData = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadFloat(1)), hasHeader: true)
                .Read(new MultiFileSource(sentimentDataPath));

            var est = new WordTokenizer(Env, "text", "text")
                .Append(new TermEstimator(Env, "text", "terms"))
                .Append(new NgramEstimator(Env, "terms", "ngrams"))
                .Append(new NgramHashEstimator(Env, "terms", "ngramshash"));
            //TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("Text", "ngrams.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                IDataView savedData = TakeFilter.Create(Env, est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                savedData = new ChooseColumnsTransform(Env, savedData, "text", "terms", "ngrams", "ngramshash");

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("Text", "ngrams.tsv");
            Done();
        }
    }
}
