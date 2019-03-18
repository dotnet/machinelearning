// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
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
            var data = TextLoaderStatic.CreateLoader(ML, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Load(sentimentDataPath);

            var invalidData = TextLoaderStatic.CreateLoader(ML, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadFloat(1)), hasHeader: true)
                .Load(sentimentDataPath)
                .AsDynamic;

            var feat = data.MakeNewEstimator()
                 .Append(row => row.text.FeaturizeText(options: new TextFeaturizingEstimator.Options { OutputTokensColumnName = "OutputTokens", }));

            TestEstimatorCore(feat.AsDynamic, data.AsDynamic, invalidInput: invalidData);

            var outputPath = GetOutputPath("Text", "featurized.tsv");
            using (var ch = ((IHostEnvironment)ML).Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true });
                var savedData = ML.Data.TakeRows(feat.Fit(data).Transform(data).AsDynamic, 4);
                savedData = ML.Transforms.SelectColumns("Data", "OutputTokens").Fit(savedData).Transform(savedData);

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
            var data = TextLoaderStatic.CreateLoader(ML, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Load(sentimentDataPath);

            var invalidData = TextLoaderStatic.CreateLoader(ML, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadFloat(1)), hasHeader: true)
                .Load(sentimentDataPath);

            var est = new WordTokenizingEstimator(ML, "words", "text")
                .Append(new TokenizingByCharactersEstimator(ML, "chars", "text"))
                .Append(new KeyToValueMappingEstimator(ML, "chars"));
            TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("Text", "tokenized.tsv");
            var savedData = ML.Data.TakeRows(est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
            savedData = ML.Transforms.SelectColumns("text", "words", "chars").Fit(savedData).Transform(savedData);

            using (var fs = File.Create(outputPath))
                ML.Data.SaveAsText(savedData, fs, headerRow: true, keepHidden: true);

            CheckEquality("Text", "tokenized.tsv");
            Done();
        }

        [Fact]
        public void TokenizeWithSeparators()
        {
            string dataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var data = TextLoaderStatic.CreateLoader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Load(dataPath).AsDynamic;

            var est = new WordTokenizingEstimator(Env, "words", "text", separators: new[] { ' ', '?', '!', '.', ',' });
            var outdata = ML.Data.TakeRows(est.Fit(data).Transform(data), 4);
            var savedData = ML.Transforms.SelectColumns("words").Fit(outdata).Transform(outdata);

            var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
            var outputPath = GetOutputPath("Text", "tokenizedWithSeparators.tsv");
            using (var ch = Env.Start("save"))
            {
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }
            CheckEquality("Text", "tokenizedWithSeparators.tsv");
            Done();
        }

        [Fact]
        public void TokenizeWithSeparatorCommandLine()
        {
            string dataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");

            TestCore(dataPath, false,
                new[] {
                    "loader=Text{col=T:TX:1} xf=take{c=4} xf=token{col=T sep=comma,s,a}"
                });

            Done();
        }

        [Fact]
        public void TextNormalizationAndStopwordRemoverWorkout()
        {
            string sentimentDataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var data = TextLoaderStatic.CreateLoader(ML, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Load(sentimentDataPath);

            var invalidData = TextLoaderStatic.CreateLoader(ML, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadFloat(1)), hasHeader: true)
                .Load(sentimentDataPath);
            var est = ML.Transforms.Text.NormalizeText("text")
                .Append(ML.Transforms.Text.TokenizeIntoWords("words", "text"))
                .Append(ML.Transforms.Text.RemoveDefaultStopWords("NoDefaultStopwords", "words"))
                .Append(ML.Transforms.Text.RemoveStopWords("NoStopWords", "words", "xbox", "this", "is", "a", "the", "THAT", "bY"));

            TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("Text", "words_without_stopwords.tsv");
            var savedData = ML.Data.TakeRows(est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
            savedData = ML.Transforms.SelectColumns("text", "NoDefaultStopwords", "NoStopWords").Fit(savedData).Transform(savedData);
            using (var fs = File.Create(outputPath))
                ML.Data.SaveAsText(savedData, fs, headerRow: true, keepHidden: true);

            CheckEquality("Text", "words_without_stopwords.tsv");
            Done();
        }

        [Fact]
        public void StopWordsRemoverFromFactory()
        {
            var factory = new PredefinedStopWordsRemoverFactory();
            string sentimentDataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var data = TextLoader.Create(ML, new TextLoader.Options()
            {
                Columns = new[]
                {
                    new TextLoader.Column("Text", DataKind.String, 1)
                }
            }, new MultiFileSource(sentimentDataPath));

            var tokenized = new WordTokenizingTransformer(ML, new[]
            {
                new WordTokenizingEstimator.ColumnOptions("Text", "Text")
            }).Transform(data);

            var xf = factory.CreateComponent(ML, tokenized,
                new[] {
                    new StopWordsRemovingTransformer.Column() { Name = "Text", Source = "Text" }
                });

            using (var cursor = xf.GetRowCursorForAllColumns())
            {
                VBuffer<ReadOnlyMemory<char>> text = default;
                var getter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(cursor.Schema["Text"]);
                while (cursor.MoveNext())
                    getter(ref text);
            }
        }

        [Fact]
        public void WordBagWorkout()
        {
            string sentimentDataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var data = TextLoaderStatic.CreateLoader(ML, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Load(sentimentDataPath);

            var invalidData = TextLoaderStatic.CreateLoader(ML, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadFloat(1)), hasHeader: true)
                .Load(sentimentDataPath);

            var est = new WordBagEstimator(ML, "bag_of_words", "text").
                Append(new WordHashBagEstimator(ML, "bag_of_wordshash", "text", maximumNumberOfInverts: -1));

            // The following call fails because of the following issue
            // https://github.com/dotnet/machinelearning/issues/969
            // TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("Text", "bag_of_words.tsv");
            var savedData = ML.Data.TakeRows(est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
            savedData = ML.Transforms.SelectColumns("text", "bag_of_words", "bag_of_wordshash").Fit(savedData).Transform(savedData);

            using (var fs = File.Create(outputPath))
                ML.Data.SaveAsText(savedData, fs, headerRow: true, keepHidden: true);

            CheckEquality("Text", "bag_of_words.tsv");
            Done();
        }

        [Fact]
        public void NgramWorkout()
        {
            string sentimentDataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var data = TextLoaderStatic.CreateLoader(ML, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Load(sentimentDataPath);

            var invalidData = TextLoaderStatic.CreateLoader(ML, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadFloat(1)), hasHeader: true)
                .Load(sentimentDataPath);

            var est = new WordTokenizingEstimator(ML, "text", "text")
                .Append(new ValueToKeyMappingEstimator(ML, "terms", "text"))
                .Append(new NgramExtractingEstimator(ML, "ngrams", "terms"))
                .Append(new NgramHashingEstimator(ML, "ngramshash", "terms"));

            TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("Text", "ngrams.tsv");
            var savedData = ML.Data.TakeRows(est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
            savedData = ML.Transforms.SelectColumns("text", "terms", "ngrams", "ngramshash").Fit(savedData).Transform(savedData);

            using (var fs = File.Create(outputPath))
                ML.Data.SaveAsText(savedData, fs, headerRow: true, keepHidden: true);

            CheckEquality("Text", "ngrams.tsv");
            Done();
        }

        [Fact]
        void TestNgramCompatColumns()
        {
            string dropModelPath = GetDataPath("backcompat/ngram.zip");
            string sentimentDataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var data = TextLoaderStatic.CreateLoader(ML, ctx => (
                    Sentiment: ctx.LoadBool(0),
                    SentimentText: ctx.LoadText(1)), hasHeader: true)
                .Load(sentimentDataPath);
            using (FileStream fs = File.OpenRead(dropModelPath))
            {
                var result = ModelFileUtils.LoadTransforms(Env, data.AsDynamic, fs);
                var featureColumn = result.Schema.GetColumnOrNull("Features");
                Assert.NotNull(featureColumn);
            }
        }

        [Fact]
        public void LdaWorkout()
        {
            IHostEnvironment env = new MLContext(seed: 42);
            string sentimentDataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var data = TextLoaderStatic.CreateLoader(env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Load(sentimentDataPath);

            var invalidData = TextLoaderStatic.CreateLoader(env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadFloat(1)), hasHeader: true)
                .Load(sentimentDataPath);

            var est = new WordBagEstimator(env, "bag_of_words", "text").
                Append(new LatentDirichletAllocationEstimator(env, "topics", "bag_of_words", 10, maximumNumberOfIterations: 10,
                    resetRandomGenerator: true));

            // The following call fails because of the following issue
            // https://github.com/dotnet/machinelearning/issues/969
            // In this test it manifests because of the WordBagEstimator in the estimator chain
            // TestEstimatorCore(est, data.AsDynamic, invalidInput: invalidData.AsDynamic);

            var outputPath = GetOutputPath("Text", "ldatopics.tsv");
            using (var ch = env.Start("save"))
            {
                var saver = new TextSaver(env, new TextSaver.Arguments { Silent = true, OutputHeader = false, Dense = true });
                var transformer = est.Fit(data.AsDynamic);
                var transformedData = transformer.Transform(data.AsDynamic);
                var savedData = ML.Data.TakeRows(transformedData, 4);
                savedData = ML.Transforms.SelectColumns("topics").Fit(savedData).Transform(savedData);

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);

                Assert.Equal(10, (savedData.Schema[0].Type as VectorType)?.Size);
            }

            // Diabling this check due to the following issue with consitency of output.
            // `seed` specified in ConsoleEnvironment has no effect.
            // https://github.com/dotnet/machinelearning/issues/1004
            // On single box, setting `s.ResetRandomGenerator = true` works but fails on build server
            // CheckEquality("Text", "ldatopics.tsv");
            Done();
        }

        [Fact]
        public void LdaWorkoutEstimatorCore()
        {
            var ml = new MLContext();

            var builder = new ArrayDataViewBuilder(Env);
            var data = new[]
            {
                new[] {  (float)1.0,  (float)0.0,  (float)0.0 },
                new[] {  (float)0.0,  (float)1.0,  (float)0.0 },
                new[] {  (float)0.0,  (float)0.0,  (float)1.0 },
            };
            builder.AddColumn("F1V", NumberDataViewType.Single, data);
            var srcView = builder.GetDataView();

            var est = ml.Transforms.Text.LatentDirichletAllocation("F1V");
            TestEstimatorCore(est, srcView);
        }

        [Fact]
        public void TestLdaCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0-10} xf=lda{col=B:A} in=f:\2.txt" }), (int)0);
        }
    }
}
