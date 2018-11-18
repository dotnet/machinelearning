// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.FeatureSelection;
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

        [Fact]
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
                .Append(new CountFeatureSelectingEstimator(Env, "bag_of_words", "bag_of_words_count", 10)
                .Append(new MutualInformationFeatureSelectionEstimator(Env, "bag_of_words", "bag_of_words_mi", labelColumn: "label")));

            var outputPath = GetOutputPath("FeatureSelection", "featureselection.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                    IDataView savedData = TakeFilter.Create(Env, est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                    savedData = ColumnSelectingTransformer.CreateKeep(Env, savedData, new[] { "bag_of_words_count", "bag_of_words_mi" });

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("FeatureSelection", "featureselection.tsv");
            Done();
        }

        [Fact]
        public void DropSlotsTransform()
        {
            var env = new ConsoleEnvironment(seed: 0);

            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoader.CreateReader(env, ctx => (
                ScalarFloat: ctx.LoadFloat(1),
                ScalarDouble: ctx.LoadDouble(1),
                VectorFloat: ctx.LoadFloat(1, 4),
                VectorDouble: ctx.LoadDouble(4, 8)
            ));

            var data = reader.Read(new MultiFileSource(dataPath)).AsDynamic;

            var columns = new[]
            {
                new DropSlotsTransform.ColumnInfo("VectorFloat", "dropped1", (min: 0, max: 1)),
                new DropSlotsTransform.ColumnInfo("VectorFloat", "dropped2"),
                new DropSlotsTransform.ColumnInfo("ScalarFloat", "dropped3", (min:0, max: 3)),
                new DropSlotsTransform.ColumnInfo("VectorFloat", "dropped4", (min: 1, max: 2)),
                new DropSlotsTransform.ColumnInfo("VectorDouble", "dropped5", (min: 1, null)),
                new DropSlotsTransform.ColumnInfo("VectorFloat", "dropped6", (min: 100, null))

            };
            var trans = new DropSlotsTransform(env, columns);

            var outputPath = GetOutputPath("FeatureSelection", "dropslots.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                IDataView savedData = TakeFilter.Create(Env, trans.Transform(data), 4);
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("FeatureSelection", "dropslots.tsv");
            Done();
        }

        [Fact]
        public void TestDropSlotsSelectionCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0-10} xf=DropSlots{col=B:A:1-4} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void CountFeatureSelectionWorkout()
        {
            var env = new ConsoleEnvironment(seed: 0);

            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoader.CreateReader(env, ctx => (
                ScalarFloat: ctx.LoadFloat(6),
                VectorFloat: ctx.LoadFloat(1, 4),
                VectorDouble: ctx.LoadDouble(4, 8)
            ));

            var data = reader.Read(new MultiFileSource(dataPath)).AsDynamic;

            var columns = new[] {
                new CountFeatureSelectingEstimator.ColumnInfo("VectorDouble", "FeatureSelectDouble", count: 1),
                new CountFeatureSelectingEstimator.ColumnInfo("ScalarFloat", "ScalFeatureSelectMissing690", count: 690),
                new CountFeatureSelectingEstimator.ColumnInfo("ScalarFloat", "ScalFeatureSelectMissing100", count: 100),
                new CountFeatureSelectingEstimator.ColumnInfo("VectorDouble", "VecFeatureSelectMissing690", count: 690),
                new CountFeatureSelectingEstimator.ColumnInfo("VectorDouble", "VecFeatureSelectMissing100", count: 100)
            };
            var est = new CountFeatureSelectingEstimator(env, "VectorFloat", "FeatureSelect", count: 1)
                .Append(new CountFeatureSelectingEstimator(env, columns));

            TestEstimatorCore(est, data);

            var outputPath = GetOutputPath("FeatureSelection", "countFeatureSelect.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                IDataView savedData = TakeFilter.Create(Env, est.Fit(data).Transform(data), 4);
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("FeatureSelection", "countFeatureSelect.tsv");
            Done();
        }

        [Fact]
        public void TestCountFeatureSelectionCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0-10} xf=CountFeatureSelection​{col=B:A c=1} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void MutualInformationSelectionWorkout()
        {
            var env = new ConsoleEnvironment(seed: 0);

            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoader.CreateReader(env, ctx => (
                Label: ctx.LoadKey(0, 0, 2),
                ScalarFloat: ctx.LoadFloat(6),
                VectorFloat: ctx.LoadFloat(1, 4),
                VectorDouble: ctx.LoadDouble(4, 8)
            ));

            var data = reader.Read(new MultiFileSource(dataPath)).AsDynamic;

            var est = new MutualInformationFeatureSelectionEstimator(env, "VectorFloat", "FeatureSelect", slotsInOutput: 1, labelColumn: "Label")
                .Append(new MutualInformationFeatureSelectionEstimator(env, labelColumn: "Label", slotsInOutput: 2, numBins: 100,
                    columns: new[] {
                        (input: "VectorFloat", output: "out1"),
                        (input: "VectorDouble", output: "out2")
                    }));
            //TestEstimatorCore(est, data);

            var outputPath = GetOutputPath("FeatureSelection", "mutualFeatureSelect.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                IDataView savedData = TakeFilter.Create(Env, est.Fit(data).Transform(data), 4);
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("FeatureSelection", "mutualFeatureSelect.tsv");
            Done();
        }

        [Fact]
        public void TestMutualInformationFeatureSelectionCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0-10} xf=MutualInformationFeatureSelection​{col=B:A} in=f:\2.txt" }), (int)0);
        }

    }
}
