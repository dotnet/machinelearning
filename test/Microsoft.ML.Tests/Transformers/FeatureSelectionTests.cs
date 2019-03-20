// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
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
            var data = TextLoaderStatic.CreateLoader(ML, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Load(sentimentDataPath);

            var invalidData = TextLoaderStatic.CreateLoader(ML, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadFloat(1)), hasHeader: true)
                .Load(sentimentDataPath);

            var est = new WordBagEstimator(ML, "bag_of_words", "text")
                .AppendCacheCheckpoint(ML)
                .Append(ML.Transforms.FeatureSelection.SelectFeaturesBasedOnCount("bag_of_words_count", "bag_of_words", 10)
                .Append(ML.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation("bag_of_words_mi", "bag_of_words", labelColumnName: "label")));

            var outputPath = GetOutputPath("FeatureSelection", "featureselection.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true });
                var savedData = ML.Data.TakeRows(est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                savedData = ML.Transforms.SelectColumns("bag_of_words_count", "bag_of_words_mi").Fit(savedData).Transform(savedData);

                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("FeatureSelection", "featureselection.tsv");
            Done();
        }

        [Fact]
        public void DropSlotsTransform()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoaderStatic.CreateLoader(ML, ctx => (
                ScalarFloat: ctx.LoadFloat(1),
                ScalarDouble: ctx.LoadDouble(1),
                VectorFloat: ctx.LoadFloat(1, 4),
                VectorDouble: ctx.LoadDouble(4, 8)
            ));

            var data = reader.Load(new MultiFileSource(dataPath)).AsDynamic;

            var columns = new[]
            {
                new SlotsDroppingTransformer.ColumnOptions("dropped1", "VectorFloat", (min: 0, max: 1)),
                new SlotsDroppingTransformer.ColumnOptions("dropped2", "VectorFloat"),
                new SlotsDroppingTransformer.ColumnOptions("dropped3", "ScalarFloat", (min:0, max: 3)),
                new SlotsDroppingTransformer.ColumnOptions("dropped4", "VectorFloat", (min: 1, max: 2)),
                new SlotsDroppingTransformer.ColumnOptions("dropped5", "VectorDouble", (min: 1, null)),
                new SlotsDroppingTransformer.ColumnOptions("dropped6", "VectorFloat", (min: 100, null))
            };
            var trans = new SlotsDroppingTransformer(ML, columns);

            var outputPath = GetOutputPath("FeatureSelection", "dropslots.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                var savedData = ML.Data.TakeRows(trans.Transform(data), 4);
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
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoaderStatic.CreateLoader(ML, ctx => (
                ScalarFloat: ctx.LoadFloat(6),
                VectorFloat: ctx.LoadFloat(1, 4),
                VectorDouble: ctx.LoadDouble(4, 8)
            ));

            var data = ML.Data.Cache(reader.Load(new MultiFileSource(dataPath)).AsDynamic);

            var columns = new[] {
                new CountFeatureSelectingEstimator.ColumnOptions("FeatureSelectDouble", "VectorDouble", count: 1),
                new CountFeatureSelectingEstimator.ColumnOptions("ScalFeatureSelectMissing690", "ScalarFloat", count: 690),
                new CountFeatureSelectingEstimator.ColumnOptions("ScalFeatureSelectMissing100", "ScalarFloat", count: 100),
                new CountFeatureSelectingEstimator.ColumnOptions("VecFeatureSelectMissing690", "VectorDouble", count: 690),
                new CountFeatureSelectingEstimator.ColumnOptions("VecFeatureSelectMissing100", "VectorDouble", count: 100)
            };
            var est = ML.Transforms.FeatureSelection.SelectFeaturesBasedOnCount("FeatureSelect", "VectorFloat", count: 1)
                .Append(ML.Transforms.FeatureSelection.SelectFeaturesBasedOnCount(columns));

            TestEstimatorCore(est, data);

            var outputPath = GetOutputPath("FeatureSelection", "countFeatureSelect.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                var savedData = ML.Data.TakeRows(est.Fit(data).Transform(data), 4);
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("FeatureSelection", "countFeatureSelect.tsv");
            Done();
        }

        [Fact]
        public void TestCountFeatureSelectionCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0-10} xf=CountFeatureSelection{col=A c=1} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void TestCountSelectOldSavingAndLoading()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoaderStatic.CreateLoader(ML, ctx => (
                Label: ctx.LoadKey(0, 3),
                VectorFloat: ctx.LoadFloat(1, 4)
            ));

            var dataView = reader.Load(new MultiFileSource(dataPath)).AsDynamic;

            var pipe = ML.Transforms.FeatureSelection.SelectFeaturesBasedOnCount("FeatureSelect", "VectorFloat", count: 1);

            var result = pipe.Fit(dataView).Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(ML, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(ML, dataView, ms);
            }
            Done();
        }

        [Fact]
        public void MutualInformationSelectionWorkout()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoaderStatic.CreateLoader(ML, ctx => (
                Label: ctx.LoadKey(0, 3),
                ScalarFloat: ctx.LoadFloat(6),
                VectorFloat: ctx.LoadFloat(1, 4),
                VectorDouble: ctx.LoadDouble(4, 8)
            ));

            var data = reader.Load(new MultiFileSource(dataPath)).AsDynamic;

            var est = ML.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation("FeatureSelect", "VectorFloat", slotsInOutput: 1, labelColumnName: "Label")
                .Append(ML.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation(labelColumnName: "Label", slotsInOutput: 2, numberOfBins: 100,
                    columns: new ColumnOptions[] {
                        ("out1", "VectorFloat"),
                        ("out2", "VectorDouble")
                    }));
            TestEstimatorCore(est, data);

            var outputPath = GetOutputPath("FeatureSelection", "mutualFeatureSelect.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                var savedData = ML.Data.TakeRows(est.Fit(data).Transform(data), 4);
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("FeatureSelection", "mutualFeatureSelect.tsv");
            Done();
        }

        [Fact]
        public void TestMutualInformationFeatureSelectionCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0-10 col=B:R4:11} xf=MutualInformationFeatureSelection{col=A lab=B} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void TestMutualInformationOldSavingAndLoading()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoaderStatic.CreateLoader(ML, ctx => (
                Label: ctx.LoadKey(0, 3),
                VectorFloat: ctx.LoadFloat(1, 4)
            ));

            var dataView = reader.Load(new MultiFileSource(dataPath)).AsDynamic;

            var pipe = ML.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation("FeatureSelect", "VectorFloat", slotsInOutput: 1, labelColumnName: "Label");

            var result = pipe.Fit(dataView).Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(ML, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(ML, dataView, ms);
            }
            Done();
        }
    }
}
