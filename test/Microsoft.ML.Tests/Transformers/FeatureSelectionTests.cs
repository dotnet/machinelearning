﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.FeatureSelection;
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
            var data = TextLoaderStatic.CreateReader(ML, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true)
                .Read(sentimentDataPath);

            var invalidData = TextLoaderStatic.CreateReader(ML, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadFloat(1)), hasHeader: true)
                .Read(sentimentDataPath);

            var est = new WordBagEstimator(ML, "text", "bag_of_words")
                .AppendCacheCheckpoint(ML)
                .Append(ML.Transforms.FeatureSelection.SelectFeaturesBasedOnCount("bag_of_words", "bag_of_words_count", 10)
                .Append(ML.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation("bag_of_words", "bag_of_words_mi", labelColumn: "label")));

            var outputPath = GetOutputPath("FeatureSelection", "featureselection.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true });
                    IDataView savedData = TakeFilter.Create(ML, est.Fit(data.AsDynamic).Transform(data.AsDynamic), 4);
                    savedData = ColumnSelectingTransformer.CreateKeep(ML, savedData, new[] { "bag_of_words_count", "bag_of_words_mi" });

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
            var reader = TextLoaderStatic.CreateReader(ML, ctx => (
                ScalarFloat: ctx.LoadFloat(1),
                ScalarDouble: ctx.LoadDouble(1),
                VectorFloat: ctx.LoadFloat(1, 4),
                VectorDouble: ctx.LoadDouble(4, 8)
            ));

            var data = reader.Read(new MultiFileSource(dataPath)).AsDynamic;

            var columns = new[]
            {
                new SlotsDroppingTransformer.ColumnInfo("VectorFloat", "dropped1", (min: 0, max: 1)),
                new SlotsDroppingTransformer.ColumnInfo("VectorFloat", "dropped2"),
                new SlotsDroppingTransformer.ColumnInfo("ScalarFloat", "dropped3", (min:0, max: 3)),
                new SlotsDroppingTransformer.ColumnInfo("VectorFloat", "dropped4", (min: 1, max: 2)),
                new SlotsDroppingTransformer.ColumnInfo("VectorDouble", "dropped5", (min: 1, null)),
                new SlotsDroppingTransformer.ColumnInfo("VectorFloat", "dropped6", (min: 100, null))
            };
            var trans = new SlotsDroppingTransformer(ML, columns);

            var outputPath = GetOutputPath("FeatureSelection", "dropslots.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                IDataView savedData = TakeFilter.Create(ML, trans.Transform(data), 4);
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
            var reader = TextLoaderStatic.CreateReader(ML, ctx => (
                ScalarFloat: ctx.LoadFloat(6),
                VectorFloat: ctx.LoadFloat(1, 4),
                VectorDouble: ctx.LoadDouble(4, 8)
            ));

            var data = ML.Data.Cache(reader.Read(new MultiFileSource(dataPath)).AsDynamic);

            var columns = new[] {
                new CountFeatureSelectingEstimator.ColumnInfo("VectorDouble", "FeatureSelectDouble", minCount: 1),
                new CountFeatureSelectingEstimator.ColumnInfo("ScalarFloat", "ScalFeatureSelectMissing690", minCount: 690),
                new CountFeatureSelectingEstimator.ColumnInfo("ScalarFloat", "ScalFeatureSelectMissing100", minCount: 100),
                new CountFeatureSelectingEstimator.ColumnInfo("VectorDouble", "VecFeatureSelectMissing690", minCount: 690),
                new CountFeatureSelectingEstimator.ColumnInfo("VectorDouble", "VecFeatureSelectMissing100", minCount: 100)
            };
            var est = new CountFeatureSelectingEstimator(ML, "VectorFloat", "FeatureSelect", minCount: 1)
                .Append(new CountFeatureSelectingEstimator(ML, columns));

            TestEstimatorCore(est, data);

            var outputPath = GetOutputPath("FeatureSelection", "countFeatureSelect.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                IDataView savedData = TakeFilter.Create(ML, est.Fit(data).Transform(data), 4);
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
            var reader = TextLoaderStatic.CreateReader(ML, ctx => (
                Label: ctx.LoadKey(0, 3),
                VectorFloat: ctx.LoadFloat(1, 4)
            ));

            var dataView = reader.Read(new MultiFileSource(dataPath)).AsDynamic;

            var pipe = new CountFeatureSelectingEstimator(ML, "VectorFloat", "FeatureSelect", minCount: 1);

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
            var reader = TextLoaderStatic.CreateReader(ML, ctx => (
                Label: ctx.LoadKey(0, 3),
                ScalarFloat: ctx.LoadFloat(6),
                VectorFloat: ctx.LoadFloat(1, 4),
                VectorDouble: ctx.LoadDouble(4, 8)
            ));

            var data = reader.Read(new MultiFileSource(dataPath)).AsDynamic;

            var est = new MutualInformationFeatureSelectingEstimator(ML, "VectorFloat", "FeatureSelect", slotsInOutput: 1, labelColumn: "Label")
                .Append(new MutualInformationFeatureSelectingEstimator(ML, labelColumn: "Label", slotsInOutput: 2, numBins: 100,
                    columns: new[] {
                        (input: "VectorFloat", output: "out1"),
                        (input: "VectorDouble", output: "out2")
                    }));
            TestEstimatorCore(est, data);

            var outputPath = GetOutputPath("FeatureSelection", "mutualFeatureSelect.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                IDataView savedData = TakeFilter.Create(ML, est.Fit(data).Transform(data), 4);
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
            var reader = TextLoaderStatic.CreateReader(ML, ctx => (
                Label: ctx.LoadKey(0, 3),
                VectorFloat: ctx.LoadFloat(1, 4)
            ));

            var dataView = reader.Read(new MultiFileSource(dataPath)).AsDynamic;

            var pipe = new MutualInformationFeatureSelectingEstimator(ML, "VectorFloat", "FeatureSelect", slotsInOutput: 1, labelColumn: "Label");

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
