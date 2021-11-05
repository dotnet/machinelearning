// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class TestApi : BaseTestClass
    {
        private const string OutputRelativePath = @"../Common/Api";

        public TestApi(ITestOutputHelper output) : base(output)
        {
        }

        private class OneIChannelWithAttribute
        {
            [CursorChannel]
            public IChannel Channel = null;
        }

        private class OneStringWithAttribute
        {
            public string OutField = "outfield is a string";

            [CursorChannelAttribute]
            public string Channel = null;
        }

        private class TwoIChannelsWithAttributes
        {
            [CursorChannelAttribute]
            public IChannel ChannelOne = null;

            [CursorChannelAttribute]
            public IChannel ChannelTwo = null;
        }

        private class TwoIChannelsOnlyOneWithAttribute
        {
            public string OutField = "Outfield is assigned";

            public IChannel ChannelOne = null;

            [CursorChannelAttribute]
            public IChannel ChannelTwo = null;
        }

        [Fact]
        public void CursorChannelExposedInMapTransform()
        {
            var env = new MLContext(seed: 0);
            // Correct use of CursorChannel attribute.
            var data1 = Utils.CreateArray(10, new OneIChannelWithAttribute());
            var idv1 = env.Data.LoadFromEnumerable(data1);
            Assert.Null(data1[0].Channel);

            var filter1 = LambdaTransform.CreateFilter<OneIChannelWithAttribute, object>(env, idv1,
                (input, state) =>
                {
                    Assert.NotNull(input.Channel);
                    return false;
                }, null);
            filter1.GetRowCursorForAllColumns().MoveNext();

            // Error case: non-IChannel field marked with attribute.
            var data2 = Utils.CreateArray(10, new OneStringWithAttribute());
            var idv2 = env.Data.LoadFromEnumerable(data2);
            Assert.Null(data2[0].Channel);

            var filter2 = LambdaTransform.CreateFilter<OneStringWithAttribute, object>(env, idv2,
                (input, state) =>
                {
                    Assert.Null(input.Channel);
                    return false;
                }, null);
            try
            {
                filter2.GetRowCursorForAllColumns().MoveNext();
                Assert.True(false, "Throw an error if attribute is applied to a field that is not an IChannel.");
            }
            catch (InvalidOperationException ex)
            {
                Assert.True(ex.IsMarked());
            }

            // Error case: multiple fields marked with attributes.
            var data3 = Utils.CreateArray(10, new TwoIChannelsWithAttributes());
            var idv3 = env.Data.LoadFromEnumerable(data3);
            Assert.Null(data3[0].ChannelOne);
            Assert.Null(data3[2].ChannelTwo);

            var filter3 = LambdaTransform.CreateFilter<TwoIChannelsWithAttributes, object>(env, idv3,
                (input, state) =>
                {
                    Assert.Null(input.ChannelOne);
                    Assert.Null(input.ChannelTwo);
                    return false;
                }, null);
            try
            {
                filter3.GetRowCursorForAllColumns().MoveNext();
                Assert.True(false, "Throw an error if attribute is applied to a field that is not an IChannel.");
            }
            catch (InvalidOperationException ex)
            {
                Assert.True(ex.IsMarked());
            }

            // Correct case: non-marked IChannel field is not touched.
            var example4 = new TwoIChannelsOnlyOneWithAttribute();
            Assert.Null(example4.ChannelTwo);
            Assert.Null(example4.ChannelOne);
            var idv4 = env.Data.LoadFromEnumerable(Utils.CreateArray(10, example4));

            var filter4 = LambdaTransform.CreateFilter<TwoIChannelsOnlyOneWithAttribute, object>(env, idv4,
                (input, state) =>
                {
                    Assert.Null(input.ChannelOne);
                    Assert.NotNull(input.ChannelTwo);
                    return false;
                }, null);
            filter1.GetRowCursorForAllColumns().MoveNext();
        }

        public class BreastCancerExample
        {
            public float Label;

            [VectorType(9)]
            public float[] Features;
        }

        [Fact]
        public void LambdaTransformCreate()
        {
            var env = new MLContext(seed: 42);
            var data = ReadBreastCancerExamples();
            var idv = env.Data.LoadFromEnumerable(data);

            var filter = LambdaTransform.CreateFilter<BreastCancerExample, object>(env, idv,
                (input, state) => input.Label == 0, null);

            Assert.Null(filter.GetRowCount());

            // test re-apply
            var applied = env.Data.LoadFromEnumerable(data);
            applied = ApplyTransformUtils.ApplyAllTransformsToData(env, filter, applied);

            var saver = new TextSaver(env, new TextSaver.Arguments());
            Assert.True(applied.Schema.TryGetColumnIndex("Label", out int label));
            using (var fs = File.Create(GetOutputPath(OutputRelativePath, "lambda-output.tsv")))
                saver.SaveData(fs, applied, label);
        }

        [Fact]
        public void TrainAveragedPerceptronWithCache()
        {
            var mlContext = new MLContext(0);
            var dataFile = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var loader = TextLoader.Create(mlContext, new TextLoader.Options(), new MultiFileSource(dataFile));
            var globalCounter = 0;
            IDataView xf = LambdaTransform.CreateFilter<object, object>(mlContext, loader,
                (i, s) => true,
                s => { globalCounter++; });
            xf = mlContext.Transforms.Conversion.ConvertType("Label", outputKind: DataKind.Boolean).Fit(xf).Transform(xf);
            // The baseline result of this was generated with everything cached in memory. As auto-cache is removed,
            // an explicit step of caching is required to make this test ok.
            var cached = mlContext.Data.Cache(xf);

            var estimator = mlContext.BinaryClassification.Trainers.AveragedPerceptron(
                new AveragedPerceptronTrainer.Options { NumberOfIterations = 2 });

            estimator.Fit(cached).Transform(cached);

            // Make sure there were 2 cursoring events.
            Assert.Equal(1, globalCounter);
        }

        [Fact]
        public void MetadataSupportInDataViewConstruction()
        {
            var data = ReadBreastCancerExamples();
            var autoSchema = SchemaDefinition.Create(typeof(BreastCancerExample));

            var mlContext = new MLContext(0);

            // Create Metadata.
            var kindFloat = "Testing float as metadata.";
            float valueFloat = 10;
            var coltypeFloat = NumberDataViewType.Single;
            var kindString = "Testing string as metadata.";
            var valueString = "Strings have value.";
            var coltypeString = TextDataViewType.Instance;
            var kindStringArray = "Testing string array as metadata.";
            var valueStringArray = "I really have no idea what these features entail.".Split(' ');
            var coltypeStringArray = new VectorDataViewType(coltypeString, valueStringArray.Length);
            var kindFloatArray = "Testing float array as metadata.";
            var valueFloatArray = new float[] { 1, 17, 7, 19, 25, 0 };
            var coltypeFloatArray = new VectorDataViewType(coltypeFloat, valueFloatArray.Length);
            var kindVBuffer = "Testing VBuffer as metadata.";
            var valueVBuffer = new VBuffer<float>(4, new float[] { 4, 6, 89, 5 });
            var coltypeVBuffer = new VectorDataViewType(coltypeFloat, valueVBuffer.Length);

            // Add Metadata.
            var labelColumn = autoSchema[0];
            labelColumn.AddAnnotation(kindFloat, valueFloat, coltypeFloat);
            labelColumn.AddAnnotation(kindString, valueString, coltypeString);

            var featureColumn = autoSchema[1];
            featureColumn.AddAnnotation(kindStringArray, valueStringArray, coltypeStringArray);
            featureColumn.AddAnnotation(kindFloatArray, valueFloatArray, coltypeFloatArray);
            featureColumn.AddAnnotation(kindVBuffer, valueVBuffer, coltypeVBuffer);

            var idv = mlContext.Data.LoadFromEnumerable(data, autoSchema);

            Assert.True(idv.Schema[0].Annotations.Schema.Count == 2);
            Assert.True(idv.Schema[0].Annotations.Schema[0].Name == kindFloat);
            Assert.True(idv.Schema[0].Annotations.Schema[0].Type == coltypeFloat);
            Assert.True(idv.Schema[0].Annotations.Schema[1].Name == kindString);
            Assert.True(idv.Schema[0].Annotations.Schema[1].Type == TextDataViewType.Instance);

            Assert.True(idv.Schema[1].Annotations.Schema.Count == 3);
            Assert.True(idv.Schema[1].Annotations.Schema[0].Name == kindStringArray);
            Assert.True(idv.Schema[1].Annotations.Schema[0].Type is VectorDataViewType vectorType && vectorType.ItemType is TextDataViewType);
            Assert.Throws<ArgumentOutOfRangeException>(() => idv.Schema[1].Annotations.Schema[kindFloat]);

            float retrievedFloat = 0;
            idv.Schema[0].Annotations.GetValue(kindFloat, ref retrievedFloat);
            Assert.True(Math.Abs(retrievedFloat - valueFloat) < .000001);

            ReadOnlyMemory<char> retrievedReadOnlyMemory = new ReadOnlyMemory<char>();
            idv.Schema[0].Annotations.GetValue(kindString, ref retrievedReadOnlyMemory);
            Assert.True(retrievedReadOnlyMemory.Span.SequenceEqual(valueString.AsMemory().Span));

            VBuffer<ReadOnlyMemory<char>> retrievedReadOnlyMemoryVBuffer = new VBuffer<ReadOnlyMemory<char>>();
            idv.Schema[1].Annotations.GetValue(kindStringArray, ref retrievedReadOnlyMemoryVBuffer);
            Assert.True(retrievedReadOnlyMemoryVBuffer.DenseValues().Select((s, i) => s.ToString() == valueStringArray[i]).All(b => b));

            VBuffer<float> retrievedFloatVBuffer = new VBuffer<float>(1, new float[] { 2 });
            idv.Schema[1].Annotations.GetValue(kindFloatArray, ref retrievedFloatVBuffer);
            VBuffer<float> valueFloatVBuffer = new VBuffer<float>(valueFloatArray.Length, valueFloatArray);
            Assert.True(retrievedFloatVBuffer.Items().SequenceEqual(valueFloatVBuffer.Items()));

            VBuffer<float> retrievedVBuffer = new VBuffer<float>();
            idv.Schema[1].Annotations.GetValue(kindVBuffer, ref retrievedVBuffer);
            Assert.True(retrievedVBuffer.Items().SequenceEqual(valueVBuffer.Items()));

            Assert.Throws<InvalidOperationException>(() => idv.Schema[1].Annotations.GetValue(kindFloat, ref retrievedReadOnlyMemoryVBuffer));
        }

        private List<BreastCancerExample> ReadBreastCancerExamples()
        {
            var dataFile = GetDataPath(TestDatasets.breastCancer.trainFilename);

            // read the data programmatically into memory
            var data = File.ReadAllLines(dataFile)
                .Where(line => !string.IsNullOrEmpty(line) && !line.StartsWith("//"))
                .Select(
                    line =>
                    {
                        var parts = line.Split('\t');
                        var ex = new BreastCancerExample
                        {
                            Features = new float[9]
                        };
                        var span = new ReadOnlySpan<char>(parts[0].ToCharArray());
                        DoubleParser.Parse(span, out ex.Label);
                        for (int j = 0; j < 9; j++)
                        {
                            span = new ReadOnlySpan<char>(parts[j + 1].ToCharArray());
                            DoubleParser.Parse(span, out ex.Features[j]);
                        }
                        return ex;
                    })
                .ToList();
            return data;
        }

        [Fact]
        public void TestSplitsSchema()
        {

            var mlContext = new MLContext(0);
            var dataPath = GetDataPath("adult.tiny.with-schema.txt");

            var fullInput = mlContext.Data.LoadFromTextFile(dataPath, new[] {
                            new TextLoader.Column("Label", DataKind.Boolean, 0),
                            new TextLoader.Column("Workclass", DataKind.String, 1),
                            new TextLoader.Column("Education", DataKind.String,2),
                            new TextLoader.Column("Age", DataKind.Single,9)
            }, hasHeader: true);

            var ttSplit = mlContext.Data.TrainTestSplit(fullInput);
            var ttSplitWithSeed = mlContext.Data.TrainTestSplit(fullInput, seed: 10);
            var ttSplitWithSeedAndSamplingKey = mlContext.Data.TrainTestSplit(fullInput, seed: 10, samplingKeyColumnName: "Workclass");

            var cvSplit = mlContext.Data.CrossValidationSplit(fullInput);
            var cvSplitWithSeed = mlContext.Data.CrossValidationSplit(fullInput, seed: 10);
            var cvSplitWithSeedAndSamplingKey = mlContext.Data.CrossValidationSplit(fullInput, seed: 10, samplingKeyColumnName: "Workclass");

            var splits = new[]
            {
                ttSplit.TrainSet,
                ttSplit.TestSet,
                ttSplitWithSeed.TrainSet,
                ttSplitWithSeed.TestSet,
                ttSplitWithSeedAndSamplingKey.TrainSet,
                ttSplitWithSeedAndSamplingKey.TestSet,
                cvSplit.First().TrainSet,
                cvSplit.First().TestSet,
                cvSplitWithSeed.First().TrainSet,
                cvSplitWithSeed.First().TestSet,
                cvSplitWithSeedAndSamplingKey.First().TrainSet,
                cvSplitWithSeedAndSamplingKey.First().TestSet
            };

            // Splitting a dataset shouldn't affect its schema
            foreach (var split in splits)
            {
                Assert.Equal(fullInput.Schema.Count, split.Schema.Count);
                foreach (var col in fullInput.Schema)
                {
                    Assert.Equal(col.Name, split.Schema[col.Index].Name);
                }
            }
        }

        [Fact]
        public void TestTrainTestSplit()
        {
            var mlContext = new MLContext(0);
            var dataPath = GetDataPath("adult.tiny.with-schema.txt");
            // Create the reader: define the data columns and where to find them in the text file.
            var input = mlContext.Data.LoadFromTextFile(dataPath, new[] {
                            new TextLoader.Column("Label", DataKind.Boolean, 0),
                            new TextLoader.Column("Workclass", DataKind.String, 1),
                            new TextLoader.Column("Education", DataKind.String,2),
                            new TextLoader.Column("Age", DataKind.Single,9)
            }, hasHeader: true);
            // this function will accept dataview and return content of "Workclass" column as List of strings.
            Func<IDataView, List<string>> getWorkclass = (IDataView view) =>
            {
                return view.GetColumn<ReadOnlyMemory<char>>(view.Schema["Workclass"]).Select(x => x.ToString()).ToList();
            };

            // Let's test what train test properly works with seed.
            // In order to do that, let's split same dataset, but in one case we will use default seed value,
            // and in other case we set seed to be specific value.
            var simpleSplit = mlContext.Data.TrainTestSplit(input);
            var splitWithSeed = mlContext.Data.TrainTestSplit(input, seed: 10);

            // Since test fraction is 0.1, it's much faster to compare test subsets of split.
            var simpleTestWorkClass = getWorkclass(simpleSplit.TestSet);

            var simpleWithSeedTestWorkClass = getWorkclass(splitWithSeed.TestSet);
            // Validate we get different test sets.
            Assert.NotEqual(simpleTestWorkClass, simpleWithSeedTestWorkClass);

            // Now let's do same thing but with presence of stratificationColumn.
            // Rows with same values in this stratificationColumn should end up in same subset (train or test).
            // So let's break dataset by "Workclass" column.
            var stratSplit = mlContext.Data.TrainTestSplit(input, samplingKeyColumnName: "Workclass");
            var stratTrainWorkclass = getWorkclass(stratSplit.TrainSet);
            var stratTestWorkClass = getWorkclass(stratSplit.TestSet);
            // Let's get unique values for "Workclass" column from train subset.
            var uniqueTrain = stratTrainWorkclass.GroupBy(x => x.ToString()).Select(x => x.First()).ToList();
            // and from test subset.
            var uniqueTest = stratTestWorkClass.GroupBy(x => x.ToString()).Select(x => x.First()).ToList();
            // Validate we don't have intersection between workclass values since we use that column as stratification column
            Assert.True(Enumerable.Intersect(uniqueTrain, uniqueTest).Count() == 0);

            // Let's do same thing, but this time we will choose different seed.
            // Stratification column should still break dataset properly without same values in both subsets.
            var stratSeed = mlContext.Data.TrainTestSplit(input, samplingKeyColumnName: "Workclass", seed: 1000000);
            var stratTrainWithSeedWorkclass = getWorkclass(stratSeed.TrainSet);
            var stratTestWithSeedWorkClass = getWorkclass(stratSeed.TestSet);
            // Let's get unique values for "Workclass" column from train subset.
            var uniqueSeedTrain = stratTrainWithSeedWorkclass.GroupBy(x => x.ToString()).Select(x => x.First()).ToList();
            // and from test subset.
            var uniqueSeedTest = stratTestWithSeedWorkClass.GroupBy(x => x.ToString()).Select(x => x.First()).ToList();

            // Validate we don't have intersection between workclass values since we use that column as stratification column
            Assert.True(Enumerable.Intersect(uniqueSeedTrain, uniqueSeedTest).Count() == 0);
            // Validate we got different test results on same stratification column with different seeds
            Assert.NotEqual(uniqueTest, uniqueSeedTest);
        }

        private sealed class Input
        {
            public int Id { get; set; }
            public string TextStrat { get; set; }
            public float FloatStrat { get; set; }
            [VectorType(4)]
            public float[] VectorStrat { get; set; }
            public DateTime DateTimeStrat { get; set; }
            public DateTimeOffset DateTimeOffsetStrat { get; set; }
            public TimeSpan TimeSpanStrat { get; set; }
        }

        [Fact]
        public void TestSplitsWithSamplingKeyColumn()
        {
            var mlContext = new MLContext(0);
            var input = mlContext.Data.LoadFromEnumerable(new[]
            {
                new Input() {
                    Id = 0, TextStrat = "a", FloatStrat = 3, VectorStrat = new float[]{ 2, 3, 4, 5 }, DateTimeStrat = new DateTime(2002, 2, 23),
                    DateTimeOffsetStrat = new DateTimeOffset(2002, 2, 23, 3, 30, 0, new TimeSpan(1, 0, 0)), TimeSpanStrat = new TimeSpan(2, 0, 0)
                },
                new Input() {
                    Id = 1, TextStrat = "b", FloatStrat = 3, VectorStrat = new float[]{ 1, 2, 3, 4 }, DateTimeStrat = new DateTime(2020, 2, 23),
                    DateTimeOffsetStrat = new DateTimeOffset(2002, 2, 23, 3, 30, 0, new TimeSpan(2, 0, 0)), TimeSpanStrat = new TimeSpan(2, 0, 10)
                },
                new Input() {
                    Id = 2, TextStrat = "c", FloatStrat = 4, VectorStrat = new float[]{ 3, 4, 5, 6 }, DateTimeStrat = new DateTime(2018, 2, 23),
                    DateTimeOffsetStrat = new DateTimeOffset(2002, 2, 23, 3, 30, 0, new TimeSpan(1, 0, 0)), TimeSpanStrat = new TimeSpan(2, 0, 10)
                },
                new Input() {
                    Id = 3, TextStrat = "d", FloatStrat = 4, VectorStrat = new float[]{ 4, 5, 6, 7 }, DateTimeStrat = new DateTime(2016, 2, 23),
                    DateTimeOffsetStrat = new DateTimeOffset(2002, 2, 23, 3, 30, 0, new TimeSpan(2, 0, 0)), TimeSpanStrat = new TimeSpan(2, 0, 0)
                },
                new Input() {
                    Id = 4, TextStrat = "a", FloatStrat = -493.28f, VectorStrat = new float[]{ 2, 3, 4, 5 }, DateTimeStrat = new DateTime(2016, 2, 23),
                    DateTimeOffsetStrat = new DateTimeOffset(2002, 2, 23, 3, 30, 0, new TimeSpan(3, 0, 0)), TimeSpanStrat = new TimeSpan(2, 0, 20)
                },
                new Input() {
                    Id = 5, TextStrat = "b", FloatStrat = -493.28f, VectorStrat = new float[]{ 1, 2, 3, 4 }, DateTimeStrat = new DateTime(2018, 2, 23),
                    DateTimeOffsetStrat = new DateTimeOffset(2002, 2, 23, 3, 30, 0, new TimeSpan(4, 0, 0)), TimeSpanStrat = new TimeSpan(2, 0, 30)
                },
                new Input() {
                    Id = 6, TextStrat = "c", FloatStrat = 6, VectorStrat = new float[]{ 3, 4, 5, 6 }, DateTimeStrat = new DateTime(2020,2 , 23),
                    DateTimeOffsetStrat = new DateTimeOffset(2002, 2, 23, 3, 30, 0, new TimeSpan(3, 0, 0)), TimeSpanStrat = new TimeSpan(2, 0, 30)
                },
                new Input() {
                    Id = 7, TextStrat = "d", FloatStrat = 6, VectorStrat = new float[]{ 4, 5, 6, 7 }, DateTimeStrat = new DateTime(2002, 2, 23),
                    DateTimeOffsetStrat = new DateTimeOffset(2002, 2, 23, 3, 30, 0, new TimeSpan(4, 0, 0)), TimeSpanStrat = new TimeSpan(2, 0, 20)
                },
            });

            // TEST TRAINTESTSPLIT
            var split = mlContext.Data.TrainTestSplit(input, 0.5, nameof(Input.TextStrat));
            var ids = split.TestSet.GetColumn<int>(split.TestSet.Schema[nameof(Input.Id)]);
            Assert.Contains(1, ids);
            Assert.Contains(5, ids);
            split = mlContext.Data.TrainTestSplit(input, 0.5, nameof(Input.FloatStrat));
            ids = split.TestSet.GetColumn<int>(split.TestSet.Schema[nameof(Input.Id)]);
            Assert.Contains(4, ids);
            Assert.Contains(5, ids);
            split = mlContext.Data.TrainTestSplit(input, 0.5, nameof(Input.VectorStrat));
            ids = split.TestSet.GetColumn<int>(split.TestSet.Schema[nameof(Input.Id)]);
            Assert.Contains(0, ids);
            Assert.Contains(4, ids);
            split = mlContext.Data.TrainTestSplit(input, 0.5, nameof(Input.DateTimeStrat));
            ids = split.TestSet.GetColumn<int>(split.TestSet.Schema[nameof(Input.Id)]);
            Assert.Contains(5, ids);
            Assert.Contains(6, ids);
            split = mlContext.Data.TrainTestSplit(input, 0.5, nameof(Input.DateTimeOffsetStrat));
            ids = split.TrainSet.GetColumn<int>(split.TrainSet.Schema[nameof(Input.Id)]);
            Assert.Contains(4, ids);
            Assert.Contains(7, ids);
            split = mlContext.Data.TrainTestSplit(input, 0.5, nameof(Input.TimeSpanStrat));
            ids = split.TestSet.GetColumn<int>(split.TestSet.Schema[nameof(Input.Id)]);
            Assert.Contains(1, ids);
            Assert.Contains(2, ids);

            var inputWithKey = mlContext.Transforms.Conversion.MapValueToKey("KeyStrat", "TextStrat").Fit(input).Transform(input);
            split = mlContext.Data.TrainTestSplit(inputWithKey, 0.5, "KeyStrat");
            ids = split.TestSet.GetColumn<int>(split.TestSet.Schema[nameof(Input.Id)]);
            Assert.Contains(1, ids);
            Assert.Contains(5, ids);
            Assert.NotNull(split.TrainSet.Schema.GetColumnOrNull("KeyStrat")); // Check that the key column used as SamplingKeyColumn wasn't deleted by the split

            // TEST CROSSVALIDATIONSPLIT
            var colnames = new[] {
                nameof(Input.TextStrat),
                nameof(Input.FloatStrat),
                nameof(Input.VectorStrat),
                nameof(Input.DateTimeStrat),
                nameof(Input.DateTimeOffsetStrat),
                nameof(Input.TimeSpanStrat),
                "KeyStrat" };

            foreach (var colname in colnames)
            {
                var cvSplits = mlContext.Data.CrossValidationSplit(inputWithKey, numberOfFolds: 2, samplingKeyColumnName: colname);
                var idsTest1 = cvSplits[0].TestSet.GetColumn<int>(cvSplits[0].TestSet.Schema[nameof(Input.Id)]);
                var idsTest2 = cvSplits[1].TestSet.GetColumn<int>(cvSplits[1].TestSet.Schema[nameof(Input.Id)]);
                Assert.True(Enumerable.Intersect(idsTest1, idsTest2).Count() == 0);
                Assert.True(idsTest1.Count() > 0, $"CV Split 0 for Column {colname} was empty");
                Assert.True(idsTest2.Count() > 0, $"CV Split 1 for Column {colname} was empty");

                // Check that using CV didn't remove the SamplingKeyColumn
                Assert.NotNull(split.TrainSet.Schema.GetColumnOrNull(colname));
            }
        }
    }
}
