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
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class TestApi : BaseTestClass
    {
        private const string OutputRelativePath = @"..\Common\Api";

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
            var dataFile = GetDataPath("breast-cancer.txt");
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
            var valueFloat = 10;
            var coltypeFloat = NumberDataViewType.Single;
            var kindString = "Testing string as metadata.";
            var valueString = "Strings have value.";
            var kindStringArray = "Testing string array as metadata.";
            var valueStringArray = "I really have no idea what these features entail.".Split(' ');
            var kindFloatArray = "Testing float array as metadata.";
            var valueFloatArray = new float[] { 1, 17, 7, 19, 25, 0 };
            var kindVBuffer = "Testing VBuffer as metadata.";
            var valueVBuffer = new VBuffer<float>(4, new float[] { 4, 6, 89, 5 });

            var metaFloat = new AnnotationInfo<float>(kindFloat, valueFloat, coltypeFloat);
            var metaString = new AnnotationInfo<string>(kindString, valueString);

            // Add Metadata.
            var labelColumn = autoSchema[0];
            var labelColumnWithMetadata = new SchemaDefinition.Column(mlContext, labelColumn.MemberName, labelColumn.ColumnType,
                annotationInfos: new AnnotationInfo[] { metaFloat, metaString });

            var featureColumnWithMetadata = autoSchema[1];
            featureColumnWithMetadata.AddAnnotation(kindStringArray, valueStringArray);
            featureColumnWithMetadata.AddAnnotation(kindFloatArray, valueFloatArray);
            featureColumnWithMetadata.AddAnnotation(kindVBuffer, valueVBuffer);

            var mySchema = new SchemaDefinition { labelColumnWithMetadata, featureColumnWithMetadata };
            var idv = mlContext.Data.LoadFromEnumerable(data, mySchema);

            Assert.True(idv.Schema[0].Annotations.Schema.Count == 2);
            Assert.True(idv.Schema[0].Annotations.Schema[0].Name == kindFloat);
            Assert.True(idv.Schema[0].Annotations.Schema[0].Type == coltypeFloat);
            Assert.True(idv.Schema[0].Annotations.Schema[1].Name == kindString);
            Assert.True(idv.Schema[0].Annotations.Schema[1].Type == TextDataViewType.Instance);

            Assert.True(idv.Schema[1].Annotations.Schema.Count == 3);
            Assert.True(idv.Schema[1].Annotations.Schema[0].Name == kindStringArray);
            Assert.True(idv.Schema[1].Annotations.Schema[0].Type is VectorType vectorType && vectorType.ItemType is TextDataViewType);
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
            var dataFile = GetDataPath("breast-cancer.txt");

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
            var stratSeed = mlContext.Data.TrainTestSplit(input, samplingKeyColumnName:"Workclass", seed: 1000000);
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

    }
}
