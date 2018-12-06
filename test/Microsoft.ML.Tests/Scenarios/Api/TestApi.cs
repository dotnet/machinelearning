// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Trainers.Online;
using Microsoft.ML.TestFramework;
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
            var idv1 = env.CreateDataView(data1);
            Assert.Null(data1[0].Channel);

            var filter1 = LambdaTransform.CreateFilter<OneIChannelWithAttribute, object>(env, idv1,
                (input, state) =>
                {
                    Assert.NotNull(input.Channel);
                    return false;
                }, null);
            filter1.GetRowCursor(col => true).MoveNext();

            // Error case: non-IChannel field marked with attribute.
            var data2 = Utils.CreateArray(10, new OneStringWithAttribute());
            var idv2 = env.CreateDataView(data2);
            Assert.Null(data2[0].Channel);

            var filter2 = LambdaTransform.CreateFilter<OneStringWithAttribute, object>(env, idv2,
                (input, state) =>
                {
                    Assert.Null(input.Channel);
                    return false;
                }, null);
            try
            {
                filter2.GetRowCursor(col => true).MoveNext();
                Assert.True(false, "Throw an error if attribute is applied to a field that is not an IChannel.");
            }
            catch (InvalidOperationException ex)
            {
                Assert.True(ex.IsMarked());
            }

            // Error case: multiple fields marked with attributes.
            var data3 = Utils.CreateArray(10, new TwoIChannelsWithAttributes());
            var idv3 = env.CreateDataView(data3);
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
                filter3.GetRowCursor(col => true).MoveNext();
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
            var idv4 = env.CreateDataView(Utils.CreateArray(10, example4));

            var filter4 = LambdaTransform.CreateFilter<TwoIChannelsOnlyOneWithAttribute, object>(env, idv4,
                (input, state) =>
                {
                    Assert.Null(input.ChannelOne);
                    Assert.NotNull(input.ChannelTwo);
                    return false;
                }, null);
            filter1.GetRowCursor(col => true).MoveNext();
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
            var idv = env.CreateDataView(data);

            var filter = LambdaTransform.CreateFilter<BreastCancerExample, object>(env, idv,
                (input, state) => input.Label == 0, null);

            Assert.Null(filter.GetRowCount());

            // test re-apply
            var applied = env.CreateDataView(data);
            applied = ApplyTransformUtils.ApplyAllTransformsToData(env, filter, applied);

            var saver = new TextSaver(env, new TextSaver.Arguments());
            Assert.True(applied.Schema.TryGetColumnIndex("Label", out int label));
            using (var fs = File.Create(GetOutputPath(OutputRelativePath, "lambda-output.tsv")))
                saver.SaveData(fs, applied, label);
        }

        [Fact]
        public void TrainAveragedPerceptronWithCache()
        {
            var env = new MLContext(0);
            var dataFile = GetDataPath("breast-cancer.txt");
            var loader = TextLoader.Create(env, new TextLoader.Arguments(), new MultiFileSource(dataFile));
            var globalCounter = 0;
            var xf = LambdaTransform.CreateFilter<object, object>(env, loader,
                (i, s) => true,
                s => { globalCounter++; });

            new AveragedPerceptronTrainer(env, "Label", "Features", numIterations: 2).Fit(xf).Transform(xf);

            // Make sure there were 2 cursoring events.
            Assert.Equal(1, globalCounter);
        }

        [Fact]
        public void MetadataSupportInDataViewConstruction()
        {
            var data = ReadBreastCancerExamples();
            using (var env = new ConsoleEnvironment(0))
            {
                var autoSchema = SchemaDefinition.Create(typeof(BreastCancerExample));

                // Create Metadata.
                var kindFloat = "Testing float as metadata.";
                var valueFloat = 10;
                var coltypeFloat = NumberType.Float;
                var kindString = "Testing string as metadata.";
                var valueString = "Strings have value.";
                var kindStringArray = "Testing string array as metadata.";
                var valueStringArray = "I really have no idea what these features entail.".Split(' ');
                var kindFloatArray = "Testing float array as metadata.";
                var valueFloatArray = new float[] { 1, 17, 7, 19, 25, 0 };
                var kindVBuffer = "Testing VBuffer as metadata.";
                var valueVBuffer = new VBuffer<float>(4, new float[] { 4, 6, 89, 5 });

                var metaFloat = new Microsoft.ML.Runtime.Api.MetadataInfo<float>(kindFloat, valueFloat, coltypeFloat);
                var metaString = new Microsoft.ML.Runtime.Api.MetadataInfo<string>(kindString, valueString);

                // Add Metadata.
                var labelColumn = autoSchema[0];
                var labelColumnWithMetadata = new SchemaDefinition.Column(env, labelColumn.MemberName, labelColumn.ColumnType,
                    metadataInfos: new Microsoft.ML.Runtime.Api.MetadataInfo[] { metaFloat, metaString });

                var featureColumnWithMetadata = autoSchema[1];
                featureColumnWithMetadata.AddMetadata(kindStringArray, valueStringArray);
                featureColumnWithMetadata.AddMetadata(kindFloatArray, valueFloatArray);
                featureColumnWithMetadata.AddMetadata(kindVBuffer, valueVBuffer);

                var mySchema = new SchemaDefinition { labelColumnWithMetadata, featureColumnWithMetadata };
                var idv = env.CreateDataView(data, mySchema);

                // Test GetMetadataTypes.
                var internalSchemaLabelMetadataTypes = idv.Schema.GetMetadataTypes(0).ToArray();
                var internalSchemaFeatureMetadataTypes = idv.Schema.GetMetadataTypes(1).ToArray();

                Assert.True(internalSchemaLabelMetadataTypes.SequenceEqual(
                    new[] { new KeyValuePair<string, ColumnType>(kindFloat, coltypeFloat) ,
                            new KeyValuePair<string, ColumnType>(kindString, TextType.Instance)}));

                Assert.True(internalSchemaFeatureMetadataTypes.Length == 3);
                Assert.Equal(internalSchemaFeatureMetadataTypes[0].Key, kindStringArray);
                Assert.True(internalSchemaFeatureMetadataTypes[0].Value.IsVector && internalSchemaFeatureMetadataTypes[0].Value.ItemType.IsText);

                // Test GetMetaDataTypeOrNull.
                Assert.True(idv.Schema.GetMetadataTypeOrNull(kindFloat, 0) == coltypeFloat);
                Assert.Null(idv.Schema.GetMetadataTypeOrNull(kindFloat, 1));

                // Test GetMetadata.
                float retrievedFloat = 0;
                idv.Schema.GetMetadata(kindFloat, 0, ref retrievedFloat);
                Assert.True(Math.Abs(retrievedFloat - valueFloat) < .000001);

                ReadOnlyMemory<char> retrievedReadOnlyMemory = new ReadOnlyMemory<char>();
                idv.Schema.GetMetadata(kindString, 0, ref retrievedReadOnlyMemory);
                Assert.True(retrievedReadOnlyMemory.Span.SequenceEqual(valueString.AsMemory().Span));

                VBuffer<ReadOnlyMemory<char>> retrievedReadOnlyMemoryVBuffer = new VBuffer<ReadOnlyMemory<char>>();
                idv.Schema.GetMetadata(kindStringArray, 1, ref retrievedReadOnlyMemoryVBuffer);
                Assert.True(retrievedReadOnlyMemoryVBuffer.DenseValues().Select((s, i) => s.ToString() == valueStringArray[i]).All(b => b));

                VBuffer<float> retrievedFloatVBuffer = new VBuffer<float>(1, new float[] { 2 });
                idv.Schema.GetMetadata(kindFloatArray, 1, ref retrievedFloatVBuffer);
                VBuffer<float> valueFloatVBuffer = new VBuffer<float>(valueFloatArray.Length, valueFloatArray);
                Assert.True(retrievedFloatVBuffer.Items().SequenceEqual(valueFloatVBuffer.Items()));

                VBuffer<float> retrievedVBuffer = new VBuffer<float>();
                idv.Schema.GetMetadata(kindVBuffer, 1, ref retrievedVBuffer);
                Assert.True(retrievedVBuffer.Items().SequenceEqual(valueVBuffer.Items()));

                try
                {
                    idv.Schema.GetMetadata(kindFloat, 1, ref retrievedReadOnlyMemoryVBuffer);
                    Assert.True(false, "Throw an error if attribute is applied to a field that is not an IChannel.");
                }
                catch (Exception ex)
                {
                    Assert.True(ex.IsMarked());
                }
            }
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
    }
}
