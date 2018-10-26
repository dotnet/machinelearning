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
            public string OutField;

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
            public string OutField;

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
            using (var env = new ConsoleEnvironment(0))
            {
                // Correct use of CursorChannel attribute.
                var data1 = Utils.CreateArray(10, new OneIChannelWithAttribute());
                var idv1 = env.CreateDataView(data1);
                Assert.Null(data1[0].Channel);

                var map1 = new CustomMappingTransformer<OneIChannelWithAttribute, OneIChannelWithAttribute>(env,
                   (input, output) =>
                   {
                       output.OutField = input.OutField + input.OutField;
                   }, null).Transform(idv1);
                map1.GetRowCursor(col => true);

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

                var map2 = new CustomMappingTransformer<OneStringWithAttribute, OneStringWithAttribute>(env,
                    (input, output) =>
                    {
                        output.OutField = input.OutField + input.OutField;
                    }, null).Transform(idv2);
                try
                {
                    map2.GetRowCursor(col => true);
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

                var map3 = new CustomMappingTransformer<TwoIChannelsWithAttributes, TwoIChannelsWithAttributes>(env,
                    (input, output) =>
                    {
                        output.OutField = input.OutField + input.OutField;
                    }, null).Transform(idv3);
                try
                {
                    map3.GetRowCursor(col => true);
                    Assert.True(false, "Throw an error if attribute is applied to a multiple fields.");
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

                var map4 = new CustomMappingTransformer<TwoIChannelsOnlyOneWithAttribute, TwoIChannelsOnlyOneWithAttribute>(env,
                    (input, output) => { }, null).Transform(idv4);
                map4.GetRowCursor(col => true);

                var filter4 = LambdaTransform.CreateFilter<TwoIChannelsOnlyOneWithAttribute, object>(env, idv4,
                    (input, state) =>
                    {
                        Assert.Null(input.ChannelOne);
                        Assert.NotNull(input.ChannelTwo);
                        return false;
                    }, null);
                filter1.GetRowCursor(col => true).MoveNext();
            }
        }

        public class BreastCancerExample
        {
            public float Label;

            [VectorType(9)]
            public float[] Features;
        }

        private class LambdaOutput
        {
            public string OutField;
        }

        [Fact]
        public void LambdaTransformCreate()
        {
            using (var env = new ConsoleEnvironment(42))
            {
                var data = ReadBreastCancerExamples();
                var idv = env.CreateDataView(data);

                var map = new CustomMappingTransformer<BreastCancerExample, LambdaOutput>(env,
                    (input, output) =>
                    {
                        output.OutField = string.Join(";", input.Features);
                    }, null).Transform(idv);

                var filter = LambdaTransform.CreateFilter<BreastCancerExample, object>(env, map,
                    (input, state) => input.Label == 0, null);

                Assert.Null(filter.GetRowCount(false));

                // test re-apply
                var applied = env.CreateDataView(data);
                applied = ApplyTransformUtils.ApplyAllTransformsToData(env, filter, applied);

                var saver = new TextSaver(env, new TextSaver.Arguments());
                Assert.True(applied.Schema.TryGetColumnIndex("Label", out int label));
                Assert.True(applied.Schema.TryGetColumnIndex("OutField", out int outField));
                using (var fs = File.Create(GetOutputPath(OutputRelativePath, "lambda-output.tsv")))
                    saver.SaveData(fs, applied, label, outField);
            }
        }

        [Fact]
        public void TrainAveragedPerceptronWithCache()
        {
            using (var env = new ConsoleEnvironment(0))
            {
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
