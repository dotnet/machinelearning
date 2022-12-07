// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Transforms.Text;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public sealed class WordBagTransformerTests : TestDataPipeBase
    {
        public WordBagTransformerTests(ITestOutputHelper helper) : base(helper)
        {
        }

        [Fact]
        public void WordBagsPreDefined()
        {
            var mlContext = new MLContext(1);
            var samples = new List<TextData>()
            {
                new TextData(){ Text = "div:12;strong:9;span:13;br:1;a:2" },
                new TextData(){ Text = "p:5;strong:1;code:2;br:2;a:2;img:1;span:6;script:1" },
                new TextData(){ Text = "p:5" },
                new TextData(){ Text = "p" },
                new TextData(){ Text = "-1" },
            };

            var dataview = mlContext.Data.LoadFromEnumerable(samples);
            var textPipeline =
                mlContext.Transforms.Text.ProduceWordBags("Text", termSeparator: ';', freqSeparator: ':');


            var textTransformer = textPipeline.Fit(dataview);
            var pred = textTransformer.Preview(dataview);

            var expected = new float[] { 12, 9, 13, 1, 2, 0, 0, 0, 0, 0 };

            Assert.Equal(expected, ((VBuffer<float>)pred.ColumnView[4].Values[0]).DenseValues().ToArray());

            TestEstimatorCore(textPipeline, dataview);
            Done();
        }

        [Fact]
        public void WordBagsPreDefinedNonDefault()
        {
            var mlContext = new MLContext(1);
            var samples = new List<TextData>()
            {
                new TextData(){ Text = "div;12:strong;9:span;13:br;1:a;2" },
                new TextData(){ Text = "p;5:strong;1:code;2:br;2:a;2:img;1:span;6:script;1" },
                new TextData(){ Text = "p;5" },
                new TextData(){ Text = "p" },
                new TextData(){ Text = "-1" },
            };

            var dataview = mlContext.Data.LoadFromEnumerable(samples);
            var textPipeline =
                mlContext.Transforms.Text.ProduceWordBags("Text", termSeparator: ':', freqSeparator: ';');


            var textTransformer = textPipeline.Fit(dataview);
            var pred = textTransformer.Preview(dataview);
            var expected = new float[] { 12, 9, 13, 1, 2, 0, 0, 0, 0, 0 };

            Assert.Equal(expected, ((VBuffer<float>)pred.ColumnView[4].Values[0]).DenseValues().ToArray());

            TestEstimatorCore(textPipeline, dataview);
            Done();
        }

        [Fact]
        public void WordBagsPreDefinedMakeSureDefaultAndNonDefaultHaveSameOutput()
        {
            var mlContext = new MLContext(1);

            var samplesDefault = new List<TextData>()
            {
                new TextData(){ Text = "div:12;strong:9;span:13;br:1;a:2" },
                new TextData(){ Text = "p:5;strong:1;code:2;br:2;a:2;img:1;span:6;script:1" },
                new TextData(){ Text = "p:5" },
                new TextData(){ Text = "p" },
                new TextData(){ Text = "-1" },
            };

            var samplesNonDefault = new List<TextData>()
            {
                new TextData(){ Text = "div;12:strong;9:span;13:br;1:a;2" },
                new TextData(){ Text = "p;5:strong;1:code;2:br;2:a;2:img;1:span;6:script;1" },
                new TextData(){ Text = "p;5" },
                new TextData(){ Text = "p" },
                new TextData(){ Text = "-1" },
            };

            var dataviewDefault = mlContext.Data.LoadFromEnumerable(samplesDefault);
            var dataviewNonDefault = mlContext.Data.LoadFromEnumerable(samplesNonDefault);
            var textPipelineDefault = mlContext.Transforms.Text.ProduceWordBags("Text", termSeparator: ';', freqSeparator: ':');
            var textPipelineNonDefault = mlContext.Transforms.Text.ProduceWordBags("Text", termSeparator: ':', freqSeparator: ';');


            var textTransformerDefault = textPipelineDefault.Fit(dataviewDefault);
            var textTransformerNonDefault = textPipelineNonDefault.Fit(dataviewNonDefault);
            var predDefault = textTransformerDefault.Preview(dataviewDefault);
            var predNonDefault = textTransformerNonDefault.Preview(dataviewNonDefault);

            Assert.Equal(((VBuffer<float>)predDefault.ColumnView[4].Values[0]).DenseValues().ToArray(), ((VBuffer<float>)predNonDefault.ColumnView[4].Values[0]).DenseValues().ToArray());

            Done();
        }

        private class TextData
        {
            public string Text { get; set; }
        }

        private class TransformedTextData
        {
            public float[] Text { get; set; }
        }
    }

}
