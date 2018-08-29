// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using FakeStaticPipes;
using Microsoft.ML.Data.StaticPipe;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.TestFramework;
using System;
using System.IO;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.StaticPipelineTesting
{

    public abstract class MakeConsoleWork : IDisposable
    {
        private readonly ITestOutputHelper _output;
        private readonly TextWriter _originalOut;
        private readonly TextWriter _textWriter;

        public MakeConsoleWork(ITestOutputHelper output)
        {
            _output = output;
            _originalOut = Console.Out;
            _textWriter = new StringWriter();
            Console.SetOut(_textWriter);
        }

        public void Dispose()
        {
            _output.WriteLine(_textWriter.ToString());
            Console.SetOut(_originalOut);
        }
    }

    public sealed class StaticPipeTests : MakeConsoleWork
    {
        public StaticPipeTests(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void SimpleTextLoaderCopyColumnsTest()
        {
            var env = new TlcEnvironment(new SysRandom(0), verbose: true);

            const string data = "0 hello 3.14159 -0 2\n"
                + "1 1 2 4 15";
            var dataSource = new BytesStreamSource(data);

            var text = TextLoader.CreateReader(env, ctx => (
                label: ctx.LoadBool(0),
                text: ctx.LoadText(1),
                numericFeatures: ctx.LoadFloat(2, null)), // If fit correctly, this ought to be equivalent to max of 4, that is, length of 3.
                dataSource, separator: ' ');

            // While we have a type-safe wrapper for `IDataView` it is utterly useless except as an input to the `Fit` functions
            // of the other statically typed wrappers. We perhaps ought to make it useful in its own right, but perhaps not now.
            // For now, just operate over the actual `IDataView`.
            var textData = text.Read(dataSource).Wrapped;

            var schema = textData.Schema;
            // First verify that the columns are there. There ought to be at least one column corresponding to the identifiers in the tuple.
            Assert.True(schema.TryGetColumnIndex("label", out int labelIdx), "Could not find column 'label'");
            Assert.True(schema.TryGetColumnIndex("text", out int textIdx), "Could not find column 'text'");
            Assert.True(schema.TryGetColumnIndex("numericFeatures", out int numericFeaturesIdx), "Could not find column 'numericFeatures'");
            // Next verify they have the expected types.
            Assert.Equal(BoolType.Instance, schema.GetColumnType(labelIdx));
            Assert.Equal(TextType.Instance, schema.GetColumnType(textIdx));
            Assert.Equal(new VectorType(NumberType.R4, 3), schema.GetColumnType(numericFeaturesIdx));
            // Next actually inspect the data.
            using (var cursor = textData.GetRowCursor(c => true))
            {
                var labelGetter = cursor.GetGetter<DvBool>(labelIdx);
                var textGetter = cursor.GetGetter<DvText>(textIdx);
                var numericFeaturesGetter = cursor.GetGetter<VBuffer<float>>(numericFeaturesIdx);

                DvBool labelVal = default;
                DvText textVal = default;
                VBuffer<float> numVal = default;

                Assert.True(cursor.MoveNext(), "Could not move even to first row");

                labelGetter(ref labelVal);
                textGetter(ref textVal);
                numericFeaturesGetter(ref numVal);

                Assert.Equal(DvBool.False, labelVal);
                Assert.Equal(new DvText("hello"), textVal);
                Assert.Equal(3, numVal.Length);
                Assert.Equal(3.14159f, numVal.GetItemOrDefault(0));
                Assert.Equal(-0f, numVal.GetItemOrDefault(1));
                Assert.Equal(2f, numVal.GetItemOrDefault(2));

                Assert.True(cursor.MoveNext(), "Could not move to second row");

                labelGetter(ref labelVal);
                textGetter(ref textVal);
                numericFeaturesGetter(ref numVal);

                Assert.Equal(DvBool.True, labelVal);
                Assert.Equal(new DvText("1"), textVal);
                Assert.Equal(3, numVal.Length);
                Assert.Equal(2f, numVal.GetItemOrDefault(0));
                Assert.Equal(4f, numVal.GetItemOrDefault(1));
                Assert.Equal(15f, numVal.GetItemOrDefault(2));

                Assert.False(cursor.MoveNext(), "Moved to third row, but there should have been only two");
            }

            // The next step where we shuffle the names around a little bit is one where we are
            // testing out the implicit usage of copy columns.

            var est = text.CreateEstimator(r => (text: r.label, label: r.numericFeatures));
            var newText = text.Append(est);
            var newTextData = newText.Fit(dataSource).Read(dataSource);

            schema = newTextData.Wrapped.Schema;
            // First verify that the columns are there. There ought to be at least one column corresponding to the identifiers in the tuple.
            Assert.True(schema.TryGetColumnIndex("label", out labelIdx), "Could not find column 'label'");
            Assert.True(schema.TryGetColumnIndex("text", out textIdx), "Could not find column 'text'");
            // Next verify they have the expected types.
            Assert.Equal(BoolType.Instance, schema.GetColumnType(textIdx));
            Assert.Equal(new VectorType(NumberType.R4, 3), schema.GetColumnType(labelIdx));
        }
    }
}
