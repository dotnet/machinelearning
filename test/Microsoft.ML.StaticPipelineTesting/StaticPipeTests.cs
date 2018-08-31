// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data.StaticPipe;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.TestFramework;
using System;
using System.IO;
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

        private void CheckSchemaHasColumn(ISchema schema, string name, out int idx)
            => Assert.True(schema.TryGetColumnIndex(name, out idx), "Could not find column '" + name + "'");

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
            var textData = text.Read(dataSource).AsDynamic;

            var schema = textData.Schema;
            // First verify that the columns are there. There ought to be at least one column corresponding to the identifiers in the tuple.
            CheckSchemaHasColumn(schema, "label", out int labelIdx);
            CheckSchemaHasColumn(schema, "text", out int textIdx);
            CheckSchemaHasColumn(schema, "numericFeatures", out int numericFeaturesIdx);
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

                void CheckValuesSame(bool bl, string tx, float v0, float v1, float v2)
                {
                    labelGetter(ref labelVal);
                    textGetter(ref textVal);
                    numericFeaturesGetter(ref numVal);

                    Assert.Equal((DvBool)bl, labelVal);
                    Assert.Equal(new DvText(tx), textVal);
                    Assert.Equal(3, numVal.Length);
                    Assert.Equal(v0, numVal.GetItemOrDefault(0));
                    Assert.Equal(v1, numVal.GetItemOrDefault(1));
                    Assert.Equal(v2, numVal.GetItemOrDefault(2));
                }

                Assert.True(cursor.MoveNext(), "Could not move even to first row");
                CheckValuesSame(false, "hello", 3.14159f, -0f, 2f);
                Assert.True(cursor.MoveNext(), "Could not move to second row");
                CheckValuesSame(true, "1", 2f, 4f, 15f);
                Assert.False(cursor.MoveNext(), "Moved to third row, but there should have been only two");
            }

            // The next step where we shuffle the names around a little bit is one where we are
            // testing out the implicit usage of copy columns.

            var est = Estimator.MakeNew(text).Append(r => (text: r.label, label: r.numericFeatures));
            var newText = text.Append(est);
            var newTextData = newText.Fit(dataSource).Read(dataSource);

            schema = newTextData.AsDynamic.Schema;
            // First verify that the columns are there. There ought to be at least one column corresponding to the identifiers in the tuple.
            CheckSchemaHasColumn(schema, "label", out labelIdx);
            CheckSchemaHasColumn(schema, "text", out textIdx);
            // Next verify they have the expected types.
            Assert.Equal(BoolType.Instance, schema.GetColumnType(textIdx));
            Assert.Equal(new VectorType(NumberType.R4, 3), schema.GetColumnType(labelIdx));
        }
    }
}
