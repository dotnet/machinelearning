// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using FakeStaticPipes;
using Microsoft.ML.Data.StaticPipe;
using Microsoft.ML.Data.StaticPipe.Runtime;
using System;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Scenarios
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

    public sealed class StaticPipe : MakeConsoleWork
    {
        public StaticPipe(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void Test()
        {
            var text = TextLoader.Create(
                ctx => (
                label: ctx.LoadBool(0),
                text: ctx.LoadText(1),
                numericFeatures: ctx.LoadFloat(2, 9)
                ));

            var transform = text.CreateTransform(r => (
                r.label,
                features: r.numericFeatures.ConcatWith(r.text.Tokenize().Dictionarize().BagVectorize())
                ));

            var train = transform.CreateTransform(r => (
                r.label.TrainLinearClassification(r.features)
            ));
        }

        [Fact]
        public void Test1()
        {
            // We load two columns, the boolean "label" and the textual "sentimentText".
            var text = TextLoader.Create(
                c => (label: c.LoadBool(0), sentimentText: c.LoadText(1)));

            // We apply the text featurizer transform to "sentimentText" producing the column "features".

            var transformation = text.CreateTransform(r =>
                (r.label, features: r.sentimentText.TextFeaturizer()));

            // We apply a learner to learn "label" given "features", which will in turn produce
            // float "score", float "probability", and boolean "predictedLabel".
            var training = transformation.CreateTransform(r =>
                r.label.TrainLinearClassification(r.features));

        }

        //[Fact]
        //public void TestUnpack()
        //{
        //    UnpackHelper(() => (a: 1, b: 2, c: (d: 3, e: 4)));
        //}

        //private void UnpackHelper<T>(Func<T> func)
        //{
        //    var t = func();
        //    var v = PipelineColumnAnalyzer.GetNames<T, int>(t, func.Method.ReturnParameter);
        //}

        //[Fact]
        //public void TestAnalysisInstance()
        //{
        //    PipelineColumnAnalyzer.Analyze((Key<uint, string> k) =>
        //        (a: default(Scalar<int>), b: default(Vector<float>)));
        //}
    }
}
