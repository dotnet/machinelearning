// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Transforms;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class TextNormalizerTests : TestDataPipeBase
    {
        public TextNormalizerTests(ITestOutputHelper output) : base(output)
        {
        }

        private class TestClass
        {
            public string A;
            [VectorType(2)]
            public string[] B;
        }


        private class TestClassB
        {
            public float A;
            [VectorType(2)]
            public float[] B;
        }

        [Fact]
        public void TextNormalizerWorkout()
        {
            var data = new[] { new TestClass() { A = "A 1, b. c! йЁ 24 ", B = new string[2] { "~``ё 52ds й vc", "6ksj94 vd ё dakl Юds Ё q й" } },
                               new TestClass() { A = null, B =new string[2]  { null, string.Empty }  } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new TextNormalizerEstimator(Env, columns: new[] { ("A", "NormA"), ("B", "NormB") });

            var invalidData = new[] { new TestClassB() { A = 1, B = new float[2] { 1,4 } },
                               new TestClassB() { A = 2, B =new float[2]  { 3,4 }  } };
            var invalidDataView = ComponentCreation.CreateDataView(Env, invalidData);
            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataView);

            var dataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var reader = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true);
            var dataSource = new MultiFileSource(dataPath);
            dataView = reader.Read(dataSource).AsDynamic;

            var pipeVariations = new TextNormalizerEstimator(Env, columns: new[] { ("text", "NormText") }).Append(
                                new TextNormalizerEstimator(Env, textCase: TextNormalizerEstimator.CaseNormalizationMode.Upper, columns: new[] { ("text", "UpperText") })).Append(
                                new TextNormalizerEstimator(Env, keepDiacritics: true, columns: new[] { ("text", "WithDiacriticsText") })).Append(
                                new TextNormalizerEstimator(Env, keepNumbers: false, columns: new[] { ("text", "NoNumberText") })).Append(
                                new TextNormalizerEstimator(Env, keepPunctuations: false, columns: new[] { ("text", "NoPuncText") }));

            var outputPath = GetOutputPath("Text", "Normalized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                var savedData = TakeFilter.Create(Env, pipeVariations.Fit(dataView).Transform(dataView), 5);
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("Text", "Normalized.tsv");
            Done();
        }

        [Fact]
        public void TestCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:TX:0} xf=TextNorm{col=B:A} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = "A 1, b. c! йЁ 24 ", B = new string[2] { "~``ё 52ds й vc", "6ksj94 vd ё dakl Юds Ё q й" } } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new TextNormalizerEstimator(Env, columns: new[] { ("A", "NormA"), ("B", "NormB") });

            var result = pipe.Fit(dataView).Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);
            }
        }
    }
}
