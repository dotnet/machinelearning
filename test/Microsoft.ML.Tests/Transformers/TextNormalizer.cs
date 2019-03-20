// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms.Text;
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
            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = new TextNormalizingEstimator(ML, columns: new[] { ("NormA", "A"), ("NormB", "B") });

            var invalidData = new[] { new TestClassB() { A = 1, B = new float[2] { 1,4 } },
                               new TestClassB() { A = 2, B =new float[2]  { 3,4 }  } };
            var invalidDataView = ML.Data.LoadFromEnumerable(invalidData);
            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataView);

            var dataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var reader = TextLoaderStatic.CreateLoader(ML, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true);
            var dataSource = new MultiFileSource(dataPath);
            dataView = reader.Load(dataSource).AsDynamic;

            var pipeVariations = new TextNormalizingEstimator(ML, columns: new[] { ("NormText", "text") }).Append(
                                new TextNormalizingEstimator(ML, caseMode: TextNormalizingEstimator.CaseMode.Upper, columns: new[] { ("UpperText", "text") })).Append(
                                new TextNormalizingEstimator(ML, keepDiacritics: true, columns: new[] { ("WithDiacriticsText", "text") })).Append(
                                new TextNormalizingEstimator(ML, keepNumbers: false, columns: new[] { ("NoNumberText", "text") })).Append(
                                new TextNormalizingEstimator(ML, keepPunctuations: false, columns: new[] { ("NoPuncText", "text") }));

            var outputPath = GetOutputPath("Text", "Normalized.tsv");
                var savedData = ML.Data.TakeRows(pipeVariations.Fit(dataView).Transform(dataView), 5);
                using (var fs = File.Create(outputPath))
                    ML.Data.SaveAsText(savedData, fs, headerRow: true, keepHidden: true);

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
            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = new TextNormalizingEstimator(Env, columns: new[] { ("NormA", "A"), ("NormB", "B") });

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
