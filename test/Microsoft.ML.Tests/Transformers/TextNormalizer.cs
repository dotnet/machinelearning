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
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class TextNormalizer : TestDataPipeBase
    {
        public TextNormalizer(ITestOutputHelper output) : base(output)
        {
        }

        private class TestClass
        {
            public string A;
            [VectorType(2)]
            public string[] B;
        }

        [Fact]
        public void TextNormalizerWorkout()
        {
            var data = new[] { new TestClass() { A = "A 1, b. c! йЁ 24 ", B = new string[2] { "~``ё 52ds й vc", "6ksj94 vd ё dakl Юds Ё q й" } } };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new TextNormalizerEstimator(Env, new[]{
                    new TextNormalizerTransform.ColumnInfo("A", "NormA"),
                    new TextNormalizerTransform.ColumnInfo("B", "NormB"),
                });

            TestEstimatorCore(pipe, dataView);
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
            var pipe = new TextNormalizerEstimator(Env, new[]{
                    new TextNormalizerTransform.ColumnInfo("A", "NormA"),
                    new TextNormalizerTransform.ColumnInfo("B", "NormB"),
                });
            var result = pipe.Fit(dataView).Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);
            }
        }

        [Fact]
        public void TextNormalizeStatic()
        {
            var dataPath = GetDataPath("wikipedia-detox-250-line-data.tsv");
            var reader = TextLoader.CreateReader(Env, ctx => (
                    label: ctx.LoadBool(0),
                    text: ctx.LoadText(1)), hasHeader: true);
            var dataSource = new MultiFileSource(dataPath);
            var data = reader.Read(dataSource);

            var est = data.MakeNewEstimator()
                .Append(r => (
                    r.label,
                    norm: r.text.NormalizeText(),
                    norm_Upper: r.text.NormalizeText(textCase:TextNormalizerEstimator.CaseNormalizationMode.Upper),
                    norm_KeepDiacritics: r.text.NormalizeText(keepDiacritics:true),
                    norm_NoPuctuations: r.text.NormalizeText(keepPunctuations: false),
                    norm_NoNumbers: r.text.NormalizeText(keepNumbers:false)));

            var outputPath = GetOutputPath("Text", "Normalized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                var savedData = TakeFilter.Create(Env, est.Fit(data).Transform(data).AsDynamic, 5);
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("Text", "Normalized.tsv");
            Done();
        }
    }
}
