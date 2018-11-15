// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.FeatureSelection;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public sealed class DropSlotsTests : TestDataPipeBase
    {
        public DropSlotsTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void DropSlotsTransform()
        {
            var env = new ConsoleEnvironment(seed: 0);

            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoader.CreateReader(env, ctx => (
                ScalarFloat: ctx.LoadFloat(1),
                ScalarDouble: ctx.LoadDouble(1),
                VectorFloat: ctx.LoadFloat(1, 4),
                VectorDoulbe: ctx.LoadDouble(1, 4)
            ));

            var data = reader.Read(new MultiFileSource(dataPath)).AsDynamic;

            var trans = new DropSlotsTransform(env, "features", "dropped", max: 5);

            var outputPath = GetOutputPath("DropSlots", "dropslots.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true, OutputHeader = false });
                IDataView savedData = TakeFilter.Create(Env, trans.Transform(data), 4);
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("DropSlots", "dropslots.tsv");
            Done();
        }

        //[Fact]
        //public void TestWhiteningCommandLine()
        //{
        //    Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0-10} xf=whitening{col=B:A} in=f:\2.txt" }), (int)0);
        //}

        //[Fact]
        //public void TestWhiteningOldSavingAndLoading()
        //{
        //    var env = new ConsoleEnvironment(seed: 0);
        //    string dataSource = GetDataPath("generated_regression_dataset.csv");
        //    var dataView = TextLoader.CreateReader(env,
        //        c => (label: c.LoadFloat(11), features: c.LoadFloat(0, 10)),
        //        separator: ';', hasHeader: true)
        //        .Read(dataSource).AsDynamic;
        //    var pipe = new VectorWhiteningEstimator(env, "features", "whitened");

        //    var result = pipe.Fit(dataView).Transform(dataView);
        //    var resultRoles = new RoleMappedData(result);
        //    using (var ms = new MemoryStream())
        //    {
        //        TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
        //        ms.Position = 0;
        //        var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);
        //    }
        //    Done();
        //}
    }
}
