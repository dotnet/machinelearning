// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Transforms;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public sealed class KeyToValueTests : TestDataPipeBase
    {
        public KeyToValueTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void KeyToValueWorkout()
        {
            string dataPath = GetDataPath("iris.txt");

            var reader = new TextLoader(Env, new TextLoader.Arguments
            {
                Column = new[]
                {
                    new TextLoader.Column("ScalarString", DataKind.TX, 1),
                    new TextLoader.Column("VectorString", DataKind.TX, new[] {new TextLoader.Range(1, 4) }),
                    new TextLoader.Column
                    {
                        Name="BareKey",
                        Source = new[] { new TextLoader.Range(0) },
                        Type = DataKind.U4,
                        KeyRange = new KeyRange(0, 5),
                    }
                }
            });

            var data = reader.Read(dataPath);

            data = new TermEstimator(Env,
                new TermTransform.ColumnInfo("ScalarString", "A"),
                new TermTransform.ColumnInfo("VectorString", "B")).Fit(data).Transform(data);

            var badData1 = new CopyColumnsTransform(Env, ("BareKey", "A")).Transform(data);
            var badData2 = new CopyColumnsTransform(Env, ("VectorString", "B")).Transform(data);

            var est = new KeyToValueEstimator(Env, ("A", "A_back"), ("B", "B_back"));
            TestEstimatorCore(est, data, invalidInput: badData1);
            TestEstimatorCore(est, data, invalidInput: badData2);


            var outputPath = GetOutputPath("KeyToValue", "featurized.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true });
                IDataView savedData = est.Fit(data).Transform(data);
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, savedData, fs, keepHidden: true);
            }

            CheckEquality("KeyToValue", "featurized.tsv");
            Done();
        }

        [Fact]
        public void KeyToValuePigsty()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var reader = TextLoader.CreateReader(Env, ctx => (
                ScalarString: ctx.LoadText(1),
                VectorString: ctx.LoadText(1, 4)
            ));

            var data = reader.Read(dataPath);

            // Non-pigsty Term.
            var dynamicData = new TermEstimator(Env,
                new TermTransform.ColumnInfo("ScalarString", "A"),
                new TermTransform.ColumnInfo("VectorString", "B"))
                .Fit(data.AsDynamic).Transform(data.AsDynamic);

            var data2 = dynamicData.AssertStatic(Env, ctx => (
                A: ctx.KeyU4.TextValues.Scalar,
                B: ctx.KeyU4.TextValues.Vector));

            var est = data2.MakeNewEstimator()
                .Append(row => (
                ScalarString: row.A.ToValue(),
                VectorString: row.B.ToValue()));

            TestEstimatorCore(est.AsDynamic, data2.AsDynamic, invalidInput: data.AsDynamic);

            // Check that term and ToValue are round-trippable.
            var dataLeft = new ChooseColumnsTransform(Env, data.AsDynamic, "ScalarString", "VectorString");
            var dataRight = new ChooseColumnsTransform(Env, est.Fit(data2).Transform(data2).AsDynamic, "ScalarString", "VectorString");
            CheckSameSchemas(dataLeft.Schema, dataRight.Schema);
            CheckSameValues(dataLeft, dataRight);
            Done();
        }
    }
}
