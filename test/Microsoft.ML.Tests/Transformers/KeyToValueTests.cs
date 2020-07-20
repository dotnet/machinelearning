// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.Transforms;
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

            var reader = new TextLoader(Env, new TextLoader.Options
            {
                Columns = new[]
                {
                    new TextLoader.Column("ScalarString", DataKind.String, 1),
                    new TextLoader.Column("VectorString", DataKind.String, new[] {new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("BareKey", DataKind.UInt32, new[] { new TextLoader.Range(0) }, new KeyCount(6))
                }
            });

            var data = reader.Load(dataPath);

            data = new ValueToKeyMappingEstimator(Env, new[] {
                new ValueToKeyMappingEstimator.ColumnOptions("A", "ScalarString"),
                new ValueToKeyMappingEstimator.ColumnOptions("B", "VectorString") }).Fit(data).Transform(data);

            var badData1 = new ColumnCopyingTransformer(Env, ("A", "BareKey")).Transform(data);
            var badData2 = new ColumnCopyingTransformer(Env, ("B", "VectorString")).Transform(data);

            var est = new KeyToValueMappingEstimator(Env, ("A_back", "A"), ("B_back", "B"));
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
        public void KeyToValue()
        {
            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var data = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("ScalarString", DataKind.String, 0),
                new TextLoader.Column("VectorString", DataKind.String, 1, 4),
            });

            var transformedData = new ValueToKeyMappingEstimator(Env, new[] {
                new ValueToKeyMappingEstimator.ColumnOptions("A", "ScalarString"),
                new ValueToKeyMappingEstimator.ColumnOptions("B", "VectorString") })
                .Fit(data).Transform(data);

            var est = ML.Transforms.Conversion.MapKeyToValue("ScalarString", "A")
                .Append(ML.Transforms.Conversion.MapKeyToValue("VectorString", "B"));

            TestEstimatorCore(est, transformedData, invalidInput: data);

            var data2Transformed = est.Fit(transformedData).Transform(transformedData);
            // Check that term and ToValue are round-trippable.
            var dataLeft = ML.Transforms.SelectColumns(new[] { "ScalarString", "VectorString" }).Fit(data).Transform(data);
            var dataRight = ML.Transforms.SelectColumns(new[] { "ScalarString", "VectorString" }).Fit(data2Transformed).Transform(data2Transformed);

            TestCommon.CheckSameSchemas(dataLeft.Schema, dataRight.Schema);
            CheckSameValues(dataLeft, dataRight);
            Done();
        }
    }
}
