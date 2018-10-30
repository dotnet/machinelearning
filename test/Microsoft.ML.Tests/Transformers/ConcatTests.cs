// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public sealed class ConcatTests : TestDataPipeBase
    {
        public ConcatTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        void TestConcat()
        {
            string dataPath = GetDataPath("adult.test");

            var source = new MultiFileSource(dataPath);
            var loader = new TextLoader(Env, new TextLoader.Arguments
            {
                Column = new[]{
                    new TextLoader.Column("float1", DataKind.R4, 0),
                    new TextLoader.Column("float4", DataKind.R4, new[]{new TextLoader.Range(0), new TextLoader.Range(2), new TextLoader.Range(4), new TextLoader.Range(10) }),
                    new TextLoader.Column("float6", DataKind.R4, new[]{new TextLoader.Range(0), new TextLoader.Range(2), new TextLoader.Range(4), new TextLoader.Range(10, 12) }),
                    new TextLoader.Column("vfloat", DataKind.R4, new[]{new TextLoader.Range(14, null) { AutoEnd = false, VariableEnd = true } })
                },
                Separator = ",",
                HasHeader = true
            }, new MultiFileSource(dataPath));
            var data = loader.Read(source);

            ColumnType GetType(Schema schema, string name)
            {
                Assert.True(schema.TryGetColumnIndex(name, out int cIdx), $"Could not find '{name}'");
                return schema.GetColumnType(cIdx);
            }
            var pipe = new ConcatEstimator(Env, "f1", "float1")
                .Append(new ConcatEstimator(Env, "f2", "float1", "float1"))
                .Append(new ConcatEstimator(Env, "f3", "float4", "float1"))
                .Append(new ConcatEstimator(Env, "f4", "float6", "vfloat", "float1"));

            data = TakeFilter.Create(Env, data, 10);
            data = pipe.Fit(data).Transform(data);

            ColumnType t;
            t = GetType(data.Schema, "f1");
            Assert.True(t.IsVector && t.ItemType == NumberType.R4 && t.VectorSize == 1);
            t = GetType(data.Schema, "f2");
            Assert.True(t.IsVector && t.ItemType == NumberType.R4 && t.VectorSize == 2);
            t = GetType(data.Schema, "f3");
            Assert.True(t.IsVector && t.ItemType == NumberType.R4 && t.VectorSize == 5);
            t = GetType(data.Schema, "f4");
            Assert.True(t.IsVector && t.ItemType == NumberType.R4 && t.VectorSize == 0);

            data = new ChooseColumnsTransform(Env, data, "f1", "f2", "f3", "f4");

            var subdir = Path.Combine("Transform", "Concat");
            var outputPath = GetOutputPath(subdir, "Concat1.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true, Dense = true });
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, data, fs, keepHidden: false);
            }

            CheckEquality(subdir, "Concat1.tsv");
            Done();
        }

        [Fact]
        public void ConcatWithAliases()
        {
            string dataPath = GetDataPath("adult.test");

            var source = new MultiFileSource(dataPath);
            var loader = new TextLoader(Env, new TextLoader.Arguments
            {
                Column = new[]{
                    new TextLoader.Column("float1", DataKind.R4, 0),
                    new TextLoader.Column("float4", DataKind.R4, new[]{new TextLoader.Range(0), new TextLoader.Range(2), new TextLoader.Range(4), new TextLoader.Range(10) }),
                    new TextLoader.Column("vfloat", DataKind.R4, new[]{new TextLoader.Range(0), new TextLoader.Range(2), new TextLoader.Range(4), new TextLoader.Range(10, null) { AutoEnd = false, VariableEnd = true } })
                },
                Separator = ",",
                HasHeader = true
            }, new MultiFileSource(dataPath));
            var data = loader.Read(source);

            ColumnType GetType(Schema schema, string name)
            {
                Assert.True(schema.TryGetColumnIndex(name, out int cIdx), $"Could not find '{name}'");
                return schema.GetColumnType(cIdx);
            }

            data = TakeFilter.Create(Env, data, 10);

            var concater = new ConcatTransform(Env,
                new ConcatTransform.ColumnInfo("f2", new[] { ("float1", "FLOAT1"), ("float1", "FLOAT2") }),
                new ConcatTransform.ColumnInfo("f3", new[] { ("float4", "FLOAT4"), ("float1", "FLOAT1") }));
            data = concater.Transform(data);

            ColumnType t;
            t = GetType(data.Schema, "f2");
            Assert.True(t.IsVector && t.ItemType == NumberType.R4 && t.VectorSize == 2);
            t = GetType(data.Schema, "f3");
            Assert.True(t.IsVector && t.ItemType == NumberType.R4 && t.VectorSize == 5);

            data = new ChooseColumnsTransform(Env, data, "f2", "f3");

            var subdir = Path.Combine("Transform", "Concat");
            var outputPath = GetOutputPath(subdir, "Concat2.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(Env, new TextSaver.Arguments { Silent = true, Dense = true });
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, data, fs, keepHidden: false);
            }

            CheckEquality(subdir, "Concat2.tsv");
            Done();
        }
    }
}
