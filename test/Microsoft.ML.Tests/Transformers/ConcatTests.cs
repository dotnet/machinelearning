﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.RunTests;
using Microsoft.ML.Transforms;
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
            string dataPath = GetDataPath("adult.tiny.with-schema.txt");

            var source = new MultiFileSource(dataPath);
            var loader = new TextLoader(ML, new TextLoader.Arguments
            {
                Columns = new[]{
                    new TextLoader.Column("float1", DataKind.R4, 9),
                    new TextLoader.Column("float4", DataKind.R4, new[]{new TextLoader.Range(9), new TextLoader.Range(10), new TextLoader.Range(11), new TextLoader.Range(12) }),
                    new TextLoader.Column("float6", DataKind.R4, new[]{new TextLoader.Range(9), new TextLoader.Range(10), new TextLoader.Range(11), new TextLoader.Range(12, 14) }),
                    new TextLoader.Column("vfloat", DataKind.R4, new[]{new TextLoader.Range(14, null) { AutoEnd = false, VariableEnd = true } })
                },
                Separator = "\t",
                HasHeader = true
            }, new MultiFileSource(dataPath));
            var data = loader.Read(source);

            DataViewType GetType(DataViewSchema schema, string name)
            {
                Assert.True(schema.TryGetColumnIndex(name, out int cIdx), $"Could not find '{name}'");
                return schema[cIdx].Type;
            }

            var pipe = ML.Transforms.Concatenate("f1", "float1")
                .Append(ML.Transforms.Concatenate("f2", "float1", "float1"))
                .Append(ML.Transforms.Concatenate("f3", "float4", "float1"))
                .Append(ML.Transforms.Concatenate("f4", "float6", "vfloat", "float1"));

            data = ML.Data.TakeRows(data, 10);
            data = pipe.Fit(data).Transform(data);

            DataViewType t;
            t = GetType(data.Schema, "f1");
            Assert.True(t is VectorType vt1 && vt1.ItemType == NumberDataViewType.Single && vt1.Size == 1);
            t = GetType(data.Schema, "f2");
            Assert.True(t is VectorType vt2 && vt2.ItemType == NumberDataViewType.Single && vt2.Size == 2);
            t = GetType(data.Schema, "f3");
            Assert.True(t is VectorType vt3 && vt3.ItemType == NumberDataViewType.Single && vt3.Size == 5);
            t = GetType(data.Schema, "f4");
            Assert.True(t is VectorType vt4 && vt4.ItemType == NumberDataViewType.Single && vt4.Size == 0);

            data = ML.Transforms.SelectColumns("f1", "f2", "f3", "f4").Fit(data).Transform(data);

            var subdir = Path.Combine("Transform", "Concat");
            var outputPath = GetOutputPath(subdir, "Concat1.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, Dense = true });
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, data, fs, keepHidden: false);
            }

            CheckEquality(subdir, "Concat1.tsv");
            Done();
        }

        [Fact]
        public void ConcatWithAliases()
        {
            string dataPath = GetDataPath("adult.tiny.with-schema.txt");

            var source = new MultiFileSource(dataPath);
            var loader = new TextLoader(ML, new TextLoader.Arguments
            {
                Columns = new[]{
                    new TextLoader.Column("float1", DataKind.R4, 9),
                    new TextLoader.Column("float4", DataKind.R4, new[]{new TextLoader.Range(9), new TextLoader.Range(10), new TextLoader.Range(11), new TextLoader.Range(12) }),
                    new TextLoader.Column("vfloat", DataKind.R4, new[]{new TextLoader.Range(9), new TextLoader.Range(10), new TextLoader.Range(11), new TextLoader.Range(12, null) { AutoEnd = false, VariableEnd = true } })
                },
                Separator = "\t",
                HasHeader = true
            }, new MultiFileSource(dataPath));
            var data = loader.Read(source);

            DataViewType GetType(DataViewSchema schema, string name)
            {
                Assert.True(schema.TryGetColumnIndex(name, out int cIdx), $"Could not find '{name}'");
                return schema[cIdx].Type;
            }

            data = ML.Data.TakeRows(data, 10);

            var concater = new ColumnConcatenatingTransformer(ML,
                new ColumnConcatenatingTransformer.ColumnInfo("f2", new[] { ("float1", "FLOAT1"), ("float1", "FLOAT2") }),
                new ColumnConcatenatingTransformer.ColumnInfo("f3", new[] { ("float4", "FLOAT4"), ("float1", "FLOAT1") }));
            data = concater.Transform(data);

            // Test Columns property.
            var columns = concater.Columns;
            var colEnumerator = columns.GetEnumerator();
            colEnumerator.MoveNext();
            Assert.True(colEnumerator.Current.outputColumnName == "f2" && 
                colEnumerator.Current.inputColumnNames[0] == "float1" && 
                colEnumerator.Current.inputColumnNames[1] == "float1");
            colEnumerator.MoveNext();
            Assert.True(colEnumerator.Current.outputColumnName == "f3" &&
                colEnumerator.Current.inputColumnNames[0] == "float4" &&
                colEnumerator.Current.inputColumnNames[1] == "float1");

            DataViewType t;
            t = GetType(data.Schema, "f2");
            Assert.True(t is VectorType vt2 && vt2.ItemType == NumberDataViewType.Single && vt2.Size == 2);
            t = GetType(data.Schema, "f3");
            Assert.True(t is VectorType vt3 && vt3.ItemType == NumberDataViewType.Single && vt3.Size == 5);

            data = ML.Transforms.SelectColumns("f2", "f3" ).Fit(data).Transform(data);

            var subdir = Path.Combine("Transform", "Concat");
            var outputPath = GetOutputPath(subdir, "Concat2.tsv");
            using (var ch = Env.Start("save"))
            {
                var saver = new TextSaver(ML, new TextSaver.Arguments { Silent = true, Dense = true });
                using (var fs = File.Create(outputPath))
                    DataSaverUtils.SaveDataView(ch, saver, data, fs, keepHidden: false);
            }

            CheckEquality(subdir, "Concat2.tsv");
            Done();
        }
    }
}
