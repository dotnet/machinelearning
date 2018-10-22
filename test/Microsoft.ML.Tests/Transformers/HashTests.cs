// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Conversions;
using System;
using System.IO;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class HashTests : TestDataPipeBase
    {
        public HashTests(ITestOutputHelper output) : base(output)
        {
        }

        private class TestClass
        {
            public float A;
            public float B;
            public float C;
        }

        private class TestMeta
        {
            [VectorType(2)]
            public float[] A;
            public float B;
            [VectorType(2)]
            public double[] C;
            public double D;
        }

        [Fact]
        public void HashWorkout()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new HashEstimator(Env, new[]{
                    new HashTransformer.ColumnInfo("A", "HashA", hashBits:4, invertHash:-1),
                    new HashTransformer.ColumnInfo("B", "HashB", hashBits:3, ordered:true),
                    new HashTransformer.ColumnInfo("C", "HashC", seed:42),
                    new HashTransformer.ColumnInfo("A", "HashD"),
                });

            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestMetadata()
        {

            var data = new[] {
                new TestMeta() { A=new float[2] { 3.5f, 2.5f}, B=1, C= new double[2] { 5.1f, 6.1f}, D= 7},
                new TestMeta() { A=new float[2] { 3.5f, 2.5f}, B=1, C= new double[2] { 5.1f, 6.1f}, D= 7},
                new TestMeta() { A=new float[2] { 3.5f, 2.5f}, B=1, C= new double[2] { 5.1f, 6.1f}, D= 7}};


            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new HashEstimator(Env, new[] {
                new HashTransformer.ColumnInfo("A", "HashA", invertHash:1, hashBits:10),
                new HashTransformer.ColumnInfo("A", "HashAUnlim", invertHash:-1, hashBits:10),
                new HashTransformer.ColumnInfo("A", "HashAUnlimOrdered", invertHash:-1, hashBits:10, ordered:true)
            });
            var result = pipe.Fit(dataView).Transform(dataView);
            ValidateMetadata(result);
            Done();
        }

        private void ValidateMetadata(IDataView result)
        {

            Assert.True(result.Schema.TryGetColumnIndex("HashA", out int HashA));
            Assert.True(result.Schema.TryGetColumnIndex("HashAUnlim", out int HashAUnlim));
            Assert.True(result.Schema.TryGetColumnIndex("HashAUnlimOrdered", out int HashAUnlimOrdered));
            VBuffer<ReadOnlyMemory<char>> keys = default;
            var types = result.Schema.GetMetadataTypes(HashA);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.KeyValues });
            result.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, HashA, ref keys);
            Assert.True(keys.Length == 1024);
            //REVIEW: This is weird. I specified invertHash to 1 so I expect only one value to be in key values, but i got two.
            Assert.Equal(keys.Items().Select(x => x.Value.ToString()), new string[2] {"2.5", "3.5" });

            types = result.Schema.GetMetadataTypes(HashAUnlim);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.KeyValues });
            result.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, HashA, ref keys);
            Assert.True(keys.Length == 1024);
            Assert.Equal(keys.Items().Select(x => x.Value.ToString()), new string[2] { "2.5", "3.5" });

            types = result.Schema.GetMetadataTypes(HashAUnlimOrdered);
            Assert.Equal(types.Select(x => x.Key), new string[1] { MetadataUtils.Kinds.KeyValues });
            result.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, HashA, ref keys);
            Assert.True(keys.Length == 1024);
            Assert.Equal(keys.Items().Select(x => x.Value.ToString()), new string[2] { "2.5", "3.5" });
        }

        [Fact]
        public void TestCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:R4:0} xf=Hash{col=B:A} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = 1, B = 2, C = 3, }, new TestClass() { A = 4, B = 5, C = 6 } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new HashEstimator(Env, new[]{
                    new HashTransformer.ColumnInfo("A", "HashA", hashBits:4, invertHash:-1),
                    new HashTransformer.ColumnInfo("B", "HashB", hashBits:3, ordered:true),
                    new HashTransformer.ColumnInfo("C", "HashC", seed:42),
                    new HashTransformer.ColumnInfo("A", "HashD"),
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
    }
}
