// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Transforms.Text;
using System.IO;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class WordTokenizeTests : TestDataPipeBase
    {
        public WordTokenizeTests(ITestOutputHelper output) : base(output)
        {
        }

        private class TestClass
        {
            public string A;
            [VectorType(2)]
            public string[] B;
        }

        [Fact]
        public void WordTokenizeWorkout()
        {
            var data = new[] { new TestClass() { A = "This is a good sentence.", B = new string[2] { "Much words", "Wow So Cool" } } };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new DelimitedTokenizeEstimator(Env, new[]{
                    new DelimitedTokenizeTransform.ColumnInfo("A", "TokenizeA"),
                    new DelimitedTokenizeTransform.ColumnInfo("B", "TokenizeB"),
                });

            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:TX:0} xf=WordToken{col=B:A} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = "This is a good sentence.", B = new string[2] { "Much words", "Wow So Cool" } } };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new DelimitedTokenizeEstimator(Env, new[]{
                    new DelimitedTokenizeTransform.ColumnInfo("A", "TokenizeA"),
                    new DelimitedTokenizeTransform.ColumnInfo("B", "TokenizeB"),
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
