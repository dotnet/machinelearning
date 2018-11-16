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
    public class CharTokenizeTests : TestDataPipeBase
    {
        public CharTokenizeTests(ITestOutputHelper output) : base(output)
        {
        }

        private class TestClass
        {
            public string A;
            [VectorType(2)]
            public string[] B;
        }

        private class TestWrong
        {
            public float A;
            [VectorType(2)]
            public float[] B;
        }


        [Fact]
        public void CharTokenizeWorkout()
        {
            var data = new[] { new TestClass() { A = "This is a good sentence.", B = new string[2] { "Much words", "Wow So Cool" } } };
            var dataView = ComponentCreation.CreateDataView(Env, data);
            var invalidData = new[] { new TestWrong() { A = 1, B = new float[2] { 2,3} } };
            var invalidDataView = ComponentCreation.CreateDataView(Env, invalidData);
            var pipe = new TokenizeByCharactersEstimator(Env, columns: new[] { ("A", "TokenizeA"), ("B", "TokenizeB") });

            TestEstimatorCore(pipe, dataView, invalidInput:invalidDataView);
            Done();
        }

        [Fact]
        public void TestCommandLine()
        {
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=A:TX:0} xf=CharToken{col=B:A} in=f:\2.txt" }), (int)0);
        }

        [Fact]
        public void TestOldSavingAndLoading()
        {
            var data = new[] { new TestClass() { A = "This is a good sentence.", B = new string[2] { "Much words", "Wow So Cool" } } };

            var dataView = ComponentCreation.CreateDataView(Env, data);
            var pipe = new TokenizeByCharactersEstimator(Env, columns: new[] { ("A", "TokenizeA"), ("B", "TokenizeB") });
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
