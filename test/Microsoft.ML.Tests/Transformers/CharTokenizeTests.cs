// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms.Text;
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
            var dataView = ML.Data.LoadFromEnumerable(data);
            var invalidData = new[] { new TestWrong() { A = 1, B = new float[2] { 2,3} } };
            var invalidDataView = ML.Data.LoadFromEnumerable(invalidData);
            var pipe = new TokenizingByCharactersEstimator(Env, columns: new[] { ("TokenizeA", "A"), ("TokenizeB", "B") });

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

            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = new TokenizingByCharactersEstimator(Env, columns: new[] { ("TokenizeA", "A"), ("TokenizeB", "B") });
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
