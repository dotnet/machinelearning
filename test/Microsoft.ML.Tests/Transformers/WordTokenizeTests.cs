// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms.Text;
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

// Visual Studio complains because the following class members are not never assigned. That is wrong because that class 
// will be implicitly created in runtime and therefore we disable warning 169.
#pragma warning disable 169
        // This is a C# native data structure used to capture the output of ML.NET tokenizer in the test below.
        public class NativeResult
        {
            public string A;
            public string[] B;
            public string[] TokenizeA;
            public string[] TokenizeB;
        }
#pragma warning restore 169


        private class TestWrong
        {
            public float A;
            [VectorType(2)]
            public float[] B;
        }
        [Fact]

        public void WordTokenizeWorkout()
        {
            var data = new[] { new TestClass() { A = "This is a good sentence.", B = new string[2] { "Much words", "Wow So Cool" } } };
            var dataView = ML.Data.LoadFromEnumerable(data);
            var invalidData = new[] { new TestWrong() { A =1, B = new float[2] { 2,3 } } };
            var invalidDataView = ML.Data.LoadFromEnumerable(invalidData);
            var pipe = new WordTokenizingEstimator(Env, new[]{
                    new WordTokenizingEstimator.ColumnOptions("TokenizeA", "A"),
                    new WordTokenizingEstimator.ColumnOptions("TokenizeB", "B"),
                });

            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataView);

            // Reuse the pipe trained on dataView in TestEstimatorCore to make prediction.
            var result = pipe.Fit(dataView).Transform(dataView);

            // Extract the transformed result of the first row (the only row we have because data contains only one TestClass) as a native class.
            var nativeResult = ML.Data.CreateEnumerable<NativeResult>(result, false).First();

            // Check the tokenization of A. Expected result is { "This", "is", "a", "good", "sentence." }.
            var tokenizeA = new[] { "This", "is", "a", "good", "sentence." };
            Assert.True(tokenizeA.Length == nativeResult.TokenizeA.Length);
            for (int i = 0; i < tokenizeA.Length; ++i)
                Assert.Equal(tokenizeA[i], nativeResult.TokenizeA[i]);

            // Check the tokenization of B. Expected result is { "Much", "words", "Wow", "So", "Cool" }. One may think that the expected output
            // should be a 2-D array { { "Much", "words"}, { "Wow", "So", "Cool" } }, but please note that ML.NET may flatten all outputs if
            // they are high-dimension tensors.
            var tokenizeB = new[] { "Much", "words", "Wow", "So", "Cool" };
            Assert.True(tokenizeB.Length == nativeResult.TokenizeB.Length);
            for (int i = 0; i < tokenizeB.Length; ++i)
                Assert.Equal(tokenizeB[i], nativeResult.TokenizeB[i]);

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

            var dataView = ML.Data.LoadFromEnumerable(data);
            var pipe = new WordTokenizingEstimator(Env, new[]{
                    new WordTokenizingEstimator.ColumnOptions("TokenizeA", "A"),
                    new WordTokenizingEstimator.ColumnOptions("TokenizeB", "B"),
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
