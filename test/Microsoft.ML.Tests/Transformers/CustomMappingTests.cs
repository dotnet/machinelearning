// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.ComponentModel.Composition;
using System.ComponentModel.Composition.Hosting;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public sealed class CustomMappingTests : TestDataPipeBase
    {
        public CustomMappingTests(ITestOutputHelper helper) : base(helper)
        {
        }

        public class MyInput
        {
            public float Float1 { get; set; }
            public float[] Float4 { get; set; }
        }

        public class MyOutput
        {
            public string Together { get; set; }
        }

        public class MyLambda
        {
            [Export("MyLambda")]
            public ITransformer MyTransformer => ML.Transforms.CustomMappingTransformer<MyInput, MyOutput>(MyAction, "MyLambda");

            [Import]
            public MLContext ML { get; set; }

            public static void MyAction(MyInput input, MyOutput output)
            {
                output.Together = $"{input.Float1} + {string.Join(", ", input.Float4)}";
            }
        }

        [Fact]
        public void TestCustomTransformer()
        {
            string dataPath = GetDataPath("adult.tiny.with-schema.txt");
            var source = new MultiFileSource(dataPath);
            var loader = ML.Data.CreateTextReader(new[] {
                    new TextLoader.Column("Float1", DataKind.R4, 9),
                    new TextLoader.Column("Float4", DataKind.R4, new[]{new TextLoader.Range(9), new TextLoader.Range(10), new TextLoader.Range(11), new TextLoader.Range(12) })
            }, hasHeader: true);

            var data = loader.Read(source);

            IDataView transformedData;
            // We create a temporary environment to instantiate the custom transformer. This is to ensure that we don't need the same
            // environment for saving and loading.
            var tempoEnv = new MLContext();
            var customEst = new CustomMappingEstimator<MyInput, MyOutput>(tempoEnv, MyLambda.MyAction, "MyLambda");

            try
            {
                TestEstimatorCore(customEst, data);
                Assert.True(false, "Cannot work without MEF injection");
            }
            catch (Exception)
            {
                // REVIEW: we should have a common mechanism that will make sure this is 'our' exception thrown.
            }
            ML.CompositionContainer = new CompositionContainer(new TypeCatalog(typeof(MyLambda)));
            TestEstimatorCore(customEst, data);
            transformedData = customEst.Fit(data).Transform(data);

            var inputs = transformedData.AsEnumerable<MyInput>(ML, true);
            var outputs = transformedData.AsEnumerable<MyOutput>(ML, true);

            Assert.True(inputs.Zip(outputs, (x, y) => y.Together == $"{x.Float1} + {string.Join(", ", x.Float4)}").All(x => x));

            Done();
        }

        [Fact]
        public void TestSchemaPropagation()
        {
            string dataPath = GetDataPath("adult.test");
            var source = new MultiFileSource(dataPath);
            var loader = ML.Data.CreateTextReader(new[] {
                    new TextLoader.Column("Float1", DataKind.R4, 0),
                    new TextLoader.Column("Float4", DataKind.R4, new[]{new TextLoader.Range(0), new TextLoader.Range(2), new TextLoader.Range(4), new TextLoader.Range(10) }),
                    new TextLoader.Column("Text1", DataKind.Text, 0)
            }, hasHeader: true, separatorChar: ',' );

            var data = loader.Read(source);

            Action<MyInput, MyOutput> mapping = (input, output) => output.Together = input.Float1.ToString();
            var est = ML.Transforms.CustomMapping(mapping, null);

            // Make sure schema propagation works for valid data.
            est.GetOutputSchema(SchemaShape.Create(data.Schema));

            var badData1 = ML.Transforms.CopyColumns("Text1", "Float1").Fit(data).Transform(data);
            try
            {
                est.GetOutputSchema(SchemaShape.Create(badData1.Schema));
                Assert.True(false);
            }
            catch (Exception) { }

            var badData2 = ML.Transforms.SelectColumns(new[] { "Float1" }).Fit(data).Transform(data);
            try
            {
                est.GetOutputSchema(SchemaShape.Create(badData2.Schema));
                Assert.True(false);
            }
            catch (Exception) { }

            Done();
        }
    }
}
