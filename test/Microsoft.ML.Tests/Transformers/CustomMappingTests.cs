// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFrameworkCommon;
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

        [CustomMappingFactoryAttribute("MyLambda")]
        public class MyLambda : CustomMappingFactory<MyInput, MyOutput>
        {
            public static void MyAction(MyInput input, MyOutput output)
            {
                output.Together = $"{input.Float1} + {string.Join(", ", input.Float4)}";
            }

            public override Action<MyInput, MyOutput> GetMapping()
            {
                return MyAction;
            }
        }

        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void TestCustomTransformer(bool registerAssembly)
        {
            string dataPath = GetDataPath("adult.tiny.with-schema.txt");
            var source = new MultiFileSource(dataPath);
            var loader = ML.Data.CreateTextLoader(new[] {
                    new TextLoader.Column("Float1", DataKind.Single, 9),
                    new TextLoader.Column("Float4", DataKind.Single, new[]{new TextLoader.Range(9), new TextLoader.Range(10), new TextLoader.Range(11), new TextLoader.Range(12) })
            }, hasHeader: true);

            var data = loader.Load(source);

            IDataView transformedData;
            // We create a temporary environment to instantiate the custom transformer. This is to ensure that we don't need the same
            // environment for saving and loading.
            var tempoEnv = new MLContext(1);
            var customEst = new CustomMappingEstimator<MyInput, MyOutput>(tempoEnv, MyLambda.MyAction, "MyLambda");

            // Before 1.5-preview3 it was required to register the assembly. 
            // Now, the assembly information is automatically saved in the model and the assembly is registered
            // when loading.
            // This tests the case that the CustomTransformer still works even if you explicitly register the assembly
            if (registerAssembly)
                ML.ComponentCatalog.RegisterAssembly(typeof(MyLambda).Assembly);

            TestEstimatorCore(customEst, data);
            transformedData = customEst.Fit(data).Transform(data);

            var inputs = ML.Data.CreateEnumerable<MyInput>(transformedData, true);
            var outputs = ML.Data.CreateEnumerable<MyOutput>(transformedData, true);

            Assert.True(inputs.Zip(outputs, (x, y) => y.Together == $"{x.Float1} + {string.Join(", ", x.Float4)}").All(x => x));

            Done();
        }

        [Fact]
        public void TestSchemaPropagation()
        {
            string dataPath = GetDataPath("adult.test");
            var source = new MultiFileSource(dataPath);
            var loader = ML.Data.CreateTextLoader(new[] {
                    new TextLoader.Column("Float1", DataKind.Single, 0),
                    new TextLoader.Column("Float4", DataKind.Single, new[]{new TextLoader.Range(0), new TextLoader.Range(2), new TextLoader.Range(4), new TextLoader.Range(10) }),
                    new TextLoader.Column("Text1", DataKind.String, 0)
            }, separatorChar: ',', hasHeader: true);

            var data = loader.Load(source);

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

        public class MyStatefulInput
        {
            public float Value { get; set; }
        }

        public class MyState
        {
            public HashSet<float> SeenValues;
        }

        public class MyStatefulOutput
        {
            public bool FirstAppearance { get; set; }
        }

        [CustomMappingFactoryAttribute(nameof(MyStatefulLambda))]
        public class MyStatefulLambda : StatefulCustomMappingFactory<MyStatefulInput, MyStatefulOutput, MyState>
        {
            public override Action<MyStatefulInput, MyStatefulOutput, MyState> GetMapping()
            {
                return MyStatefulAction;
            }

            public override Action<MyState> GetStateInitAction()
            {
                return MyStateInit;
            }

            public static void MyStatefulAction(MyStatefulInput input, MyStatefulOutput output, MyState state)
            {
                output.FirstAppearance = !state.SeenValues.Contains(input.Value);
                state.SeenValues.Add(input.Value);
            }

            public static void MyStateInit(MyState state)
            {
                state.SeenValues = new HashSet<float>();
            }
        }

        [Fact]
        public void TestStatefulCustomMappingTransformer()
        {
            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var source = new MultiFileSource(dataPath);
            var loader = ML.Data.CreateTextLoader(new[] {
                new TextLoader.Column("Features", DataKind.Single, 1, 9),
                new TextLoader.Column("Label", DataKind.String, 0),
                new TextLoader.Column("Value", DataKind.Single, 2),
            });
            var data = loader.Load(source);

            // We create a temporary environment to instantiate the custom transformer. This is to ensure that we don't need the same
            // environment for saving and loading.
            var tempoEnv = new MLContext();
            var customEst = tempoEnv.Transforms.StatefulCustomMapping<MyStatefulInput, MyStatefulOutput, MyState>(MyStatefulLambda.MyStatefulAction, MyStatefulLambda.MyStateInit, nameof(MyStatefulLambda));

            TestEstimatorCore(customEst, data);
            var transformedData = customEst.Fit(data).Transform(data);
            var outputs = transformedData.GetColumn<bool>(transformedData.Schema[nameof(MyStatefulOutput.FirstAppearance)]);
            Assert.Equal(10, outputs.Count(output => output));

            Done();
        }

        [Fact]
        public void TestCustomFilter()
        {
            string dataPath = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var source = new MultiFileSource(dataPath);
            var loader = ML.Data.CreateTextLoader(new[] {
                new TextLoader.Column("Float4", DataKind.Single, 1, 4),
                new TextLoader.Column("Float1", DataKind.Single, 2),
            });
            var data = loader.Load(source);

            var filteredData = ML.Data.FilterByCustomPredicate<MyInput>(data, input => input.Float1 % 2 == 0);
            Assert.True(filteredData.GetColumn<float>(filteredData.Schema[nameof(MyInput.Float1)]).All(x => x % 2 == 1));
        }

        private sealed class MyFilterState
        {
            public int Count { get; set; }
        }

        private sealed class MyFilterInput
        {
            public int Counter { get; set; }
            public int Value { get; set; }
        }

        [Fact]
        public void TestStatefulCustomFilter()
        {
            var data = ML.Data.LoadFromEnumerable(new[]
            {
                new MyFilterInput() { Counter = 0, Value = 1 },
                new MyFilterInput() { Counter = 1, Value = 1 },
                new MyFilterInput() { Counter = 2, Value = 2 },
                new MyFilterInput() { Counter = 3, Value = 0 },
                new MyFilterInput() { Counter = 4, Value = 2 },
                new MyFilterInput() { Counter = 5, Value = 4 },
                new MyFilterInput() { Counter = 6, Value = 1 },
                new MyFilterInput() { Counter = 7, Value = 1 },
                new MyFilterInput() { Counter = 8, Value = 2 },
            });

            var filteredData = ML.Data.FilterByStatefulCustomPredicate<MyFilterInput, MyFilterState>(data,
                (input, state) =>
                {
                    if (state.Count++ % 2 == 0)
                        return input.Value % 2 == 0;
                    else
                        return input.Value % 2 == 1;
                }, state => state.Count = 0);

            var values = filteredData.GetColumn<int>(filteredData.Schema[nameof(MyFilterInput.Value)]);
            var counter = filteredData.GetColumn<int>(filteredData.Schema[nameof(MyFilterInput.Counter)]);
            Assert.Equal(new[] { 0, 3, 5, 6 }, counter);
            Assert.Equal(new[] { 1, 0, 4, 1 }, values);
        }
    }
}
