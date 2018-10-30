// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.Tools;
using System.ComponentModel.Composition;
using System.ComponentModel.Composition.Hosting;
using System.IO;
using Xunit;
using Xunit.Abstractions;
using System.Linq;

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
            public ITransformer MyTransformer => new CustomMappingTransformer<MyInput, MyOutput>(ML, MyAction, "MyLambda");

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
            ML.PartCatalog.Catalogs.Add(new TypeCatalog(typeof(MyLambda)));

            string dataPath = GetDataPath("adult.test");
            var source = new MultiFileSource(dataPath);
            var loader = ML.Data.TextReader(new[] {
                    new TextLoader.Column("Float1", DataKind.R4, 0),
                    new TextLoader.Column("Float4", DataKind.R4, new[]{new TextLoader.Range(0), new TextLoader.Range(2), new TextLoader.Range(4), new TextLoader.Range(10) })
            }, s => { s.Separator = ","; s.HasHeader = true; });

            var data = loader.Read(source);

            var customEst = new CustomMappingEstimator<MyInput, MyOutput>(ML, MyLambda.MyAction, "MyLambda");
            TestEstimatorCore(customEst, data);

            var transformedData = customEst.Fit(data).Transform(data);

            var inputs = transformedData.AsEnumerable<MyInput>(ML, true);
            var outputs = transformedData.AsEnumerable<MyOutput>(ML, true);

            Assert.True(inputs.Zip(outputs, (x, y) => y.Together == $"{x.Float1} + {string.Join(", ", x.Float4)}").All(x => x));

            Done();
        }
    }
}
