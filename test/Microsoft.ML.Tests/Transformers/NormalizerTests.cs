// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public sealed class NormalizerTests : TestDataPipeBase
    {
        public NormalizerTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void NormalizerWorkout()
        {
            string dataPath = GetDataPath("iris.txt");

            var loader = new TextLoader(Env, new TextLoader.Arguments
            {
                Column = new[]{
                    new TextLoader.Column("float1", DataKind.R4, 1),
                    new TextLoader.Column("float4", DataKind.R4, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("float4", DataKind.R4, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("double1", DataKind.R8, 1),
                    new TextLoader.Column("double4", DataKind.R8, new[]{new TextLoader.Range(1, 4) }),
                    new TextLoader.Column("int1", DataKind.I4, 0),
                    new TextLoader.Column("float0", DataKind.R4, new[]{ new TextLoader.Range { Min = 1, VariableEnd = true } }),
                },
                HasHeader = true
            }, new MultiFileSource(dataPath));

            var est = new Normalizer(Env, new Normalizer.MinMaxColumn("float1"),
                new Normalizer.MinMaxColumn("float4"),
                new Normalizer.MinMaxColumn("double1"),
                new Normalizer.MinMaxColumn("double4"));

            var data = loader.Read(new MultiFileSource(dataPath));

            var badData1 = new CopyColumnsTransform(Env, ("int1", "float1")).Transform(data);
            var badData2 = new CopyColumnsTransform(Env, ("float0", "float4")).Transform(data);

            TestEstimatorCore(est, data, null, badData1);
            TestEstimatorCore(est, data, null, badData2);
        }
    }
}
