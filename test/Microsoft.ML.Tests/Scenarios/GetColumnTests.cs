// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFrameworkCommon;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Scenarios
{
    public sealed class GetColumnTests : BaseTestClass
    {
        public GetColumnTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TestGetColumn()
        {
            var path = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var mlContext = new MLContext(1);
            var data = mlContext.Data.LoadFromTextFile(path, new[] {
                new TextLoader.Column("floatScalar", DataKind.Single, 1),
                new TextLoader.Column("floatVector", DataKind.Single, 2, 6),
                new TextLoader.Column("stringScalar", DataKind.String, 4),
                new TextLoader.Column("stringVector", DataKind.String, 5, 7)
            });

            var enum1 = data.GetColumn<float>(data.Schema["floatScalar"]).ToArray();
            var enum2 = data.GetColumn<float[]>(data.Schema["floatVector"]).ToArray();
            var enum3 = data.GetColumn<VBuffer<float>>(data.Schema["floatVector"]).ToArray();

            var enum4 = data.GetColumn<string>(data.Schema["stringScalar"]).ToArray();
            var enum5 = data.GetColumn<string[]>(data.Schema["stringVector"]).ToArray();

            var mustFail = GetMustFail();
            mustFail(() => data.GetColumn<float[]>(data.Schema["floatScalar"]));
            mustFail(() => data.GetColumn<int[]>(data.Schema["floatVector"]));
            mustFail(() => data.GetColumn<int>(data.Schema["floatScalar"]));
            mustFail(() => data.GetColumn<int?>(data.Schema["floatScalar"]));
            mustFail(() => data.GetColumn<string>(data.Schema["floatScalar"]));



            var data1 = mlContext.Data.LoadFromTextFile(path, new[] {
                new TextLoader.Column("floatScalar", DataKind.String, 1),
                new TextLoader.Column("anotherFloatVector", DataKind.Single, 2, 6),
                new TextLoader.Column("stringVector", DataKind.String, 5, 7)
            });

            // Type wrong. Load float as string.
            mustFail(() => data.GetColumn<float>(data1.Schema["floatScalar"]));
            // Name wrong. Load anotherFloatVector from floatVector column.
            mustFail(() => data.GetColumn<float>(data1.Schema["anotherFloatVector"]));
            // Index wrong. stringVector is indexed by 3 in data but 2 in data1.
            mustFail(() => data.GetColumn<string[]>(data1.Schema["stringVector"]).ToArray());
        }

        [Fact]
        public void TestGetColumnSelectedByString()
        {
            var path = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var mlContext = new MLContext(1);
            var data = mlContext.Data.LoadFromTextFile(path, new[] {
                new TextLoader.Column("floatScalar", DataKind.Single, 1),
                new TextLoader.Column("floatVector", DataKind.Single, 2, 6),
                new TextLoader.Column("stringScalar", DataKind.String, 4),
                new TextLoader.Column("stringVector", DataKind.String, 5, 7)
            });

            var enum1 = data.GetColumn<float>("floatScalar").ToArray();
            var enum2 = data.GetColumn<float[]>("floatVector").ToArray();
            var enum3 = data.GetColumn<VBuffer<float>>("floatVector").ToArray();

            var enum4 = data.GetColumn<string>("stringScalar").ToArray();
            var enum5 = data.GetColumn<string[]>("stringVector").ToArray();

            var mustFail = GetMustFail();
            mustFail(() => data.GetColumn<float[]>("floatScalar"));
            mustFail(() => data.GetColumn<int[]>("floatVector"));
            mustFail(() => data.GetColumn<int>("floatScalar"));
            mustFail(() => data.GetColumn<int?>("floatScalar"));
            mustFail(() => data.GetColumn<string>("floatScalar"));
        }

        private static Action<Action> GetMustFail()
        {
            return (Action action) =>
            {
                try
                {
                    action();
                    Assert.False(true);
                }
                catch (ArgumentOutOfRangeException) { }
                catch (InvalidOperationException) { }
                catch (TargetInvocationException ex)
                {
                    Exception e;
                    for (e = ex; e.InnerException != null; e = e.InnerException)
                    {
                    }
                    Assert.True(e is ArgumentOutOfRangeException || e is InvalidOperationException);
                    Assert.True(e.IsMarked());
                }
            };
        }
    }
}
