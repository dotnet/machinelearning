// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.TestFramework;
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
            var env = new MLContext();
            var data = TextLoaderStatic.CreateLoader(env, ctx => (
                floatScalar: ctx.LoadFloat(1),
                floatVector: ctx.LoadFloat(2, 6),
                stringScalar: ctx.LoadText(4),
                stringVector: ctx.LoadText(5, 7)
            )).Load(path);

            var enum1 = data.AsDynamic.GetColumn<float>(data.AsDynamic.Schema["floatScalar"]).ToArray();
            var enum2 = data.AsDynamic.GetColumn<float[]>(data.AsDynamic.Schema["floatVector"]).ToArray();
            var enum3 = data.AsDynamic.GetColumn<VBuffer<float>>(data.AsDynamic.Schema["floatVector"]).ToArray();

            var enum4 = data.AsDynamic.GetColumn<string>(data.AsDynamic.Schema["stringScalar"]).ToArray();
            var enum5 = data.AsDynamic.GetColumn<string[]>(data.AsDynamic.Schema["stringVector"]).ToArray();

            var mustFail = GetMustFail();
            mustFail(() => data.AsDynamic.GetColumn<float[]>(data.AsDynamic.Schema["floatScalar"]));
            mustFail(() => data.AsDynamic.GetColumn<int[]>(data.AsDynamic.Schema["floatVector"]));
            mustFail(() => data.AsDynamic.GetColumn<int>(data.AsDynamic.Schema["floatScalar"]));
            mustFail(() => data.AsDynamic.GetColumn<int?>(data.AsDynamic.Schema["floatScalar"]));
            mustFail(() => data.AsDynamic.GetColumn<string>(data.AsDynamic.Schema["floatScalar"]));


            // Static types.
            var enum8 = data.GetColumn(r => r.floatScalar);
            var enum9 = data.GetColumn(r => r.floatVector);
            var enum10 = data.GetColumn(r => r.stringScalar);
            var enum11 = data.GetColumn(r => r.stringVector);

            var data1 = TextLoaderStatic.CreateLoader(env, ctx => (
                floatScalar: ctx.LoadText(1),
                anotherFloatVector: ctx.LoadFloat(2, 6),
                stringVector: ctx.LoadText(5, 7)
            )).Load(path);

            // Type wrong. Load float as string.
            mustFail(() => data.AsDynamic.GetColumn<float>(data1.AsDynamic.Schema["floatScalar"]));
            // Name wrong. Load anotherFloatVector from floatVector column.
            mustFail(() => data.AsDynamic.GetColumn<float>(data1.AsDynamic.Schema["anotherFloatVector"]));
            // Index wrong. stringVector is indexed by 3 in data but 2 in data1.
            mustFail(() => data.AsDynamic.GetColumn<string[]>(data1.AsDynamic.Schema["stringVector"]).ToArray());
        }

        [Fact]
        public void TestGetColumnSelectedByString()
        {
            var path = GetDataPath(TestDatasets.breastCancer.trainFilename);
            var env = new MLContext();
            var data = TextLoaderStatic.CreateLoader(env, ctx => (
                floatScalar: ctx.LoadFloat(1),
                floatVector: ctx.LoadFloat(2, 6),
                stringScalar: ctx.LoadText(4),
                stringVector: ctx.LoadText(5, 7)
            )).Load(path);

            var enum1 = data.AsDynamic.GetColumn<float>("floatScalar").ToArray();
            var enum2 = data.AsDynamic.GetColumn<float[]>("floatVector").ToArray();
            var enum3 = data.AsDynamic.GetColumn<VBuffer<float>>("floatVector").ToArray();

            var enum4 = data.AsDynamic.GetColumn<string>("stringScalar").ToArray();
            var enum5 = data.AsDynamic.GetColumn<string[]>("stringVector").ToArray();

            var mustFail = GetMustFail();
            mustFail(() => data.AsDynamic.GetColumn<float[]>("floatScalar"));
            mustFail(() => data.AsDynamic.GetColumn<int[]>("floatVector"));
            mustFail(() => data.AsDynamic.GetColumn<int>("floatScalar"));
            mustFail(() => data.AsDynamic.GetColumn<int?>("floatScalar"));
            mustFail(() => data.AsDynamic.GetColumn<string>("floatScalar"));
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
