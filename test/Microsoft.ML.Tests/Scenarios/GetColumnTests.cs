// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Api;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.TestFramework;
using System;
using System.Linq;
using System.Reflection;
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
            var path = GetDataPath("breast-cancer.txt");
            var env = new LocalEnvironment();
            var data = TextLoader.CreateReader(env, ctx => (
                floatScalar: ctx.LoadFloat(1),
                floatVector: ctx.LoadFloat(2, 6),
                stringScalar: ctx.LoadText(4),
                stringVector: ctx.LoadText(5, 7)
            )).Read(new MultiFileSource(path));

            Action<Action> mustFail = (Action action) =>
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

            var enum1 = data.AsDynamic.GetColumn<float>(env, "floatScalar").ToArray();
            var enum2 = data.AsDynamic.GetColumn<float[]>(env, "floatVector").ToArray();
            var enum3 = data.AsDynamic.GetColumn<VBuffer<float>>(env, "floatVector").ToArray();

            var enum4 = data.AsDynamic.GetColumn<string>(env, "stringScalar").ToArray();
            var enum5 = data.AsDynamic.GetColumn<string[]>(env, "stringVector").ToArray();

            var enum6 = data.AsDynamic.GetColumn<float?>(env, "floatScalar").ToArray();
            var enum7 = data.AsDynamic.GetColumn<float?[]>(env, "floatVector").ToArray();

            mustFail(() => data.AsDynamic.GetColumn<float[]>(env, "floatScalar"));
            mustFail(() => data.AsDynamic.GetColumn<int[]>(env, "floatVector"));
            mustFail(() => data.AsDynamic.GetColumn<int>(env, "floatScalar"));
            mustFail(() => data.AsDynamic.GetColumn<int?>(env, "floatScalar"));
            mustFail(() => data.AsDynamic.GetColumn<string>(env, "floatScalar"));
        }
    }
}
