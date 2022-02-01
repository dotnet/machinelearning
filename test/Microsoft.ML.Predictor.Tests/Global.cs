// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Internal.Internallearn.Test;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.RunTests
{
    public sealed class Global : BaseTestClass
    {
        // See https://github.com/dotnet/machinelearning/issues/1095
        public const string AutoInferenceAndPipelineSweeperTestCollectionName = "TestPipelineSweeper and TestAutoInference should not be run at the same time since it causes deadlocks";

        public Global(ITestOutputHelper output) : base(output)
        {
        }

        [Fact(Skip = "Disabled")]
        public void AssertHandlerTest()
        {
            GlobalBase.AssertHandlerTest();
        }
    }
}
