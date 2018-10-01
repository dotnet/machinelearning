// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Internal.Internallearn.Test;
using Xunit;

namespace Microsoft.ML.Runtime.RunTests
{
    public sealed class Global
    {
        // See https://github.com/dotnet/machinelearning/issues/1095
        public const string AutoInferenceAndPipelineSweeperTestCollectionName = "TestPipelineSweeper and TestAutoInference should not be run at the same time since it causes deadlocks";

        [Fact(Skip = "Disabled")]
        public void AssertHandlerTest()
        {
            GlobalBase.AssertHandlerTest();
        }
    }
}
