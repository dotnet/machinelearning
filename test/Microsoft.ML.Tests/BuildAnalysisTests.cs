// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class BuildAnalysisTests : BaseTestClass
    {
        public BuildAnalysisTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void CiModeExperiment()
        {
            Assert.Fail("MLNET_BUILD_ANALYSIS_CI_MODE_EXPERIMENT_7344");
        }
    }
}
