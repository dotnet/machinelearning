// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.SamplesUtils;
using Microsoft.ML.Trainers.HalLearners;
using Xunit;

namespace Microsoft.ML.Functional.Tests
{
    internal static class Common
    {
        public static void CheckMetrics(RegressionMetrics metrics)
        {
            // Perform sanity checks on the metrics
            Assert.True(metrics.Rms >= 0);
            Assert.True(metrics.L1 >= 0);
            Assert.True(metrics.L2 >= 0);
            Assert.True(metrics.RSquared <= 1);
        }
    }
}
