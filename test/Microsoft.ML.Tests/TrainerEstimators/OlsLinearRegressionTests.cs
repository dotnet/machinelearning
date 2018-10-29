// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Trainers.HalLearners;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void TestEstimatorOlsLinearRegression()
        {
            var dataView = GetRegressionPipeline();
            var pipe = new OlsLinearRegressionTrainer(Env, "Features", "Label");
            TestEstimatorCore(pipe, dataView);
            Done();
        }
    }
}
