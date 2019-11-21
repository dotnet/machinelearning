// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void TestEstimatorOlsLinearRegression()
        {
            var dataView = GetRegressionPipeline();
            var trainer = ML.Regression.Trainers.Ols(new OlsTrainer.Options());
            TestEstimatorCore(trainer, dataView);

            var model = trainer.Fit(dataView);
            Assert.True(model.Model.HasStatistics);
            Assert.NotEmpty(model.Model.StandardErrors);
            Assert.NotEmpty(model.Model.PValues);
            Assert.NotEmpty(model.Model.TValues);
            trainer = ML.Regression.Trainers.Ols(new OlsTrainer.Options() { CalculateStatistics = false });
            model = trainer.Fit(dataView);
            Assert.False(model.Model.HasStatistics);
            Done();
        }
    }
}
