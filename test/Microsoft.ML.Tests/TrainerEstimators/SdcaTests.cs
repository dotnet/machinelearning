﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void SdcaWorkout()
        {
            var dataPath = GetDataPath("breast-cancer.txt");

            var data = TextLoader.CreateReader(Env, ctx => (Label: ctx.LoadFloat(0), Features: ctx.LoadFloat(1, 10)))
                .Read(dataPath);
            IEstimator<ITransformer> est = new LinearClassificationTrainer(Env, "Features", "Label", advancedSettings: (s) => s.ConvergenceTolerance = 1e-2f);
            TestEstimatorCore(est, data.AsDynamic);

            est = new SdcaRegressionTrainer(Env, "Features", "Label", advancedSettings: (s) => s.ConvergenceTolerance = 1e-2f);
            TestEstimatorCore(est, data.AsDynamic);

            est = new SdcaMultiClassTrainer(Env, "Features", "Label", advancedSettings: (s) => s.ConvergenceTolerance = 1e-2f);
            TestEstimatorCore(est, data.AsDynamic);

            Done();
        }
    }
}
