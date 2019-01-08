// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void SdcaWorkout()
        {
            var dataPath = GetDataPath("breast-cancer.txt");

            var data = TextLoaderStatic.CreateReader(Env, ctx => (Label: ctx.LoadFloat(0), Features: ctx.LoadFloat(1, 10)))
                .Read(dataPath).Cache();

            var binaryTrainer = new SdcaBinaryTrainer(Env, "Label", "Features", advancedSettings: (s) => s.ConvergenceTolerance = 1e-2f);
            TestEstimatorCore(binaryTrainer, data.AsDynamic);

            var regressionTrainer = new SdcaRegressionTrainer(Env, "Label", "Features", advancedSettings: (s) => s.ConvergenceTolerance = 1e-2f);
            TestEstimatorCore(regressionTrainer, data.AsDynamic);

            var mcTrainer = new SdcaMultiClassTrainer(Env, "Label", "Features", advancedSettings: (s) => s.ConvergenceTolerance = 1e-2f);
            TestEstimatorCore(mcTrainer, data.AsDynamic);

            Done();
        }
    }
}
