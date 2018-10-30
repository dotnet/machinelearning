// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void TestEstimatorLogisticRegression()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetBinaryClassificationPipeline();
            pipe = pipe.Append(new LogisticRegression(Env, "Features", "Label"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestEstimatorMulticlassLogisticRegression()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetMultiClassPipeline();
            pipe = pipe.Append(new MulticlassLogisticRegression(Env, "Features", "Label"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestEstimatorPoissonRegression()
        {
            var dataView = GetRegressionPipeline();
            var pipe = new PoissonRegression(Env, "Features", "Label");
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestLogisticRegressionStats()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetBinaryClassificationPipeline();

            var trainer = new LogisticRegression(Env, "Features", "Label", advancedSettings: s=> { s.ShowTrainingStats = true; });
            pipe = pipe.Append(trainer);
            var transformer = pipe.Fit(dataView);

            using (var ch = Env.Start("Calcuating STD for LR."))
                trainer.ComputeExtendedTrainingStatistics(ch);

            Done();
        }
    }
}
