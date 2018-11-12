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
            pipe = pipe.Append(new LogisticRegression(Env, "Label", "Features"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestEstimatorMulticlassLogisticRegression()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetMultiClassPipeline();
            pipe = pipe.Append(new MulticlassLogisticRegression(Env, "Label", "Features"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestEstimatorPoissonRegression()
        {
            var dataView = GetRegressionPipeline();
            var pipe = new PoissonRegression(Env, "Label", "Features");
            TestEstimatorCore(pipe, dataView);
            Done();
        }
    }
}
