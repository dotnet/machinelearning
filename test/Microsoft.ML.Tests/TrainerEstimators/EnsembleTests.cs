// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void TestEstimatorEnsembleTrainer()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetBinaryClassificationPipeline();
            pipe.Append(new EnsembleTrainer(Env, "Features", "Label"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestEstimatorMulticlassDataPartitionEnsembleTrainer()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetMultiClassPipeline();
            pipe.Append(new MulticlassDataPartitionEnsembleTrainer(Env, "Features", "Label"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestEstimatorRegressionEnsembleTrainer()
        {
            var dataView = GetRegressionPipeline();
            var pipe = new RegressionEnsembleTrainer(Env, "Features", "Label");
            TestEstimatorCore(pipe, dataView);
            Done();
        }
    }
}
