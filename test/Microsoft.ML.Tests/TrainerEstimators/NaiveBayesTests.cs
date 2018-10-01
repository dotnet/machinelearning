// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void TestEstimatorMultiClassNaiveBayesTrainer()
        {
            (IEstimator<ITransformer> pipe, IDataView dataView) = GetMultiClassPipeline();
            pipe.Append(new MultiClassNaiveBayesTrainer(Env, "Features", "Label"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }
    }
}