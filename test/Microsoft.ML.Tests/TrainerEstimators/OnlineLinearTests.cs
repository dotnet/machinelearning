// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public sealed class OnlineLinearTests : TestDataPipeBase
    {
        public OnlineLinearTests(ITestOutputHelper helper) : base(helper)
        {
        }

        [Fact(Skip = "AP is now uncalibrated but advertises as calibrated")]
        public void OnlineLinearWorkout()
        {
            var dataPath = GetDataPath("breast-cancer.txt");

            var data = TextLoader.CreateReader(Env, ctx => (Label: ctx.LoadFloat(0), Features: ctx.LoadFloat(1, 10)))
                .Read(new MultiFileSource(dataPath));

            var pipe = data.MakeNewEstimator()
                .Append(r => (r.Label, Features: r.Features.Normalize()));

            var trainData = pipe.Fit(data).Transform(data).AsDynamic;

            IEstimator<ITransformer> est = new OnlineGradientDescentTrainer(Env, new OnlineGradientDescentTrainer.Arguments());
            TestEstimatorCore(est, trainData);

            est = new AveragedPerceptronTrainer(Env, new AveragedPerceptronTrainer.Arguments());
            TestEstimatorCore(est, trainData);

            Done();

        }
    }
}
