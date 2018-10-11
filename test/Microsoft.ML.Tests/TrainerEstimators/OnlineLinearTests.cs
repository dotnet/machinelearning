// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FactorizationMachine;
using Microsoft.ML.Runtime.Learners;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void OnlineLinearWorkout()
        {
            var dataPath = GetDataPath("breast-cancer.txt");

            var data = TextLoader.CreateReader(Env, ctx => (Label: ctx.LoadFloat(0), Features: ctx.LoadFloat(1, 10)))
                .Read(new MultiFileSource(dataPath));

            var pipe = data.MakeNewEstimator()
                .Append(r => (r.Label, Features: r.Features.Normalize()));

            var trainData = pipe.Fit(data).Transform(data).AsDynamic;

            IEstimator<ITransformer> est = new OnlineGradientDescentTrainer(Env, "Label", "Features");
            TestEstimatorCore(est, trainData);

            est = new AveragedPerceptronTrainer(Env, "Label", "Features", lossFunction:new HingeLoss.Arguments(), advancedSettings: s =>
            {
                s.LearningRate = 0.5f;
            });
            TestEstimatorCore(est, trainData);

            Done();

        }
    }
}
