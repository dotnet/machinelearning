// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.SymSgd;
using System.Linq;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void TestEstimatorSymSgdClassificationTrainer()
        {
            (var pipe, var dataView) = GetBinaryClassificationPipeline();
            pipe = pipe.Append(new SymSgdClassificationTrainer(Env, "Features", "Label"));
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestEstimatorSymSgdInitPredictor()
        {
            (var pipe, var dataView) = GetBinaryClassificationPipeline();
            var transformedData = pipe.Fit(dataView).Transform(dataView);

            var initPredictor = new SdcaBinaryTrainer(Env, "Features", "Label").Fit(transformedData);
            var data = initPredictor.Transform(transformedData);

            var withInitPredictor = new SymSgdClassificationTrainer(Env, "Features", "Label").Train(transformedData, initialPredictor: initPredictor.Model);
            var outInitData = withInitPredictor.Transform(transformedData);

            var notInitPredictor = new SymSgdClassificationTrainer(Env, "Features", "Label").Train(transformedData);
            var outNoInitData = notInitPredictor.Transform(transformedData);

            int numExamples = 10;
            var col1 = data.GetColumn<float>(Env, "Score").Take(numExamples).ToArray();
            var col2 = outInitData.GetColumn<float>(Env, "Score").Take(numExamples).ToArray();
            var col3 = outNoInitData.GetColumn<float>(Env, "Score").Take(numExamples).ToArray();

            bool col12Diff = default;
            bool col23Diff = default;
            bool col13Diff = default;

            for (int i = 0; i < numExamples; i++)
            {
                col12Diff = col12Diff || (col1[i] != col2[i]);
                col23Diff = col23Diff || (col2[i] != col3[i]);
                col13Diff = col13Diff || (col1[i] != col3[i]);
            }
            Contracts.Assert(col12Diff && col23Diff && col13Diff);
            Done();
        }
    }
}