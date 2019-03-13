// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void TestEstimatorSymSgdClassificationTrainer()
        {
            (var pipe, var dataView) = GetBinaryClassificationPipeline();
            var trainer = new SymbolicSgdTrainer(Env, new SymbolicSgdTrainer.Options());
            var pipeWithTrainer = pipe.Append(trainer);
            TestEstimatorCore(pipeWithTrainer, dataView);

            var transformedDataView = pipe.Fit(dataView).Transform(dataView);
            var model = trainer.Fit(transformedDataView);
            trainer.Fit(transformedDataView, model.Model.SubModel);
            Done();
        }

        [Fact]
        public void TestEstimatorSymSgdInitPredictor()
        {
            (var pipe, var dataView) = GetBinaryClassificationPipeline();
            var transformedData = pipe.Fit(dataView).Transform(dataView);

            var initPredictor = ML.BinaryClassification.Trainers.SdcaCalibrated().Fit(transformedData);
            var data = initPredictor.Transform(transformedData);

            var withInitPredictor = new SymbolicSgdTrainer(Env, new SymbolicSgdTrainer.Options()).Fit(transformedData,
                modelParameters: initPredictor.Model.SubModel);
            var outInitData = withInitPredictor.Transform(transformedData);

            var notInitPredictor = new SymbolicSgdTrainer(Env, new SymbolicSgdTrainer.Options()).Fit(transformedData);
            var outNoInitData = notInitPredictor.Transform(transformedData);

            int numExamples = 10;
            var col1 = data.GetColumn<float>(data.Schema["Score"]).Take(numExamples).ToArray();
            var col2 = outInitData.GetColumn<float>(outInitData.Schema["Score"]).Take(numExamples).ToArray();
            var col3 = outNoInitData.GetColumn<float>(outNoInitData.Schema["Score"]).Take(numExamples).ToArray();

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