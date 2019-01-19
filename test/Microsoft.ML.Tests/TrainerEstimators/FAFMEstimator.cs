﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.FactorizationMachine;
using Microsoft.ML.RunTests;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators : TestDataPipeBase
    {
        [Fact]
        public void FieldAwareFactorizationMachine_Estimator()
        {
            var data = new TextLoader(Env, GetFafmBCLoaderArgs())
                    .Read(GetDataPath(TestDatasets.breastCancer.trainFilename));

            var est = new FieldAwareFactorizationMachineTrainer(Env, new[] { "Feature1", "Feature2", "Feature3", "Feature4" }, "Label",
                advancedSettings: s =>
                 {
                     s.Shuffle = false;
                     s.Iters = 3;
                     s.LatentDim = 7;
                 });

            TestEstimatorCore(est, data);
            var model = est.Fit(data);
            var anotherModel = est.Train(data, data, model.Model);

            Done();
        }

        private TextLoader.Arguments GetFafmBCLoaderArgs()
        {
            return new TextLoader.Arguments()
            {
                Separator = "\t",
                HasHeader = false,
                Column = new[]
                {
                    new TextLoader.Column("Feature1", DataKind.R4, new [] { new TextLoader.Range(1, 2) }),
                    new TextLoader.Column("Feature2", DataKind.R4, new [] { new TextLoader.Range(3, 4) }),
                    new TextLoader.Column("Feature3", DataKind.R4, new [] { new TextLoader.Range(5, 6) }),
                    new TextLoader.Column("Feature4", DataKind.R4, new [] { new TextLoader.Range(7, 9) }),
                    new TextLoader.Column("Label", DataKind.BL, 0)
                }
            };
        }
    }
}
