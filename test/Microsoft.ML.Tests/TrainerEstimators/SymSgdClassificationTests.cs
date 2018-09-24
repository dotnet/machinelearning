// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Runtime.SymSgd;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        private IDataView GetBreastCancerDataview()
        {
            return new TextLoader(Env,
            new TextLoader.Arguments()
            {
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.R4, 0),
                    new TextLoader.Column("Features", DataKind.R4, new [] { new TextLoader.Range(1, 9) } )
                }
            }).Read(new MultiFileSource(GetDataPath(TestDatasets.breastCancer.trainFilename)));
        }

        [Fact]
        public void TestEstimatorSymSgdClassificationTrainer()
        {
            var dataView = GetBreastCancerDataview();
            var pipe = new SymSgdClassificationTrainer(Env, "Features", "Label");
            TestEstimatorCore(pipe, dataView);
            Done();
        }
    }
}