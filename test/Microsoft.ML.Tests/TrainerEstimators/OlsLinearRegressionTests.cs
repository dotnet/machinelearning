// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.HalLearners;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        private IDataView GetGeneratedRegressionDataview()
        {
            return new TextLoader(Env,
            new TextLoader.Arguments()
            {
                Separator = ";",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.R4, 11),
                    new TextLoader.Column("Features", DataKind.R4, new [] { new TextLoader.Range(0, 10) } )
                }
            }).Read(new MultiFileSource(GetDataPath(TestDatasets.generatedRegressionDataset.trainFilename)));
        }

        [Fact]
        public void TestEstimatorOlsLinearRegression()
        {
            var dataView = GetGeneratedRegressionDataview();
            var pipe = new OlsLinearRegressionTrainer(Env, "Features", "Label");
            TestEstimatorCore(pipe, dataView);
            Done();
        }
    }
}
