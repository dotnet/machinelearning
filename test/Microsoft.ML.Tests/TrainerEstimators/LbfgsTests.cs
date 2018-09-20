// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
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


        private IDataView GetIrisDataview()
        {
            return new TextLoader(Env,
            new TextLoader.Arguments()
            {
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.R4, 0),
                    new TextLoader.Column("Features", DataKind.R4, new [] { new TextLoader.Range(1, 4) } )
                }
            }).Read(new MultiFileSource(GetDataPath(TestDatasets.irisLoader.trainFilename)));
        }

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
        public void TestEstimatorLogisticRegression()
        {
            var dataView = GetBreastCancerDataview();
            //var args = new LogisticRegression.Arguments();
            var pipe = new LogisticRegression(Env, "Features", "Label");
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestEstimatorMulticlassLogisticRegression()
        {
            var dataView = GetIrisDataview();
            var pipe = new MulticlassLogisticRegression(Env, "Features", "Label");
            TestEstimatorCore(pipe, dataView);
            Done();
        }

        [Fact]
        public void TestEstimatorPoissonRegression()
        {
            var dataView = GetGeneratedRegressionDataview();
            var pipe = new PoissonRegression(Env, "Features", "Label");
            TestEstimatorCore(pipe, dataView);
            Done();
        }
    }
}
