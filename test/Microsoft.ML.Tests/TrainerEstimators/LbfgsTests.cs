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
                            new TextLoader.Column("Label", type: null, 0),
                            new TextLoader.Column("F1", DataKind.Text, 1),
                            new TextLoader.Column("F2", DataKind.I4, 2),
                            new TextLoader.Column("Rest", type: null, new [] { new TextLoader.Range(3, 9) })
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
                    new TextLoader.Column("SepalLength", DataKind.R4, 1),
                    new TextLoader.Column("SepalWidth", DataKind.R4, 2),
                    new TextLoader.Column("PetalLength", DataKind.R4, 3),
                    new TextLoader.Column("PetalWidth", DataKind.R4, 4)
                }
            }).Read(new MultiFileSource(GetDataPath(TestDatasets.irisData.trainFilename)));
        }

        private IDataView GetGeneratedRegressionDataview()
        {
            return new TextLoader(Env,
            new TextLoader.Arguments()
            {
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.R4, 11),
                    new TextLoader.Column("Features", DataKind.R4, new [] { new TextLoader.Range(0, 10) } )
                }
            }).Read(new MultiFileSource(GetDataPath(TestDatasets.generatedRegressionDatasetmacro.trainFilename)));
        }

        [Fact]
        public void TestEstimatorLogisticRegression()
        {
            var dataView = GetBreastCancerDataview();
            dataView = Env.CreateTransform("Term{col=F1}", dataView);
            var data = FeatureCombiner.PrepareFeatures(Env, new FeatureCombiner.FeatureCombinerInput() { Data = dataView, Features = new[] { "F1", "F2", "Rest" } }).OutputData;

            //var args = new LogisticRegression.Arguments();
            var pipe = new LogisticRegression(Env, "Features", "Label");
            TestEstimatorCore(pipe, data);
            Done();
        }

        [Fact]
        public void TestEstimatorMulticlassLogisticRegression()
        {
            var dataView = GetIrisDataview();
            var data = FeatureCombiner.PrepareFeatures(Env, new FeatureCombiner.FeatureCombinerInput() { Data = dataView, Features = new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" } }).OutputData;

            var pipe = new LogisticRegression(Env, "Features", "Label");
            TestEstimatorCore(pipe, data);
            Done();
        }

        [Fact]
        public void TestEstimatorPoissonRegression()
        {
            var dataView = GetGeneratedRegressionDataview();
            var pipe = new LogisticRegression(Env, "Features", "Label");
            TestEstimatorCore(pipe, dataView);
            Done();
        }
    }
}
