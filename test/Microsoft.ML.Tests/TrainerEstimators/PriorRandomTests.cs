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
        private IDataView GetBreastCancerDataviewWithTextColumns()
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
                    }).Read(GetDataPath(TestDatasets.breastCancer.trainFilename));
        }

        [Fact]
        public void TestEstimatorRandom()
        {
            var dataView = GetBreastCancerDataviewWithTextColumns();
            var pipe = new RandomTrainer(Env);

            // Test only that the schema propagation works.
            // REVIEW: the save/load is not preserving the full state of the random predictor. This is unfortunate, but we don't care too much at this point.
            TestEstimatorCore(pipe, new EmptyDataView(Env, dataView.Schema));
            Done();
        }

        [Fact]
        public void TestEstimatorPrior()
        {
            var dataView = GetBreastCancerDataviewWithTextColumns();
            
            var pipe = new PriorTrainer(Contracts.CheckRef(Env, nameof(Env)).Register("PriorPredictor"), "Label");
            TestEstimatorCore(pipe, dataView);
            Done();
        }
    }
}
