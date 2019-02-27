﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        private IDataView GetBreastCancerDataviewWithTextColumns()
        {
            return new TextLoader(Env,
                    new TextLoader.Options()
                    {
                        HasHeader = true,
                        Columns = new[]
                        {
                            new TextLoader.Column("Label", DataKind.Single, 0),
                            new TextLoader.Column("F1", DataKind.String, 1),
                            new TextLoader.Column("F2", DataKind.Int32, 2),
                            new TextLoader.Column("Rest", DataKind.Single, new [] { new TextLoader.Range(3, 9) })
                        }
                    }).Load(GetDataPath(TestDatasets.breastCancer.trainFilename));
        }

        [Fact]
        public void TestEstimatorRandom()
        {
            var dataView = GetBreastCancerDataviewWithTextColumns();
            var pipe = ML.BinaryClassification.Trainers.Random();

            // Test only that the schema propagation works.
            // REVIEW: the save/load is not preserving the full state of the random predictor. This is unfortunate, but we don't care too much at this point.
            TestEstimatorCore(pipe, new EmptyDataView(Env, dataView.Schema));
            Done();
        }

        [Fact]
        public void TestEstimatorPrior()
        {
            var dataView = GetBreastCancerDataviewWithTextColumns();

            var pipe = ML.BinaryClassification.Trainers.Prior("Label");
            TestEstimatorCore(pipe, dataView);
            Done();
        }
    }
}
