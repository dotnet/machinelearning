// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class SimpleEstimatorTests : TestDataPipeBase
    {
        private IDataView GetBreastCancerDataviewWithTextColumns()
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            return ImportTextData.TextLoader(Env, new ImportTextData.LoaderInput()
            {
                Arguments =
                {
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("Label", type: null, 0),
                        new TextLoader.Column("F1", DataKind.Text, 1),
                        new TextLoader.Column("F2", DataKind.I4, 2),
                        new TextLoader.Column("Rest", type: null, new [] { new TextLoader.Range(3, 9) })
                    }
                },

                InputFile = inputFile
            }).Data;
        }

        private static SchemaShape.Column MakeFeatureColumn(string featureColumn)
            => new SchemaShape.Column(featureColumn, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false);

        private static SchemaShape.Column MakeLabelColumn(string labelColumn)
            => new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false);

        public SimpleEstimatorTests(ITestOutputHelper output) : base(output)
        {
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
            using (var env = new TlcEnvironment())
            {
                var dataView = GetBreastCancerDataviewWithTextColumns();
                dataView = env.CreateTransform("Term{col=F1}", dataView);
                var result = FeatureCombiner.PrepareFeatures(env, new FeatureCombiner.FeatureCombinerInput() { Data = dataView, Features = new[] { "F1", "F2", "Rest" } }).OutputData;

                var pipe = new PriorTrainer(Contracts.CheckRef(env, nameof(env)).Register("PriorPredictor"), MakeFeatureColumn(DefaultColumnNames.Features), MakeLabelColumn(DefaultColumnNames.Label), null);
                TestEstimatorCore(pipe, result, invalidInput: dataView);
            }
            Done();
        }
    }
}
