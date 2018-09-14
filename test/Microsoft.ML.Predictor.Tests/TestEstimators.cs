// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

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
    public class EstimatorTests: TestDataPipeBase
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

        public EstimatorTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TestEstimatorRandom()
        {
            using (var env = new TlcEnvironment())
            {
                var dataView = GetBreastCancerDataviewWithTextColumns();
                //dataView = env.CreateTransform("Term{col=F1}", dataView);
                //var result = FeatureCombiner.PrepareFeatures(env, new FeatureCombiner.FeatureCombinerInput() { Data = dataView, Features = new[] { "F1", "F2", "Rest" } }).OutputData;

                var pipe = new RandomTrainer(env, new RandomTrainer.Arguments());
                TestEstimatorCore(pipe, dataView);
            }
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

                var pipe = new PriorTrainer(env, new PriorTrainer.Arguments());
                TestEstimatorCore(pipe, result, invalidInput: dataView);
            }
            Done();
        }
    }
}
