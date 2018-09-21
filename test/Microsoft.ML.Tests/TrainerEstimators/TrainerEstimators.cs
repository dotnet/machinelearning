// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.KMeans;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.RunTests;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators : TestDataPipeBase
    {
        public TrainerEstimators(ITestOutputHelper helper) : base(helper)
        {
        }

        /// <summary>
        /// KMeans TrainerEstimator test 
        /// </summary>
        [Fact]
        public void KMeansEstimator()
        {
            string featureColumn = "NumericFeatures";
            string weightColumn = "Weights";

            var reader = new TextLoader(Env, new TextLoader.Arguments
            {
                HasHeader = true,
                Separator = "\t",
                Column = new[]
                {
                    new TextLoader.Column(featureColumn, DataKind.R4, new [] { new TextLoader.Range(1, 784) }),
                    new TextLoader.Column(weightColumn, DataKind.R4, 0)
                }
            });
            var data = reader.Read(new MultiFileSource(GetDataPath(TestDatasets.mnistTiny28.trainFilename)));


            // Pipeline.
            var pipeline = new KMeansPlusPlusTrainer(Env, featureColumn, weightColumn,
                            advancedSettings: s => { s.InitAlgorithm = KMeansPlusPlusTrainer.InitAlgorithm.KMeansParallel; });

            TestEstimatorCore(pipeline, data);
        }
    }
}
