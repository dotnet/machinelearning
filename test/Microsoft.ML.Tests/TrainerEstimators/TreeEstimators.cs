// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Xunit;
using System.Linq;
using Microsoft.ML.Runtime.FastTree;
using Xunit.Abstractions;
using Microsoft.ML.Runtime.RunTests;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TreeEstimators : TestDataPipeBase
    {

        public TreeEstimators(ITestOutputHelper output) : base(output)
        {
        }


        /// <summary>
        /// FastTreeBinaryClassification TrainerEstimator test 
        /// </summary>
        [Fact]
        public void FastTreeBinaryEstimator()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                var reader = new TextLoader(env, MakeSentimentTextLoaderArgs());
                var data = reader.Read(new MultiFileSource(dataPath));

                // Pipeline.
                var pipeline = new TextTransform(env, "SentimentText", "Features")
                  .Append(new FastTreeBinaryClassificationTrainer(env, "Label", "Features", advancedSettings: s => { s.NumTrees = 10; }));

                TestEstimatorCore(pipeline, data);
            }
        }

        private static TextLoader.Arguments MakeSentimentTextLoaderArgs()
        {
            return new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.BL, 0),
                    new TextLoader.Column("SentimentText", DataKind.Text, 1)
                }
            };
        }
    }
}
